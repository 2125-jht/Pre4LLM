# -*- coding: utf-8 -*-
"""
多兴趣推荐模型 - 基于Transformer的编码器-解码器架构
用于用户兴趣建模和内容推荐

主要功能：
1. 基于用户静态特征和点击行为建模用户兴趣
2. 使用Transformer编码器提取用户兴趣表示
3. 使用Transformer解码器生成推荐sid
4. 支持束搜索(Beam Search)进行sid生成
"""
import tensorflow as tf
import sys
from feature_attr_extract import *
from modulesV2 import *
from modules_ import *

# 用户静态特征名称列表
# 包含用户的基本属性信息
user_static_fea_names = [
    "user_id",          # 用户ID
    "user_gender",      # 用户性别
    "user_age_segment", # 用户年龄段
    "user_level"        # 用户等级
]

# 用户点击行为特征名称列表
# 包含用户的历史交互行为数据
user_click_fea_names = [
    "user_profile_v1_click_pid_list",  # 用户点击的视频ID列表
    "user_profile_v1_click_aid_list"   # 用户点击的作者ID列表
]

class MultiInterestModel(object):
    """
    多兴趣推荐模型类
    
    该模型使用Transformer架构，通过编码器-解码器结构：
    1. 编码器：将用户特征和行为序列编码为多个兴趣表示
    2. 解码器：基于兴趣表示生成推荐的语义ID序列
    """
    
    def __init__(self, feature_emb_dict, feature_emb_size_dict, dim=256, vocab_sizes=[8192, 8192, 8192], print_ops=None):
        """
        初始化多兴趣模型
        
        Args:
            feature_emb_dict: 特征嵌入字典，存储各特征的嵌入向量
            feature_emb_size_dict: 特征嵌入维度字典
            dim: 模型隐藏层维度，默认64
            vocab_sizes: 各个词汇表大小的列表，对应不同语义层级，默认[8192, 8192, 8192]
            print_ops: 用于调试的打印操作列表
        """
        self._feature_emb_dict = feature_emb_dict
        self._feature_emb_size_dict = feature_emb_size_dict
        self._print_ops = print_ops
        self._vocab_sizes = vocab_sizes  # 三个语义层级的词汇表大小
        self._total_vocab_size = sum(self._vocab_sizes)  # 总词汇表大小
        # 诊断性简化实验：先把多兴趣头数从 8 降到 4，降低 router 学习难度
        self._query_token_numb=4
        self._stage_one_dim=dim
        self._stage_two_dim=dim
        self._dim = dim
        self._router_token_id = self._total_vocab_size + self._query_token_numb
        self._start_token_id = self._router_token_id + 1
        self._posterior_temperature = 1
        self._router_loss_weight = 3
        self._router_infer_top_k = 2
        
        # query token、sid token 和 decoder start token 共用一套 embedding
        self._embedding = tf.get_variable(
            shape=[self._start_token_id + 1, dim],
            name='embedding',
            initializer=tf.random_uniform_initializer(minval=-1.0/dim, maxval=1.0/dim),
            trainable=True
        )

    def _masked_mean(self, values, mask):
        mask = tf.cast(mask, tf.float32)
        return tf.reduce_sum(values * mask) / (tf.reduce_sum(mask) + 1e-9)

    def _gather_by_head(self, tensor, head_idx):
        batch_size = tf.shape(tensor)[0]
        gather_idx = tf.stack([tf.range(batch_size, dtype=tf.int32), head_idx], axis=1)
        return tf.gather_nd(tensor, gather_idx)

    def _build_prior_router(self, router_token_output, head_state):
        """
        prior router 只看用户上下文和全部候选 head state。
        router token 先汇总用户行为，再和每个 head 的状态做 matching 打分。
        """
        router_embed = tf.layers.dense(
            router_token_output,
            self._stage_two_dim,
            name='router_project',
            activation=tf.nn.leaky_relu,
            reuse=tf.AUTO_REUSE
        )  # [b, dim]
        router_embed = tf.expand_dims(router_embed, axis=1)  # [b,1,dim]
        router_embed_tiled = tf.tile(router_embed, [1, self._query_token_numb, 1])  # [b,q,dim]

        router_input = tf.concat(
            [
                router_embed_tiled,
                head_state,
                router_embed_tiled * head_state,
                router_embed_tiled - head_state
            ],
            axis=-1
        )
        router_hidden = mlp(
            'prior_router',
            router_input,
            [2 * self._stage_two_dim],
            self._stage_two_dim,
            activation=tf.nn.leaky_relu
        )
        router_logits = tf.layers.dense(
            router_hidden,
            1,
            name='prior_router_logit',
            reuse=tf.AUTO_REUSE
        )  # [b,q,1]
        router_logits = tf.squeeze(router_logits, axis=-1)  # [b,q]
        router_probs = tf.nn.softmax(router_logits, axis=-1)
        return router_logits, router_probs
        

    def model(self, photo_sid, label, photo_semantic_id_int):
        """
        主训练模型前向传播
        
        Args:
            photo_sid: 视频语义ID序列，shape=[batch_size, seq_len]
            label: 真实标签，shape=[batch_size, 3]，对应三个语义层级
            photo_semantic_id_int: 视频语义ID整数序列，用于计算loss mask
            
        Returns:
            loss: 训练损失值
        """
        print_tensor("user_click_list_length", self._feature_emb_size_dict['user_profile_v1_click_pid_list'])
        # === 1. 用户静态特征处理 ===
        # 拼接所有用户静态特征
        user_static_fea = tf.concat([self._feature_emb_dict[fea] for fea in user_static_fea_names], axis=1)
        # 通过MLP将静态特征映射到指定维度
        user_static_emb = mlp('user_static_emb', user_static_fea, [2*self._stage_one_dim], self._stage_one_dim, activation=tf.nn.leaky_relu)
        batch_size = tf.shape(user_static_emb)[0]
        # 调整形状为[batch_size, 1, dim]，作为序列的一个元素
        user_static_emb = tf.reshape(user_static_emb, [batch_size, 1,self._stage_one_dim])

        # === 2. 用户点击行为特征处理 ===
        # 2.1 拼接原始点击特征
        user_click_fea = tf.concat(
            [self._feature_emb_dict[fea] for fea in user_click_fea_names], axis=2)   # [B, L, dim]
        
        # 2.2 仅保留“有效序列的最后 selected_size 条”
        max_len  = 200                                             
        raw_len  = tf.cast(
            self._feature_emb_size_dict['user_profile_v1_click_pid_list'],
            tf.int32)                 # 可能是 [B,1] 也可能是 [B]
        valid_len = tf.reshape(raw_len, [-1])      # 强制展平成 [B]
        print_tensor("valid_len", valid_len)
        
        max_len_i  = tf.constant(max_len, dtype=tf.int32)           # 256
        used_len   = tf.minimum(valid_len, max_len_i)               # [B]  小于 256 保留原值，≥256 置 256
        print_tensor("used_len", used_len)                          # 打印结果

        # 通过MLP处理点击特征
        user_click_emb = mlp('user_click_emb', user_click_fea, [4*self._stage_one_dim], self._stage_one_dim, activation=tf.nn.leaky_relu)
        
        # debug
        self._print_ops.append(tf.print("user_click_emb first sample:", user_click_emb[0,:,1], summarize=100))
        
        # === 3. 构建query token ===
        query_token_indice_1d = tf.range(
            start=self._total_vocab_size,
            limit=self._total_vocab_size + self._query_token_numb,
            delta=1,
            dtype=tf.int32)  # [q]
        query_token_indice_2d = tf.expand_dims(query_token_indice_1d, axis=0)  # [1,q]
        query_token_indice = tf.tile(query_token_indice_2d, multiples=[batch_size, 1])  # [b,q]
        query_token_emb = tf.nn.embedding_lookup(self._embedding, query_token_indice)  # [b,q,dim]

        print_tensor('query_token_sim/input', calc_sim_cos_btd(query_token_emb))

        # router special token：用于汇总用户上下文并给兴趣头打分
        router_token_indice = tf.fill([batch_size, 1], tf.constant(self._router_token_id, dtype=tf.int32))  # [b,1]
        router_token_emb = tf.nn.embedding_lookup(self._embedding, router_token_indice)  # [b,1,dim]

        # === 4. 构建编码器输入：query token + router token 前置 ===
        encoder_input = tf.concat([query_token_emb, router_token_emb, user_static_emb, user_click_emb], axis=1)

        encoder_input_sim = tf.reshape(encoder_input[:,self._query_token_numb+1:,:], [batch_size, -1])
        print_tensor("encoder_input_sim", calc_sim_cos(encoder_input_sim))

        # === 4-A. 构建 Encoder 的 padding mask =============================
        # 整个序列长度 = query_token_numb + 1（router token）+ 1（user token）+ max_len（点击序列）
        total_len  = self._query_token_numb + 1 + 1 + max_len
        B          = tf.shape(used_len)[0]

        click_mask = tf.sequence_mask(
            lengths=used_len,
            maxlen=max_len,
            dtype=tf.int8)  # [B, max_len]

        query_tok  = tf.ones([B, self._query_token_numb], dtype=tf.int8)  # [B, Q]
        router_tok = tf.ones([B, 1], dtype=tf.int8)  # [B,1]
        user_tok   = tf.ones([B, 1], dtype=tf.int8)  # [B,1]

        seq_mask   = tf.concat([query_tok, router_tok, user_tok, click_mask], axis=1)  # [B, total_len]

        self._print_ops.append(tf.print("seq_mask first sample:", seq_mask[0], summarize=100))

        src_mask = tf.reshape(seq_mask, [B, 1, 1, total_len])  # [B, 1, 1, total_len]

        encoder_model = EncoderModel(num_layers=4, dim=self._stage_one_dim, num_heads=8, dropout_rate=0.1, hidden_dim=self._stage_one_dim*2, training=True)
        encoder_output = encoder_model.forward(encoder_input, src_mask, training=True)  # [batch_size, total_len, dim]
        # posterior teacher 分支关闭 dropout，给 router 一个更稳定的蒸馏目标
        encoder_output_teacher = encoder_model.forward(encoder_input, src_mask, training=False)  # [batch_size, total_len, dim]

        encoder_output_sim = tf.reshape(encoder_output[:,self._query_token_numb+1:,:], [batch_size, -1])
        print_tensor("encoder_output_sim", calc_sim_cos(encoder_output_sim))

        # === 5. 取前 Q 个位置作为多兴趣表示 ===
        coarse_interest = encoder_output[:, :self._query_token_numb, :]  # [b, q, dim]
        router_token_output = tf.squeeze(encoder_output[:, self._query_token_numb:self._query_token_numb+1, :], axis=1)  # [b, dim]
        coarse_interest_teacher = encoder_output_teacher[:, :self._query_token_numb, :]  # [b, q, dim]
        router_token_output_teacher = tf.squeeze(encoder_output_teacher[:, self._query_token_numb:self._query_token_numb+1, :], axis=1)  # [b, dim]

        print_tensor('query_token_sim/output', calc_sim_cos_btd(coarse_interest))

        interest_embed = tf.layers.dense(coarse_interest, self._stage_two_dim, name='project', activation=tf.nn.leaky_relu, reuse=tf.AUTO_REUSE) # [b, q, dim]从兴趣提取语义到sid生成
        interest_embed_teacher = tf.layers.dense(coarse_interest_teacher, self._stage_two_dim, name='project', activation=tf.nn.leaky_relu, reuse=tf.AUTO_REUSE) # [b, q, dim]

        #计算output query token之间的相似度
        print_tensor('query_token_sim/interest_embed', calc_sim_cos_btd(interest_embed)) 

        # 使用 router token 构造 prior router，per-head 输入改成第一步解码 hidden state
        # 单阶段训练下，先切断 router -> backbone 的梯度，避免 prior router 反向扰动 encoder / interest head
        prior_router_input = tf.stop_gradient(router_token_output_teacher)  # [b,dim]

        #构造fine_item_input的输入，训练时每个 head 单独解码一遍 GT，用于构造 posterior teacher
        start_token_indice = tf.tile(tf.constant(self._start_token_id, shape=(1, 1), dtype=tf.int32), [batch_size, 1])
        photo_with_start_token = tf.concat([start_token_indice, photo_sid[:,:-1]], axis=1) 
        input_sid_embed = tf.nn.embedding_lookup(self._embedding, photo_with_start_token) #[b,len(vocab_sizes),_stage_two_dim]

        interest_embed_tiled = tf.expand_dims(interest_embed, axis=2)  # [b,q,1,dim]
        input_sid_embed_tiled = tf.tile(tf.expand_dims(input_sid_embed, axis=1), [1, self._query_token_numb, 1, 1])  # [b,q,len(vocab_sizes),dim]
        fine_item_input = tf.concat([interest_embed_tiled, input_sid_embed_tiled], axis=2)  # [b,q,1+len(vocab_sizes),dim]
        fine_item_input = tf.reshape(fine_item_input, [batch_size * self._query_token_numb, 1 + len(self._vocab_sizes), self._stage_two_dim])  # [b*q,1+len(vocab_sizes),dim]
        interest_embed_teacher_tiled = tf.expand_dims(interest_embed_teacher, axis=2)  # [b,q,1,dim]
        fine_item_input_teacher = tf.concat([interest_embed_teacher_tiled, input_sid_embed_tiled], axis=2)  # [b,q,1+len(vocab_sizes),dim]
        fine_item_input_teacher = tf.reshape(fine_item_input_teacher, [batch_size * self._query_token_numb, 1 + len(self._vocab_sizes), self._stage_two_dim])  # [b*q,1+len(vocab_sizes),dim]

        fine_item_model = DecoderModel(num_layers=4, dim=self._stage_two_dim, num_heads=8, dropout_rate=0.1, hidden_dim=self._stage_two_dim*2, training=True, query_token_numb=1)
        decoder_output = fine_item_model.forward(fine_item_input, training=True)  # [b*q, 1+len(vocab_sizes), dim]
        decoder_output = tf.reshape(decoder_output, [batch_size, self._query_token_numb, 1 + len(self._vocab_sizes), self._stage_two_dim])  # [b,q,1+len(vocab_sizes),dim]
        decoder_output_teacher = fine_item_model.forward(fine_item_input_teacher, training=False)  # [b*q, 1+len(vocab_sizes), dim]
        decoder_output_teacher = tf.reshape(decoder_output_teacher, [batch_size, self._query_token_numb, 1 + len(self._vocab_sizes), self._stage_two_dim])  # [b,q,1+len(vocab_sizes),dim]
        prior_head_input = tf.stop_gradient(decoder_output_teacher[:, :, 1, :])  # [b,q,dim]，只看到 interest + start token
        prior_router_logits, prior_router_probs = self._build_prior_router(prior_router_input, prior_head_input)  # [b,q]

        # === 7. 损失计算 ===
        losses = []
        # 创建损失掩码，只对有效的语义ID计算损失
        loss_mask = tf.where(
            photo_semantic_id_int > 0,  
            tf.ones_like(photo_semantic_id_int, dtype=tf.float32), 
            tf.zeros_like(photo_semantic_id_int, dtype=tf.float32)
        )
        loss_mask = tf.reshape(loss_mask, [-1])

        result_dict = {}
        temperature = 1
        per_head_losses = []
        per_head_losses_teacher = []
        per_head_logits = []
        
        # 对每个语义层级分别计算每个 head 的 logits 和 loss
        for step in range(len(self._vocab_sizes)):
            with tf.variable_scope('proj_%d' % step):
                # 使用MLP将每个 head 的解码输出映射到对应词汇表大小的logits
                pred_logit_all = tf.layers.dense(decoder_output[:, :, 1 + step, :], self._vocab_sizes[step], name='pred', reuse=tf.AUTO_REUSE) # [batch_size, q, vocab_size]
                pred_logit_teacher_all = tf.layers.dense(decoder_output_teacher[:, :, 1 + step, :], self._vocab_sizes[step], name='pred', reuse=tf.AUTO_REUSE) # [batch_size, q, vocab_size]
                per_head_logits.append(pred_logit_all)

                # 转换标签为one-hot编码，并为每个 head 复制一份
                one_hot_labels = tf.one_hot(label[:, step], self._vocab_sizes[step])  # [B, V]
                one_hot_labels = tf.tile(tf.expand_dims(one_hot_labels, axis=1), [1, self._query_token_numb, 1])  # [B, q, V]

                # 训练分支的每个 head loss，供主生成 loss 使用
                loss_i_all = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels, logits=pred_logit_all/temperature)  # [B, q]
                per_head_losses.append(loss_i_all)
                # teacher 分支关闭 dropout，用更稳定的 NLL 构造 posterior teacher
                loss_i_teacher_all = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels, logits=pred_logit_teacher_all/temperature)  # [B, q]
                per_head_losses_teacher.append(loss_i_teacher_all)

        # posterior teacher：先只用 sid0 的 teacher loss 选头，让 teacher 定义更贴近第一步解码 hidden state
        teacher_step0_loss = per_head_losses_teacher[0]  # [B, q]
        posterior_router_logits = -teacher_step0_loss / self._posterior_temperature  # [B, q]
        posterior_router_probs = tf.stop_gradient(tf.nn.softmax(posterior_router_logits, axis=-1))  # [B, q]
        posterior_best_head = tf.argmax(posterior_router_probs, axis=1, output_type=tf.int32)  # [B]
        prior_best_head = tf.argmax(prior_router_probs, axis=1, output_type=tf.int32)  # [B]

        # router 蒸馏损失：先用 posterior top1 head 做 hard label CE，降低 teacher 分布过软带来的学习难度
        router_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=posterior_best_head, logits=prior_router_logits)  # [B]

        # 打印 router 统计信息
        posterior_top2 = tf.nn.top_k(posterior_router_probs, k=2).values
        prior_top2 = tf.nn.top_k(prior_router_probs, k=2).values
        posterior_margin = posterior_top2[:, 0] - posterior_top2[:, 1]
        prior_margin = prior_top2[:, 0] - prior_top2[:, 1]
        posterior_entropy = -tf.reduce_sum(posterior_router_probs * tf.log(posterior_router_probs + 1e-9), axis=-1)
        prior_entropy = -tf.reduce_sum(prior_router_probs * tf.log(prior_router_probs + 1e-9), axis=-1)
        print_tensor("router/posterior_margin", posterior_margin)
        print_tensor("router/prior_margin", prior_margin)
        print_tensor("router/posterior_entropy", posterior_entropy)
        print_tensor("router/prior_entropy", prior_entropy)
        print_tensor("router/posterior_top1_prob", tf.reduce_max(posterior_router_probs, axis=-1))
        print_tensor("router/prior_top1_prob", tf.reduce_max(prior_router_probs, axis=-1))
        prior_teacher_agreement = self._masked_mean(tf.cast(tf.equal(prior_best_head, posterior_best_head), tf.float32), loss_mask)
        print_tensor("router/prior_teacher_agreement", prior_teacher_agreement)

        for i in range(self._query_token_numb):
            posterior_usage = self._masked_mean(tf.cast(tf.equal(posterior_best_head, i), tf.float32), loss_mask)
            prior_usage = self._masked_mean(tf.cast(tf.equal(prior_best_head, i), tf.float32), loss_mask)
            print_tensor("router/posterior_usage/query_%d" % i, posterior_usage)
            print_tensor("router/prior_usage/query_%d" % i, prior_usage)

        # 默认使用 prior router 选中的 head 作为指标统计对象，更接近线上效果
        decoder_output_selected = self._gather_by_head(decoder_output, prior_best_head)  # [B, 1+len(vocab_sizes), dim]

        # 计算解码器各步输出的余弦相似度（用于调试）
        for i in range(len(self._vocab_sizes)):
            similarity = calc_sim_cos(decoder_output_selected[:, 1 + i, :])
            print_tensor('decoder_sim/decoder_output_%d' % i, similarity)

        # 对每个语义层级分别计算 loss 和指标
        for step in range(len(self._vocab_sizes)):
            pred_logit_all = per_head_logits[step]  # [B, q, V]
            pred_logit = self._gather_by_head(pred_logit_all, prior_best_head)  # [B, V]
            print_tensor("logits/pred_logit_%d" % step, pred_logit)

            # 使用 posterior teacher 对所有 head 的 loss 做软加权
            loss_i = tf.reduce_sum(per_head_losses[step] * posterior_router_probs, axis=1)  # [B]
            losses.append(loss_i)

            # 1. 求 softmax 概率
            pred_prob = tf.nn.softmax(pred_logit/temperature, axis=-1)  # [B, V]

            # 2. 取出正确 label 的概率
            #    先拼出索引对 (batch_idx, label_id)
            batch_idx = tf.range(tf.shape(pred_prob)[0], dtype=tf.int32)
            indices   = tf.stack([batch_idx, label[:, step]], axis=1)  # [B, 2]
            correct_p = tf.gather_nd(pred_prob, indices)               # [B]
            # 3. 打印
            print_tensor("probs/correct_token_prob_%d" % step, tf.reduce_sum(correct_p * loss_mask) / (tf.reduce_sum(loss_mask) + 1e-9))
            result_dict['truth%d_probs' % step] = correct_p

            greater = tf.cast(pred_prob > tf.expand_dims(correct_p, 1),tf.float32)  # [B,V]
            correct_token_rank = 1 + tf.reduce_sum(greater, axis=1)      # [B], 1=top1
            trim_rank = masked_trimmed_mean(correct_token_rank, loss_mask, trim_ratio=0.05,name="trim_rank_%d" % step)
            print_tensor("probs/correct_token_rank_%d" % step,trim_rank)
            
            max_probs, _ = tf.nn.top_k(pred_prob, k=1)
            print_tensor("probs/max_token_prob_%d" % step, tf.reduce_sum(tf.squeeze(max_probs, -1) * loss_mask) / (tf.reduce_sum(loss_mask) + 1e-9))
            
            max_16_probs, max_16_indices = tf.nn.top_k(pred_prob, k=16, sorted=True)
            result_dict["sid%d_probs" % step] = max_16_probs
            result_dict["sid%d_indices" % step] = max_16_indices

            # 打印每个层次的损失
            print_tensor("loss/loss_%d" % step, tf.reduce_sum(loss_i * loss_mask) / (tf.reduce_sum(loss_mask) + 1e-9))
            # 计算各种recall指标
            recall_at_k(pred_logit, label[:, step], loss_mask, self._print_ops, top_k=[1, 16, 128, 256], name="predict_recall_%d" % step)
                
        print_tensor("loss_mask", loss_mask)
        # 计算加权平均损失
        router_loss_mean = tf.reduce_sum(router_loss * loss_mask) / (tf.reduce_sum(loss_mask) + 1e-9)
        print_tensor("loss/router_loss", router_loss_mean)
        loss = tf.reduce_sum((losses[0] + losses[1] + losses[2] + self._router_loss_weight * router_loss) * loss_mask) / (tf.reduce_sum(loss_mask) + 1e-9)
        result_dict["posterior_router_probs"] = posterior_router_probs
        result_dict["prior_router_probs"] = prior_router_probs
        result_dict["posterior_best_head"] = posterior_best_head
        result_dict["prior_best_head"] = prior_best_head
        return loss, result_dict

    def beam_search_fast(self, beam_size=512, temperature=1):
        """
        O(batch·beam·logV) 近似复杂度的束搜索（显存与 beam_size 线性）

        改进版本：
        * **step=0** 仅用 1 条 beam，从 |V_0| 里直接选 top‑k 形成不同路径，
        避免所有 beam 被同一起点锁死。
        * step>0 时保持固定 beam_size。

        返回：
            gen_part_loc  – shape [B, beam_size, seq_len] 的推荐 sid 局部 id 序列
            probs         – 同形状，逐 token 的 softmax 概率（便于做温度/多样性分析）
        """
        # ------------- 常量 & 子模型 -------------
        offsets = [0,
                self._vocab_sizes[0],
                self._vocab_sizes[0] + self._vocab_sizes[1]]       # 局部→全局 id 偏移

        # ---------- ① 预处理：编码用户 ----------
        # === 1. 用户静态特征处理 ===
        # 拼接所有用户静态特征
        user_static_fea = tf.concat([self._feature_emb_dict[fea] for fea in user_static_fea_names], axis=1)
        # 通过MLP将静态特征映射到指定维度
        user_static_emb = mlp('user_static_emb', user_static_fea, [2*self._dim], self._dim, activation=tf.nn.leaky_relu)
        batch_size = tf.shape(user_static_emb)[0]
        # 调整形状为[batch_size, 1, dim]，作为序列的一个元素
        user_static_emb = tf.reshape(user_static_emb, [batch_size, 1, self._dim])

        # === 2. 用户点击行为特征处理 ===
        # 2.1 拼接原始点击特征
        user_click_fea = tf.concat(
            [self._feature_emb_dict[fea] for fea in user_click_fea_names], axis=2)   # [B, L, dim]
        
        # 2.2 仅保留“有效序列的最后 selected_size 条”
        max_len  = 200                                             
        raw_len  = tf.cast(
            self._feature_emb_size_dict['user_profile_v1_click_pid_list'],
            tf.int32)                 # 可能是 [B,1] 也可能是 [B]
        valid_len = tf.reshape(raw_len, [-1])      # 强制展平成 [B]
        
        max_len_i  = tf.constant(max_len, dtype=tf.int32)           # 256
        used_len   = tf.minimum(valid_len, max_len_i)               # [B]  小于 256 保留原值，≥256 置 256

        # 通过MLP处理点击特征
        user_click_emb = mlp('user_click_emb', user_click_fea, [4*self._dim], self._dim, activation=tf.nn.leaky_relu)
        
        # === 3. 构建编码器输入 ===
        # 将静态特征和点击行为特征拼接作为编码器输入
        encoder_input = tf.concat([user_static_emb, user_click_emb], axis=1)
        
        # === 3-A. 构建 Encoder/Decoder 的 padding mask =============================
        # 整个序列长度 = 1（user token）+ max_len（点击序列）
        total_len  = 1 + max_len                       # int, e.g. 6 when max_len=5
        B          = tf.shape(used_len)[0]            # batch_size 动态

        # ① 为点击序列生成左对齐的 0/1 mask：左侧有效=1，右侧padding=0
        #    sequence_mask 默认左对齐 -> [1 1 0 0 0]
        click_mask = tf.sequence_mask(
            lengths=used_len,         # [B]
            maxlen=max_len,           # =5
            dtype=tf.int8)            # [B, max_len]

        # ② user 静态 token 永远有效，直接补 1
        user_tok   = tf.ones([B, 1], dtype=tf.int8)  # [B,1]

        # ③ 拼成整条序列的 mask 向量，形状 [B, total_len]
        seq_mask   = tf.concat([user_tok, click_mask], axis=1)  # 例: [1 1 0 0 0 0]
        
        # debug
        self._print_ops.append(tf.print("seq_mask first sample:", seq_mask[0], summarize=100))

        # ④ 扩展到head和Tq
        src_mask = tf.reshape(seq_mask, [B, 1, 1, total_len])  # [B, 1, 1, total_len]        
        
        enc_out_base = encoder_input  # [B, L_enc, C]

        # === 5 构建query token ===
        query_token_indice_1d = tf.range(start=self._total_vocab_size, limit=self._total_vocab_size + self._query_token_numb, delta=1, dtype=tf.int32)#[q]
        query_token_indice_2d = tf.expand_dims(query_token_indice_1d, axis=0)#[1,q]
        query_token_indice = tf.tile(query_token_indice_2d, multiples=[batch_size, 1])#[b,q]
        coarse_interest_input = tf.nn.embedding_lookup(self._embedding, query_token_indice)#[b,q,dim]
                 
        # === 6. 粗粒度解码器 ===
        # 使用4层Transformer解码器生成序列表示
        coarse_interest_model = QFormer(num_layers=4, dim=self._dim, num_heads=8, dropout_rate=0.1, hidden_dim=self._dim*2)
        coarse_interest = coarse_interest_model.forward(coarse_interest_input, enc_out_base, src_mask, training=True) # [b, q, dim]


        # ---------- ② Beam 状态初始化 ----------

        start_tok = tf.fill([B, 1], self._start_token_id)   # global id of <START> [b,1]
        initial_seqs   = tf.expand_dims(start_tok, 1)                 # [B, 1, 1]  每条路径，第一维b，第二维是beam_size，第三维是输出的seq_len(输出每一个具体的token),
        initial_probs  = tf.ones_like(initial_seqs, dtype=tf.float32)         # [B, 1, 1]  每条路径得分，第一维b，第二维是beam_size，第三维是输出的seq_len(输出每一个token的概率),
        initial_scores = tf.zeros([B, 1], dtype=tf.float32)           # [B, 1]  初始得分为0，第一位B，第二维是beam_size(每条路径的概率和)，相当于reduce_mean(probs,axis=-1)

        # 每个头推理出的结果
        all_seqs=[]
        all_probs=[]
        all_scores=[]

        #控制self._query_token_numb个粗粒度解码器是否共享参数
        SHARE_FINE_ITEM_PARAMS=True

        for i in range(self._query_token_numb):

            seqs=initial_seqs
            probs=initial_probs
            scores=initial_scores
            cur_beam = 1  # 当前 beam 数

            cache = {}                    # 全层 KV

            query_token_embed=coarse_interest[:, i,  :] # [b,dim]
        
            # ---------- ③ 逐层解码 ----------
            for step, V in enumerate(self._vocab_sizes):

                fine_item_input = query_token_embed if step==0 else tf.nn.embedding_lookup(self._embedding, seqs[:, :, -1])  # [B,cur_beam,1,dim]
                fine_item_input = tf.reshape(fine_item_input, [B*cur_beam, 1, self._dim])

                if SHARE_FINE_ITEM_PARAMS:
                    scope_name = "fine_item_decoder"
                else:
                    scope_name = "fine_item_decoder_{}".format(i)

                with tf.variable_scope(scope_name):
                    fine_item_model = DecoderModel(num_layers=4, dim=self._dim, num_heads=8, dropout_rate=0.1, hidden_dim=self._dim*2)

                    dec_out, cache = fine_item_model.step(fine_item_input, cur_beam, enc_out_base, src_mask, cache) #[b*beam,1,dim]
                    last_h = tf.reshape(dec_out, [B, cur_beam, self._dim]) #[b,beam,dim]

                with tf.variable_scope('proj_%d' % step):
                    logits = tf.layers.dense(last_h, V, name=f'pred', reuse=tf.AUTO_REUSE)

                logp = tf.nn.log_softmax(logits / temperature)                     # [B, cur_beam, V]


                # --- 本轮候选：parent_beam × top‑V → (cur_beam*V)
                k = beam_size                                                      # 第 0 步从 |V| 里挑 beam_size
                topk_logp, topk_tok = tf.nn.top_k(logp, k=k)                       # [B, cur_beam, k] 
                topk_prob = tf.exp(topk_logp)                                      #下一个token的预测分数
                cand_scores = tf.expand_dims(scores, -1) + topk_logp               # [B, cur_beam, k] ，总分数

                # --- 选全局 top‑beam_size ---
                flat_scores = tf.reshape(cand_scores, [B, -1])                     # [B, cur_beam*k]
                best_scores, best_idx = tf.nn.top_k(flat_scores, k=beam_size)      # 取新的 beam

                parent_beam = best_idx // k                                        # index in 0..cur_beam‑1
                tok_rank    = best_idx %  k                                        # index in 0..k‑1

                batch_idx = tf.tile(tf.expand_dims(tf.range(B), 1), [1, beam_size]) #batch中的每条取beam size个

                # gather 父路径
                gather_parent = tf.stack([batch_idx, parent_beam], axis=2)         # [B, beam, 2]
                parent_seq   = tf.gather_nd(seqs,  gather_parent)                  # [B, beam, T]
                parent_prob  = tf.gather_nd(probs, gather_parent)

                def gather_cache(old_cache, gp):
                    new_cache = {}
                    for k, v in old_cache.items():
                        if k.startswith(("k_self_", "v_self_")):
                            new_cache[k] = tf.gather_nd(v, gp)   # [B, beam, H, T, Dh] → 重新排序
                        else:
                            new_cache[k] = v                     # k_enc / v_enc 原样保留
                    return new_cache
                cache = gather_cache(cache, gather_parent)
                
                # gather 新 token
                tok_gather = tf.stack([batch_idx, parent_beam, tok_rank], axis=2)
                next_tok   = tf.gather_nd(topk_tok,  tok_gather)                   # [B, beam]
                next_prob  = tf.gather_nd(topk_prob, tok_gather)                   # [B, beam]
                
                # map 到全局 id
                next_tok_glb = next_tok + offsets[step]

                # 更新序列
                seqs  = tf.concat([parent_seq, tf.expand_dims(next_tok_glb, -1)], axis=-1)  # [B, beam, T+1]
                probs = tf.concat([parent_prob, tf.expand_dims(next_prob,  -1)], axis=-1)   #
                scores = best_scores                                                # [B, beam]

                cur_beam = beam_size            # 以后固定


            # 去掉 <START>
            seqs  = seqs[:, :, 1:]  #[b,beam,seq]
            probs = probs[:, :, 1:]  #[b,beam,seq]

            # 转回局部 id
            offsets_t = tf.constant(offsets, dtype=seqs.dtype)          # [3]
            gen_part_loc = seqs - offsets_t

            all_seqs.append(gen_part_loc)
            all_probs.append(probs)
            all_scores.append(scores)
        
        #  axis=1 堆叠 + 按 scores 全局排序
        all_seqs   = tf.concat(all_seqs,   axis=1)   # [B, Q*beam, seq]
        all_probs  = tf.concat(all_probs,  axis=1)   # [B, Q*beam, seq]
        all_scores = tf.concat(all_scores, axis=1)   # [B, Q*beam]

        K = tf.shape(all_scores)[1]                 # K = Q*beam
        sorted_scores, sorted_idx = tf.nn.top_k(all_scores, k=K)  # desc, [B,K]

        # 用 sorted_idx 重排 seqs/probs
        batch_idx = tf.tile(tf.expand_dims(tf.range(B), 1), [1, K])             # [B,K]
        gather_2d = tf.stack([batch_idx, sorted_idx], axis=2)                   # [B,K,2]

        all_seqs  = tf.gather_nd(all_seqs,  gather_2d)   # [B,K,seq]
        all_probs = tf.gather_nd(all_probs, gather_2d)   # [B,K,seq]
        all_scores = sorted_scores                        # [B,K]


        
        return all_seqs,all_probs


    def beam_search_fast_share_head(self, beam_size=512, temperature=1, head_top_k=None):
        """
        O(batch·beam·logV) 近似复杂度的束搜索（显存与 beam_size 线性）

        改进版本：
        * **step=0** 仅用 1 条 beam，从 |V_0| 里直接选 top‑k 形成不同路径，
        避免所有 beam 被同一起点锁死。
        * step>0 时保持固定 beam_size。

        返回：
            gen_part_loc  – shape [B, topk_head*beam_size, seq_len] 的推荐 sid 局部 id 序列
            probs         – 同形状，逐 token 的 softmax 概率（便于做温度/多样性分析）
            query_indices – shape [B, topk_head*beam_size]，标记每个序列对应的 query 索引
            query_probs   – shape [B, topk_head*beam_size]，标记每个序列对应的 router 概率
        """
        if head_top_k is None:
            head_top_k = self._router_infer_top_k
        head_top_k = max(1, min(head_top_k, self._query_token_numb))

        # ------------- 常量 & 子模型 -------------
        offsets = [0,
                self._vocab_sizes[0],
                self._vocab_sizes[0] + self._vocab_sizes[1]]       # 局部→全局 id 偏移
        

        # ---------- ① 预处理：编码用户 ----------
        # === 1. 用户静态特征处理 ===
        # 拼接所有用户静态特征
        user_static_fea = tf.concat([self._feature_emb_dict[fea] for fea in user_static_fea_names], axis=1)
        # 通过MLP将静态特征映射到指定维度
        user_static_emb = mlp('user_static_emb', user_static_fea, [2*self._dim], self._dim, activation=tf.nn.leaky_relu)
        batch_size = tf.shape(user_static_emb)[0]
        # 调整形状为[batch_size, 1, dim]，作为序列的一个元素
        user_static_emb = tf.reshape(user_static_emb, [batch_size, 1, self._dim])

        # === 2. 用户点击行为特征处理 ===
        # 2.1 拼接原始点击特征
        user_click_fea = tf.concat(
            [self._feature_emb_dict[fea] for fea in user_click_fea_names], axis=2)   # [B, L, dim]
        
        # 2.2 仅保留“有效序列的最后 selected_size 条”
        max_len  = 200                                             
        raw_len  = tf.cast(
            self._feature_emb_size_dict['user_profile_v1_click_pid_list'],
            tf.int32)                 # 可能是 [B,1] 也可能是 [B]
        valid_len = tf.reshape(raw_len, [-1])      # 强制展平成 [B]
        
        max_len_i  = tf.constant(max_len, dtype=tf.int32)           # 256
        used_len   = tf.minimum(valid_len, max_len_i)               # [B]  小于 256 保留原值，≥256 置 256

        # 通过MLP处理点击特征
        user_click_emb = mlp('user_click_emb', user_click_fea, [4*self._dim], self._dim, activation=tf.nn.leaky_relu)
        
        # === 3. 构建 decoder cross-attention 使用的基础序列 ===
        enc_out_base = tf.concat([user_static_emb, user_click_emb], axis=1)

        # === 3-A. 构建 decoder cross-attention 的 padding mask =============================
        # 整个序列长度 = 1（user token）+ max_len（点击序列）
        total_len  = 1 + max_len                       # int, e.g. 6 when max_len=5
        B          = tf.shape(used_len)[0]            # batch_size 动态

        # ① 为点击序列生成左对齐的 0/1 mask：左侧有效=1，右侧padding=0
        #    sequence_mask 默认左对齐 -> [1 1 0 0 0]
        click_mask = tf.sequence_mask(
            lengths=used_len,         # [B]
            maxlen=max_len,           # =5
            dtype=tf.int8)            # [B, max_len]

        # ② user 静态 token 永远有效，直接补 1
        user_tok   = tf.ones([B, 1], dtype=tf.int8)  # [B,1]

        # ③ 拼成整条序列的 mask 向量，形状 [B, total_len]
        seq_mask   = tf.concat([user_tok, click_mask], axis=1)  # 例: [1 1 0 0 0 0]
        
        # debug
        self._print_ops.append(tf.print("seq_mask first sample:", seq_mask[0], summarize=100))

        # ④ 扩展到head和Tq
        src_mask = tf.reshape(seq_mask, [B, 1, 1, total_len])  # [B, 1, 1, total_len]        

        # === 4. 构建 query token + router token，并走 encoder 得到兴趣头 ===
        query_token_indice_1d = tf.range(
            start=self._total_vocab_size,
            limit=self._total_vocab_size + self._query_token_numb,
            delta=1,
            dtype=tf.int32)  # [q]
        query_token_indice_2d = tf.expand_dims(query_token_indice_1d, axis=0)  # [1,q]
        query_token_indice = tf.tile(query_token_indice_2d, multiples=[batch_size, 1])  # [b,q]
        query_token_emb = tf.nn.embedding_lookup(self._embedding, query_token_indice)  # [b,q,dim]

        router_token_indice = tf.fill([batch_size, 1], tf.constant(self._router_token_id, dtype=tf.int32))  # [b,1]
        router_token_emb = tf.nn.embedding_lookup(self._embedding, router_token_indice)  # [b,1,dim]

        encoder_input = tf.concat([query_token_emb, router_token_emb, user_static_emb, user_click_emb], axis=1)

        encoder_total_len = self._query_token_numb + 1 + 1 + max_len
        query_tok  = tf.ones([B, self._query_token_numb], dtype=tf.int8)  # [B, Q]
        router_tok = tf.ones([B, 1], dtype=tf.int8)  # [B,1]
        encoder_seq_mask = tf.concat([query_tok, router_tok, user_tok, click_mask], axis=1)  # [B, total_len]
        encoder_src_mask = tf.reshape(encoder_seq_mask, [B, 1, 1, encoder_total_len])  # [B,1,1,total_len]

        encoder_model = EncoderModel(num_layers=4, dim=self._stage_one_dim, num_heads=8, dropout_rate=0.1, hidden_dim=self._stage_one_dim*2, training=True)
        encoder_output = encoder_model.forward(encoder_input, encoder_src_mask, training=True)  # [b,total_len,dim]

        coarse_interest = encoder_output[:, :self._query_token_numb, :]  # [b,q,dim]
        router_token_output = tf.squeeze(encoder_output[:, self._query_token_numb:self._query_token_numb+1, :], axis=1)  # [b,dim]

        interest_embed = tf.layers.dense(coarse_interest, self._stage_two_dim, name='project', activation=tf.nn.leaky_relu, reuse=tf.AUTO_REUSE)  # [b,q,dim]

        # === 5. prior router：先构造 [interest, start] 的第一步解码 hidden state，再从 8 个兴趣头中选 top-k ===
        start_token_indice = tf.tile(tf.constant(self._start_token_id, shape=(1, 1), dtype=tf.int32), [batch_size, 1])  # [b,1]
        start_token_embed = tf.nn.embedding_lookup(self._embedding, start_token_indice)  # [b,1,dim]
        start_token_embed_tiled = tf.tile(tf.expand_dims(start_token_embed, axis=1), [1, self._query_token_numb, 1, 1])  # [b,q,1,dim]
        interest_embed_tiled = tf.expand_dims(interest_embed, axis=2)  # [b,q,1,dim]
        router_decoder_input = tf.concat([interest_embed_tiled, start_token_embed_tiled], axis=2)  # [b,q,2,dim]
        router_decoder_input = tf.reshape(router_decoder_input, [batch_size * self._query_token_numb, 2, self._stage_two_dim])  # [b*q,2,dim]

        fine_item_router_model = DecoderModel(num_layers=4, dim=self._stage_two_dim, num_heads=8, dropout_rate=0.1, hidden_dim=self._stage_two_dim*2, training=False, query_token_numb=1)
        router_decoder_output = fine_item_router_model.forward(router_decoder_input, training=False)  # [b*q,2,dim]
        router_decoder_output = tf.reshape(router_decoder_output, [batch_size, self._query_token_numb, 2, self._stage_two_dim])  # [b,q,2,dim]
        prior_head_state = router_decoder_output[:, :, 1, :]  # [b,q,dim]

        _, prior_router_probs = self._build_prior_router(router_token_output, prior_head_state)  # [b,q]
        selected_query_probs, selected_query_indices = tf.nn.top_k(prior_router_probs, k=head_top_k)  # [b,topk]

        batch_idx = tf.tile(tf.expand_dims(tf.range(B), 1), [1, head_top_k])  # [b,topk]
        gather_interest_idx = tf.stack([batch_idx, selected_query_indices], axis=2)  # [b,topk,2]
        selected_interest_embed = tf.gather_nd(interest_embed, gather_interest_idx)  # [b,topk,dim]

        # ========== 0) 把 decoder 维折叠进 batch：B*topk ==========
        query_token_embed_all = tf.reshape(selected_interest_embed, [B * head_top_k, -1]) # [B*topk, dim]
        ## enc_out_base: [B, Lenc, D] -> [B*topk, Lenc, D]（每个 decoder复用同一个 enc_out）
        Lenc = tf.shape(enc_out_base)[1]
        enc_out_all = tf.reshape(tf.tile(tf.expand_dims(enc_out_base, 1), [1, head_top_k, 1, 1]), [B * head_top_k, Lenc, -1])
        src_mask_all = tf.reshape(tf.tile(tf.expand_dims(src_mask, 1), [1, head_top_k, 1, 1, 1]), [B * head_top_k, 1, 1, -1])

        # ---------- ② Beam 状态初始化 ----------
        BQ = B * head_top_k
        start_tok = tf.fill([BQ, 1], self._start_token_id)   # global id of <START> [b,1]
        initial_seqs   = tf.expand_dims(start_tok, 1)                 # [BQ, 1, 1]
        initial_probs  = tf.ones_like(initial_seqs, dtype=tf.float32) # [BQ, 1, 1]
        initial_scores = tf.reshape(tf.log(tf.maximum(selected_query_probs, 1e-9)), [BQ, 1])  # [BQ, 1]

        seqs=initial_seqs
        probs=initial_probs
        scores=initial_scores
        cur_beam = 1  # 当前 beam 数
        cache = {}    # 全层 KV

        with tf.variable_scope('fine_item_decoder', reuse=tf.AUTO_REUSE):
            fine_item_model = DecoderModel(num_layers=4, dim=self._dim, num_heads=8, dropout_rate=0.1, hidden_dim=self._dim*2)
        
        # ---------- ③ 逐层解码 ----------
        for step, V in enumerate(self._vocab_sizes):

            fine_item_input = query_token_embed_all if step==0 else tf.nn.embedding_lookup(self._embedding, seqs[:, :, -1])  # [BQ,cur_beam,1,dim]
            fine_item_input = tf.reshape(fine_item_input, [BQ*cur_beam, 1, self._dim])

            with tf.variable_scope('fine_item_decoder', reuse=tf.AUTO_REUSE):
                dec_out, cache = fine_item_model.step(fine_item_input, cur_beam, enc_out_all, src_mask_all, cache) #[BQ*beam,1,dim]
            
            last_h = tf.reshape(dec_out, [BQ, cur_beam, self._dim]) #[BQ,beam,dim]

            with tf.variable_scope('proj_%d' % step):
                logits = tf.layers.dense(last_h, V, name=f'pred', reuse=tf.AUTO_REUSE)

            logp = tf.nn.log_softmax(logits / temperature)                     # [BQ, cur_beam, V]

            # --- 本轮候选：parent_beam × top‑V → (cur_beam*V)
            k = beam_size
            topk_logp, topk_tok = tf.nn.top_k(logp, k=k)                       # [BQ, cur_beam, k] 
            topk_prob = tf.exp(topk_logp)
            cand_scores = tf.expand_dims(scores, -1) + topk_logp               # [BQ, cur_beam, k]

            # --- 选全局 top‑beam_size ---
            flat_scores = tf.reshape(cand_scores, [BQ, -1])                    # [BQ, cur_beam*k]
            best_scores, best_idx = tf.nn.top_k(flat_scores, k=beam_size)

            parent_beam = best_idx // k
            tok_rank    = best_idx %  k

            batch_idx = tf.tile(tf.expand_dims(tf.range(BQ), 1), [1, beam_size])

            # gather 父路径
            gather_parent = tf.stack([batch_idx, parent_beam], axis=2)         # [BQ, beam, 2]
            parent_seq   = tf.gather_nd(seqs,  gather_parent)                  # [BQ, beam, T]
            parent_prob  = tf.gather_nd(probs, gather_parent)

            def gather_cache(old_cache, gp):
                new_cache = {}
                for kk, vv in old_cache.items():
                    if kk.startswith(("k_self_", "v_self_")):
                        new_cache[kk] = tf.gather_nd(vv, gp)
                    else:
                        new_cache[kk] = vv
                return new_cache
            cache = gather_cache(cache, gather_parent)
            
            # gather 新 token
            tok_gather = tf.stack([batch_idx, parent_beam, tok_rank], axis=2)
            next_tok   = tf.gather_nd(topk_tok,  tok_gather)                   # [BQ, beam]
            next_prob  = tf.gather_nd(topk_prob, tok_gather)                   # [BQ, beam]
            
            # map 到全局 id
            next_tok_glb = next_tok + offsets[step]

            # 更新序列
            seqs  = tf.concat([parent_seq, tf.expand_dims(next_tok_glb, -1)], axis=-1)  # [BQ, beam, T+1]
            probs = tf.concat([parent_prob, tf.expand_dims(next_prob,  -1)], axis=-1)
            scores = best_scores

            cur_beam = beam_size

        # 去掉 <START>
        seqs  = seqs[:, :, 1:]  #[BQ,beam,seq]
        probs = probs[:, :, 1:] #[BQ,beam,seq]

        # 转回局部 id
        offsets_t = tf.constant(offsets, dtype=seqs.dtype)          # [3]
        gen_part_loc = seqs - offsets_t

        # 合并 topk_head*beam
        all_seqs   = tf.reshape(gen_part_loc, [B, head_top_k * beam_size, -1])   # [B, topk*beam, Seq]
        all_probs  = tf.reshape(probs,        [B, head_top_k * beam_size, -1])   # [B, topk*beam, Seq]
        all_scores = tf.reshape(scores,       [B, head_top_k * beam_size])       # [B, topk*beam]

        query_indices = tf.reshape(
            tf.tile(tf.expand_dims(selected_query_indices, axis=-1), [1, 1, beam_size]),
            [B, head_top_k * beam_size]
        )  # [B, topk*beam]
        query_probs = tf.reshape(
            tf.tile(tf.expand_dims(selected_query_probs, axis=-1), [1, 1, beam_size]),
            [B, head_top_k * beam_size]
        )  # [B, topk*beam]

        K = tf.shape(all_scores)[1]                 # K = topk*beam
        sorted_scores, sorted_idx = tf.nn.top_k(all_scores, k=K)  # desc, [B,K]

        # 用 sorted_idx 重排 seqs/probs/query 信息
        batch_idx = tf.tile(tf.expand_dims(tf.range(B), 1), [1, K])             # [B,K]
        gather_2d = tf.stack([batch_idx, sorted_idx], axis=2)                   # [B,K,2]

        all_seqs  = tf.gather_nd(all_seqs,  gather_2d)   # [B,K,seq]
        all_probs = tf.gather_nd(all_probs, gather_2d)   # [B,K,seq]
        query_indices = tf.gather_nd(query_indices, gather_2d)  # [B,K]
        query_probs = tf.gather_nd(query_probs, gather_2d)      # [B,K]
        all_scores = sorted_scores                              # [B,K]

        return all_seqs, all_probs, query_indices, query_probs
