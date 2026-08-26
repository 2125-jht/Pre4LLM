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
        self._query_token_numb=1
        
        # 创建统一的嵌入矩阵，包含所有语义ID的嵌入向量
        # 使用均匀分布初始化，范围为[-1/dim, 1/dim]
        self._embedding = tf.get_variable(
            shape=[self._total_vocab_size+self._query_token_numb, dim], 
            name='embedding',
            dtype=tf.float16,
            initializer=tf.random_uniform_initializer(minval=-1.0/dim, maxval=1.0/dim), 
            trainable=False
        )
        
        self._dim = dim

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
        print_tensor("valid_len", valid_len)
        
        max_len_i  = tf.constant(max_len, dtype=tf.int32)           # 256
        used_len   = tf.minimum(valid_len, max_len_i)               # [B]  小于 256 保留原值，≥256 置 256
        print_tensor("used_len", used_len)                          # 打印结果

        # 通过MLP处理点击特征
        user_click_emb = mlp('user_click_emb', user_click_fea, [4*self._dim], self._dim, activation=tf.nn.leaky_relu)
        
        # debug
        self._print_ops.append(tf.print("user_click_emb first sample:", user_click_emb[0,:,1], summarize=100))
        
        # === 3. 构建编码器输入 ===
        # 将静态特征和点击行为特征拼接作为编码器输入
        encoder_input = tf.concat([user_static_emb, user_click_emb], axis=1)
        
        # 计算编码器输入的余弦相似度（用于调试）
        encoder_input_sim = tf.reshape(encoder_input, [batch_size, -1])
        print_tensor("encoder_input_sim", calc_sim_cos(encoder_input_sim))
        
        # === 3-A. 构建 Encoder/Decoder 的 padding mask =============================
        # 整个序列长度 = 1（user token）+ max_len（点击序列）
        total_len  = 1 + max_len                       # int, e.g. 6 when max_len=5
        B          = tf.shape(used_len)[0]            # batch_size 动态

        # ① 为点击序列生成右对齐的 0/1 mask：左侧 padding=0，右侧有效=1
        #    sequence_mask 默认左对齐 -> [1 1 0 0 0]
        click_mask = tf.sequence_mask(
            lengths=used_len,         # [B]
            maxlen=max_len,            # =5
            dtype=tf.int8)          # [B, max_len]

        # ② user 静态 token 永远有效，直接补 1
        user_tok   = tf.ones([B, 1], dtype=tf.int8)  # [B,1]

        # ③ 拼成整条序列的 mask 向量，形状 [B, total_len]
        seq_mask   = tf.concat([user_tok, click_mask], axis=1)  # 例: [1 1 0 0 0 0]
        
        # debug
        self._print_ops.append(tf.print("seq_mask first sample:", seq_mask[0], summarize=100))

        # ④ 扩展到head和Tq
        src_mask = tf.reshape(seq_mask, [B, 1, 1, total_len])  # [B, 1, 1, total_len]

        encoder_output = encoder_input
                
        # photo sid embedding
        sid_embedding = tf.nn.embedding_lookup(self._embedding, photo_sid) #[b,code_size,dim]
        preference_embedding = tf.reduce_sum(sid_embedding ,axis=1) #[b,dim]


        # === 5 构建query token ===
        query_token_id = tf.constant(self._total_vocab_size, dtype=tf.int32)     # 标量
        query_token_indice = tf.fill([batch_size, 1], query_token_id)           # [b,1]
        coarse_interest_input = tf.nn.embedding_lookup(self._embedding, query_token_indice)  # [b,1,dim]
        
                
        # === 6. 粗粒度解码器 ===
        # 使用4层Transformer解码器生成序列表示
        coarse_interest_model = QFormer(num_layers=4, dim=self._dim, num_heads=8, dropout_rate=0.1, hidden_dim=self._dim*2, training=True)
        coarse_interest = coarse_interest_model.forward(coarse_interest_input, encoder_output, src_mask, training=True) # [b, 1, dim]
        


        #构造fine_item_input的输入，
        fine_item_input = tf.concat([coarse_interest, sid_embedding[:, :-1, :]], axis=1)  # [b, code_size, dim]


        with tf.variable_scope("fine_item_decoder", reuse=tf.AUTO_REUSE):

            fine_item_model = DecoderModel(num_layers=1, dim=self._dim, num_heads=8, dropout_rate=0.1, hidden_dim=self._dim*2, training=True)
            decoder_output = fine_item_model.forward(fine_item_input, encoder_output, src_mask, training=True)  # [b, code_size, dim]

        coarse_norm = tf.nn.l2_normalize(tf.squeeze(coarse_interest), axis=-1)        # [b, dim]
        pref_norm   = tf.nn.l2_normalize(preference_embedding, axis=-1)   # [b, dim]

        cos_sim = tf.reduce_sum(coarse_norm * pref_norm, axis=-1)         # [b]
        print_tensor("interest_pref_sim", tf.reduce_mean(cos_sim))



        
        # 计算解码器各步输出的余弦相似度（用于调试）
        for i in range(len(self._vocab_sizes)):
            similarity = calc_sim_cos(decoder_output[:, i, :])
            print_tensor('decoder_sim/decoder_output_%d' % i, similarity)

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
        
        # 对每个语义层级分别计算损失
        for step in range(len(self._vocab_sizes)):
            with tf.variable_scope('proj_%d' % step):
                # 使用MLP将解码器输出映射到对应词汇表大小的logits
                # pred_logit = mlp("pred", decoder_output[:, step, :], [self._vocab_sizes[step]], self._vocab_sizes[step], activation=tf.nn.leaky_relu)
                pred_logit = tf.layers.dense(decoder_output[:, step, :], self._vocab_sizes[step], name='pred') # [batch_size, vocab_size]
                print_tensor("logits/pred_logit_%d" % step, pred_logit)
                
                temperature = 1
                
                # 1. 求 softmax 概率
                pred_prob = tf.nn.softmax(pred_logit/temperature, axis=-1)  # [B, V]

                # print_tensor("probs/pred_prob_%d" % step, pred_prob)

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

                # 转换标签为one-hot编码
                one_hot_labels = tf.one_hot(label[:, step], self._vocab_sizes[step])
                # 计算交叉熵损失，使用温度缩放(temperature=2.0)
                loss_i = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels, logits=pred_logit/temperature)
                losses.append(loss_i)
            
                # 打印每个层次的损失
                print_tensor("loss/loss_%d" % step, tf.reduce_sum(loss_i * loss_mask) / (tf.reduce_sum(loss_mask) + 1e-9))
                # 计算各种recall指标
                recall_at_k(pred_logit, label[:, step], loss_mask, self._print_ops, top_k=[1, 16, 128, 256], name="predict_recall_%d" % step)
                
        print_tensor("loss_mask", loss_mask)
        # 计算加权平均损失
        loss = tf.reduce_sum((losses[0] + losses[1] + losses[2]) * loss_mask) / (tf.reduce_sum(loss_mask) + 1e-9)
        return loss, result_dict


    def beam_search_fast_share_head(self, beam_size=512, temperature=1):
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
        

        # ④ 扩展到head和Tq
        src_mask = tf.reshape(seq_mask, [B, 1, 1, total_len])  # [B, 1, 1, total_len]        
        
        enc_out_base = encoder_input  # [B, L_enc, C]

        # === 5 构建query token ===
        query_token_id = tf.constant(self._total_vocab_size, dtype=tf.int32)     # 标量
        query_token_indice = tf.fill([batch_size, 1], query_token_id)           # [b,1]
        coarse_interest_input = tf.nn.embedding_lookup(self._embedding, query_token_indice)  # [b,1,dim]
                 
        # === 6. 粗粒度解码器 ===
        # 使用4层Transformer解码器生成序列表示
        coarse_interest_model = QFormer(num_layers=4, dim=self._dim, num_heads=8, dropout_rate=0.1, hidden_dim=self._dim*2, training=False)
        coarse_interest = coarse_interest_model.forward(coarse_interest_input, enc_out_base, src_mask, training=False) # [b, 1, dim]


        # ========== 初始化 ==========
        query_token_embed_all = tf.reshape(coarse_interest, [B * self._query_token_numb, -1]) # [B*1, dim],实现并行推理
        start_tok = tf.fill([B, 1], self._total_vocab_size)   # global id of <START> [b,1]
        seqs   = tf.expand_dims(start_tok, 1)                 # [B, 1, 1]
        probs  = tf.ones_like(seqs, dtype=tf.float32)         # [B, 1, 1]
        scores = tf.zeros([B, 1], dtype=tf.float32)           # [B, 1]

        cur_beam = 1  # 当前 beam 数
        cache = {}                    # 全层 KV
       

        with tf.variable_scope('fine_item_decoder', reuse=tf.AUTO_REUSE):
            fine_item_model = DecoderModel(num_layers=1, dim=self._dim, num_heads=8, dropout_rate=0.1, hidden_dim=self._dim*2, training=False)
        
        # ---------- ③ 逐层解码 ----------
        for step, V in enumerate(self._vocab_sizes):



            fine_item_input = query_token_embed_all if step==0 else tf.nn.embedding_lookup(self._embedding, seqs[:, :, -1])  # [B,cur_beam,1,dim]
            fine_item_input = tf.reshape(fine_item_input, [B*cur_beam, 1, self._dim])

            with tf.variable_scope('fine_item_decoder', reuse=tf.AUTO_REUSE):
                dec_out, cache = fine_item_model.step(fine_item_input, cur_beam, enc_out_base, src_mask, cache) #[b*beam,1,dim]
            
            last_h = tf.reshape(dec_out, [B, cur_beam, self._dim])
            # # 仅取最后一个 time‑step（上一 token）输出做投影
            # dec_out = tf.reshape(dec_out, [B, cur_beam, -1, self._dim])
            # last_h = dec_out[:, :, -1, :]                                      # [B, cur_beam, C]
            
            with tf.variable_scope('proj_%d' % step):
                logits = tf.layers.dense(last_h, V, name=f'pred', reuse=tf.AUTO_REUSE,trainable=False)

            logp = tf.nn.log_softmax(tf.cast(logits, tf.float32) / temperature)  # [B, cur_beam, V]

            # --- 本轮候选：parent_beam × top‑V → (cur_beam*V)
            k =  beam_size                          # 第 0 步从 |V| 里挑 beam_size
            topk_logp, topk_tok = tf.nn.top_k(logp, k=k)                       # [B, cur_beam, k]
            topk_prob = tf.exp(topk_logp)

            # 累积得分
            cand_scores = tf.expand_dims(scores, -1) + topk_logp               # [B, cur_beam, k]

            # --- 选全局 top‑beam_size ---
            flat_scores = tf.reshape(cand_scores, [B, -1])                     # [B, cur_beam*k]
            best_scores, best_idx = tf.nn.top_k(flat_scores, k=beam_size)      # 取新的 beam

            parent_beam = best_idx // k                                        # index in 0..cur_beam‑1
            tok_rank    = best_idx %  k                                        # index in 0..k‑1

            batch_idx = tf.tile(tf.expand_dims(tf.range(B), 1), [1, beam_size])

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
            probs = tf.concat([parent_prob, tf.expand_dims(next_prob,  -1)], axis=-1)
            scores = best_scores                                                # [B, beam]

            cur_beam = beam_size            # 以后固定

        # 去掉 <START>
        seqs  = seqs[:, :, 1:]
        probs = probs[:, :, 1:]

        # 转回局部 id
        offsets_t = tf.constant(offsets, dtype=seqs.dtype)          # [3]
        gen_part_loc = seqs - offsets_t

        return gen_part_loc, probs
