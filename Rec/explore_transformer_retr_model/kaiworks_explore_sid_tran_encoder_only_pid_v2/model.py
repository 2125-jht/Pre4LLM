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
    
    def __init__(self, feature_emb_dict, feature_emb_size_dict, dim=512, vocab_sizes=[8192, 8192, 8192], print_ops=None):
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
        
        # 创建统一的嵌入矩阵，包含所有语义ID的嵌入向量
        # 使用均匀分布初始化，范围为[-1/dim, 1/dim]
        self._embedding = tf.get_variable(
            shape=[self._total_vocab_size+1, dim], 
            name='embedding',
            initializer=tf.random_uniform_initializer(minval=-1.0/dim, maxval=1.0/dim), 
            trainable=True
        )
        # self._embedding = tf.get_variable(
        #     name="embedding",                       
        #     shape=[self._total_vocab_size + 1, dim],
        #     initializer=tf.glorot_uniform_initializer(),           # ★ Xavier Uniform ★
        #     trainable=True
        # )
        
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
        
        # === 1. 用户静态特征处理 ===
        # 拼接所有用户静态特征
        user_static_fea = tf.concat([self._feature_emb_dict[fea] for fea in user_static_fea_names], axis=1)
        # 通过MLP将静态特征映射到指定维度
        user_static_emb = mlp('user_static_emb', user_static_fea, [2*self._dim], self._dim, activation=tf.nn.leaky_relu)
        batch_size = tf.shape(user_static_emb)[0]
        # 调整形状为[batch_size, 1, dim]，作为序列的一个元素
        user_static_emb = tf.reshape(user_static_emb, [batch_size, 1, self._dim])

        # === 2. 用户点击行为特征处理 ===
        # 拼接用户点击特征（视频ID和作者ID）
        user_click_fea = tf.concat([self._feature_emb_dict[fea] for fea in user_click_fea_names], axis=2)
        # 通过MLP处理点击特征
        user_click_emb = mlp('user_click_emb', user_click_fea, [4*self._dim], self._dim, activation=tf.nn.leaky_relu)
        
        # === 3. 构建编码器输入 ===
        # 将静态特征和点击行为特征拼接作为编码器输入
        encoder_input = tf.concat([user_static_emb, user_click_emb], axis=1)
        
        # need_pe = True
        
        # if need_pe:
        # # 获取位置编码
        #     seq_len = tf.shape(encoder_input)[1]  # 获取序列长度
        #     position_encoding = get_encoder_position_encoding(seq_len, self._dim)  # 获取位置编码
        #     # 添加位置编码到输入嵌入
        #     encoder_input += position_encoding  # 添加位置编码
        
        # === 4. 构建解码器输入 ===
        # 添加起始token（使用总词汇表大小作为特殊标记）
        start_token_indice = tf.tile(tf.constant(self._total_vocab_size, shape=(1, 1), dtype=tf.int32), [batch_size, 1])
        # 将起始token与视频语义ID拼接
        photo_with_start_token = tf.concat([start_token_indice, photo_sid], axis=1)
        # 查找嵌入向量
        decoder_input = tf.nn.embedding_lookup(self._embedding, photo_with_start_token)

        # if need_pe:
        #     # 获取解码器位置编码
        #     decoder_seq_len = tf.shape(decoder_input)[1]  # 获取解码器序列长度
        #     decoder_position_encoding = get_decoder_position_encoding(decoder_seq_len, self._dim)
        #     # 添加位置编码到解码器输入
        #     decoder_input += decoder_position_encoding
        
        # === 5. Transformer编码器 ===
        # 使用4层Transformer编码器处理用户特征
        encoder_model = EncoderModel(num_layers=4, dim=self._dim, num_heads=8, dropout_rate=0.1, hidden_dim=self._dim*2, training=True)
        encoder_output = encoder_model.forward(encoder_input, training=True) # [batch_size, seq_len, dim]
        
        # 计算编码器输出的余弦相似度（用于调试）
        encoder_output_sim = tf.reshape(encoder_output, [batch_size, -1])
        print_tensor("encoder_output_sim", calc_sim_cos(encoder_output_sim))
        
        # === 6. Transformer解码器 ===
        # 使用4层Transformer解码器生成序列表示
        decoder_model = DecoderModel(num_layers=4, dim=self._dim, num_heads=8, dropout_rate=0.1, hidden_dim=self._dim*2, training=True)
        decoder_output = decoder_model.forward(decoder_input, encoder_output, training=True) # [batch_size, seq_len, dim]
        
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

        # 对每个语义层级分别计算损失
        for step in range(len(self._vocab_sizes)):
            with tf.variable_scope('proj_%d' % step):
                # 使用MLP将解码器输出映射到对应词汇表大小的logits
                # pred_logit = mlp("pred", decoder_output[:, step, :], [self._vocab_sizes[step]], self._vocab_sizes[step], activation=tf.nn.leaky_relu)
                pred_logit = tf.layers.dense(decoder_output[:, step, :], self._vocab_sizes[step], name='pred') # [batch_size, vocab_size]
                print_tensor("logits/pred_logit_%d" % step, pred_logit)
                
                temperature = 20
                
                # 1. 求 softmax 概率
                pred_prob = tf.nn.softmax(pred_logit*temperature, axis=-1)  # [B, V]

                # 2. 取出正确 label 的概率
                #    先拼出索引对 (batch_idx, label_id)
                batch_idx = tf.range(tf.shape(pred_prob)[0], dtype=tf.int32)
                indices   = tf.stack([batch_idx, label[:, step]], axis=1)  # [B, 2]
                correct_p = tf.gather_nd(pred_prob, indices)               # [B]

                # 3. 打印
                print_tensor("probs/correct_token_prob_%d" % step, correct_p)
                
                max_probs, _ = tf.nn.top_k(pred_prob, k=1)
                print_tensor("probs/max_token_prob_%d" % step, tf.reduce_sum(tf.squeeze(max_probs, -1) * loss_mask) / (tf.reduce_sum(loss_mask) + 1e-9))
        
                # 转换标签为one-hot编码
                one_hot_labels = tf.one_hot(label[:, step], self._vocab_sizes[step])
                # 计算交叉熵损失，使用温度缩放(temperature=2.0)
                loss_i = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels, logits=pred_logit*temperature)
                losses.append(loss_i)
            
                # 打印每个层次的损失
                print_tensor("loss/loss_%d" % step, tf.reduce_sum(loss_i * loss_mask) / (tf.reduce_sum(loss_mask) + 1e-9))
                
                # 计算各种recall指标
                recall_at_k(pred_logit, label[:, step], loss_mask, self._print_ops, top_k=[1, 16, 128], name="predict_recall_%d" % step)
        
        # 打印每个层次的损失
        print_tensor("revised_loss", tf.reduce_sum((losses[0] + losses[1] + losses[2]) * loss_mask) / (tf.reduce_sum(loss_mask) + 1e-9))
        
        print_tensor("loss_mask", loss_mask)
        # 计算加权平均损失
        loss = tf.reduce_sum((losses[0] + losses[1]*2 + losses[2]*3) * loss_mask) / (tf.reduce_sum(loss_mask) + 1e-9)
        return loss

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

        encoder_model = EncoderModel(num_layers=4, dim=self._dim, num_heads=8,
                                    dropout_rate=0.1, hidden_dim=self._dim * 2)
        decoder_model = DecoderModel(num_layers=4, dim=self._dim, num_heads=8,
                                    dropout_rate=0.1, hidden_dim=self._dim * 2)

        # ---------- ① 预处理：编码用户 ----------
        user_static = tf.concat([self._feature_emb_dict[f] for f in user_static_fea_names], axis=1)
        user_static = mlp('user_static_emb', user_static, [2 * self._dim], self._dim,
                        activation=tf.nn.leaky_relu)

        B = tf.shape(user_static)[0]
        user_static = tf.reshape(user_static, [B, 1, self._dim])

        user_click = tf.concat([self._feature_emb_dict[f] for f in user_click_fea_names], axis=2)
        user_click = mlp('user_click_emb', user_click, [4 * self._dim], self._dim,
                        activation=tf.nn.leaky_relu)

        enc_in_base = tf.concat([user_static, user_click], axis=1)            # [B, L_enc, C]
        enc_out_base = encoder_model.forward(enc_in_base, training=False)     # [B, L_enc, C]

        # ---------- ② Beam 状态初始化 ----------
        start_tok = tf.fill([B, 1], self._total_vocab_size)   # global id of <START>
        seqs   = tf.expand_dims(start_tok, 1)                 # [B, 1, 1]
        probs  = tf.ones_like(seqs, dtype=tf.float32)         # [B, 1, 1]
        scores = tf.zeros([B, 1], dtype=tf.float32)           # [B, 1]

        cur_beam = 1  # 当前 beam 数

        # ---------- ③ 逐层解码 ----------
        for step, V in enumerate(self._vocab_sizes):
            # --- 准备 decoder 输入（embedding + encoder 复制）
            dec_in = tf.nn.embedding_lookup(self._embedding, seqs)            # [B, cur_beam, T, C]
            dec_in = tf.reshape(dec_in, [B * cur_beam, -1, self._dim])

            # tile encoder 输出到 cur_beam
            enc_out = tf.tile(tf.expand_dims(enc_out_base, 1), [1, cur_beam, 1, 1])
            enc_out = tf.reshape(enc_out, [B * cur_beam, -1, self._dim])

            dec_out = decoder_model.forward(dec_in, enc_out, training=False)   # [B*cur_beam, T, C]
            dec_out = tf.reshape(dec_out, [B, cur_beam, -1, self._dim])

            # 仅取最后一个 time‑step（上一 token）输出做投影
            last_h = dec_out[:, :, -1, :]                                      # [B, cur_beam, C]
            
            with tf.variable_scope('proj_%d' % step):
                logits = tf.layers.dense(last_h, V, name='pred', reuse=tf.AUTO_REUSE)  # [B, cur_beam, V]
        
            logp = tf.nn.log_softmax(logits / temperature)                     # [B, cur_beam, V]

            # --- 本轮候选：parent_beam × top‑V → (cur_beam*V)
            k = beam_size if step == 0 else beam_size                          # 第 0 步从 |V| 里挑 beam_size
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


    def beam_search_fast_v2(self, beam_sizes=(16, 128, 1024), temperature=1):
        """Tree‑style beam search.

        Each decoding step uses a (possibly) different beam width:
            step 0 -> beam_sizes[0]
            step 1 -> beam_sizes[1]
            step 2 -> beam_sizes[2]
        …and so on.  Therefore the final output contains `beam_sizes[-1]` paths.

        Args:
            beam_sizes: tuple/list with length = len(self._vocab_sizes)
            temperature: softmax temperature.

        Returns:
            gen_part_loc: [B, beam_sizes[-1], seq_len] local‑id sequence
            probs       : same shape, token‑level probabilities.
        """
        # ----------- sanity check -----------
        num_levels = len(self._vocab_sizes)
        beam_sizes = list(beam_sizes)
        assert len(beam_sizes) == num_levels, "beam_sizes length must match number of vocab levels"

        offsets = [0,
                self._vocab_sizes[0],
                self._vocab_sizes[0] + self._vocab_sizes[1]]

        encoder_model = EncoderModel(num_layers=4, dim=self._dim, num_heads=8,
                                    dropout_rate=0.1, hidden_dim=self._dim * 2)
        decoder_model = DecoderModel(num_layers=4, dim=self._dim, num_heads=8,
                                    dropout_rate=0.1, hidden_dim=self._dim * 2)

        # ---------- encode user ----------
        user_static = tf.concat([self._feature_emb_dict[f] for f in user_static_fea_names], axis=1)
        user_static = mlp('user_static_emb', user_static, [2 * self._dim], self._dim,
                        activation=tf.nn.leaky_relu)
        B = tf.shape(user_static)[0]
        user_static = tf.reshape(user_static, [B, 1, self._dim])

        user_click = tf.concat([self._feature_emb_dict[f] for f in user_click_fea_names], axis=2)
        user_click = mlp('user_click_emb', user_click, [4 * self._dim], self._dim,
                        activation=tf.nn.leaky_relu)

        enc_in_base = tf.concat([user_static, user_click], axis=1)
        enc_out_base = encoder_model.forward(enc_in_base, training=False)      # [B, L_enc, C]

        # ---------- init beam ----------
        start_tok = tf.fill([B, 1], self._total_vocab_size)
        seqs   = tf.expand_dims(start_tok, 1)           # [B, 1, 1]
        probs  = tf.ones_like(seqs, tf.float32)
        scores = tf.zeros([B, 1], tf.float32)
        cur_beam = 1

        # ---------- decode levels ----------
        for step, V in enumerate(self._vocab_sizes):
            k_target = beam_sizes[step]                 # desired beam width this level

            # ---- prepare decoder input ----
            dec_in = tf.nn.embedding_lookup(self._embedding, seqs)            # [B, cur_beam, T, C]
            dec_in = tf.reshape(dec_in, [B * cur_beam, -1, self._dim])

            enc_out = tf.tile(tf.expand_dims(enc_out_base, 1), [1, cur_beam, 1, 1])
            enc_out = tf.reshape(enc_out, [B * cur_beam, -1, self._dim])

            dec_out = decoder_model.forward(dec_in, enc_out, training=False)  # [B*cur_beam, T, C]
            dec_out = tf.reshape(dec_out, [B, cur_beam, -1, self._dim])
            last_h  = dec_out[:, :, -1, :]                                    # [B, cur_beam, C]

            with tf.variable_scope('proj_%d' % step):
                logits = tf.layers.dense(last_h, V, name=f'pred', reuse=tf.AUTO_REUSE)
            
            logp   = tf.nn.log_softmax(logits / temperature)                  # [B, cur_beam, V]

            # top‑k over vocabulary
            topk_logp, topk_tok = tf.nn.top_k(logp, k=V)  # keep full vocab first
            # reshape for candidate enumeration: [B, cur_beam * V]
            cand_scores = tf.reshape(scores[..., None] + topk_logp, [B, -1])
            best_scores, best_idx = tf.nn.top_k(cand_scores, k=k_target)       # pick target beam

            parent_beam = best_idx // V
            tok_rank    = best_idx %  V
            batch_idx   = tf.tile(tf.expand_dims(tf.range(B), 1), [1, k_target])

            gather_parent = tf.stack([batch_idx, parent_beam], axis=2)
            parent_seq  = tf.gather_nd(seqs,  gather_parent)
            parent_prob = tf.gather_nd(probs, gather_parent)

            tok_gather = tf.stack([batch_idx, parent_beam, tok_rank], axis=2)
            next_tok   = tf.gather_nd(topk_tok,  tok_gather)
            next_prob  = tf.gather_nd(tf.exp(topk_logp), tok_gather)

            # local→global id
            next_tok_glb = next_tok + offsets[step]

            seqs   = tf.concat([parent_seq, tf.expand_dims(next_tok_glb, -1)], axis=-1)
            probs  = tf.concat([parent_prob, tf.expand_dims(next_prob,   -1)], axis=-1)
            scores = best_scores
            cur_beam = k_target

        # strip <START>
        seqs  = seqs[:, :, 1:]
        probs = probs[:, :, 1:]

        offsets_t = tf.constant(offsets, dtype=seqs.dtype)          # [3]
        gen_part_loc = seqs - offsets_t

        return gen_part_loc, probs
