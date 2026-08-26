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
    
    def __init__(self, feature_emb_dict, feature_emb_size_dict, dim=128, vocab_sizes=[8192, 8192, 8192], print_ops=None):
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
        
        # === 4. 构建解码器输入 ===
        # 添加起始token（使用总词汇表大小作为特殊标记）
        start_token_indice = tf.tile(tf.constant(self._total_vocab_size, shape=(1, 1), dtype=tf.int32), [batch_size, 1])
        # 将起始token与视频语义ID拼接
        photo_with_start_token = tf.concat([start_token_indice, photo_sid], axis=1)
        # 查找嵌入向量
        decoder_input = tf.nn.embedding_lookup(self._embedding, photo_with_start_token)

        # === 5. Transformer编码器 ===
        # 使用4层Transformer编码器处理用户特征
        encoder_model = EncoderModel(num_layers=8, dim=self._dim, num_heads=8, dropout_rate=0.1, hidden_dim=self._dim*4, training=True)
        encoder_output = encoder_model.forward(encoder_input, src_mask, training=True) # [batch_size, seq_len, dim]
        
        # 计算编码器输出的余弦相似度（用于调试）
        encoder_output_sim = tf.reshape(encoder_output, [batch_size, -1])
        print_tensor("encoder_output_sim", calc_sim_cos(encoder_output_sim))
        
        # === 6. Transformer解码器 ===
        # 使用4层Transformer解码器生成序列表示
        decoder_model = DecoderModel(num_layers=8, dim=self._dim, num_heads=8, dropout_rate=0.1, hidden_dim=self._dim*4, training=True)
        decoder_output = decoder_model.forward(decoder_input, encoder_output, src_mask, training=True) # [batch_size, seq_len, dim]
        
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

                # 2. 取出正确 label 的概率
                #    先拼出索引对 (batch_idx, label_id)
                batch_idx = tf.range(tf.shape(pred_prob)[0], dtype=tf.int32)
                indices   = tf.stack([batch_idx, label[:, step]], axis=1)  # [B, 2]
                correct_p = tf.gather_nd(pred_prob, indices)               # [B]
                # 3. 打印
                print_tensor("probs/correct_token_prob_%d" % step, tf.reduce_sum(correct_p * loss_mask) / (tf.reduce_sum(loss_mask) + 1e-9))
                result_dict['truth%d_probs' % step] = correct_p
                
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
                recall_at_k(pred_logit, label[:, step], loss_mask, self._print_ops, top_k=[1, 16, 128], name="predict_recall_%d" % step)
                
        print_tensor("loss_mask", loss_mask)
        # 计算加权平均损失
        loss = tf.reduce_sum((losses[0] + losses[1] + losses[2]) * loss_mask) / (tf.reduce_sum(loss_mask) + 1e-9)
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

        encoder_model = EncoderModel(num_layers=8, dim=self._dim, num_heads=8,
                                    dropout_rate=0.1, hidden_dim=self._dim*4)
        decoder_model = DecoderModel(num_layers=8, dim=self._dim, num_heads=8,
                                    dropout_rate=0.1, hidden_dim=self._dim*4)

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
        
        enc_out_base = encoder_model.forward(encoder_input, src_mask, training=False)     # [B, L_enc, C]

        # ---------- ② Beam 状态初始化 ----------
        start_tok = tf.fill([B, 1], self._total_vocab_size)   # global id of <START>
        seqs   = tf.expand_dims(start_tok, 1)                 # [B, 1, 1]
        probs  = tf.ones_like(seqs, dtype=tf.float32)         # [B, 1, 1]
        scores = tf.zeros([B, 1], dtype=tf.float32)           # [B, 1]

        cur_beam = 1  # 当前 beam 数
        cache = {}                    # 全层 KV

        # ---------- ③ 逐层解码 ----------
        for step, V in enumerate(self._vocab_sizes):
            # 只 embed 当前 token
            dec_in = tf.nn.embedding_lookup(self._embedding, seqs[:, :, -1])  # [B,beam,1,C]
            dec_in = tf.reshape(dec_in, [B*cur_beam, 1, self._dim])

            dec_out, cache = decoder_model.step(
                dec_in, cur_beam, enc_out_base, src_mask, cache)            # 只算一步

            last_h = tf.reshape(dec_out, [B, cur_beam, self._dim])
            # # 仅取最后一个 time‑step（上一 token）输出做投影
            # dec_out = tf.reshape(dec_out, [B, cur_beam, -1, self._dim])
            # last_h = dec_out[:, :, -1, :]                                      # [B, cur_beam, C]
            
            with tf.variable_scope('proj_%d' % step):
                logits = tf.layers.dense(last_h, V, name=f'pred', reuse=tf.AUTO_REUSE)

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
    

    # def beam_search_fast(self, beam_size=64, temperature=20):
    #     """
    #     O(batch·beam·logV) 近似复杂度的束搜索（显存与 beam_size 线性）
    #     返回 shape=[B, beam_size, 3] 的推荐 sid 序列
    #     """
        
    #     offsets   = [0,
    #         self._vocab_sizes[0],
    #         self._vocab_sizes[0] + self._vocab_sizes[1]]   # 把局部 id ↦ 全局 id
                
    #     encoder_model = EncoderModel(num_layers=8, dim=self._dim, num_heads=8, dropout_rate=0.1, hidden_dim=self._dim*4)
    #     decoder_model = DecoderModel(num_layers=8, dim=self._dim, num_heads=8, dropout_rate=0.1, hidden_dim=self._dim*4)
        
    #     # ---------- ① 预处理：编码用户 ---------
    #     # 静态 + 点击特征
    #     user_static = tf.concat([self._feature_emb_dict[f] for f in user_static_fea_names], 1)
    #     user_static = mlp('user_static_emb', user_static, [2*self._dim], self._dim,
    #                     activation=tf.nn.leaky_relu)
    #     B = tf.shape(user_static)[0]

    #     user_static = tf.reshape(user_static, [B, 1, self._dim])

    #     user_click  = tf.concat([self._feature_emb_dict[f] for f in user_click_fea_names], 2)
    #     user_click  = mlp('user_click_emb', user_click, [4*self._dim], self._dim,
    #                     activation=tf.nn.leaky_relu)

    #     enc_in      = tf.concat([user_static, user_click], 1)               # [B, L_enc, C]

    #     enc_out = encoder_model.forward(enc_in, training=False) # [batch_size, seq_len, dim]

    #     # ---------- Beam 状态 ----------
    #     start_tok  = tf.fill([B, 1], self._total_vocab_size)
    #     seqs       = tf.tile(tf.expand_dims(start_tok, 1), [1, beam_size, 1])   # token
    #     probs      = tf.ones_like(seqs, dtype=tf.float32)                       # p=1.0
    #     scores     = tf.zeros([B, beam_size], tf.float32)                       # logP

    #     enc_out = tf.tile(tf.expand_dims(enc_out, 1), [1, beam_size, 1, 1])
    #     enc_out = tf.reshape(enc_out, [B*beam_size, -1, self._dim])

    #     # ---------- 逐层解码 ----------
    #     for step, V in enumerate(self._vocab_sizes):
            
    #         dec_in  = tf.nn.embedding_lookup(self._embedding, seqs)
    #         dec_in = tf.reshape(dec_in, [B*beam_size, -1, self._dim])
    #         dec_out = decoder_model.forward(dec_in, enc_out, training=False) # [batch_size, seq_len, dim]
    #         dec_out = tf.reshape(dec_out, [B, beam_size, -1, self._dim])
            
    #         logits  = tf.layers.dense(dec_out[:, :, step, :], V,
    #                                 name=f'proj_{step}', reuse=tf.AUTO_REUSE)

    #         logp    = tf.nn.log_softmax(logits * temperature)   # [B, beam, V]
    #         topk_logp, topk_tok = tf.nn.top_k(logp, k=beam_size)
    #         topk_prob = tf.exp(topk_logp)                       # 转回概率

    #         cand_scores = tf.expand_dims(scores, -1) + topk_logp   # 累积 logP

    #         # ---- 选全局 top-beam ----
    #         flat_scores = tf.reshape(cand_scores, [B, -1])         # [B, beam*beam]
    #         best_scores, best_idx = tf.nn.top_k(flat_scores, k=beam_size)

    #         parent_beam = best_idx // beam_size
    #         tok_rank    = best_idx %  beam_size

    #         batch_idx   = tf.tile(tf.expand_dims(tf.range(B), 1), [1, beam_size])

    #         # ---- gather parent 序列 & 概率 ----
    #         gather_idx  = tf.stack([batch_idx, parent_beam], 2)      # [B, beam, 2]
    #         parent_seq  = tf.gather_nd(seqs,  gather_idx)            # token
    #         parent_prob = tf.gather_nd(probs, gather_idx)            # 对应概率

    #         # ---- gather 具体 token 及其概率 ----
    #         tok_gather  = tf.stack([batch_idx, parent_beam, tok_rank], 2)
    #         next_tok    = tf.gather_nd(topk_tok,   tok_gather)       # [B, beam]
    #         next_prob   = tf.gather_nd(topk_prob,  tok_gather)       # [B, beam]

    #         # 映射到全局 vocabulary id
    #         next_tok_glb = next_tok + offsets[step]         # [B,beam]
            
    #         # ---- 更新路径 ----
    #         seqs  = tf.concat([parent_seq,  tf.expand_dims(next_tok_glb,  -1)], -1)
    #         probs = tf.concat([parent_prob, tf.expand_dims(next_prob, -1)], -1)
    #         scores = best_scores

    #     # [B,beam,1+3]
    #     seqs = seqs[:, :, -3:]
    #     probs = probs[:, :, -3:]
        
    #     # —— offsets = [0, 8192, 16384] 已在前面定义 ——
    #     offsets_t = tf.constant(offsets, dtype=seqs.dtype)          # [3]
    #     gen_part_loc  = seqs - offsets_t                    # 转回局部
        
    #     return gen_part_loc, probs


    # def beam_search_v2(self, beam_sizes=1024):
    #     """
    #     束搜索推理方法 - 版本2
        
    #     支持自适应beam_size，可以在不同步骤使用不同的beam大小
        
    #     Args:
    #         beam_sizes: 可以是单个数值或列表，指定各步骤的beam大小
            
    #     Returns:
    #         selected_sequences: 生成的序列，shape=[batch_size, final_beam_size, seq_len]
    #     """
        
    #     # === 1. 用户特征处理（与版本1相同） ===
    #     user_static_fea = tf.concat([self._feature_emb_dict[fea] for fea in user_static_fea_names], axis=1)
    #     user_static_emb = mlp('user_static_emb', user_static_fea, [2*self._dim], self._dim, activation=tf.nn.leaky_relu)
    #     batch_size = tf.shape(user_static_emb)[0]
    #     user_static_emb = tf.reshape(user_static_emb, [batch_size, 1, self._dim])
    #     user_click_fea = tf.concat([self._feature_emb_dict[fea] for fea in user_click_fea_names], axis=2)
    #     user_click_fea = tf.reshape(user_click_fea, [-1, self._dim * 2])
    #     user_click_emb = mlp('user_click_emb', user_click_fea, [4*self._dim], self._dim, activation=tf.nn.leaky_relu)
    #     user_click_emb_reshape = tf.reshape(user_click_emb, [batch_size, -1, self._dim])
    #     encoder_input = tf.concat([user_static_emb, user_click_emb_reshape], axis=1)

    #     # === 2. 编码器处理 ===
    #     encoder_output = transformer_encoder_layer(encoder_input, 4, dim=self._dim)
    #     # encoder_output = hstu_encoder_layer(encoder_input, 4, dim=self._dim)  # [batch_size, seq_len, dim]
    #     batch_size = tf.shape(encoder_output)[0]

    #     # === 3. 初始化束搜索状态 ===
    #     scores = tf.zeros([batch_size, 1])  # 初始只有一个beam
    #     selected_sequences = tf.tile(tf.constant(self._total_vocab_size, shape=[1, 1, 1]), [batch_size, 1, 1])

    #     # === 4. 自适应束搜索 ===
    #     for step in range(len(beam_sizes)):
    #         seq_len = tf.shape(selected_sequences)[2]

    #         # 解码器前向传播
    #         decoder_input = tf.nn.embedding_lookup(self._embedding, selected_sequences)
    #         decoder_input_beam_size = tf.shape(decoder_input)[1]
    #         encoder_output_expand = tf.tile(tf.expand_dims(encoder_output, axis=1), [1, decoder_input_beam_size, 1, 1])
    #         decoder_output = transformer_decoder_layer(encoder_output_expand, decoder_input, 4, dim=self._dim)
    #         # decoder_output = hstu_decoder_layer(encoder_output_expand, decoder_input, 4, dim=self._dim)
    #         self._print_ops.append(tf.print("decoder_output_shape_%d" % step, tf.shape(decoder_output), summarize=-1, output_stream=sys.stdout))

    #         # 预测概率分布
    #         with tf.variable_scope('proj_%d' % step, reuse=tf.AUTO_REUSE):
    #             logits = tf.layers.dense(decoder_output[:, :, step, :], self._vocab_sizes[step])
            
    #         next_token_probs = tf.nn.softmax(logits/4.0, axis=-1)  # 使用更大的温度参数
    #         log_probs = tf.math.log(next_token_probs)

    #         # === 5. 不同步骤的处理逻辑 ===
    #         if step == 0:
    #             # 第一步：从单个起始状态扩展到vocab_size[0]个候选
    #             append_selected_sequences = tf.tile(tf.expand_dims(tf.range(self._vocab_sizes[0], dtype=tf.int32), axis=0), [batch_size, 1])
    #             append_selected_sequences = tf.expand_dims(append_selected_sequences, axis=2)
    #             selected_sequences = tf.tile(selected_sequences, [1, self._vocab_sizes[0], 1])
    #             selected_sequences = tf.concat([selected_sequences, append_selected_sequences], axis=2)
    #             scores = tf.reshape(log_probs, [batch_size, self._vocab_sizes[0]])
    #         else:
    #             # 后续步骤：动态调整beam大小
    #             last_beam_size = tf.shape(selected_sequences)[1]
    #             cur_beam_size = beam_sizes[step]
    #             cur_num = tf.cast(cur_beam_size / last_beam_size, dtype=tf.int32)  # 每个beam扩展的候选数
    #             candidate_scores = tf.expand_dims(scores, -1) + log_probs
    #             # 构建候选序列
    #             candidate_sequences = tf.expand_dims(selected_sequences, axis=2)
    #             candidate_sequences = tf.tile(candidate_sequences, [1, 1, self._vocab_sizes[step], 1])

    #             add_token = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.range(self._vocab_sizes[step]), axis=1), axis=0), axis=0)
    #             add_token = tf.tile(add_token, [batch_size, last_beam_size, 1, 1])
    #             candidate_sequences = tf.concat([candidate_sequences, add_token], axis=-1)
                
    #             # 选择top candidates
    #             top_k_scores, top_k_indices = tf.math.top_k(candidate_scores, k=cur_num, sorted=True)
    
    #             # 构建3D索引进行gather操作
    #             batch_idx = tf.reshape(tf.range(batch_size), [batch_size, 1, 1])
    #             batch_idx = tf.tile(batch_idx, [1, last_beam_size, cur_num])
    #             beam_idx = tf.reshape(tf.range(last_beam_size), [1, last_beam_size, 1])
    #             beam_idx = tf.tile(beam_idx, [batch_size, 1, cur_num])
    #             gather_indices = tf.stack([batch_idx, beam_idx, top_k_indices], axis=-1)

    #             # 获取新序列
    #             new_sequence = tf.gather_nd(candidate_sequences, gather_indices)

    #             # 更新状态
    #             scores = tf.reshape(top_k_scores, [batch_size, cur_beam_size])
    #             selected_sequences = tf.reshape(new_sequence, [batch_size, cur_beam_size, -1])
                
    #     # === 6. 处理剩余的语义层级 ===
    #     # 对于未在beam_sizes中指定的层级，使用随机初始化（可能用于测试）
    #     for step in range(len(beam_sizes), 4):
    #         with tf.variable_scope('proj_%d' % step, reuse=tf.AUTO_REUSE):
    #             logits = tf.layers.dense(tf.random_uniform([batch_size, self._dim]), self._vocab_sizes[step])
                
    #     return selected_sequences
