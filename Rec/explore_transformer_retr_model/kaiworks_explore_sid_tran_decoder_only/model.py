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
        
    def get_print_ops(self):
        return [tf.group(*self._print_ops)]

    def model(self, user_sid_list, photo_sid, label, photo_semantic_id_int):
        """
        主训练模型前向传播
        
        Args:
            photo_sid: 视频语义ID序列，shape=[batch_size, seq_len]
            label: 真实标签，shape=[batch_size, 3]，对应三个语义层级
            photo_semantic_id_int: 视频语义ID整数序列，用于计算loss mask
            
        Returns:
            loss: 训练损失值
        """
        
        PAD_IDX = self._total_vocab_size
        # ① 右移一格 —— 去掉 photo_sid 的最后一个 token
        #    等价于在最前面插一个 <START>（或任何占位符），
        #    这里直接把最后一个切掉就行
        decoder_sid = tf.concat([user_sid_list, photo_sid[:, :-1]], axis=1)
        
        # debug
        self._print_ops.append(tf.print("all_sid_list first sample:", decoder_sid[0], summarize=100))
        self._print_ops.append(tf.print("all_sid_list shape:", tf.shape(decoder_sid), summarize=100))
        
        # 把 -1 映射到 PAD_IDX
        sid_idx = tf.where(
                    tf.equal(decoder_sid, -1),
                    tf.fill(tf.shape(decoder_sid), PAD_IDX),
                    decoder_sid)                          # [B, L]
        # debug
        self._print_ops.append(tf.print("sid_idx first sample:", sid_idx[0], summarize=100))
        
        # ② embedding lookup
        dec_embedding = tf.nn.embedding_lookup(self._embedding, sid_idx)  # [B, L, D]

        # --------------------------------------------------------
        # ③ Attention mask / bias  (右下三角 + pad)
        #      attn_bias 直接加到 QK^T / √d 结果上 —— 常见写法
        # --------------------------------------------------------
        B, L = tf.shape(sid_idx)[0], tf.shape(sid_idx)[1]
        
        # 3.1  padding mask
        nonpad = tf.not_equal(sid_idx, PAD_IDX)           # [B, L]  bool

        # 3.2  causal 下三角  (True 允许，False 屏蔽)
        tri = tf.linalg.band_part(tf.ones([L, L], tf.bool), -1, 0)    # [L, L]

        # 3.3  综合 mask = 非pad_key & causal
        key_mask  = tf.expand_dims(nonpad, 1)             # [B, 1, L]
        mask_bool = tf.logical_and(tri, key_mask)         # [B, L, L]
        
        # debug
        self._print_ops.append(tf.print("mask_bool first sample:", mask_bool[0], summarize=100))
        
        # 3.4  转 int：允许=1，屏蔽=0
        atten_mask = tf.cast(mask_bool, tf.int64)     # [B, L, L]
        
        # --- 注意力 bias 扩充到 head 维 ---
        atten_mask = tf.expand_dims(atten_mask, 1)   # [B,1,L,L]
        
        decoder_model = DecoderOnlyModel(num_layers=4, dim=self._dim, num_heads=8, hidden_dim=self._dim*2, dropout_rate=0.1, training=True)
        decoder_output = decoder_model.forward(dec_embedding, atten_mask, training=True) # [batch_size, seq_len, dim]
        
        # 计算解码器各步输出的余弦相似度（用于调试）
        for step in range(len(self._vocab_sizes)):
            # --- slice decoder_output 最后 3 步 ---
            L  = tf.shape(decoder_output)[1]          # Tensor
            idx = L - 3 + step                       # Tensor
            similarity = calc_sim_cos(decoder_output[:, idx, :])
            print_tensor('decoder_sim/decoder_output_%d' % step, similarity)

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
            with tf.variable_scope('proj_%d' % step, reuse=tf.AUTO_REUSE):
                # --- slice decoder_output 最后 3 步 ---
                L  = tf.shape(decoder_output)[1]          # Tensor
                idx = L - 3 + step                       # Tensor
                pred_logit = tf.layers.dense(decoder_output[:, idx, :], self._vocab_sizes[step], name='pred')

                print_tensor("logits/pred_logit_%d" % step, pred_logit)
                
                # 1. 求 softmax 概率
                pred_prob = tf.nn.softmax(pred_logit, axis=-1)  # [B, V]

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
                loss_i = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels, logits=pred_logit)
                losses.append(loss_i)
            
                # 打印每个层次的损失
                print_tensor("loss/loss_%d" % step, tf.reduce_sum(loss_i * loss_mask) / tf.reduce_sum(loss_mask + 1e-6))
                
                # 计算各种recall指标
                recall_at_k(pred_logit, label[:, step], loss_mask, self._print_ops, top_k=[1, 16, 128], name="predict_recall_%d" % step)
                
        print_tensor("loss_mask", loss_mask)
        # 计算加权平均损失
        loss = tf.reduce_sum((losses[0] + losses[1] + losses[2]) * loss_mask) / tf.reduce_sum(loss_mask + 1e-6)
        return loss
    
    
    def beam_search_fast(self, user_sid_list, beam_size=64, temperature=1.0):
        """
        O(batch·beam·logV) 近似复杂度的束搜索（显存与 beam_size 线性）
        返回 shape=[B, beam_size, seq_len] 的推荐 sid 序列
        """
        
        decoder_model = DecoderOnlyModel(num_layers=4, dim=self._dim, num_heads=8, hidden_dim=self._dim*2, dropout_rate=0.1, training=False)
        
        # ---------- ① 预处理：编码用户 ---------
        PAD_IDX = self._total_vocab_size
        
        offsets = [0,
                self._vocab_sizes[0],
                self._vocab_sizes[0] + self._vocab_sizes[1]]   # 把局部 id ↦ 全局 id
        
        # 把 -1 映射到 PAD_IDX
        sid_idx = tf.where(
                    tf.equal(user_sid_list, -1),
                    tf.fill(tf.shape(user_sid_list), PAD_IDX),
                    user_sid_list)                          # [B, L]
        
        # --------------------------------------------------------
        # ③ Attention mask / bias  (右下三角 + pad)
        #      attn_bias 直接加到 QK^T / √d 结果上 —— 常见写法
        # --------------------------------------------------------
        B, L = tf.shape(sid_idx)[0], tf.shape(sid_idx)[1]
        gen_steps = 3-1                      # 要补的位置
        
        # 3.1  padding mask
        nonpad = tf.not_equal(sid_idx, PAD_IDX)           # [B, L]  bool
        
        # 把待生成的 3 个位置也补成 non-pad=True
        gen_pad = tf.ones([tf.shape(sid_idx)[0], gen_steps], dtype=tf.bool)
        nonpad_ext = tf.concat([nonpad, gen_pad], axis=1)      # [B, L+2]

        # 3.2  causal 下三角  (True 允许，False 屏蔽)
        L_total = tf.shape(sid_idx)[1] + gen_steps            # L+2 (张量)
        tri = tf.linalg.band_part(tf.ones([L_total, L_total], tf.bool),
                                -1, 0)                      # [L+2, L+2]

        # 3.3  综合 mask = 非pad_key & causal
        key_mask  = tf.expand_dims(nonpad_ext, 1)             # [B, 1, L+2]
        mask_bool = tf.logical_and(tri, key_mask)             # [B, L+2, L+2]
        
        # 3.4  转 int：允许=1，屏蔽=0
        atten_mask = tf.cast(mask_bool, tf.int64)     # [B, L+2, L+2]
        
        # --- 注意力 bias 扩充到 head 维 ---
        atten_mask = tf.expand_dims(atten_mask, 1)   # [B,1,L+2,L+2]
        
        # ---------- Beam 状态 ----------
        seqs   = tf.expand_dims(sid_idx, 1)                      # [B,1,L]
        probs  = tf.ones_like(seqs, dtype=tf.float32)                     # 对应位置概率
        scores = tf.zeros([B, beam_size], tf.float32)                     # 累积 logP
        
        cur_beam = 1  # 当前 beam 数
        
        # ---------- 逐层解码 ----------
        for step, V in enumerate(self._vocab_sizes):
            
            t_len  = L + step               # 当前序列长度  (不含将要生成的位置)
            atten_t = atten_mask[:, :, :t_len, :t_len]          # [B,1,t,t]
        
            # ---- (B*beam, t) 送进 Decoder ----
            flat_seq  = tf.reshape(seqs,   [-1, t_len])          # [B*beam,t]
            flat_mask = tf.tile(atten_t, [cur_beam, 1, 1, 1])    # [B*beam,t,t]
            flat_emb  = tf.nn.embedding_lookup(self._embedding, flat_seq)
            
            dec_out = decoder_model.forward(flat_emb, flat_mask, training=False) # [B*beam,t,dim]
            
            dec_out = tf.reshape(dec_out, [B, cur_beam, -1, self._dim])

            # 仅取最后一个 time‑step（上一 token）输出做投影
            last_h = dec_out[:, :, -1, :]                                      # [B, cur_beam, C]
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
        seqs  = seqs[:, :, -3:]
        probs = probs[:, :, -3:]

        # 转回局部 id
        offsets_t = tf.constant(offsets, dtype=seqs.dtype)          # [3]
        gen_part_loc = seqs - offsets_t

        return gen_part_loc, probs
    
    
    