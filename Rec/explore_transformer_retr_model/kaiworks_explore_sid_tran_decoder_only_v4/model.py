# -*- coding: utf-8 -*-
"""
用于用户兴趣建模和内容推荐
"""
import tensorflow as tf
import sys
from feature_attr_extract import *
from modulesV2 import *
from modules_ import *

from util import *

# 用户静态特征名称列表
# 包含用户的基本属性信息
# user_static_fea_names = [
#     "user_id",          # 用户ID
#     "user_gender",      # 用户性别
#     "user_age_segment", # 用户年龄段
#     "user_level"        # 用户等级
# ]

# # 用户点击行为特征名称列表
# # 包含用户的历史交互行为数据
# user_click_fea_names = [
#     "user_profile_v1_click_pid_list",  # 用户点击的视频ID列表
#     "user_profile_v1_click_aid_list"   # 用户点击的作者ID列表
# ]

class SIDRecModel(object):
    """
    SID推荐模型
    """
    
    def __init__(self, feature_emb_dict, feature_emb_size_dict, dim=512, select_size=64, 
                 vocab_sizes=[8192, 8192, 8192], print_ops=None):
        """
        初始化模型
        
        Args:
            feature_emb_dict: 特征嵌入字典，存储各特征的嵌入向量
            feature_emb_size_dict: 特征嵌入维度字典
            dim: 模型隐藏层维度
            select_size: 选择的序列长度
            vocab_sizes: 各个词汇表大小的列表，对应不同语义层级
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
            shape=[self._total_vocab_size+2, dim], 
            name='vocab_embedding',
            initializer=tf.random_uniform_initializer(minval=-1.0/dim, maxval=1.0/dim), 
            trainable=True
        )
        
        # 常数 id
        self.PAD_IDX = self._total_vocab_size         # =24576
        self.BOS_IDX = self._total_vocab_size + 1     # =24577
        
        self.offsets = [0,
                   self._vocab_sizes[0],
                   self._vocab_sizes[0] + self._vocab_sizes[1]]
        
        self._dim = dim
        self._select_size = select_size
        
    def get_print_ops(self):
        return [tf.group(*self._print_ops)]
    
    def _select_recent(self, photo_ids):
        """
        Args
        ----
        photo_ids : int64 [B, L]   0 表示无效
        Returns
        -------
        photo_sel : int64 [B, k]   左 PAD = -1，右侧 oldest→newest
        valid_len : int32 [B, 1]   每条样本真实 token 数 (<= k)
        """
        B, L = tf.shape(photo_ids)[0], tf.shape(photo_ids)[1]

        # ---------- 1. 构造筛选掩码 ----------
        mask = tf.not_equal(photo_ids, 0)

        # ---------- 2. 把满足条件的位置编号；不满足 = -1 ----------
        idx_all = tf.tile(tf.expand_dims(tf.range(L), 0), [B, 1])     # [B,L]
        idx_val = tf.where(mask, idx_all, tf.fill([B, L], -1))        # [B,L]

        # ---------- 3. 取最近 k 个（索引越大越新） ----------
        #    top_k 先给出降序(新→旧)，再按升序排成 oldest→newest
        topk_val, _ = tf.nn.top_k(idx_val, k=self._select_size)       # [B,k], 含 -1
        topk_val_sorted = tf.sort(topk_val, direction='ASCENDING')    # [-1, … idx_old → idx_new]

        # ---------- 4. 统计有效条数 ----------
        valid_flags = tf.not_equal(topk_val_sorted, -1)               # [B,k]
        valid_len   = tf.reduce_sum(tf.cast(valid_flags, tf.int32), axis=1, keepdims=True)  # [B,1]

        # ---------- 5. 把 -1 改成 0 才能 gather (0 位置一定存在) ----------
        safe_idx = tf.maximum(topk_val_sorted, 0)                     # [B,k]

        batch = tf.tile(tf.expand_dims(tf.range(B), 1), [1, self._select_size])
        gather = tf.gather_nd(photo_ids, tf.stack([batch, safe_idx], 2))        # [B,k]

        # ---------- 6. 左 PAD = -1 ----------
        pad_cnt   = self._select_size - valid_len                      # [B,1]
        idx_range = tf.range(self._select_size)                        # [k]
        mask_pad  = idx_range < pad_cnt                                # [B,k]  broadcast
        pad_vals  = tf.fill([B, self._select_size], tf.cast(-1, gather.dtype))

        photo_sel = tf.where(mask_pad, pad_vals, gather)               # [B,k]  (-1, … s1,s2,s3)

        return photo_sel, valid_len    # oldest→newest，左 PAD

    # --------------------------- 前向 & loss ----------------------------
    def model(self, user_sid, tgt_sid):            # shapes: [B,200] / [B,1]
        
        B = tf.shape(tgt_sid)[0]
        user_sid = tf.reverse(user_sid, axis=[1])     # reverse 使得 最旧→最新
        # debug
        self._print_ops.append(tf.print("user_sid(first):",
                                        user_sid[0], summarize=100))

        # 1) 拼接当前目标
        photo_ids = tf.concat([user_sid, tgt_sid], 1)       # [B,201] 最旧→最新

        # 2) 最近 k 条
        photo_sel, valid_len = self._select_recent(photo_ids)   # [B,k] [B,1]
        
        # debug
        self._print_ops.append(tf.print("photo_sel(first):",
                                        photo_sel[0], summarize=100))
        
        tokens_flat = processInput(photo_sel)                                      # [B,3k]
        
        # debug
        print_tensor("valid_len", valid_len)           # 实际使用条数
        
        # ------------------------------------------------------------
        # 已有：
        #   tokens_flat  : [B, 3k]   右侧 PAD = -1，左侧 pad_len 个 -1
        #   valid_len    : [B, 1]    每条样本真实 SID 数
        # ------------------------------------------------------------
        B         = tf.shape(tokens_flat)[0]
        L_tokens  = tf.shape(tokens_flat)[1]              # = 3k
        
        pad_len1  = L_tokens - tf.reshape(valid_len * 3, [-1])   # [B]  左 pad 的列数
        bos_col   = tf.fill([B, 1], tf.cast(self.BOS_IDX, tf.int32))

        sid0    = tf.concat([bos_col, tokens_flat], axis=1)     # [B, L_tokens+1]

        # ② 计算“每行把 BOS 插到位置 pad_len1[b]”所需的源索引映射
        L1   = L_tokens + 1
        j    = tf.tile(tf.expand_dims(tf.range(L1), 0), [B, 1])                 # [B, L1]
        R    = tf.cast(tf.expand_dims(pad_len1, 1), dtype=j.dtype)              # [B, 1]

        # j < R 位置来自 sid0 的 j+1（也就是原 tokens 的左半段）
        # j = R 位置来自 sid0 的 0（BOS）
        # j > R 位置来自 sid0 的 j（原 tokens 的右半段）
        src_idx = tf.where(j < R, j + 1, j)
        src_idx = tf.where(tf.equal(j, R), tf.zeros_like(src_idx), src_idx)     # [B, L1]

        # ③ 按映射批量 gather 出最终序列 [B, L_tokens+1]
        batch_ids = tf.tile(tf.expand_dims(tf.range(B), 1), [1, L1])            # [B, L1]
        gather_nd_idx = tf.stack([batch_ids, src_idx], axis=2)                  # [B, L1, 2]
        sid_idx = tf.gather_nd(sid0, gather_nd_idx)                             # [B, L1]

        # debug
        self._print_ops.append(tf.print("sid_idx(first):",
                                        sid_idx[0], summarize=100))
        
        # 把所有 -1 → PAD_IDX
        sid_idx = tf.where(
            tf.equal(sid_idx, tf.cast(-1, sid_idx.dtype)),
            tf.fill(tf.shape(sid_idx), tf.cast(self.PAD_IDX, sid_idx.dtype)),
            sid_idx
        )                                  # [B, 3k+1]

        # 4) Embedding & mask
        dec_in  = tf.nn.embedding_lookup(self._embedding, sid_idx)             # [B,3k+1,D]
        nonpad  = tf.not_equal(sid_idx, self.PAD_IDX)  # [B,3k+1]
        L       = tf.shape(sid_idx)[1]
        causal  = tf.linalg.band_part(tf.ones([L, L], tf.bool), -1, 0)
        attn_m  = tf.cast(tf.expand_dims(nonpad, 1) & causal, tf.int8)
        attn_m  = tf.expand_dims(attn_m, 1)                                    # [B,1,3k+1,3k+1]

        # mask为PAD的输入 tf.where 只对 True 分支回传梯度，PAD 行不会回到 embedding
        dec_in  = tf.where(tf.broadcast_to(tf.expand_dims(nonpad, -1), [B, L, self._dim]), dec_in, tf.zeros_like(dec_in))
                
        # 5) Decoder
        decoder = DecoderOnlyModel(num_layers=4, dim=self._dim, num_heads=8,
                                   hidden_dim=self._dim*2, dropout_rate=0.1, training=True)
        dec_out = decoder.forward(dec_in, attn_m, training=True)                              # [B,3k+1,D]
        
        # 计算decoder输出的余弦相似度（用于调试）
        dec_out_sim = tf.reshape(dec_out, [B, -1])
        print_tensor("decoder_output_sim", calc_sim_cos(dec_out_sim))
        
        # 对最后3级（除最后一级输入的解码结果的前面三级）分别计算余弦相似度
        for step in range(len(self._vocab_sizes)):
            L  = tf.shape(dec_out)[1]          # Tensor
            idx = L - 3 + step - 1             # Tensor
            similarity = calc_sim_cos(dec_out[:, idx, :])
            print_tensor('decoder_sim/decoder_output_%d' % step, similarity)
        
        # 6) Teacher-Forcing & 单 BOS 层级判定
        dec_pred = dec_out[:, :-1, :]                   # hidden 0…L-2 [B, 3k, D]
        tgt_abs  = sid_idx[:, 1:]                       # labels 1…L-1 [B, 3k, D]
        tgt_mask = nonpad[:, :-1]                       # bool

        T_pred = tf.shape(dec_pred)[1]
        pos_mod3 = tf.math.floormod(tf.range(T_pred, dtype=tf.int32), 3)   # [3k]
        level_target = [0, 1, 2]                       # BOS→0, level-0→1, level-1→2

        losses = []
        losses_last = []
        eps = 1e-6
        
        for lvl in range(3):
            
            # -------- 1) 选出本层级对应的时间位置（所有 batch 共用同一子序列） --------
            time_sel = tf.equal(pos_mod3, level_target[lvl])            # [3k] bool
            # 把 [B,T,...] → 仅保留本层级步：[B, T_l, ...]  T=3k T_l=k
            dec_l  = tf.boolean_mask(dec_pred, time_sel, axis=1)        # [B, k, D]
            lab_l  = tf.boolean_mask(tgt_abs,  time_sel, axis=1)        # [B, k]
            mask_l = tf.boolean_mask(tgt_mask, time_sel, axis=1)        # [B, k] bool
            w_l    = tf.cast(mask_l, tf.float32)                        # [B, k]
            
            # -------- 2) 投影到该层级 vocab（只算 T_l 个 time step） --------
            with tf.variable_scope('proj_%d' % lvl, reuse=tf.AUTO_REUSE):
                
                logits_l = tf.layers.dense(dec_l, self._vocab_sizes[lvl], name='pred')  # [B,T_l,V_l]

                # -------- 3) 相对标签 + 仅对 padding 做 mask --------
                lab_rel = tf.cast(lab_l - self.offsets[lvl], tf.int32)                              # [B,T_l]
                lab_rel_safe = tf.where(mask_l, lab_rel, tf.zeros_like(lab_rel))               # [B,T_l]

                # -------- 4) 逐位交叉熵（只对有效位加权），避免被非本层时间步拖累 --------
                ce_bt = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=lab_rel_safe,
                                                                    logits=logits_l)        # [B,T_l]
                denom = tf.reduce_sum(w_l) + eps
                loss_l = tf.reduce_sum(ce_bt * w_l) / denom
                losses.append(loss_l)
                
                print_tensor("loss_all/loss_%d" % lvl, loss_l)
                
                # 最后一个预测位的loss
                denom_last = tf.reduce_sum(w_l[:,-1]) + eps
                loss_last = tf.reduce_sum(ce_bt[:,-1] * w_l[:,-1]) / denom_last
                losses_last.append(loss_last)
                
                print_tensor("loss_last/loss_%d" % lvl, loss_last)
                
                # 最后一个预测位的各种指标
                pred_logit = logits_l[:,-1,:] # [B, V]
                label = lab_rel_safe[:,-1] # [B]
                loss_mask = w_l[:,-1] # [B]
                
                # 1. 求 softmax 概率
                pred_prob = tf.nn.softmax(pred_logit, axis=-1)  # [B, V]

                # 2. 取出正确 label 的概率
                #    先拼出索引对 (batch_idx, label_id)
                batch_idx = tf.range(tf.shape(pred_prob)[0], dtype=tf.int32)
                indices   = tf.stack([batch_idx, label], axis=1)  # [B, 2]
                correct_p = tf.gather_nd(pred_prob, indices)               # [B]
                print_tensor("probs/correct_token_prob_%d" % step, tf.reduce_sum(correct_p * loss_mask) / (tf.reduce_sum(loss_mask) + eps))
                
                # 3. 最大的 prob 概率
                max_probs, _ = tf.nn.top_k(pred_prob, k=1)
                print_tensor("probs/max_token_prob_%d" % step, tf.reduce_sum(tf.squeeze(max_probs, -1) * loss_mask) / (tf.reduce_sum(loss_mask) + eps))
                
                # 计算各种recall指标
                recall_at_k(pred_logit, label, loss_mask, self._print_ops, top_k=[1, 16, 128], name="predict_recall_%d" % step)
            
        print_tensor("loss_all/loss_all", tf.add_n(losses))
        print_tensor("loss_last/loss_last", tf.add_n(losses_last))
        
        return tf.add_n(losses) / 3.0
    
    
    def beam_search_fast(self, user_sid, beam_size=64, temperature=1.0):
        """
        基于 step+KV cache 的自回归束搜索，生成后续 3 个 token。
        返回:
        gen_part_loc:  [B, beam_size, 3]  (局部 sid)
        probs: [B, beam_size, 3]  (对应 softmax 概率)
        """
                
        # ---------- 0) 取最近 k 条 → 3k tokens，并在左侧插入 BOS ----------
        user_sid = tf.reverse(user_sid, axis=[1])     # reverse 使得 最旧→最新
        photo_sel, valid_len = self._select_recent(user_sid)   # [B,k], [B,1]
        tokens_flat = processInput(photo_sel)                                       # [B,3k]，左 PAD=-1
        B        = tf.shape(tokens_flat)[0]
        L_tokens = tf.shape(tokens_flat)[1]                    # = 3k
        pad_len1 = tf.reshape(L_tokens - valid_len * 3, [-1])  # [B]

        # 批量把 BOS 插到每行 pad_len1[b] 位置
        bos_col = tf.fill([B, 1], tf.cast(self.BOS_IDX, tf.int32))
        sid0    = tf.concat([bos_col, tokens_flat], axis=1)    # [B, 3k+1]
        L1      = tf.shape(sid0)[1]
        j       = tf.tile(tf.expand_dims(tf.range(L1), 0), [B, 1])     # [B, L1]
        R       = tf.expand_dims(pad_len1, 1)                          # [B, 1]
        src_idx = tf.where(j < R, j + 1, j)
        src_idx = tf.where(tf.equal(j, R), tf.zeros_like(src_idx), src_idx)
        
        batch_ids = tf.tile(tf.expand_dims(tf.range(B), 1), [1, L1])
        sid_idx = tf.gather_nd(sid0, tf.stack([batch_ids, src_idx], axis=2))        # [B,3k+1]

        # 把所有 -1 → PAD_IDX
        sid_idx = tf.where(
            tf.equal(sid_idx, tf.cast(-1, sid_idx.dtype)),
            tf.fill(tf.shape(sid_idx), tf.cast(self.PAD_IDX, sid_idx.dtype)),
            sid_idx
        )                                  # [B, 3k+1]

        # ---------- 1) 准备模型与 prefix 的 KV cache ----------
        decoder = DecoderOnlyModel(num_layers=4, dim=self._dim, num_heads=8,
                                hidden_dim=self._dim*2, dropout_rate=0.1, training=False)
        
        B = tf.shape(sid_idx)[0]
        
        # 对前面L-1长度 填充每层的 KV cache
        cache = {}
        emb_prefix = tf.nn.embedding_lookup(self._embedding, sid_idx[:, :-1])  # [B, L-1, D]
        nonpad_prefix = tf.not_equal(sid_idx[:, :-1], self.PAD_IDX)                  # [B, L-1]
        L       = tf.shape(sid_idx[:, :-1])[1]
        causal  = tf.linalg.band_part(tf.ones([L, L], tf.bool), -1, 0)
        attn_m  = tf.cast(tf.expand_dims(nonpad_prefix, 1) & causal, tf.int8)
        attn_m  = tf.expand_dims(attn_m, 1)                                    # [B,1,L-1,L-1]
        
        # mask为PAD的输入
        emb_prefix  = tf.where(tf.broadcast_to(tf.expand_dims(nonpad_prefix, -1), [B, L, self._dim]), emb_prefix, tf.zeros_like(emb_prefix))
        
        _, cache = decoder.forward_with_cache(emb_prefix, 1, attn_m, cache)       # 更新KV，并得到输出
        
        # ---------- 2) Beam 初始化 ----------
        # 序列 / 概率 / 分数
        seqs   = tf.expand_dims(sid_idx, 1)                  # [B, 1, L]
        probs  = tf.ones_like(seqs, dtype=tf.float32)         # [B, 1, L]
        scores = tf.zeros([B, 1], tf.float32)
        
        cur_beam = 1

        atten_mask = tf.cast(tf.expand_dims(tf.expand_dims(nonpad_prefix, 1), 1), tf.int8) # [B,1,1,L-1]
        
        for step, V in enumerate(self._vocab_sizes):
            
            # 只 embed 当前 token
            dec_in = tf.nn.embedding_lookup(self._embedding, seqs[:, :, -1])  # [B,beam,1,C]
            dec_in = tf.reshape(dec_in, [B*cur_beam, 1, self._dim])
            
            # 在最后一个维度添加一个1
            ones_to_add = tf.ones([B, 1, 1, 1], dtype=tf.int8)
            atten_mask = tf.concat([atten_mask, ones_to_add], axis=-1)  # [B,1,1,cur_len]
            
            dec_out, cache = decoder.step(
                dec_in, cur_beam, atten_mask, cache)            # 只算一步

            last_h = tf.reshape(dec_out, [B, cur_beam, self._dim])
            
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
            next_tok_glb = next_tok + self.offsets[step]

            # 更新序列
            seqs  = tf.concat([parent_seq, tf.expand_dims(next_tok_glb, -1)], axis=-1)  # [B, beam, T+1]
            probs = tf.concat([parent_prob, tf.expand_dims(next_prob,  -1)], axis=-1)
            scores = best_scores                                                # [B, beam]

            cur_beam = beam_size            # 以后固定

        # 只取最后三个
        seqs  = seqs[:, :, -3:]
        probs = probs[:, :, -3:]

        # 转回局部 id
        offsets_t = tf.constant(self.offsets, dtype=seqs.dtype)          # [3]
        gen_part_loc = seqs - offsets_t

        return gen_part_loc, probs
