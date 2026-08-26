# -*- coding: utf-8 -*-

import tensorflow as tf
import sys
from feature_attr_extract import *
from modulesV2 import *
from modules_ import *

from util import *

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
# user_click_fea_names = [
#     "user_profile_v1_click_pid_list",  # 用户点击的视频ID列表
#     "user_profile_v1_click_aid_list"   # 用户点击的作者ID列表
# ]

user_colossus_fea_names = [
    "user_colossus_pid_list",  # 用户点击视频ID列表
    "user_colossus_aid_list"   # 用户点击的作者ID列表
]

class SIDRecModel(object):
    """
    SID推荐模型类 生成推荐的语义ID序列
    """
    
    def __init__(self, feature_emb_dict, feature_emb_size_dict, dim=512, vocab_sizes=[8192, 8192, 8192], print_ops=None):
        """
        初始化模型
        
        Args:
            feature_emb_dict: 特征嵌入字典，存储各特征的嵌入向量
            feature_emb_size_dict: 特征嵌入维度字典
            dim: 模型隐藏层维度，默认64
            vocab_sizes: 各个词汇表大小的列表，对应不同语义层级，默认[8192, 8192, 8192]
            print_ops: 用于调试的打印操作列表
        """
        self._feature_emb_dict = feature_emb_dict
        self._feature_emb_size_dict = feature_emb_size_dict
        self._dim = dim
        self._vocab_sizes = vocab_sizes  # 三个语义层级的词汇表大小
        self._print_ops = print_ops
        
        self._total_vocab_size = sum(self._vocab_sizes)  # 总词汇表大小
        
        # 创建统一的嵌入矩阵，包含所有语义ID的嵌入向量
        # 使用均匀分布初始化，范围为[-1/dim, 1/dim]
        self._vocab_embedding = tf.get_variable(
            shape=[self._total_vocab_size+1, dim], 
            name='vocab_embedding',
            initializer=tf.random_uniform_initializer(minval=-1.0/dim, maxval=1.0/dim), 
            trainable=True
        )
        
    def model(self, click_semantic_id_list, colossus_semantic_id_list, photo_sid, label, photo_semantic_id_int):
        """
        单次前向 + 分步mask：
        - enc = user_static + click(200) + colossus_sel(≤K_sel)
        - t=0: 只看 click(长/短 200/64 逐样本二选一)
        - t=1/2: 只看 colossus_sel(与 label[0] 同类且最近的 K_sel)
        """
        # === 常量 ===
        K_tail = 64     # 第一级短期窗口
        K_sel  = 50     # 二/三级 colossus 选择窗口
        temperature = 1.0

        def expand_src_mask(mask_BxTqxS):
            # mask_BxTqxS -> [B,1,Tq,S]
            return tf.expand_dims(mask_BxTqxS, axis=1)

        # === 1) 用户静态 ===
        user_static_fea = tf.concat([self._feature_emb_dict[fea] for fea in user_static_fea_names], axis=1)
        user_static_emb = mlp('user_static_emb', user_static_fea, [2*self._dim], self._dim, activation=tf.nn.leaky_relu)
        B = tf.shape(user_static_emb)[0]
        user_static_emb = tf.reshape(user_static_emb, [B, 1, self._dim])  # [B,1,C]
        user_tok_mask = tf.ones([B, 1], dtype=tf.int8)

        # === 2) Click(1000) 最旧->最新 右padding ===
        user_click_fea = tf.concat([self._feature_emb_dict[fea] for fea in user_colossus_fea_names], axis=2)
        click_max_len = 1000
        click_raw_len = tf.cast(self._feature_emb_size_dict['user_colossus_pid_list'], tf.int32)
        click_used_len = tf.minimum(tf.reshape(click_raw_len, [-1]), tf.constant(click_max_len, tf.int32))  # [B]

        # debug
        print_tensor("click_len/raw", click_raw_len)
        print_tensor("click_len/used", click_used_len) 
        
        # user_click_fea = tf.reverse(user_click_fea, axis=[1])                 # [B,Lc,Fea]
        # user_click_sid = tf.reverse(colossus_semantic_id_list, axis=[1])         # [B,Lc] int64
        user_click_emb = mlp('user_click_emb', user_click_fea, [4*self._dim], self._dim, activation=tf.nn.leaky_relu)

        # click 有效位（按长度）∧ sid!=0
        click_mask_left = tf.sequence_mask(click_used_len, maxlen=click_max_len, dtype=tf.int8)  # [B,Lc] 左对齐
        # click_mask = tf.reverse(click_mask_left, axis=[1])                                       # [B,Lc]

        # === 2.1 第一级 长/短 1000/64 二选一（按与 label[0] 的相似度） ===
        Lc = tf.shape(colossus_semantic_id_list)[1]
        pos_idx = tf.expand_dims(tf.range(Lc, dtype=tf.int32), 0)            # [1,Lc]
        tail64  = tf.cast(pos_idx >= (Lc - K_tail), tf.int8)                 # [1,Lc]
        click_mask_short = tf.bitwise.bitwise_and(click_mask_left, tail64)        # [B,Lc]

        seq_mask_all_long  = tf.concat([user_tok_mask, click_mask_left],        axis=1)  # [B,1+Lc]
        seq_mask_all_short = tf.concat([user_tok_mask, click_mask_short], axis=1)   # [B,1+Lc]

        # 用 label[0] 与长/短池化求相似度
        lvl1_click = tf.cast(tf.bitwise.bitwise_and(
            tf.bitwise.right_shift(tf.cast(colossus_semantic_id_list, tf.int64), tf.constant(30, dtype=tf.int64)),
            tf.constant(0x7FFF, dtype=tf.int64)
        ), tf.int32)                                                          # [B,Lc]
        click_sid_emb = tf.nn.embedding_lookup(self._vocab_embedding, lvl1_click)   # [B,Lc,C]

        def masked_avg(x, m):
            m_f = tf.cast(m, tf.float32)
            denom = tf.reduce_sum(m_f, axis=1, keepdims=True) + 1e-9
            return tf.reduce_sum(x * m_f[:, :, None], axis=1) / denom

        valid_sid_mask = tf.cast(tf.not_equal(colossus_semantic_id_list, 0), tf.int8)
        
        pool_long  = masked_avg(click_sid_emb, tf.bitwise.bitwise_and(click_mask_left, valid_sid_mask))
        pool_short = masked_avg(click_sid_emb, tf.bitwise.bitwise_and(click_mask_short, valid_sid_mask))

        label_lvl1 = tf.cast(label[:, 0], tf.int32)
        label_emb0 = tf.nn.embedding_lookup(self._vocab_embedding, label_lvl1)     # [B,C]

        def cos(a, b): return tf.reduce_sum(tf.nn.l2_normalize(a, -1)*tf.nn.l2_normalize(b, -1), axis=-1)

        bias = 0.0
        sim_long  = cos(label_emb0, pool_long)
        sim_short = cos(label_emb0, pool_short) + bias
        choose_short = sim_short > sim_long                                       # [B] bool

        pos0_long  = tf.expand_dims(seq_mask_all_long,  axis=1)                  # [B,1,1+Lc]
        pos0_short = tf.expand_dims(seq_mask_all_short, axis=1)                  # [B,1,1+Lc]
        pos0_click = tf.where(tf.broadcast_to(choose_short[:, None, None], tf.shape(pos0_long)),
                            pos0_short, pos0_long)                              # [B,1,1+Lc]
        
        # debug
        print_tensor("choose_sim/long", sim_long)
        print_tensor("choose_sim/short", sim_short)
        
        print_tensor("choose_count/total", B)
        print_tensor("choose_count/long", tf.reduce_sum(1-tf.cast(choose_short, tf.int32)))
        print_tensor("choose_count/short", tf.reduce_sum(tf.cast(choose_short, tf.int32)))
        
        # === 3) Colossus：选同类(=label[0])最近 K_sel（不足补0），只对选出的过 MLP ===
        user_colossus_fea = tf.concat([self._feature_emb_dict[fea] for fea in user_colossus_fea_names], axis=2)  # [B,Lcol,Fea]
        user_colossus_sid = colossus_semantic_id_list                                                            # [B,Lcol] int64
        Lcol = tf.shape(user_colossus_sid)[1]
        col_valid = tf.cast(tf.not_equal(user_colossus_sid, 0), tf.int8)

        lvl1_col = tf.cast(tf.bitwise.bitwise_and(
            tf.bitwise.right_shift(tf.cast(user_colossus_sid, tf.int64), tf.constant(30, dtype=tf.int64)),
            tf.constant(0x7FFF, dtype=tf.int64)
        ), tf.int32)  # [B,Lcol]

        same_lvl1 = tf.equal(lvl1_col, label_lvl1[:, None])                       # [B,Lcol] bool
        cond_mask = tf.bitwise.bitwise_and(tf.cast(same_lvl1, tf.int8), col_valid)# [B,Lcol]

        recency = tf.cast(tf.range(Lcol)[None, :], tf.float32)                    # 越大越新
        scores_sel = tf.cast(cond_mask, tf.float32)*recency + (1.0-tf.cast(cond_mask, tf.float32))*(-1e9)
        _, top_idx = tf.nn.top_k(scores_sel, k=K_sel)                              # [B,K_sel]

        sel_fea     = tf.batch_gather(user_colossus_fea, tf.cast(top_idx, tf.int32))           # [B,K_sel,Fea]
        sel_present = tf.cast(tf.batch_gather(tf.cast(cond_mask, tf.float32), top_idx) > 0.5, tf.int8)  # [B,K_sel]

        sel_emb = mlp('user_click_emb', sel_fea, [4*self._dim], self._dim, activation=tf.nn.leaky_relu)  # 复用 click 的 MLP
       
        # debug
        print_tensor("sid_count/valid_colossus_sid", tf.reduce_sum(tf.cast(col_valid, tf.int32), axis=1))
        print_tensor("sid_count/select_colossus_sid", tf.reduce_sum(tf.cast(cond_mask, tf.int32), axis=1))
        
        # === 4) Encoder 一次性拼接：user | click(200) | colossus_sel(≤K_sel) ===
        encoder_input = tf.concat([user_static_emb, user_click_emb, sel_emb], axis=1)          # [B, 1+Lc+K_sel, C]

        # === 5) Decoder 输入（teacher forcing） ===
        start_tok = tf.fill([B, 1], self._total_vocab_size)             # [B,1]
        # 只用到前两个 label（预测 y0,y1,y2）：输入为 BOS, y0_gt, y1_gt
        dec_in_ids = tf.concat([start_tok, photo_sid[:, :2]], axis=1)   # [B,3]
        decoder_input = tf.nn.embedding_lookup(self._vocab_embedding, dec_in_ids)  # [B,3,C]
        Tq = tf.shape(decoder_input)[1]                                  # =3

        # === 6) 按 step 构造跨注意力 mask（一次 forward） ===
        # t=0：看 user+click(长/短)；colossus_sel 屏蔽
        zeros_col = tf.zeros([B, K_sel], dtype=tf.int8)
        pos0 = tf.concat([pos0_click, tf.expand_dims(zeros_col, 1)], axis=2)                    # [B,1, 1+Lc+K_sel]

        # t>=1：看 user + colossus_sel；click 屏蔽
        zeros_click = tf.zeros([B, Lc], dtype=tf.int8)
        sel_mask = tf.concat([user_tok_mask, zeros_click, sel_present], axis=1)                 # [B, 1+Lc+K_sel]
        rest_len = tf.maximum(Tq - 1, 0)
        rest = tf.tile(tf.expand_dims(sel_mask, axis=1), [1, rest_len, 1])                      # [B,Tq-1,S]

        seq_mask_per_t = tf.concat([pos0, rest], axis=1)                                        # [B,Tq,S]
        src_mask = expand_src_mask(seq_mask_per_t)                                              # [B,1,Tq,S]

        # === 7) Decoder 一次 forward ===
        decoder_model = DecoderModel(num_layers=4, dim=self._dim, num_heads=8,
                                    dropout_rate=0.1, hidden_dim=self._dim*2, training=True)
        decoder_output = decoder_model.forward(decoder_input, encoder_input, src_mask, training=True)  # [B,3,C]

        # === 8) 三个投影 & 损失 ===
        logits = []
        for step_i in range(3):
            with tf.variable_scope(f'proj_{step_i}'):
                logits_i = tf.layers.dense(decoder_output[:, step_i, :], self._vocab_sizes[step_i], name='pred')
            logits.append(logits_i)

        # step 掩码（每个位置是否有效）
        step_mask = tf.where(
            photo_semantic_id_int > 0,  
            tf.ones_like(photo_semantic_id_int, dtype=tf.float32), 
            tf.zeros_like(photo_semantic_id_int, dtype=tf.float32)
        )
        step_mask = tf.reshape(step_mask, [-1])

        def ce_loss(logits_i, y_i, w_i):
            oh = tf.one_hot(y_i, tf.shape(logits_i)[-1])
            loss_i = tf.nn.softmax_cross_entropy_with_logits(labels=oh, logits=logits_i/temperature)
            denom = tf.reduce_sum(w_i) + 1e-9
            return tf.reduce_sum(loss_i * w_i) / denom

        loss0 = ce_loss(logits[0], label[:, 0], step_mask)
        loss1 = ce_loss(logits[1], label[:, 1], step_mask)
        loss2 = ce_loss(logits[2], label[:, 2], step_mask)
        loss = loss0 + loss1 + loss2

        # 调试
        # 计算enc的余弦相似度
        enc_all_sim = tf.reshape(encoder_input, [B, -1])
        enc_click_sim = tf.reshape(user_click_emb, [B, -1])
        enc_sel_colossus_sim = tf.reshape(sel_emb, [B, -1])
        print_tensor("enc_sim/enc_all_sim", calc_sim_cos(enc_all_sim))
        print_tensor("enc_sim/enc_click_sim", calc_sim_cos(enc_click_sim))
        print_tensor("enc_sim/enc_sel_colossus_sim", calc_sim_cos(enc_sel_colossus_sim))
        # 计算dec out的余弦相似度
        print_tensor("dec_sim/dec_out0", calc_sim_cos(decoder_output[:, 0, :]))
        print_tensor("dec_sim/dec_out1", calc_sim_cos(decoder_output[:, 1, :]))
        print_tensor("dec_sim/dec_out2", calc_sim_cos(decoder_output[:, 2, :]))

        def log_probs_and_recall(logits_i, y_i, w_i, name):
            prob = tf.nn.softmax(logits_i, axis=-1)
            batch_idx = tf.range(tf.shape(prob)[0], dtype=tf.int32)
            idx = tf.stack([batch_idx, y_i], axis=1)
            true_p = tf.gather_nd(prob, idx)
            print_tensor(f"probs/true_token_prob_{name}",
                        tf.reduce_sum(true_p * w_i) / (tf.reduce_sum(w_i) + 1e-9))
            max_probs, _ = tf.nn.top_k(prob, k=1)
            print_tensor(f"probs/max_token_prob_{name}",
                        tf.reduce_sum(tf.squeeze(max_probs, -1) * w_i) / (tf.reduce_sum(w_i) + 1e-9))
            recall_at_k(logits_i, y_i, w_i, self._print_ops, top_k=[1, 16, 128], name=f"predict_recall_{name}")

        with tf.variable_scope('step_0'):
            print_tensor("loss/loss_0", loss0)
            log_probs_and_recall(logits[0], label[:, 0], step_mask, "0")
        with tf.variable_scope('step_1'):
            print_tensor("loss/loss_1", loss1)
            log_probs_and_recall(logits[1], label[:, 1], step_mask, "1")
        with tf.variable_scope('step_2'):
            print_tensor("loss/loss_2", loss2)
            log_probs_and_recall(logits[2], label[:, 2], step_mask, "2")

        return loss

    def beam_search_fast(self, click_semantic_id_list, colossus_semantic_id_list,
                        beam_size=512, temperature=1):
        """
        双路分开解码：
        - step=0：encoder 仅用 user_static + click(200)，long/short 两路独立取 top-k（建议各 256）。
        - step=1,2：各路按自身 sid0 在 colossus 中筛“同类最近200”，构建该路 per-beam 的 enc-KV，
                    然后继续该路的 beam 维扩展与重排。
        - 最终把两路沿 beam 维拼接。
        """
        offsets = [0,
                self._vocab_sizes[0],
                self._vocab_sizes[0] + self._vocab_sizes[1]]
        num_heads = 8
        decoder_model = DecoderModel(num_layers=4, dim=self._dim, num_heads=num_heads,
                                    dropout_rate=0.1, hidden_dim=self._dim * 2)

        # ---------------- ① 编码用户：user_static + click(1000) ----------------
        # user static
        user_static_fea = tf.concat([self._feature_emb_dict[fea] for fea in user_static_fea_names], axis=1)
        user_static_emb = mlp('user_static_emb', user_static_fea, [2 * self._dim], self._dim, activation=tf.nn.leaky_relu)
        B = tf.shape(user_static_emb)[0]
        user_static_emb = tf.reshape(user_static_emb, [B, 1, self._dim])  # [B,1,C]

        # click（最旧->最新）
        user_click_fea = tf.concat([self._feature_emb_dict[fea] for fea in user_colossus_fea_names], axis=2)  # [B,Lc,F]
        click_max_len = 1000
        click_raw_len = tf.cast(self._feature_emb_size_dict['user_colossus_pid_list'], tf.int32)
        click_valid_len = tf.reshape(click_raw_len, [-1])
        click_used_len = tf.minimum(click_valid_len, tf.constant(click_max_len, dtype=tf.int32))
        # user_click_fea = tf.reverse(user_click_fea, axis=[1])                      # [B,Lc,F]
        # user_click_sid = tf.reverse(click_semantic_id_list, axis=[1])              # [B,Lc] int64
        user_click_sid = colossus_semantic_id_list
        user_click_emb = mlp('user_click_emb', user_click_fea, [4 * self._dim], self._dim, activation=tf.nn.leaky_relu)

        # colossus（最旧->最新）
        # user_colossus_fea = tf.concat([self._feature_emb_dict[fea] for fea in user_colossus_fea_names], axis=2)  # [B,L_col,Fea]
        # user_colossus_emb = mlp('user_click_emb', user_colossus_fea, [4*self._dim], self._dim, activation=tf.nn.leaky_relu)  # [B,M,C]
        user_colossus_emb = user_click_emb

        enc_step0 = tf.concat([user_static_emb, user_click_emb], axis=1)           # [B, 1+Lc, C]
        enc_out_step0 = enc_step0

        # ---------------- ② step0 的两路 mask（long/short） ----------------
        L_click = tf.shape(user_click_sid)[1]
        user_tok_mask = tf.ones([B, 1], dtype=tf.int8)
        click_mask_left = tf.sequence_mask(click_used_len, maxlen=click_max_len, dtype=tf.int8)  # [B,Lc]
        # click_mask = tf.reverse(click_mask_left, axis=[1])                                       # [B,Lc]

        # step0: long / short
        K_short = 64
        pos = tf.expand_dims(tf.range(L_click), 0)                           # [1,Lc]
        tailK = tf.cast(pos >= (L_click - K_short), tf.int8)                 # [1,Lc]
        click_mask_short = tf.bitwise.bitwise_and(click_mask_left, tailK)         # [B,Lc]

        mask_long  = tf.concat([user_tok_mask, click_mask_left],       axis=1)    # [B,1+Lc]
        mask_short = tf.concat([user_tok_mask, click_mask_short], axis=1)    # [B,1+Lc]

        # ---------------- ③ colossus 解析（后续两路都会用到） ----------------
        user_colossus_sid = colossus_semantic_id_list  # [B,Lcol] int64（最旧->最新）
        L_col = tf.shape(user_colossus_sid)[1]
        colossus_valid_mask = tf.cast(tf.not_equal(user_colossus_sid, 0), tf.int8)                                # [B,Lcol]
        lvl1_colossus = tf.bitwise.bitwise_and(
            tf.bitwise.right_shift(tf.cast(user_colossus_sid, tf.int64), tf.constant(30, dtype=tf.int64)),
            tf.constant(0x7FFF, dtype=tf.int64)
        )  # [B,Lcol] int64
        lvl1_colossus = tf.cast(lvl1_colossus, tf.int32)

        # ---------------- ④ step=0：两路分开取 top-k，得到各自初始 beam ----------------
        # <START> 作为当前解码输入
        start_tok = tf.fill([B, 1], self._total_vocab_size)
        # 统一的 dec_in
        dec_in0 = tf.nn.embedding_lookup(self._vocab_embedding, start_tok)  # [B,1,C]
        dec_in0 = tf.reshape(dec_in0, [B, 1, self._dim])

        # helper：单步前向 + 取 top-k
        def step0_pick(mask_1S, k_pick, cache={}):
            src_mask = tf.reshape(mask_1S, [B, 1, 1, tf.shape(mask_1S)[-1]])   # [B,1,1,S]
            dec_out, cache_once = decoder_model.step(dec_in0, 1, enc_out_step0, src_mask, cache=cache)
            h = tf.reshape(dec_out, [B, 1, self._dim])
            with tf.variable_scope('proj_0', reuse=tf.AUTO_REUSE):
                logits = tf.layers.dense(h, self._vocab_sizes[0], name='pred')
            logp = tf.nn.log_softmax(logits / temperature)                     # [B,1,V0]
            topk_logp, topk_tok = tf.nn.top_k(logp, k=tf.maximum(1, k_pick))   # [B,1,k]
            topk_prob = tf.exp(topk_logp)
            return (topk_tok[:,0,:], topk_prob[:,0,:], topk_logp[:,0,:], cache_once)   # [B,k], [B,k], [B,k], cache

        # 均分 beam
        k_long  = beam_size // 2
        k_short = beam_size - k_long
        
        # long step0
        tok_long,  prob_long,  score_long,  cache_long_once  = step0_pick(mask_long,  k_long)
        # 抽取 encoder KV
        cache_enc = {k: v for k, v in cache_long_once.items() if k.startswith("k_enc_") or k.startswith("v_enc_")}
        # short step0（复用 encoder KV，避免重复算）
        tok_short, prob_short, score_short, cache_short_once = step0_pick(mask_short, k_short, cache_enc)

        # 初始序列/概率/分数
        seqs_long  = tf.concat([tf.tile(tf.expand_dims(start_tok, 1), [1, k_long, 1]),
                                tf.expand_dims(tok_long + offsets[0], -1)], axis=-1)          # [B,k_long,2]
        probs_long = tf.concat([tf.ones([B, k_long, 1], tf.float32),
                                tf.expand_dims(prob_long, -1)], axis=-1)                      # [B,k_long,2]
        scores_long = score_long                                                              # [B,k_long]

        seqs_short  = tf.concat([tf.tile(tf.expand_dims(start_tok, 1), [1, k_short, 1]),
                                tf.expand_dims(tok_short + offsets[0], -1)], axis=-1)        # [B,k_short,2]
        probs_short = tf.concat([tf.ones([B, k_short, 1], tf.float32),
                                tf.expand_dims(prob_short, -1)], axis=-1)                    # [B,k_short,2]
        scores_short = score_short                                                            # [B,k_short]

        # step0 的 self-KV 复制到 beam 维（两路各自持有）
        cache_long  = {}
        cache_short = {}
        for kname, val in cache_long_once.items():
            if kname.startswith(("k_self_", "v_self_")):
                cache_long[kname] = tf.tile(val, [1, k_long, 1, 1, 1])    # [B,k_long,H,T,Dh]
        for kname, val in cache_short_once.items():
            if kname.startswith(("k_self_", "v_self_")):
                cache_short[kname] = tf.tile(val, [1, k_short, 1, 1, 1])  # [B,k_short,H,T,Dh]

        # ---------------- ⑤ step=1：为两路各自构建 per-beam encoder（colossus 同类最近200）并缓存 KV ----------------
        K_sel = 50

        def build_branch_enc(seqs_branch, k_branch):
            # 该路 sid0（局部 id）
            sid0_loc = seqs_branch[:, :, 1] - offsets[0]                                        # [B,beam]
            # 同类 & 有效
            same_lvl1_b = tf.equal(lvl1_colossus[:, None, :], sid0_loc[:, :, None])             # [B,beam,Lcol]
            same_lvl1_b = tf.cast(same_lvl1_b, tf.int8)
            col_valid_b = tf.tile(colossus_valid_mask[:, None, :], [1, k_branch, 1])            # [B,beam,Lcol]
            same_and_valid = tf.bitwise.bitwise_and(same_lvl1_b, col_valid_b)                   # [B,beam,Lcol]

            # 选“最近的 200”
            idx = tf.cast(tf.range(L_col)[None, None, :], tf.float32)                           # [1,1,Lcol]
            scores_sel = tf.cast(same_and_valid, tf.float32) * idx + (1. - tf.cast(same_and_valid, tf.float32)) * (-1e9)
            _, topk_idx = tf.nn.top_k(scores_sel, k=K_sel)                                      # [B,beam,K]

            Bb = B * k_branch
            topk_idx_flat = tf.reshape(topk_idx, [Bb, K_sel])                                   # [Bb,K]

            # 扩展 colossus emb 到 [Bb, L_col, C]
            emb_col = user_colossus_emb                                                   # [B,L_col,C]
            emb_col = tf.expand_dims(emb_col, 1)                                          # [B,1,L_col,C]
            emb_col = tf.tile(emb_col, [1, k_branch, 1, 1])                               # [B,beam,L_col,C]
            emb_col = tf.reshape(emb_col, [Bb, L_col, tf.shape(user_colossus_emb)[2]])    # [Bb,L_col,C]
            sel_emb = tf.batch_gather(emb_col, topk_idx_flat)  

            # presence mask
            same_and_valid_f = tf.cast(same_and_valid, tf.float32)
            same_and_valid_f = tf.reshape(same_and_valid_f, [Bb, L_col])
            sel_present = tf.batch_gather(same_and_valid_f, topk_idx_flat)                      # [Bb,K]
            sel_present = tf.cast(sel_present > 0.5, tf.int8)                                   # [Bb,K]

            # 拼接 user_static 到每个 beam
            user_stat_beam = tf.tile(user_static_emb, [1, k_branch, 1])                         # [B,beam,C]
            user_stat_beam = tf.reshape(user_stat_beam, [Bb, 1, self._dim])                     # [Bb,1,C]
            enc_beam = tf.concat([user_stat_beam, sel_emb], axis=1)                             # [Bb,1+K,C]

            # per-beam src mask（user=1 + sel_present）
            sel_mask = tf.reshape(sel_present, [B, k_branch, K_sel])                            # [B,beam,K]
            src_mask_5d = tf.reshape(tf.concat([tf.ones([B, k_branch, 1], tf.int8), sel_mask], axis=2),
                                    [B, k_branch, 1, 1, 1 + K_sel])                            # [B,beam,1,1,S]

            # 预计算每层的 enc-KV，并还原到 [B,beam,H,L,Dh]
            def split_heads(x, h):
                depth = x.get_shape().as_list()[-1]
                Dh = depth // h
                reshaped = tf.reshape(x, [Bb, -1, h, Dh])       # [Bb,L,H,Dh]
                return tf.transpose(reshaped, [0, 2, 1, 3])     # [Bb,H,L,Dh]

            branch_cache = {}
            for i in range(decoder_model.num_layers):
                with tf.variable_scope(f"decoder_layer_{i}/multi_head_attention", reuse=tf.AUTO_REUSE):
                    k_enc_lin = tf.layers.dense(enc_beam, self._dim, use_bias=False, name="w_k")    # [Bb,L,D]
                    v_enc_lin = tf.layers.dense(enc_beam, self._dim, use_bias=False, name="w_v")    # [Bb,L,D]
                k_enc_h = split_heads(k_enc_lin, num_heads)  # [Bb,H,L,Dh]
                v_enc_h = split_heads(v_enc_lin, num_heads)  # [Bb,H,L,Dh]
                H = num_heads
                Dh = self._dim // num_heads
                L_enc = tf.shape(k_enc_h)[2]
                k_enc_h = tf.reshape(k_enc_h, [B, k_branch, H, L_enc, Dh])
                v_enc_h = tf.reshape(v_enc_h, [B, k_branch, H, L_enc, Dh])
                branch_cache[f"k_enc_{i}"] = k_enc_h
                branch_cache[f"v_enc_{i}"] = v_enc_h

            return branch_cache, src_mask_5d

        cache_enc_long,  src_mask_5d_long  = build_branch_enc(seqs_long,  k_long)
        cache_enc_short, src_mask_5d_short = build_branch_enc(seqs_short, k_short)

        # 把 enc-KV 合并进两路各自缓存
        for kname, val in cache_enc_long.items():
            cache_long[kname] = val
        for kname, val in cache_enc_short.items():
            cache_short[kname] = val

        # ---------------- ⑥ step=1,2：两路各自继续解码 ----------------
        def branch_decode_two_steps(seqs_b, probs_b, scores_b, cache_b, src_mask_5d_b, k_branch, branch_name):
            # 公共工具：重排 cache 与 mask
            def gather_cache_beam(old_cache, gp):
                new_cache = {}
                for kname, val in old_cache.items():
                    if kname.startswith(("k_self_", "v_self_", "k_enc_", "v_enc_")):
                        new_cache[kname] = tf.gather_nd(val, gp)   # [B,beam,H,*,Dh]
                    else:
                        new_cache[kname] = val
                return new_cache

            # 固定 per-beam src mask（两步共用）
            S = tf.shape(src_mask_5d_b)[-1]
            src_mask_step = tf.reshape(src_mask_5d_b, [B * k_branch, 1, 1, S])   # [B*beam,1,1,S]
            dummy_enc = tf.zeros([B, 1, self._dim])  # enc_out 参数不再使用

            # 逐 level（step=1,2）
            for lvl, V in enumerate(self._vocab_sizes[1:], start=1):
                # dec input：上一步 token
                dec_in = tf.nn.embedding_lookup(self._vocab_embedding, seqs_b[:, :, -1])  # [B,beam,1,C]
                dec_in = tf.reshape(dec_in, [B * k_branch, 1, self._dim])

                dec_out, cache_b = decoder_model.step(dec_in, k_branch, dummy_enc, src_mask_step, cache_b)
                last_h = tf.reshape(dec_out, [B, k_branch, self._dim])

                with tf.variable_scope('proj_%d' % lvl):
                    logits = tf.layers.dense(last_h, V, name='pred', reuse=tf.AUTO_REUSE)  # [B,beam,V]
                logp = tf.nn.log_softmax(logits / temperature)
                topk_logp, topk_tok = tf.nn.top_k(logp, k=k_branch)                       # [B,beam,beam]
                topk_prob = tf.exp(topk_logp)

                # 全局选 beam
                cand_scores = tf.expand_dims(scores_b, -1) + topk_logp                    # [B,beam,beam]
                flat_scores = tf.reshape(cand_scores, [B, -1])                            # [B, beam*beam]
                best_scores, best_idx = tf.nn.top_k(flat_scores, k=k_branch)              # [B,beam]

                parent_beam = best_idx // k_branch
                tok_rank    = best_idx %  k_branch
                batch_idx   = tf.tile(tf.expand_dims(tf.range(B), 1), [1, k_branch])

                gather_parent = tf.stack([batch_idx, parent_beam], axis=2)
                parent_seq   = tf.gather_nd(seqs_b,  gather_parent)                       # [B,beam,T]
                parent_prob  = tf.gather_nd(probs_b, gather_parent)
                # 重排 cache 和 src mask
                cache_b = gather_cache_beam(cache_b, gather_parent)
                src_mask_5d_b = tf.gather_nd(src_mask_5d_b, gather_parent)                    # [B,beam,1,1,S]
                src_mask_step = tf.reshape(src_mask_5d_b, [B * k_branch, 1, 1, S])

                tok_gather = tf.stack([batch_idx, parent_beam, tok_rank], axis=2)
                next_tok   = tf.gather_nd(topk_tok,  tok_gather)                           # [B,beam]
                next_prob  = tf.gather_nd(topk_prob, tok_gather)                           # [B,beam]

                next_tok_glb = next_tok + offsets[lvl]
                seqs_b  = tf.concat([parent_seq, tf.expand_dims(next_tok_glb, -1)], axis=-1)  # [B,beam,T+1]
                probs_b = tf.concat([parent_prob, tf.expand_dims(next_prob,  -1)], axis=-1)
                scores_b = best_scores

            return seqs_b, probs_b, scores_b, cache_b

        seqs_long, probs_long, scores_long, cache_long   = branch_decode_two_steps(seqs_long,  probs_long,  scores_long,  cache_long,  src_mask_5d_long,  k_long,  "long")
        seqs_short, probs_short, scores_short, cache_short = branch_decode_two_steps(seqs_short, probs_short, scores_short, cache_short, src_mask_5d_short, k_short, "short")

        # ---------------- ⑦ 拼接两路输出并还原为局部 id ----------------
        seqs  = tf.concat([seqs_long,  seqs_short],  axis=1)   # [B, beam_size, 3]
        probs = tf.concat([probs_long, probs_short], axis=1)   # [B, beam_size, 3]

        # 去掉 <START>，转回局部 id
        seqs  = seqs[:, :, 1:]                                              # [B,beam,3]
        probs = probs[:, :, 1:]
        gen_part_loc = seqs - tf.constant(offsets, dtype=seqs.dtype)        # 广播减 -> [B,beam,3]

        return gen_part_loc, probs
