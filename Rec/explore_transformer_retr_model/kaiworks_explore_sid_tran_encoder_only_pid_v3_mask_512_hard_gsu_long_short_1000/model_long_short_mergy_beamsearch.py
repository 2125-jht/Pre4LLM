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
user_click_fea_names = [
    "user_profile_v1_click_pid_list",  # 用户点击的视频ID列表
    "user_profile_v1_click_aid_list"   # 用户点击的作者ID列表
]

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
        主训练模型前向传播
        
        Args:
            click_semantic_id_list:用户点击的视频SID int列表 [batch_size, 200] int64 最新到最旧，最后padding0
            colossus_semantic_id_list:用户点击的视频SID int列表 [batch_size, 1000] int64 最旧到最新，最后padding0
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
        # 拼接原始点击特征
        user_click_fea = tf.concat(
            [self._feature_emb_dict[fea] for fea in user_click_fea_names], axis=2)   # 最新->最旧 [B, L, dim]

        user_colossus_fea = tf.concat(
            [self._feature_emb_dict[fea] for fea in user_colossus_fea_names], axis=2)  # [B, M, dim]

        # 打印有效长度                                       
        click_raw_len  = tf.cast(self._feature_emb_size_dict['user_profile_v1_click_pid_list'], tf.int32) # 可能是 [B,1] 也可能是 [B]
        click_valid_len = tf.reshape(click_raw_len, [-1])      # 强制展平成 [B]
        click_max_len = 200
        click_used_len = click_valid_len

        colossus_raw_len  = tf.cast(self._feature_emb_size_dict['user_colossus_pid_list'], tf.int32) # 可能是 [B,1] 也可能是 [B]
        colossus_valid_len = tf.reshape(colossus_raw_len, [-1])      # 强制展平成 [B]

        with tf.variable_scope('valid_len'):
            print_tensor("colossus_valid_len", colossus_valid_len)
            print_tensor("click_valid_len", click_valid_len)

        # 调整序列
        user_click_sid = tf.reverse(click_semantic_id_list, axis=[1]) # reverse 使得 最旧→最新
        user_click_fea = tf.reverse(user_click_fea, axis=[1]) # reverse 使得 最旧→最新
        user_colossus_sid = colossus_semantic_id_list # 已经是 最旧→最新

        # 通过MLP处理点击特征
        user_click_emb = mlp('user_click_emb', user_click_fea, [4*self._dim], self._dim, activation=tf.nn.leaky_relu)
        user_colossus_emb = mlp('user_click_emb', user_colossus_fea, [4*self._dim], self._dim, activation=tf.nn.leaky_relu)  # [B,L_colossus,C]

        # debug
        self._print_ops.append(tf.print("user_click_sid first sample:", user_click_sid[0], summarize=100))
        self._print_ops.append(tf.print("user_click_emb first sample:", user_click_emb[0,:,1], summarize=100))
        
        # === 3. 构建编码器输入 ===
        # encoder_input 把 colossus 加上去（顺序： user token | click 序列 | colossus 序列）
        encoder_input = tf.concat([user_static_emb, user_click_emb, user_colossus_emb], axis=1)  # [B, 1+L_click+L_colossus, C]
        
        # 计算编码器输入的余弦相似度（用于调试）
        encoder_input_sim = tf.reshape(encoder_input, [batch_size, -1])
        print_tensor("encoder_input_sim", calc_sim_cos(encoder_input_sim))
            
        # === 4. 构建解码器输入 ===
        # 添加起始token（使用总词汇表大小作为特殊标记）
        start_token_indice = tf.tile(tf.constant(self._total_vocab_size, shape=(1, 1), dtype=tf.int32), [batch_size, 1])
        # 将起始token与视频语义ID拼接，只取最后一个之前的
        photo_with_start_token = tf.concat([start_token_indice, photo_sid[:,:-1]], axis=1)
        # 查找嵌入向量
        decoder_input = tf.nn.embedding_lookup(self._vocab_embedding, photo_with_start_token)
        
        # === 5. 构建 Encoder/Decoder 的 padding mask =============================
        # 构建各类 mask
        B = tf.shape(click_used_len)[0]             # batch_size 动态
        L_click = tf.shape(user_click_sid)[1]       # 200
        L_colossus = tf.shape(user_colossus_sid)[1] # 1000
        
        # 第1级的mask        
        # click 有效位
        click_mask_left = tf.sequence_mask(lengths=click_used_len, maxlen=click_max_len, dtype=tf.int8)  # [B, L_click]
        click_mask      = tf.reverse(click_mask_left, axis=[1])  # [B, L_click] （最旧→最新 对齐）
        user_tok = tf.ones([B, 1], dtype=tf.int8)                # [B,1]

        # colossus 有效位：colossus padding 用 0 表示，所以直接 not_equal
        colossus_valid_mask = tf.cast(tf.not_equal(user_colossus_sid, tf.zeros_like(user_colossus_sid)), tf.int8)  # [B, L_colossus]
        
        # long/short for step=0 （step0 只看 click 序列）
        seq_mask_all_long  = tf.concat([user_tok, click_mask, tf.zeros([B, L_colossus], dtype=tf.int8)], axis=1)  # [B, 1+L_click+L_colossus]
        # short: 末尾 K of click
        K = 64
        pos = tf.range(L_click, dtype=tf.int32)[None, :]  # [1, L_click]
        tail64 = tf.cast(pos >= (L_click - K), tf.int8)   # [1, L_click]
        click_mask_short = tf.bitwise.bitwise_and(click_mask, tail64)  # [B, L_click]
        seq_mask_all_short = tf.concat([user_tok, click_mask_short, tf.zeros([B, L_colossus], dtype=tf.int8)], axis=1)  # [B,1+L_click+L_colossus]

        # 计算 lvl1 的表示（click & colossus），用于选择逻辑
        # 注意用 int64 位运算，再 cast 回 int32
        lvl1_click = tf.bitwise.bitwise_and(
            tf.bitwise.right_shift(tf.cast(user_click_sid, tf.int64), tf.constant(30, dtype=tf.int64)),
            tf.constant(0x7FFF, dtype=tf.int64)
        )  # [B, L_click] int64
        lvl1_click = tf.cast(lvl1_click, tf.int32)

        lvl1_colossus = tf.bitwise.bitwise_and(
            tf.bitwise.right_shift(tf.cast(user_colossus_sid, tf.int64), tf.constant(30, dtype=tf.int64)),
            tf.constant(0x7FFF, dtype=tf.int64)
        )  # [B, L_colossus] int64
        lvl1_colossus = tf.cast(lvl1_colossus, tf.int32)
        
        # 计算“长/短池”的嵌入平均，并与 label 第1级做余弦相似度
        # 用统一的 vocab embedding（全局 id=局部id+offset[0]）
        lvl1_click_glb = lvl1_click  # offset=0
        click_sid_emb  = tf.nn.embedding_lookup(self._vocab_embedding, lvl1_click_glb)      # [B,L_click,C]
        
        def masked_avg(x, m):  # x:[B,L,C], m:[B,L] int8(0/1)
            m_f = tf.cast(m, tf.float32)
            denom = tf.reduce_sum(m_f, axis=1, keepdims=True) + 1e-9
            w = m_f[:, :, None]
            return tf.reduce_sum(x * w, axis=1) / denom                                      # [B,C]
        
        valid_sid_mask = tf.cast(tf.not_equal(user_click_sid, 0), tf.int8)                   # [B,L_click]
        pool_long_sid  = masked_avg(click_sid_emb, tf.bitwise.bitwise_and(click_mask, valid_sid_mask))
        pool_short_sid = masked_avg(click_sid_emb, tf.bitwise.bitwise_and(click_mask_short, valid_sid_mask))

        # label 第1级的嵌入
        label_lvl1_glb = tf.cast(label[:, 0], tf.int32)                                      # [B]
        label_emb = tf.nn.embedding_lookup(self._vocab_embedding, label_lvl1_glb)            # [B,C]

        def cos(a, b):
            a_n = tf.nn.l2_normalize(a, axis=-1); b_n = tf.nn.l2_normalize(b, axis=-1)
            return tf.reduce_sum(a_n * b_n, axis=-1)                                         # [B]

        bias = 0.0  # 调整偏向于short的程度 例如设置为0.02
        sim_long  = cos(label_emb, pool_long_sid)                                            # [B]
        sim_short = cos(label_emb, pool_short_sid) + bias                                    # [B]

        # 基于相似度选择 step=0 的跨注意力 mask（逐样本 hard 选择）
        choose_short = sim_short > sim_long                                                  # [B] bool
        # expand 到 [B,1,S]
        pos0_long  = tf.expand_dims(seq_mask_all_long,  axis=1)                              # [B,1,S]
        pos0_short = tf.expand_dims(seq_mask_all_short, axis=1)                              # [B,1,S]
        cond = tf.cast(choose_short[:, None, None], tf.bool)                                 # [B,1,1]
        cond = tf.broadcast_to(cond, tf.shape(pos0_long))                                    # [B,1,S]    
        pos0 = tf.where(cond, pos0_short, pos0_long)                                         # [B,1,S]
        
        # debug
        with tf.variable_scope('choose_count'):
            print_tensor("total", B)
            print_tensor("long", tf.reduce_sum(1-tf.cast(choose_short, tf.int32)))
            print_tensor("short", tf.reduce_sum(tf.cast(choose_short, tf.int32)))
        
        # 构造第2/3级的 mask：**只允许 attend 到 colossus 中 lvl1 == label_lvl1 的位置**
        # 哪些 colossus lvl1 与 label 第1级相同
        target_lvl1 = tf.cast(label[:, 0], tf.int32)     # [B]
        same_lvl1_in_colossus = tf.equal(lvl1_colossus, target_lvl1[:, None])  # [B, L_colossus] bool
        same_lvl1_in_colossus = tf.cast(same_lvl1_in_colossus, tf.int8)       # [B, L_colossus]

        # 组合：colossus position 必须有效且同类
        select_colossus_mask = tf.bitwise.bitwise_and(colossus_valid_mask, same_lvl1_in_colossus)  # [B, L_colossus]

        # 最终 seq_mask_sel: user token + zeros(点击位) + select_colossus_mask
        zeros_click = tf.zeros_like(click_mask, dtype=tf.int8)  # [B, L_click]
        seq_mask_sel = tf.concat([user_tok, zeros_click, select_colossus_mask], axis=1)  # [B, 1+L_click+L_colossus]

        # 组装按解码步位的跨注意力 mask: 若 decoder_input 长度为 Tq
        Tq = tf.shape(decoder_input)[1]   # [B, Tq] == e.g. 1+2
        rest_len = tf.maximum(Tq - 1, 0)
        
        # pos0 已由 choose_short 决定（pos0: [B,1,1+L_click+L_colossus]），rest 用 seq_mask_sel（第2/3级）
        rest = tf.tile(tf.expand_dims(seq_mask_sel, axis=1), [1, rest_len, 1])  # [B, Tq-1, 1+L_click+L_colossus]

        seq_mask_per_t = tf.concat([pos0, rest], axis=1)   # [B, Tq, 1+L_click+L_colossus]
        src_mask = tf.expand_dims(seq_mask_per_t, axis=1)  # [B,1,Tq,1+L_click+L_colossus]

        # debug
        with tf.variable_scope('sid_count'):
            print_tensor("valid_click_sid",
                        tf.reduce_sum(tf.cast(valid_sid_mask, tf.int32), axis=1))
            print_tensor("valid_colossus_sid",
                        tf.reduce_sum(tf.cast(colossus_valid_mask, tf.int32), axis=1))
            print_tensor("select_colossus",
                        tf.reduce_sum(tf.cast(select_colossus_mask, tf.int32), axis=1))

        # debug
        self._print_ops.append(tf.print("lvl1_click[0]:", lvl1_click[0], summarize=100))
        self._print_ops.append(tf.print("lvl1_colossus[0]:", lvl1_colossus[0], summarize=100))
        self._print_ops.append(tf.print("label[0]:", label[0], summarize=100))
        self._print_ops.append(tf.print("mask/valid_sid_mask[0]:", valid_sid_mask[0], summarize=100))
        self._print_ops.append(tf.print("mask/colossus_valid_mask[0]:", colossus_valid_mask[0], summarize=100))
        self._print_ops.append(tf.print("mask/pos0[0]:", pos0[0], summarize=100))
        self._print_ops.append(tf.print("mask/seq_mask_sel[0]:", seq_mask_sel[0], summarize=100))
        self._print_ops.append(tf.print("mask/num_selected_clicks:", 
                                        tf.reduce_sum(tf.cast(select_colossus_mask, tf.int32), axis=1)))
        
        # === 5. no enc ===
        # encoder_output = layer_norm(encoder_input, scope="enc_ln")
        encoder_output = encoder_input
        # 计算余弦相似度（用于调试）
        encoder_output_sim = tf.reshape(encoder_output, [batch_size, -1])
        print_tensor("encoder_output_sim", calc_sim_cos(encoder_output_sim))
        
        # === 6. decoder ===
        decoder_model = DecoderModel(num_layers=4, dim=self._dim, num_heads=8, dropout_rate=0.1, hidden_dim=self._dim*2, training=True)
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
        
        # 对每个语义层级分别计算损失
        for step in range(len(self._vocab_sizes)):
            with tf.variable_scope('proj_%d' % step):
                # 使用MLP将解码器输出映射到对应词汇表大小的logits
                pred_logit = tf.layers.dense(decoder_output[:, step, :], self._vocab_sizes[step], name='pred') # [batch_size, vocab_size]
                print_tensor("logits/pred_logit_%d" % step, pred_logit)
                
                temperature = 1

                # 1. 求 softmax 概率
                pred_prob = tf.nn.softmax(pred_logit/temperature, axis=-1)  # [B, V]

                # 2. 取出正确 label 的概率
                # 先拼出索引对 (batch_idx, label_id)
                batch_idx = tf.range(tf.shape(pred_prob)[0], dtype=tf.int32)
                indices   = tf.stack([batch_idx, label[:, step]], axis=1)  # [B, 2]
                true_p = tf.gather_nd(pred_prob, indices)               # [B]

                # 3. 打印
                print_tensor("probs/true_token_prob_%d" % step, tf.reduce_sum(true_p * loss_mask) / (tf.reduce_sum(loss_mask) + 1e-9))
                max_probs, _ = tf.nn.top_k(pred_prob, k=1)
                print_tensor("probs/max_token_prob_%d" % step, tf.reduce_sum(tf.squeeze(max_probs, -1) * loss_mask) / (tf.reduce_sum(loss_mask) + 1e-9))

                # 转换标签为one-hot编码
                one_hot_labels = tf.one_hot(label[:, step], self._vocab_sizes[step])
                # 计算交叉熵损失，可选温度
                loss_i = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels, logits=pred_logit/temperature)
                losses.append(loss_i)
            
                # 打印每个层次的损失
                print_tensor("loss/loss_%d" % step, tf.reduce_sum(loss_i * loss_mask) / (tf.reduce_sum(loss_mask) + 1e-9))
                # 计算各种recall指标
                recall_at_k(pred_logit, label[:, step], loss_mask, self._print_ops, top_k=[1, 16, 128], name="predict_recall_%d" % step)
                
        print_tensor("loss_mask", loss_mask)
        # 计算加权平均损失
        loss = tf.reduce_sum((losses[0] + losses[1] + losses[2]) * loss_mask) / (tf.reduce_sum(loss_mask) + 1e-9)
        return loss

    def beam_search_fast(self, click_semantic_id_list, colossus_semantic_id_list,
                        beam_size=512, temperature=1):
        """
        step=0：encoder 仅用 user_static + click(200)
        step=1：按每条 beam 的 sid0 在 colossus 中筛最近200（不足补0），
                ，encoder 输入= user_static + colossus_sel，
                并把 encoder KV 以 [B, beam, H, L_enc, Dh] 形式缓存。
        """
        offsets = [0,
                self._vocab_sizes[0],
                self._vocab_sizes[0] + self._vocab_sizes[1]]

        num_heads = 8
        decoder_model = DecoderModel(num_layers=4, dim=self._dim, num_heads=num_heads,
                                    dropout_rate=0.1, hidden_dim=self._dim * 2)

        # ---------- ① 编码用户：user_static + click(200) ----------
        # 1) 静态特征
        user_static_fea = tf.concat([self._feature_emb_dict[fea] for fea in user_static_fea_names], axis=1)
        user_static_emb = mlp('user_static_emb', user_static_fea, [2 * self._dim], self._dim, activation=tf.nn.leaky_relu)
        B = tf.shape(user_static_emb)[0]
        user_static_emb = tf.reshape(user_static_emb, [B, 1, self._dim])  # [B,1,C]

        # 2) click（最新->最旧）反转为（最旧->最新）
        user_click_fea = tf.concat([self._feature_emb_dict[fea] for fea in user_click_fea_names], axis=2)  # [B,Lc,Cf]
        click_max_len = 200
        click_raw_len = tf.cast(self._feature_emb_size_dict['user_profile_v1_click_pid_list'], tf.int32)
        click_valid_len = tf.reshape(click_raw_len, [-1])
        click_used_len = tf.minimum(click_valid_len, tf.constant(click_max_len, dtype=tf.int32))

        user_click_fea = tf.reverse(user_click_fea, axis=[1])                   # [B,Lc,Cf]
        user_click_sid = tf.reverse(click_semantic_id_list, axis=[1])           # [B,Lc] int64

        user_click_emb = mlp('user_click_emb', user_click_fea, [4 * self._dim], self._dim, activation=tf.nn.leaky_relu)
        
        # 3）colossus（最旧->最新）
        user_colossus_fea = tf.concat([self._feature_emb_dict[fea] for fea in user_colossus_fea_names], axis=2)  # [B,L_col,Fea]
        user_colossus_emb = mlp('user_click_emb', user_colossus_fea, [4*self._dim], self._dim, activation=tf.nn.leaky_relu)  # [B,M,C]

        # encoder_input_step0：user | click
        enc_step0 = tf.concat([user_static_emb, user_click_emb], axis=1)        # [B, 1+Lc, C]
        enc_out_base = enc_step0                                                # 不做LN，保持一致

        # ---------- ② 基础 mask（step0） ----------
        L_click = tf.shape(user_click_sid)[1]
        user_tok_mask = tf.ones([B, 1], dtype=tf.int8)
        click_mask_left = tf.sequence_mask(click_used_len, maxlen=click_max_len, dtype=tf.int8)  # [B,Lc]
        click_mask = tf.reverse(click_mask_left, axis=[1])                                       # [B,Lc]

        # step0: long / short
        K_short = 64
        pos = tf.expand_dims(tf.range(L_click), 0)                      # [1, Lc]
        tailK = tf.cast(pos >= (L_click - K_short), tf.int8)            # [1, Lc]
        click_mask_short = tf.bitwise.bitwise_and(click_mask, tailK)    # [B, Lc]

        seq_mask_all_long  = tf.concat([user_tok_mask, click_mask], axis=1)         # [B, 1+Lc]
        seq_mask_all_short = tf.concat([user_tok_mask, click_mask_short], axis=1)   # [B, 1+Lc]

        # lvl1 抽取（给 step>=1 用）
        lvl1_colossus = tf.bitwise.bitwise_and(
            tf.bitwise.right_shift(tf.cast(colossus_semantic_id_list, tf.int64), tf.constant(30, dtype=tf.int64)),
            tf.constant(0x7FFF, dtype=tf.int64)
        )  # [B, L_col] int64
        lvl1_colossus = tf.cast(lvl1_colossus, tf.int32)
        user_colossus_sid = colossus_semantic_id_list                                                             # [B,L_col] int64
        L_col = tf.shape(user_colossus_sid)[1]
        colossus_valid_mask = tf.cast(tf.not_equal(user_colossus_sid, 0), tf.int8)                                # [B,L_col]

        # ---------- ③ Beam 状态 ----------
        start_tok = tf.fill([B, 1], self._total_vocab_size)   # <START>
        seqs   = tf.expand_dims(start_tok, 1)                 # [B,1,1]
        probs  = tf.ones_like(seqs, dtype=tf.float32)         # [B,1,1]
        scores = tf.zeros([B, 1], dtype=tf.float32)           # [B,1]
        cur_beam = 1
        cache = {}

        # ---------- ④ 逐级解码 ----------
        for step, V in enumerate(self._vocab_sizes):

            if step == 0:
                # ===== 双路并行：long/short 各取 top-k，合并为 beam =====
                k_long  = tf.maximum(1, beam_size // 2)
                k_short = beam_size - k_long

                # dec_in = <START>
                dec_in = tf.nn.embedding_lookup(self._vocab_embedding, seqs[:, :, -1])  # [B,1,1,C]
                dec_in = tf.reshape(dec_in, [B, 1, self._dim])

                # (1) long
                src_mask_long = tf.reshape(seq_mask_all_long,  [B, 1, 1, tf.shape(seq_mask_all_long)[-1]])
                dec_out_long, cache_long = decoder_model.step(dec_in, 1, enc_out_base, src_mask_long, cache={})
                last_h_long = tf.reshape(dec_out_long, [B, 1, self._dim])
                with tf.variable_scope('proj_0', reuse=tf.AUTO_REUSE):
                    logits_long = tf.layers.dense(last_h_long, V, name='pred')
                logp_long = tf.nn.log_softmax(logits_long / temperature)
                topk_logp_long, topk_tok_long = tf.nn.top_k(logp_long, k=k_long)   # [B,1,k_long]
                topk_prob_long = tf.exp(topk_logp_long)

                # 抽取 encoder KV（共享）
                cache_enc = {k: v for k, v in cache_long.items() if k.startswith("k_enc_") or k.startswith("v_enc_")}

                # (2) short（复用 encoder KV，避免重复算）
                src_mask_short = tf.reshape(seq_mask_all_short, [B, 1, 1, tf.shape(seq_mask_all_short)[-1]])
                dec_out_short, cache_short = decoder_model.step(dec_in, 1, enc_out_base, src_mask_short, cache=cache_enc)
                last_h_short = tf.reshape(dec_out_short, [B, 1, self._dim])
                with tf.variable_scope('proj_0', reuse=True):
                    logits_short = tf.layers.dense(last_h_short, V, name='pred')
                logp_short = tf.nn.log_softmax(logits_short / temperature)
                topk_logp_short, topk_tok_short = tf.nn.top_k(logp_short, k=k_short)  # [B,1,k_short]
                topk_prob_short = tf.exp(topk_logp_short)

                # (3) 合并
                next_tok   = tf.concat([topk_tok_long[:, 0, :],  topk_tok_short[:, 0, :]],  axis=1)  # [B,beam]
                next_prob  = tf.concat([topk_prob_long[:,0,:],   topk_prob_short[:,0,:]],   axis=1)
                scores     = tf.concat([topk_logp_long[:,0,:],   topk_logp_short[:,0,:]],   axis=1)
                next_tok_glb = next_tok + offsets[0]

                parent_seq  = tf.tile(seqs,  [1, beam_size, 1])                   # [B,beam,1]
                parent_prob = tf.tile(probs, [1, beam_size, 1])
                seqs  = tf.concat([parent_seq, tf.expand_dims(next_tok_glb, -1)], axis=-1)   # [B,beam,2]
                probs = tf.concat([parent_prob, tf.expand_dims(next_prob,   -1)], axis=-1)
                cur_beam = beam_size

                # 构建 step0 的 self-KV（long/short）→ 合并到 cache 的 beam 维
                cache = {}
                # self-KV
                for kname, val in cache_long.items():
                    if kname.startswith("k_self_") or kname.startswith("v_self_"):
                        cache[kname] = tf.tile(val, [1, k_long, 1, 1, 1])      # [B,k_long,H,T,Dh]
                for kname, val in cache_short.items():
                    if kname.startswith("k_self_") or kname.startswith("v_self_"):
                        cache[kname] = tf.concat([cache[kname], tf.tile(val, [1, k_short, 1, 1, 1])], axis=1)  # [B,beam,...]
                        
                continue  # 进入 step==1

            # ===== step == 1：先基于每条 beam 的 sid0，构建 per-beam encoder（user + colossus_sel≤200），并缓存 enc KV =====
            if step == 1:
                # 取每条 beam 的第1级局部 id（sid0）
                sid0_loc_beam = seqs[:, :, 1] - offsets[0]                         # [B,beam] int32

                # 同类位置 mask（B,beam,L_col）
                same_lvl1_b = tf.equal(lvl1_colossus[:, None, :], sid0_loc_beam[:, :, None])  # [B,beam,L_col] bool
                same_lvl1_b = tf.cast(same_lvl1_b, tf.int8)
                col_valid_b = tf.tile(colossus_valid_mask[:, None, :], [1, cur_beam, 1])      # [B,beam,L_col]
                same_and_valid = tf.bitwise.bitwise_and(same_lvl1_b, col_valid_b)             # [B,beam,L_col]

                # 选“最近的200”：用索引做分数；mask=0 给 -1e9，top_k 取最大的（即最近）
                K_sel = 200
                idx = tf.cast(tf.range(L_col)[None, None, :], tf.float32)                     # [1,1,L_col]
                scores_sel = tf.cast(same_and_valid, tf.float32) * idx + \
                            (1.0 - tf.cast(same_and_valid, tf.float32)) * (-1e9)
                _, topk_idx = tf.nn.top_k(scores_sel, k=K_sel)                                 # [B,beam,K_sel] 近→远（无序要求可省排序）

                # 准备按 (B*beam) 取子序列
                Bb = B * cur_beam
                topk_idx_flat = tf.reshape(topk_idx, [Bb, K_sel])                              # [Bb,K]
                # 扩展 colossus emb 到 [Bb, L_col, C]
                emb_col = user_colossus_emb                                                   # [B,L_col,C]
                emb_col = tf.expand_dims(emb_col, 1)                                          # [B,1,L_col,C]
                emb_col = tf.tile(emb_col, [1, cur_beam, 1, 1])                               # [B,beam,L_col,C]
                emb_col = tf.reshape(emb_col, [Bb, L_col, tf.shape(user_colossus_emb)[2]])    # [Bb,L_col,C]
                sel_emb = tf.batch_gather(emb_col, topk_idx_flat)                              # [Bb,K,C]

                # 同样取出 presence mask（是否真的同类且有效），作为 src mask 的一部分
                same_and_valid_f = tf.cast(same_and_valid, tf.float32)                         # [B,beam,L_col]
                same_and_valid_f = tf.reshape(same_and_valid_f, [Bb, L_col])
                sel_present = tf.batch_gather(same_and_valid_f, topk_idx_flat)                 # [Bb,K] (0/1)
                sel_present = tf.cast(sel_present > 0.5, tf.int8)                              # [Bb,K] -> int8

                # 拼接 user_static 到每个 beam
                user_stat_beam = tf.tile(user_static_emb, [1, cur_beam, 1])        # [B,beam,C]
                user_stat_beam = tf.reshape(user_stat_beam, [Bb, 1, self._dim])    # [Bb,1,C]
                enc_beam = tf.concat([user_stat_beam, sel_emb], axis=1)            # [Bb, 1+K, C]

                # === 构建 per-beam 的 src mask（user=1 + sel_present） ===
                sel_mask = tf.reshape(sel_present, [B, cur_beam, K_sel])           # [B,beam,K]
                src_mask_beam = tf.concat([tf.ones([B, cur_beam, 1], dtype=tf.int8), sel_mask], axis=2)  # [B,beam,1+K]
                perbeam_src_mask_5d = tf.reshape(src_mask_beam, [B, cur_beam, 1, 1, tf.shape(src_mask_beam)[-1]])  # [B,beam,1,1,S]

                # === 预计算每层的 encoder KV，按 beam 维缓存 ===
                def split_heads(x, h):
                    # x: [Bb, L, D] -> [Bb, H, L, Dh]
                    depth = x.get_shape().as_list()[-1]
                    Dh = depth // h
                    reshaped = tf.reshape(x, [Bb, -1, h, Dh])       # [Bb, L, H, Dh]
                    return tf.transpose(reshaped, [0, 2, 1, 3])     # [Bb, H, L, Dh]

                # 逐层计算（与 DecoderLayer 内的命名一致，确保复用权重）
                for i in range(decoder_model.num_layers):
                    with tf.variable_scope(f"decoder_layer_{i}/multi_head_attention", reuse=tf.AUTO_REUSE):
                        k_enc_lin = tf.layers.dense(enc_beam, self._dim, use_bias=False, name="w_k")  # [Bb,L,D]
                        v_enc_lin = tf.layers.dense(enc_beam, self._dim, use_bias=False, name="w_v")  # [Bb,L,D]
                    k_enc_h = split_heads(k_enc_lin, num_heads)  # [Bb,H,L,Dh]
                    v_enc_h = split_heads(v_enc_lin, num_heads)  # [Bb,H,L,Dh]

                    # 还原到 [B, beam, H, L, Dh]
                    H = num_heads
                    Dh = self._dim // num_heads
                    L_enc = tf.shape(k_enc_h)[2]
                    k_enc_h = tf.reshape(k_enc_h, [B, cur_beam, H, L_enc, Dh])
                    v_enc_h = tf.reshape(v_enc_h, [B, cur_beam, H, L_enc, Dh])

                    cache[f"k_enc_{i}"] = k_enc_h
                    cache[f"v_enc_{i}"] = v_enc_h

            # ====== 用 per-beam enc KV + src_mask 解码 step>=1 ======
            # dec_in: 上一步生成的 token embedding
            dec_in = tf.nn.embedding_lookup(self._vocab_embedding, seqs[:, :, -1])   # [B,beam,1,C]
            dec_in = tf.reshape(dec_in, [B * cur_beam, 1, self._dim])

            # 注意：此时 enc_out 参数将被忽略（因为我们已经把 k_enc/v_enc 放进 cache）
            dummy_enc_out = tf.zeros([B, 1, self._dim])  # 仅占位
            # 将 5D 的 per-beam mask 摊平成本步需要的 4D（与 dec_in 的 [B*beam,1,...] 对齐）
            S = tf.shape(perbeam_src_mask_5d)[-1]
            src_mask_step = tf.reshape(perbeam_src_mask_5d, [B * cur_beam, 1, 1, S])  # [B*beam,1,1,S]
            dec_out, cache = decoder_model.step(dec_in, cur_beam, dummy_enc_out, src_mask_step, cache)
            last_h = tf.reshape(dec_out, [B, cur_beam, self._dim])

            with tf.variable_scope('proj_%d' % step):
                logits = tf.layers.dense(last_h, V, name='pred', reuse=tf.AUTO_REUSE)  # [B,beam,V]

            logp = tf.nn.log_softmax(logits / temperature)
            k = beam_size
            topk_logp, topk_tok = tf.nn.top_k(logp, k=k)       # [B,beam,k]
            topk_prob = tf.exp(topk_logp)

            # 全局选 beam
            cand_scores = tf.expand_dims(scores, -1) + topk_logp                  # [B,beam,k]
            flat_scores = tf.reshape(cand_scores, [B, -1])                        # [B, beam*k]
            best_scores, best_idx = tf.nn.top_k(flat_scores, k=beam_size)         # [B,beam]

            parent_beam = best_idx // k
            tok_rank    = best_idx %  k
            batch_idx   = tf.tile(tf.expand_dims(tf.range(B), 1), [1, beam_size])

            # 重新排序 beam（seq/prob/cache）
            gather_parent = tf.stack([batch_idx, parent_beam], axis=2)            # [B,beam,2]
            parent_seq   = tf.gather_nd(seqs,  gather_parent)                     # [B,beam,T]
            parent_prob  = tf.gather_nd(probs, gather_parent)
            # 对 src_mask 也重排
            perbeam_src_mask_5d = tf.gather_nd(perbeam_src_mask_5d, gather_parent)  # 仍为 [B,beam,1,1,S]

            def gather_cache_beam(old_cache, gp):
                new_cache = {}
                for kname, val in old_cache.items():
                    if kname.startswith(("k_self_", "v_self_", "k_enc_", "v_enc_")):
                        new_cache[kname] = tf.gather_nd(val, gp)  # [B,beam,H,*,Dh]
                    else:
                        new_cache[kname] = val
                return new_cache
            cache = gather_cache_beam(cache, gather_parent)

            tok_gather = tf.stack([batch_idx, parent_beam, tok_rank], axis=2)
            next_tok   = tf.gather_nd(topk_tok,  tok_gather)                       # [B,beam]
            next_prob  = tf.gather_nd(topk_prob, tok_gather)                       # [B,beam]

            next_tok_glb = next_tok + offsets[step]
            seqs  = tf.concat([parent_seq, tf.expand_dims(next_tok_glb, -1)], axis=-1)  # [B,beam,T+1]
            probs = tf.concat([parent_prob, tf.expand_dims(next_prob,  -1)], axis=-1)
            scores = best_scores
            cur_beam = beam_size

        # 去掉 <START>，转回局部 id
        seqs  = seqs[:, :, 1:]
        probs = probs[:, :, 1:]
        gen_part_loc = seqs - tf.constant(offsets, dtype=seqs.dtype)

        return gen_part_loc, probs
