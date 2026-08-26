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
    "user_colossus_pid_list",  # 用户点击视频ID列表
    "user_colossus_aid_list"   # 用户点击的作者ID列表
]

user_realtime_click_fea_names = [
    "user_profile_v1_explore_click_pid_list",  # 用户点击的视频ID列表
    "user_profile_v1_explore_click_aid_list"   # 用户点击的作者ID列表
]

class MultiInterestModel(object):
    """
    多兴趣推荐模型类
    
    该模型使用Transformer架构，通过编码器-解码器结构：
    1. 编码器：将用户特征和行为序列编码为多个兴趣表示
    2. 解码器：基于兴趣表示生成推荐的语义ID序列
    """
    
    def __init__(self, feature_emb_dict, feature_emb_size_dict, dim=16, selected_size=256, vocab_sizes=[8192, 8192, 8192], print_ops=None):
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
            name='vocab_embedding',
            initializer=tf.random_uniform_initializer(minval=-1.0/dim, maxval=1.0/dim), 
            trainable=True
        )
        
        self._dim = dim
        self._selected_size = selected_size      

    def model(self, user_colossus_play_time_list, user_colossus_duration_list, user_colossus_channel_list, photo_sid, label, photo_semantic_id_int):
        """
        主训练模型前向传播
        
        Args:
            photo_sid: 视频语义ID序列，shape=[batch_size, seq_len]
            label: 真实标签，shape=[batch_size, 3]，对应三个语义层级
            photo_semantic_id_int: 视频语义ID整数序列，用于计算loss mask
            
        Returns:
            loss: 训练损失值
        """
        
        # debug
        # self._print_ops.append(tf.print("user_colossus_play_time_list(first):",
        #                                 user_colossus_play_time_list[0], summarize=100))
        # # debug
        # self._print_ops.append(tf.print("user_colossus_duration_list(first):",
        #                                 user_colossus_duration_list[0], summarize=100))
        # # debug
        # self._print_ops.append(tf.print("user_colossus_channel_list(first):",
        #                                 user_colossus_channel_list[0], summarize=100))
        
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
            [self._feature_emb_dict[fea] for fea in user_click_fea_names], axis=2)   # [B, 1000, dim*2]
        
        # self._print_ops.append(tf.print("user_click_fea(first):",
        #                                 user_click_fea[0, :, 0], summarize=100))
        
        # === 2.2 仅保留“长播”并取最近 selected_size 条（不足左补 -1） ===
        max_len   = self._selected_size               # 例如 256

        # play‑time / duration 原始张量 [B, 1000]，与 click‑emb 对齐
        play_time_seq = user_colossus_play_time_list*1000      # int64, ms
        duration_seq  = user_colossus_duration_list*1000       # int64, ms
        channel_seq = user_colossus_channel_list
        
        # 有效点击条数 (= 1000 序列中真实点击长度)
        raw_len   = tf.cast(self._feature_emb_size_dict['user_colossus_pid_list'],
                            tf.int32)                          # [B] 或 [B,1]
        valid_len = tf.reshape(raw_len, [-1])                 # [B]
        
        with tf.variable_scope('sample_len'):
            print_tensor("valid_len", valid_len)
        
        # ---------- ② per‑sample 过滤 + recent‑max_len + padding ----------
        def _broadcast_const(shape_like, val, dtype):
            """返回与 shape_like 同形、值恒为 val 的张量（避免 tf.fill + XLA 的 rank‑bug）"""
            return tf.broadcast_to(tf.cast(val, dtype), tf.shape(shape_like))

        # # ---------- ② per‑sample 过滤 + recent‑max_len + padding ----------
        def _const_like(x, val):
            """返回与 x 同形、值全为 val 的张量（保持 dtype）"""
            return tf.fill(tf.shape(x), tf.cast(val, x.dtype))
        
        
        def filter_long_and_crop_vec(seq_emb,
                                    play_time, duration, channel,
                                    valid_len,
                                    max_len,
                                    pad_value=0.0,
                                    name="filter_long_and_crop_vec"):
            """
            Args
            ----
            seq_emb   : [B, L, D]  float32/16   点击序列 embedding
            play_time : [B, L]     int64        播放时长 (ms)
            duration  : [B, L]     int64        视频总时长 (ms)
            channel   : [B, L]     int64        频道 id
            valid_len : [B]        int32        每条序列真实长度 (≤ L)
            max_len   : int                       只保留最近 max_len 条
            pad_value : float                     左侧 padding 常量

            Returns
            -------
            out_emb      : [B, max_len, D]   截断 + 左 PAD
            longview_len : [B] int32         满足长播条件的条数 (未截断)
            used_len     : [B] int32         min(longview_len, max_len)
            """
            with tf.name_scope(name):
                
                valid_len = tf.reshape(tf.cast(valid_len, tf.int32), [-1])        # [B]

                B = tf.shape(seq_emb)[0]
                L = tf.shape(seq_emb)[1]
                D = tf.shape(seq_emb)[2]

                # ---------- ① 计算阈值 thr ----------
                one_like_d = tf.ones_like(duration, dtype=duration.dtype)  # shape [B,L]

                thr_base = _const_like(duration, 79700)                # 默认 79.7s
                thr_base = tf.where(duration <=195000, _const_like(one_like_d, 92500), thr_base)
                thr_base = tf.where(duration <=118200, _const_like(one_like_d, 74900), thr_base)
                thr_base = tf.where(duration <= 71800, _const_like(one_like_d, 46600), thr_base)
                thr_base = tf.where(duration <= 38800, _const_like(one_like_d, 28800), thr_base)
                thr_base = tf.where(duration <= 20300, _const_like(one_like_d, 18400), thr_base)
                thr_base = tf.where(duration <= 12700, _const_like(one_like_d, 13600), thr_base)
                thr_base = tf.where(duration <=  8700, _const_like(one_like_d, 12000), thr_base)
                thr_base = tf.where(duration <=     0, _const_like(one_like_d, 13100), thr_base)
                
                thr_chan_1     = thr_base // 1
                thr_chan_other = thr_base // 1
                thr = tf.where(tf.equal(channel, 1), thr_chan_1, thr_chan_other)  # shape [B,L]

                # ---------- ② 长播掩码 + 有效长度掩码 ----------
                long_mask  = tf.greater_equal(play_time, thr)                      # [B, L] bool

                
                rng        = tf.range(L, dtype=tf.int32)[tf.newaxis, :]                    # [1,L]
                valid_mask = tf.less(rng, valid_len[:, tf.newaxis])                        # [B,L] bool
                keep_mask  = tf.logical_and(long_mask, valid_mask)                         # [B,L] bool

                longview_len = tf.reduce_sum(tf.cast(keep_mask, tf.int32), axis=1)         # [B]

                # ---------- ③ 取最近 max_len 条 ----------
                idx_full   = tf.tile(rng, [B, 1])                                          # [B,L] int32
                pad_neg1   = _broadcast_const(idx_full, -1, idx_full.dtype)                # [B,L] 常数 -1
                masked_idx = tf.where(keep_mask, idx_full, pad_neg1)                       # [B,L]

                topk_vals, _ = tf.nn.top_k(masked_idx, k=max_len)                          # [B,max_len] desc
                ordered_idx  = tf.reverse(topk_vals, axis=[1])                             # asc，左 PAD

                keep_topk = tf.cast(ordered_idx >= 0, seq_emb.dtype)                       # [B,max_len]
                used_len  = tf.reduce_sum(tf.cast(keep_topk, tf.int32), axis=1)            # [B]

                safe_idx  = tf.maximum(ordered_idx, 0)                                     # 负 idx → 0

                batch_ids = tf.tile(tf.range(B, dtype=tf.int32)[:, tf.newaxis], [1, max_len])
                gather_nd_idx = tf.stack([batch_ids, safe_idx], axis=-1)                   # [B,max_len,2]

                gathered = tf.gather_nd(seq_emb, gather_nd_idx)                            # [B,max_len,D]

                pad_tensor = _broadcast_const(gathered, pad_value, gathered.dtype)         # [B,max_len,D]
                out_emb = keep_topk[:, :, tf.newaxis] * gathered + \
                        (1.0 - keep_topk[:, :, tf.newaxis]) * pad_tensor                 # [B,max_len,D]

                return out_emb, longview_len, used_len
        
        # play_time / duration 已 ×1000 → int64(ms)
        user_click_fea, longview_len, used_len = filter_long_and_crop_vec(
            seq_emb   = user_click_fea,                 # [B, 1000, fea_dim]
            play_time = play_time_seq,                  # [B, 1000]
            duration  = duration_seq,                   # [B, 1000]
            channel   = channel_seq,                    # [B, 1000]
            valid_len = valid_len,                      # [B]
            max_len   = max_len,
            pad_value = 0.0,
            name="user_click_recent_long_vec"
        )
        
        # 通过MLP处理点击特征
        user_click_emb = mlp('user_click_emb', user_click_fea, [4*self._dim], self._dim, activation=tf.nn.leaky_relu) # [B, 1000, dim]

        with tf.variable_scope('sample_len'):
            # —— debug print —— #
            print_tensor("longview_len", longview_len)   # 每个 batch 样本长播原始条数
            print_tensor("used_len", used_len)           # 截断后实际使用条数
        
        # debug
        # self._print_ops.append(tf.print("user_click_emb(first) after long-view filter:",
        #                                 user_click_emb[0, :, 0], summarize=100))
        
        # ================ 用户最近的实时点击 ===================
        # 1) 原始特征 -> Embedding [B, 20, dim]
        user_realtime_click_fea = tf.concat(
            [self._feature_emb_dict[fea] for fea in user_realtime_click_fea_names], axis=2)
        # self._print_ops.append(tf.print("user_realtime_click_fea(first):",
        #                                 user_realtime_click_fea[0, :, 0], summarize=100))

        user_realtime_click_emb = mlp('user_click_emb',
                                    user_realtime_click_fea,
                                    [4 * self._dim], self._dim,
                                    activation=tf.nn.leaky_relu)        # [B, 20, dim]

        # 2) 序列长度（≤ 20）
        realtime_raw_len = tf.reshape(
            tf.cast(self._feature_emb_size_dict['user_profile_v1_explore_click_pid_list'], tf.int32),
            [-1])                                                           # [B]
        
        with tf.variable_scope('sample_len'):
            print_tensor("realtime_raw_len", realtime_raw_len)

        # 3) 直接 reverse：把「最新→最旧」 + 右侧 padding       ➜  「左侧 padding」 + 「最旧→最新」
        realtime_max_len = 20
        user_realtime_click_emb = tf.reverse(user_realtime_click_emb, axis=[1])  # [B,20,dim]
        user_realtime_click_emb.set_shape([None, realtime_max_len, self._dim])

        # 4) used_len = min(raw_len, 20)   （仍用于 mask）
        realtime_used_len = tf.minimum(realtime_raw_len, realtime_max_len)       # [B]

        with tf.variable_scope('sample_len'):
            print_tensor("realtime_used_len", realtime_used_len)
            print_tensor("all_used_len", used_len + realtime_used_len)
        
        # self._print_ops.append(tf.print("user_realtime_click_emb(first):",
                                        # user_realtime_click_emb[0, :, 0], summarize=100))
        
        # === 3. 构建编码器输入 ===
        # 拼接顺序： [user_static] + [长播筛后历史 max_len] + [实时序列 realtime_max_len]
        encoder_input = tf.concat([user_static_emb, user_click_emb, user_realtime_click_emb], axis=1)
        
        # 计算编码器输入的余弦相似度（用于调试）
        encoder_input_sim = tf.reshape(encoder_input, [batch_size, -1])
        print_tensor("encoder_input_sim", calc_sim_cos(encoder_input_sim))
        
        # === 3-A. padding mask 更新：使用 used_len & realtime_used_len ===============================
        # 总序列长度
        total_len = 1 + max_len + realtime_max_len   # user token + 历史 + 实时

        B = tf.shape(used_len)[0]  # batch

        # 历史（长播）mask：右对齐
        click_mask = tf.sequence_mask(used_len, maxlen=max_len, dtype=tf.int64)   # [B,max_len] 左 1 右 0
        click_mask = tf.reverse(click_mask, axis=[1])                              # 右对齐

        # 实时 mask：右对齐
        realtime_mask = tf.sequence_mask(realtime_used_len, maxlen=realtime_max_len, dtype=tf.int64)
        realtime_mask = tf.reverse(realtime_mask, axis=[1])

        # user token 永远有效
        user_tok = tf.ones([B, 1], dtype=tf.int64)

        # 拼整条 mask
        seq_mask = tf.concat([user_tok, click_mask, realtime_mask], axis=1)  # [B,total_len]

        # debug
        # self._print_ops.append(tf.print("seq_mask(first):", seq_mask[0], summarize=100))
        
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
        encoder_model = EncoderModel(num_layers=1, dim=self._dim, num_heads=8, dropout_rate=0.1, hidden_dim=self._dim*2, training=True)
        encoder_output = encoder_model.forward(encoder_input, src_mask, training=True) # [batch_size, seq_len, dim]
        
        # 计算编码器输出的余弦相似度（用于调试）
        encoder_output_sim = tf.reshape(encoder_output, [batch_size, -1])
        print_tensor("encoder_output_sim", calc_sim_cos(encoder_output_sim))
        
        # === 6. Transformer解码器 ===
        # 使用4层Transformer解码器生成序列表示
        decoder_model = DecoderModel(num_layers=1, dim=self._dim, num_heads=8, dropout_rate=0.1, hidden_dim=self._dim*2, training=True)
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

    def get_print_ops(self):
        return [tf.group(*self._print_ops)]
    