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
        
        # 创建统一的嵌入矩阵，包含所有语义ID的嵌入向量
        # 使用均匀分布初始化，范围为[-1/dim, 1/dim]
        self._vocab_embedding = tf.get_variable(
            shape=[self._total_vocab_size+1, dim], 
            name='vocab_embedding',
            initializer=tf.random_uniform_initializer(minval=-1.0/dim, maxval=1.0/dim), 
            trainable=True
        )
        
        self._dim = dim

    def model(self, photo_sid, label, photo_semantic_id_int, playing_time=None):
        """
        主训练模型前向传播
        
        Args:
            photo_sid: 视频语义ID序列，shape=[batch_size, seq_len]
            label: 真实标签，shape=[batch_size, 3]，对应三个语义层级
            photo_semantic_id_int: 视频语义ID整数序列，用于计算loss mask
            playing_time: 用户播放时长(毫秒)，用于loss加权，shape=[batch_size, 1]
                        若提供，则按 lg(2 + playing_time/1000) 对样本加权，长播放权重更高
                        若为None，退化为二值mask（推理模式）

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
        decoder_input = tf.nn.embedding_lookup(self._vocab_embedding, photo_with_start_token)

        # === 5. Transformer编码器 ===
        # encoder_output = encoder_input
        # 先看看没有encoder结构的效果？如果可以就不用加encoder了
        encoder_output = layer_norm(encoder_input, scope="enc_ln")
        
        # 计算编码器输出的余弦相似度（用于调试）
        encoder_output_sim = tf.reshape(encoder_output, [batch_size, -1])
        print_tensor("encoder_output_sim", calc_sim_cos(encoder_output_sim))
        
        # === 6. Transformer解码器 ===
        # 使用4层Transformer解码器生成序列表示
        decoder_model = DecoderModel(num_layers=2, dim=self._dim, num_heads=8, dropout_rate=0.1, hidden_dim=self._dim*2, training=True)
        decoder_output = decoder_model.forward(decoder_input, encoder_output, src_mask, training=True) # [batch_size, seq_len, dim]
        
        # 计算解码器各步输出的余弦相似度（用于调试）
        for i in range(len(self._vocab_sizes)):
            similarity = calc_sim_cos(decoder_output[:, i, :])
            print_tensor('decoder_sim/decoder_output_%d' % i, similarity)

        # === 7. 损失计算 ===
        losses = []
        # 创建损失掩码，对有效样本按播放时长进行对数加权
        # 加权公式: lg(2 + playing_time/1000)，长播放样本权重更高
        # playing_time=5s 时 lg(7)≈0.85，20s 时 lg(22)≈1.34，85s 时 lg(87)≈1.94，300s 时 lg(302)≈2.48
        valid = tf.cast(tf.reshape(photo_semantic_id_int, [-1]) > 0, tf.float32)
        if playing_time is not None:
            playing_time_flat = tf.reshape(playing_time, [-1])
            weight = tf.math.log(2.0 + playing_time_flat / 1000.0) / tf.math.log(10.0)
            loss_mask = valid * weight
        else:
            # 推理模式无 playing_time，退化为二值 mask
            loss_mask = valid

        # 打印加权后 loss_mask 的统计信息，方便监控训练时权重分布
        print_tensor("loss_mask_max", tf.reduce_max(loss_mask))
        # 打印有效样本比例（photo_semantic_id_int > 0 的样本占比）
        print_tensor("loss_mask_valid_ratio", tf.reduce_mean(valid))
        # 仅对有效样本求 loss_mask 均值，这才是真正的权重分布
        if playing_time is not None:
            valid_loss_mask_mean = tf.reduce_sum(loss_mask) / (tf.reduce_sum(valid) + 1e-9)
            print_tensor("loss_mask_valid_mean", valid_loss_mask_mean)

        # 打印 batch 中播放时长分布（分母为有效样本数）
        if playing_time is not None:
            playing_time_flat = tf.reshape(playing_time, [-1])
            is_short_play = tf.cast(tf.less_equal(playing_time_flat, 17000), tf.float32)
            is_long_play = tf.cast(tf.greater(playing_time_flat, 17000), tf.float32)
            valid_count = tf.reduce_sum(valid) + 1e-9
            # 短/长播放比例（两者之和应≈1.0，用于交叉验证）
            short_ratio = tf.reduce_sum(is_short_play * valid) / valid_count
            long_ratio = tf.reduce_sum(is_long_play * valid) / valid_count
            print_tensor("play_short_ratio", short_ratio)
            print_tensor("play_long_ratio", long_ratio)
            print_tensor("play_ratio_sum", short_ratio + long_ratio)
            # 有效样本的平均播放时长（毫秒）
            avg_playing_time = tf.reduce_sum(playing_time_flat * valid) / valid_count
            print_tensor("avg_playing_time", avg_playing_time)

        # === 7-A. PRM loss ===
        prm_model = PRMModel(dim=self._dim, num_heads=8, dropout_rate=0.1, training=True)
        # 关键：先显式建 PRM 相关变量（attention / target_score_mlp），保持训练/推理图的 dense bin 顺序一致
        # prm_model.build_variables()
        sid_embeddings = tf.nn.embedding_lookup(self._vocab_embedding, photo_sid)  # [B, 3, dim]
        # target_embeddings = tf.cumsum(sid_embeddings, axis=1)  # path sum-pool: [sid0, sid0+sid1, sid0+sid1+sid2]
        prm_temperature = 1 # 0.2
        # in-batch 负采样下，热门路径作为负样本的概率正比于其在 batch 内的频率，会引入 popularity bias
        # 按 Yi et al. "Sampling-Bias-Corrected Neural Modeling for Retrievals" (WSDM 2019) 做纠偏
        use_logq_correction = True  # 是否使用纠偏

        # === KV 投影：对 [B, L_enc, dim] 做一次，得到 [B, H, L_enc, Dh]，后续循环内只做 tile ===
        prm_K, prm_V = prm_model.project_kv(encoder_output)

        # === K/V tile 到 B²、src_mask tile 到 B*B，统一在循环外做一次，避免 3 次重复 op ===
        _H_kv  = tf.shape(prm_K)[1]
        _L_kv  = tf.shape(prm_K)[2]
        _Dh_kv = tf.shape(prm_K)[3]
        prm_K_tiled = tf.tile(tf.expand_dims(prm_K, axis=1), [1, batch_size, 1, 1, 1])  # [B,B,H,L,Dh]
        prm_K_tiled = tf.reshape(prm_K_tiled, [batch_size * batch_size, _H_kv, _L_kv, _Dh_kv])
        prm_V_tiled = tf.tile(tf.expand_dims(prm_V, axis=1), [1, batch_size, 1, 1, 1])
        prm_V_tiled = tf.reshape(prm_V_tiled, [batch_size * batch_size, _H_kv, _L_kv, _Dh_kv])
        prm_pair_src_mask = tf.tile(tf.expand_dims(src_mask, axis=1), [1, batch_size, 1, 1, 1])
        prm_pair_src_mask = tf.reshape(prm_pair_src_mask,
                                       [batch_size * batch_size, 1, 1, tf.shape(encoder_output)[1]])

        prm_losses = []
        for step in range(len(self._vocab_sizes)):
            # 路径 -> 单 token 表示：embedding_lookup + reduce_sum (sum-pool)
            # 每层 SID 通过全局 offset 位于互不相交的区间，sum 天然保留位置信息、无合法路径碰撞
            # 无参数投影层，梯度经 reduce_sum → embedding_lookup 回传至 _vocab_embedding（trainable=True）
            prefix_emb = sid_embeddings[:, :step + 1, :]                     # [B, step+1, dim]
            target_embedding = tf.reduce_sum(prefix_emb, axis=1)             # [B, dim]

            # === 提前计算 path_hash（与 logQ 复用，无额外开销）===
            prefix_tokens = tf.cast(photo_sid[:, :step + 1], tf.int64)       # [B, step+1]
            base_hash = tf.constant(self._total_vocab_size + 1, dtype=tf.int64)
            path_hash = tf.zeros([batch_size], dtype=tf.int64)
            for k in range(step + 1):
                path_hash = path_hash * base_hash + prefix_tokens[:, k]

            # === False Negative Mask ===
            # same_path[i, j] = 1 if path_hash[i] == path_hash[j]，即同路径的 pair
            ph_row = tf.reshape(path_hash, [batch_size, 1])            # [B, 1]
            ph_col = tf.reshape(path_hash, [1, batch_size])            # [1, B]
            same_path = tf.cast(tf.equal(ph_row, ph_col), tf.float32)  # [B, B]
            # 去掉对角线（对角线是正样本，必须保留）
            diag = tf.eye(batch_size, dtype=tf.float32)
            false_neg_mask = same_path * (1.0 - diag)  # [B, B]  1=false negative

            # pair_target_embedding 构造：将 batch 内所有 target embedding 两两配对
            pair_target_embedding = tf.tile(tf.expand_dims(target_embedding, axis=0), [batch_size, 1, 1])  # [B,B,dim]
            pair_target_embedding = tf.reshape(pair_target_embedding, [batch_size * batch_size, 1, self._dim])  # [B*B,1,dim]

            # K/V 和 src_mask 已在循环外 tile 完毕，直接复用 prm_K_tiled / prm_V_tiled / prm_pair_src_mask

            prm_logits = prm_model.forward_with_kv(pair_target_embedding, prm_K_tiled, prm_V_tiled, prm_pair_src_mask, training=True)
            prm_logits = tf.reshape(prm_logits, [batch_size, batch_size])

            # === Apply False Negative Mask: set false-neg positions to -1e9 ===
            prm_logits_masked = prm_logits + false_neg_mask * tf.constant(-1e9, dtype=tf.float32)

            # 监控：每行 mask 后剩余的真负样本数
            valid_neg_per_row = tf.reduce_sum(1.0 - same_path, axis=1)  # [B]
            print_tensor("prm/valid_neg_count_%d" % step, tf.reduce_mean(valid_neg_per_row))

            # === logQ correction for in-batch negative sampling ===
            prm_logits_for_loss = prm_logits_masked
            if use_logq_correction:
                # 统计 batch 内频率（path_hash 已在上面计算）
                _, idx, count = tf.unique_with_counts(path_hash)                 # [U], [B], [U]
                freq = tf.gather(count, idx)                                     # [B]
                logQ = tf.math.log(
                    tf.cast(freq, tf.float32) / tf.cast(batch_size, tf.float32))  # [B]
                print_tensor("prm/logQ_%d" % step, logQ)
                print_tensor("prm/path_freq_%d" % step, tf.cast(freq, tf.float32))
                # 减去 log Q(j)（按列广播：row i 的所有候选 j 都减 logQ[j]）
                # 已屏蔽位置（-1e9）减去 logQ 后仍极小，不影响 softmax
                prm_logits_for_loss = prm_logits_masked - tf.reshape(logQ, [1, -1])

            prm_label = tf.range(tf.shape(prm_logits)[0], dtype=tf.int32)  # 对角线为正样本
            prm_loss_i = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=prm_label, logits=prm_logits_for_loss / prm_temperature)
            prm_losses.append(prm_loss_i)
            print_tensor("loss/prm_loss_%d" % step, tf.reduce_sum(prm_loss_i * loss_mask) / (tf.reduce_sum(loss_mask) + 1e-9))
            print_tensor("prm/pos_score_%d" % step, tf.reduce_sum(tf.linalg.diag_part(prm_logits) * loss_mask) / (tf.reduce_sum(loss_mask) + 1e-9))

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
        ntp_loss = tf.reduce_sum(tf.add_n(losses) * loss_mask) / (tf.reduce_sum(loss_mask) + 1e-9)
        prm_loss = tf.reduce_sum(tf.add_n(prm_losses) * loss_mask) / (tf.reduce_sum(loss_mask) + 1e-9)
        loss = ntp_loss + prm_loss
        print_tensor("loss/ntp_loss", ntp_loss)
        print_tensor("loss/prm_loss", prm_loss)
        print_tensor("loss/total_loss", loss)
        return loss, result_dict

    def beam_search_fast(self, beam_size=512, temperature=1):
        """
        Decoder beam search + PRM 剪枝（PRM 只决定去留，不决定先后）：
        * 每个 step 先用 decoder beam search 保留 beam_size * 2 条候选 prefix
          （按 decoder 累积概率降序）。
        * 第一层（step=0）不做 PRM 打分，直接用 decoder 累积概率选 top beam_size。
        * 第二层起，候选 prefix 按训练时的 path sum-pool 构造 PRM target embedding：
          [sid0], [sid0+sid1], [sid0+sid1+sid2]。
        * PRM 只做剪枝：从 beam_size*2 条中选 PRM 分数最高的 beam_size 条留下，
          但保持候选原本的 decoder 概率降序，不按 PRM 分数重排。
        * 最终输出天然按 decoder 累积概率降序，无需末尾重排。

        返回：
            gen_part_loc  – shape [B, beam_size, seq_len] 的推荐 sid 局部 id 序列
            probs         – shape [B, beam_size]，每条 path 的 decoder 累积概率（已排序）
        """
        # ------------- 常量 & 子模型 -------------
        offsets = [0,
                self._vocab_sizes[0],
                self._vocab_sizes[0] + self._vocab_sizes[1]]       # 局部→全局 id 偏移

        decoder_model = DecoderModel(num_layers=2, dim=self._dim, num_heads=8,
                                    dropout_rate=0.1, hidden_dim=self._dim * 2)
        prm_model = PRMModel(dim=self._dim, num_heads=8, dropout_rate=0.1, training=False)
        # 训练图里 PRM 参数先于 proj_0 创建。推理图也保持同样 dense 参数顺序，
        # 避免线上 dense bin 按顺序加载时把投影层和 PRM 层权重错位。
        # prm_model.build_variables()

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
        
        enc_out_base = layer_norm(encoder_input, scope="enc_ln")     # [B, L_enc, C]

        # ---------- ② Beam 状态初始化 ----------
        start_tok = tf.fill([B, 1], self._total_vocab_size)   # global id of <START>
        seqs   = tf.expand_dims(start_tok, 1)                 # [B, 1, 1]
        # decoder 累计 log 概率（用于筛候选池 & 最终排序，不被 PRM 污染）
        dec_path_log_probs = tf.zeros([B, 1], dtype=tf.float32)

        cur_beam = 1  # 当前 beam 数
        cache = {}                    # 全层 KV

        prm_candidate_size = beam_size * 2
        prm_temperature = 1  # 0.2  与训练侧保持一致，用于 PRM logits 缩放

        # PRM K/V 延迟构建：step==0 时 project_kv 一次；step>=1 tile 到 B×cand
        prm_K = None
        prm_V = None
        prm_K_tiled = None
        prm_V_tiled = None

        # PRM src_mask 延迟构建：仅在 step==1 时构建一次
        prm_src_mask = None

        # ---------- Cache 辅助函数 ----------
        def gather_cache(old_cache, gp):
            """按 beam 选择索引 gp 从 old_cache 中重排 self-attention KV；
               cross-attention KV (k_enc/v_enc) 各 beam 共享，原样保留。"""
            new_cache = {}
            for ck, v in old_cache.items():
                if ck.startswith(("k_self_", "v_self_")):
                    new_cache[ck] = tf.gather_nd(v, gp)   # [B, beam, H, T, Dh] → 重新排序
                else:
                    new_cache[ck] = v                     # k_enc / v_enc 原样保留
            return new_cache

        def tile_cache_for_first_step(old_cache, num_beams):
            """第一层从 1 beam 扩展到 num_beams beams；
               所有新 beam 继承同一父 beam (beam 0) 的 self-attention KV，
               直接 tile 即可，比 gather_nd 更高效。"""
            new_cache = {}
            for ck, v in old_cache.items():
                if ck.startswith(("k_self_", "v_self_")):
                    # [B, 1, H, T, Dh] -> [B, num_beams, H, T, Dh]
                    new_cache[ck] = tf.tile(v, [1, num_beams, 1, 1, 1])
                else:
                    new_cache[ck] = v  # k_enc / v_enc 原样保留
            return new_cache

        # ---------- ③ 逐层解码 ----------
        for step, V in enumerate(self._vocab_sizes):
            # 只 embed 当前 token
            dec_in = tf.nn.embedding_lookup(self._vocab_embedding, seqs[:, :, -1])  # [B,beam,1,C]
            dec_in = tf.reshape(dec_in, [B*cur_beam, 1, self._dim])

            dec_out, cache = decoder_model.step(
                dec_in, cur_beam, enc_out_base, src_mask, cache)            # 只算一步

            # 训练图里 PRM 参数先于 proj_0 创建。推理图也保持同样 dense 参数顺序，
            # 避免线上 dense bin 按顺序加载时把投影层和 PRM 层权重错位。
            # ⚠️ 必须在 decoder.step() 之后、proj_0 之前调用，严格对齐训练侧建图顺序。
            if step == 0:
                prm_model.build_variables()
                # 对 enc_out_base [B, L_enc, dim] 投影一次，得到 [B, H, L_enc, Dh]
                prm_K, prm_V = prm_model.project_kv(enc_out_base)
                # 紧接着把 K/V 和 src_mask tile 到 [B*cand, H, L_enc, Dh] / [B*cand, 1, 1, L_enc]
                # 放在 step==0 分支里，整个图里只创建一次 tile op，step>=1 直接复用同一张量
                prm_enc_len = tf.shape(enc_out_base)[1]
                _H_inf  = tf.shape(prm_K)[1]
                _L_inf  = tf.shape(prm_K)[2]
                _Dh_inf = tf.shape(prm_K)[3]
                prm_K_tiled = tf.tile(
                    tf.expand_dims(prm_K, axis=1), [1, prm_candidate_size, 1, 1, 1])  # [B,cand,H,L,Dh]
                prm_K_tiled = tf.reshape(prm_K_tiled, [B * prm_candidate_size, _H_inf, _L_inf, _Dh_inf])
                prm_V_tiled = tf.tile(
                    tf.expand_dims(prm_V, axis=1), [1, prm_candidate_size, 1, 1, 1])
                prm_V_tiled = tf.reshape(prm_V_tiled, [B * prm_candidate_size, _H_inf, _L_inf, _Dh_inf])
                prm_src_mask = tf.tile(
                    tf.expand_dims(src_mask, axis=1), [1, prm_candidate_size, 1, 1, 1])
                prm_src_mask = tf.reshape(
                    prm_src_mask, [B * prm_candidate_size, 1, 1, prm_enc_len])

            last_h = tf.reshape(dec_out, [B, cur_beam, self._dim])
            
            with tf.variable_scope('proj_%d' % step):
                logits = tf.layers.dense(last_h, V, name=f'pred', reuse=tf.AUTO_REUSE)

            logp = tf.nn.log_softmax(logits / temperature)                     # [B, cur_beam, V]

            # ==============================================================
            #  第一层（step==0）：不做 PRM 打分，decoder 直接选 top beam_size
            #  - cur_beam=1，logp shape [B, 1, V]，直接从 V 选 beam_size 即可
            #  - 无需构建 cand_seqs 中间张量，直接拼出 seqs 送入下一层
            #  - 无需 PRM 大张量（prm_encoder_output / prm_src_mask）
            # ==============================================================
            if step == 0:
                # 从 [B, 1, V] 的第 0 个 beam 取 top beam_size
                topk_logp, topk_tok = tf.nn.top_k(logp[:, 0, :], k=beam_size)  # [B, beam_size]

                # decoder 累积 log 概率 = topk logp（初始为 0 + logp = logp）
                dec_path_log_probs = topk_logp                                  # [B, beam_size]

                # 直接构建 seqs：tile <START> + 拼上选中的 token
                start_tiled  = tf.tile(seqs, [1, beam_size, 1])                 # [B, beam_size, 1]
                next_tok_glb = topk_tok + offsets[0]                            # [B, beam_size]
                seqs = tf.concat(
                    [start_tiled, tf.expand_dims(next_tok_glb, -1)], axis=-1)   # [B, beam_size, 2]

                # Cache 扩展：所有 beam_size 条路径继承同一父 beam (beam 0) 的 KV
                cache = tile_cache_for_first_step(cache, beam_size)

            # ==============================================================
            #  后续层（step>=1）：decoder 选 beam_size*2 候选 → PRM 打分剪枝到 beam_size
            # ==============================================================
            else:
                # prm_K_tiled / prm_V_tiled / prm_src_mask 已在 step==0 时构建完毕，直接复用

                # --- 第一阶段：decoder 每 beam 扩展 beam_size 条路径，全局选 top-(beam_size*2) 送 PRM ---
                k = beam_size
                topk_logp, topk_tok = tf.nn.top_k(logp, k=k)                   # [B, cur_beam, k]

                # 纯 decoder 累积 log 概率筛选候选池
                # 与论文 Algorithm 1 line 13 一致：s' = s + log p_k，不掺入 PRM 分数
                cand_scores = tf.expand_dims(dec_path_log_probs, -1) + topk_logp # [B, cur_beam, k]

                # 全局 top-(beam_size*2)
                flat_scores = tf.reshape(cand_scores, [B, -1])                  # [B, cur_beam*k]
                cand_dec_log_probs, best_idx = tf.nn.top_k(
                    flat_scores, k=prm_candidate_size)                          # [B, cand]

                parent_beam = best_idx // k                                     # index in 0..cur_beam-1
                tok_rank    = best_idx %  k                                     # index in 0..k-1

                batch_idx = tf.tile(tf.expand_dims(tf.range(B), 1), [1, prm_candidate_size])

                # gather 父路径 + 新 token → 候选 prefix
                gather_parent = tf.stack([batch_idx, parent_beam], axis=2)      # [B, cand, 2]
                parent_seq   = tf.gather_nd(seqs,  gather_parent)               # [B, cand, T]

                tok_gather = tf.stack([batch_idx, parent_beam, tok_rank], axis=2)
                next_tok   = tf.gather_nd(topk_tok,  tok_gather)                # [B, cand]
                next_tok_glb = next_tok + offsets[step]

                cand_seqs  = tf.concat(
                    [parent_seq, tf.expand_dims(next_tok_glb, -1)], axis=-1)    # [B, cand, T+1]

                # --- 第二阶段：PRM 打分 + 剪枝 ---
                # cand_seqs[:, :, 1:] 去掉 <START>，长度 = step+1，即当前 prefix 长度
                cand_sid_embeddings = tf.nn.embedding_lookup(
                    self._vocab_embedding, cand_seqs[:, :, 1:])                 # [B, cand, step+1, dim]
                prm_target_embedding = tf.reduce_sum(cand_sid_embeddings, axis=2)  # [B, cand, dim]
                prm_target_embedding = tf.reshape(
                    prm_target_embedding,
                    [B * prm_candidate_size, 1, self._dim]
                )

                # prm_K_tiled / prm_V_tiled 已在 step==0 tile 好，直接复用
                prm_logits = prm_model.forward_with_kv(
                    prm_target_embedding,
                    prm_K_tiled,
                    prm_V_tiled,
                    prm_src_mask,
                    training=False
                )
                prm_logits = tf.reshape(prm_logits, [B, prm_candidate_size])

                # --- beam diagnostics (tf.print, no impact on inference logic) ---
                # 1. parent entropy: 候选池中父 beam 分布信息熵
                parent_onehot = tf.one_hot(parent_beam, depth=beam_size, dtype=tf.float32)  # [B, cand, beam_size]
                parent_count = tf.reduce_sum(parent_onehot, axis=1)  # [B, beam_size]
                p_parent = parent_count / tf.reduce_sum(parent_count, axis=1, keepdims=True)
                parent_entropy = -tf.reduce_sum(
                    p_parent * tf.math.log(tf.maximum(p_parent, 1e-10)), axis=1)  # [B]
                # 2. prm_prob entropy
                prm_prob = tf.nn.softmax(prm_logits / prm_temperature, axis=-1)
                prm_entropy = -tf.reduce_sum(
                    prm_prob * tf.math.log(tf.maximum(prm_prob, 1e-10)), axis=1)  # [B]
                # 3. top-1 parent ratio after PRM selection
                _, prm_best_idx_tmp = tf.nn.top_k(prm_logits / prm_temperature, k=beam_size)
                beam_batch_idx = tf.tile(tf.expand_dims(tf.range(B), 1), [1, beam_size])
                prm_selected_parents = tf.gather_nd(
                    parent_beam,
                    tf.stack([beam_batch_idx, prm_best_idx_tmp], axis=2)
                )  # [B, beam_size]
                sel_onehot = tf.one_hot(prm_selected_parents, depth=beam_size, dtype=tf.float32)
                sel_count = tf.reduce_sum(sel_onehot, axis=1)
                top1_parent_ratio = tf.reduce_max(sel_count, axis=1) / tf.cast(beam_size, tf.float32)

                diag_print = tf.print(
                    "beam_diag/parent_entropy_step%d" % step, tf.reduce_mean(parent_entropy),
                    "beam_diag/prm_entropy_step%d" % step, tf.reduce_mean(prm_entropy),
                    "beam_diag/prm_top1_parent_ratio_step%d" % step, tf.reduce_mean(top1_parent_ratio),
                    output_stream="stderr"
                )
                with tf.control_dependencies([diag_print]):
                    prm_logits = tf.identity(prm_logits)

                # 第二阶段：PRM 只做剪枝（决定去留），不参与排序（决定先后）
                prm_logits_scaled = prm_logits / prm_temperature
                _, prm_best_idx = tf.nn.top_k(prm_logits_scaled, k=beam_size)
                prm_best_idx = tf.sort(prm_best_idx, axis=-1)                      # [B, beam] 升序 = cand_seqs 原序
                gather_prm = tf.stack([beam_batch_idx, prm_best_idx], axis=2)  # [B, beam, 2]

                seqs  = tf.gather_nd(cand_seqs, gather_prm)                       # [B, beam, T+1] decoder 概率序
                # 留下来的 beam 继承其原本的 decoder 累积 log 概率，PRM 分数不进入排序基准
                dec_path_log_probs = tf.gather_nd(cand_dec_log_probs, gather_prm) # [B, beam] decoder 概率序

                # cache 只包含本轮已消费的 parent prefix，需按 PRM 最终留下的父 beam 重排。
                selected_parent_beam = tf.gather_nd(parent_beam, gather_prm)
                gather_parent_after_prm = tf.stack([beam_batch_idx, selected_parent_beam], axis=2)
                cache = gather_cache(cache, gather_parent_after_prm)

            cur_beam = beam_size            # 以后固定

        # PRM 剪枝已保持 cand_seqs 原始顺序（decoder 累积概率降序），
        # seqs / dec_path_log_probs 在每一步都天然按 decoder 概率降序，无需末尾重排。

        # 去掉 <START>
        seqs  = seqs[:, :, 1:]
        # 返回 decoder 概率（beam 得分即 decoder 累积概率）
        probs = tf.exp(dec_path_log_probs)

        # 转回局部 id
        offsets_t = tf.constant(offsets, dtype=seqs.dtype)          # [3]
        gen_part_loc = seqs - offsets_t

        return gen_part_loc, probs

    def beam_search_fast_no_prm(self, beam_size=512, temperature=1):
        """
        Decoder-only beam search for inference.

        注意：这里只创建 PRMModel.forward 会用到的 dense 参数，
        不执行 PRM 的 attention/MLP 计算；beam 只使用 decoder 的累积 log prob。
        """
        # ------------- 常量 & 子模型 -------------
        offsets = [0,
                self._vocab_sizes[0],
                self._vocab_sizes[0] + self._vocab_sizes[1]]       # 局部→全局 id 偏移

        decoder_model = DecoderModel(num_layers=2, dim=self._dim, num_heads=8,
                                    dropout_rate=0.1, hidden_dim=self._dim * 2)
        # 保留对象定义；PRM 参数在下面手动建，不使用 PRM 做 forward / rerank。
        # build_variables 现在只创建 attention / target_score_mlp 变量（路径表示改为 sum-pool，无参数）。
        prm_model = PRMModel(dim=self._dim, num_heads=8, dropout_rate=0.1, training=False)
        # 训练图里 PRM 参数先于 proj_0 创建。no_prm 虽然不用 PRM 计算，
        # 也要按同样顺序建变量，保证 dense 参数顺序和训练一致。
        # prm_model.build_variables()

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

        enc_out_base = layer_norm(encoder_input, scope="enc_ln")     # [B, L_enc, C]

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
            dec_in = tf.nn.embedding_lookup(self._vocab_embedding, seqs[:, :, -1])  # [B,beam,1,C]
            dec_in = tf.reshape(dec_in, [B*cur_beam, 1, self._dim])

            dec_out, cache = decoder_model.step(
                dec_in, cur_beam, enc_out_base, src_mask, cache)            # 只算一步

            if step == 0:
                prm_model.build_variables()

            last_h = tf.reshape(dec_out, [B, cur_beam, self._dim])

            with tf.variable_scope('proj_%d' % step):
                logits = tf.layers.dense(last_h, V, name=f'pred', reuse=tf.AUTO_REUSE)

            logp = tf.nn.log_softmax(logits / temperature)                     # [B, cur_beam, V]

            # --- 本轮候选：decoder-only beam search
            k = beam_size
            topk_logp, topk_tok = tf.nn.top_k(logp, k=k)                       # [B, cur_beam, k]
            topk_prob = tf.exp(topk_logp)

            # 累积得分
            cand_scores = tf.expand_dims(scores, -1) + topk_logp               # [B, cur_beam, k]

            # --- decoder 选全局 top-beam_size ---
            flat_scores = tf.reshape(cand_scores, [B, -1])                     # [B, cur_beam*k]
            best_scores, best_idx = tf.nn.top_k(flat_scores, k=beam_size)

            parent_beam = best_idx // k                                        # index in 0..cur_beam‑1
            tok_rank    = best_idx %  k                                        # index in 0..k‑1

            batch_idx = tf.tile(tf.expand_dims(tf.range(B), 1), [1, beam_size])

            # gather 父路径
            gather_parent = tf.stack([batch_idx, parent_beam], axis=2)         # [B, cand, 2]
            parent_seq   = tf.gather_nd(seqs,  gather_parent)                  # [B, cand, T]
            parent_prob  = tf.gather_nd(probs, gather_parent)

            def gather_cache(old_cache, gp):
                new_cache = {}
                for k, v in old_cache.items():
                    if k.startswith(("k_self_", "v_self_")):
                        new_cache[k] = tf.gather_nd(v, gp)   # [B, beam, H, T, Dh] → 重新排序
                    else:
                        new_cache[k] = v                     # k_enc / v_enc 原样保留
                return new_cache

            # gather 新 token
            tok_gather = tf.stack([batch_idx, parent_beam, tok_rank], axis=2)
            next_tok   = tf.gather_nd(topk_tok,  tok_gather)                   # [B, cand]
            next_prob  = tf.gather_nd(topk_prob, tok_gather)                   # [B, cand]

            # map 到全局 id
            next_tok_glb = next_tok + offsets[step]

            # decoder 候选 prefix
            cand_seqs  = tf.concat([parent_seq, tf.expand_dims(next_tok_glb, -1)], axis=-1)  # [B, cand, T+1]
            cand_probs = tf.concat([parent_prob, tf.expand_dims(next_prob,  -1)], axis=-1)

            seqs  = cand_seqs                                                 # [B, beam, T+1]
            probs = cand_probs
            scores = best_scores                                               # [B, beam]

            # cache 按 decoder 最终留下的父 beam 重排。
            beam_batch_idx = tf.tile(tf.expand_dims(tf.range(B), 1), [1, beam_size])
            gather_parent_after_decoder = tf.stack([beam_batch_idx, parent_beam], axis=2)
            cache = gather_cache(cache, gather_parent_after_decoder)

            cur_beam = beam_size            # 以后固定

        # 去掉 <START>
        seqs  = seqs[:, :, 1:]
        probs = probs[:, :, 1:]

        # 转回局部 id
        offsets_t = tf.constant(offsets, dtype=seqs.dtype)          # [3]
        gen_part_loc = seqs - offsets_t

        return gen_part_loc, probs
