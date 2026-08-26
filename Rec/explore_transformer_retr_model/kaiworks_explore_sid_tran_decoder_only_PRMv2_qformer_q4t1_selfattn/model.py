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
        
        # === QFormer 相关：K 个可学习兴趣 query token ===
        # 只对行为序列（200 tokens）压缩，静态特征（1 token）直接保留
        # 行为序列 [B, 200, D] → QFormer → [B, K, D] 画像摘要
        # 最终 enc_compressed = concat([user_static, portrait]) = [B, 1+K, D]
        # Decoder 和 PRM 的 cross-attn Tk 从 201 降至 1+K
        self._qformer_query_num = 16   # K=4，4 个 query 捕捉多兴趣画像
        # 正交初始化：4 个 query 起点互相正交（pairwise cos sim ≈ 0），消除"起点 collapse"
        # 旧版 stddev=0.01 下 4 个 query 初始几乎重合（cos sim≈1），SA 一上来就面对"相同向量"，
        # 自然学出"相同均值"。正交初始化让起点 diverse，配合 diversity loss 守住全程。
        # K=4 < D=256 时 orthogonal_initializer 生成行正交矩阵，每行 L2 norm = 1，
        # 尺度合理（QFormer 内部 RMSNorm 会归一化，尺度差异不影响训练）。
        self._qformer_queries = tf.get_variable(
            shape=[self._qformer_query_num, dim],
            name='qformer_queries',
            initializer=tf.orthogonal_initializer(),
            trainable=True
        )

    def model(self, photo_sid, label, photo_semantic_id_int, playing_time=None):
        """
        主训练模型前向传播
        
        Args:
            photo_sid: 视频语义ID序列，shape=[batch_size, seq_len]
            label: 真实标签，shape=[batch_size, 3]，对应三个语义层级
            photo_semantic_id_int: 视频语义ID整数序列，用于计算loss mask
            playing_time: 播放时长(毫秒)，用于loss加权，shape=[batch_size, 1]
                        若提供，则按 lg(2 + playing_time/1000) 对样本加权，播放时间越长权重越高
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
        # === 1-B. 静态特征 LayerNorm（与 QFormer 输出 click_portrait 尺度对齐 ~O(1)）===
        # static 不经 QFormer，需外部归一化保证与 portrait 尺度一致，避免下游 cross-attn 被某一方主导
        user_static_emb = layer_norm(user_static_emb, scope="static_ln")  # [B, 1, D]

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
        
        # === 3. 构建行为特征的 padding mask（供 QFormer 使用）===
        B = tf.shape(used_len)[0]

        # 为点击序列生成 0/1 mask：左侧有效=1，右侧padding=0
        click_mask = tf.sequence_mask(
            lengths=used_len,          # [B]
            maxlen=max_len,            # 200
            dtype=tf.int8)             # [B, max_len]

        # debug
        self._print_ops.append(tf.print("click_mask first sample:", click_mask[0], summarize=100))

        # QFormer 需要 [B, 1, 1, max_len] 格式的 src_mask
        click_src_mask = tf.reshape(click_mask, [B, 1, 1, max_len])   # [B, 1, 1, 200]

        # 计算行为特征输入的余弦相似度（调试用）
        user_click_emb_flat = tf.reshape(user_click_emb, [batch_size, -1])
        print_tensor("encoder_input_sim", calc_sim_cos(user_click_emb_flat))

        # === 4. 构建解码器输入 ===
        # 添加起始token（使用总词汇表大小作为特殊标记）
        start_token_indice = tf.tile(tf.constant(self._total_vocab_size, shape=(1, 1), dtype=tf.int32), [batch_size, 1])
        # 将起始token与视频语义ID拼接
        photo_with_start_token = tf.concat([start_token_indice, photo_sid], axis=1)
        # 查找嵌入向量
        decoder_input = tf.nn.embedding_lookup(self._vocab_embedding, photo_with_start_token)

        # === 5. 行为特征直接送入 QFormer（不做外部 LayerNorm）===
        # 设计调整：QFormer 内部对 query 做 RMSNorm，K/V 侧靠 attention 的 1/sqrt(d_k) 缩放即可
        # 去掉外部 LayerNorm 可保留 token 间幅度差异（兴趣强度/时近性信号）
        # encoder_output_sim 监控改为监控原始 user_click_emb
        encoder_output_sim = tf.reshape(user_click_emb, [batch_size, -1])
        print_tensor("encoder_output_sim", calc_sim_cos(encoder_output_sim))

        # === 5-B. QFormer 压缩行为序列 [B, 200, D] → [B, K, D]（兴趣画像摘要）===
        # 设计思路：
        #   - 静态特征（1 token）：用户身份锚点，信息密度高，直接保留，不经 QFormer
        #   - 行为序列（200 tokens）：用户兴趣，冗余较高，通过 K=4 个 query 提炼多兴趣画像
        #   - 最终 enc_compressed = [static(1), portrait(K)] = [B, 1+K, D]
        with tf.variable_scope("user_qformer", reuse=tf.AUTO_REUSE):
            K = self._qformer_query_num  # 4
            # 每个样本共享同一组 learnable query，tile 到 batch
            qformer_queries = tf.tile(
                tf.reshape(self._qformer_queries, [1, K, self._dim]),
                [batch_size, 1, 1])  # [B, K, D]
            qformer_model = QFormer(
                num_layers=4, dim=self._dim, num_heads=8,
                dropout_rate=0.1, hidden_dim=self._dim * 2, training=True)
            click_portrait = qformer_model.forward(
                qformer_queries, user_click_emb, click_src_mask, training=True)  # [B, K, D]

        # === 5-C. 监控 K 个 query 间同质化程度 ===
        # pairwise cosine similarity 均值越接近 1 → K 个 query 输出趋同（同质化），越接近 0 → 多样性好
        # 健康范围：< 0.5 为可接受；若持续 > 0.8，说明需要加 self-attention 打破同质化
        K_f = tf.constant(float(self._qformer_query_num), dtype=tf.float32)
        query_normed = tf.nn.l2_normalize(click_portrait, axis=-1)               # [B, K, D]
        query_sim_matrix = tf.matmul(query_normed, query_normed, transpose_b=True)   # [B, K, K]
        query_sim_off_diag = tf.reduce_sum(query_sim_matrix, axis=[1, 2]) - K_f      # [B]: 去掉对角线
        num_pairs = K_f * (K_f - 1.0)
        query_sim_mean = tf.reduce_mean(query_sim_off_diag / num_pairs)              # scalar
        print_tensor("qformer/query_pairwise_cos_sim", query_sim_mean)

        # === 5-C2. Diversity loss（压缩保真 framing：让 K 个 query 信息不冗余）===
        # L_div = mean_batch mean_{i≠j}(cos_sim(qi, qj)^2)
        #   - 平方：对高相似度对打压更狠（梯度 2·cos·∇cos），集中火力压最趋同的 query 对
        #   - 不加 stop_gradient：梯度回流整个 QFormer（SA/cross-attn/FFN/query embedding），
        #     让网络整体学会产出 diverse 输出，避免只推 query embedding 被 SA 重新 smoothing 抹平
        # 复用 query_sim_matrix（对角线为 1，平方后仍为 1，减去 K 即得 off-diag 平方和）
        # 权重由 loss 聚合处控制（起步 0.05），此处只算原始 L_div
        query_sim_sq = tf.square(query_sim_matrix)                                   # [B, K, K]
        off_diag_sq_sum = tf.reduce_sum(query_sim_sq, axis=[1, 2]) - K_f            # [B]: 减去 K 个对角线 1^2
        diversity_loss = tf.reduce_mean(off_diag_sq_sum / num_pairs)                 # scalar
        print_tensor("loss/qformer_diversity_loss", diversity_loss)

        # === 5-D. 尺度监控：验证 static 与 portrait 是否平衡 ===
        # 期望：static_l2_norm ≈ portrait_l2_norm，ratio ≈ 1
        # K=4 时 portrait_l2 对 K 个 token 的 L2 norm 取均值（RMS across K and D），反映 K 个 token 整体尺度
        static_l2 = tf.sqrt(tf.reduce_mean(tf.square(user_static_emb[:, 0, :]), axis=-1))     # [B]
        portrait_l2 = tf.sqrt(tf.reduce_mean(tf.square(click_portrait), axis=[1, 2]))           # [B]
        print_tensor("qformer/static_l2_norm", tf.reduce_mean(static_l2))
        print_tensor("qformer/portrait_l2_norm", tf.reduce_mean(portrait_l2))
        print_tensor("qformer/static_portrait_norm_ratio",
                     tf.reduce_mean(static_l2 / (portrait_l2 + 1e-9)))

        # === 5-D2. 压缩保真度监控：portrait 与用户行为序列均值的对齐 ===
        # QFormer 职责是压缩用户行为信息成画像，portrait 应保留序列的主要方向
        # 只对有效 token 求均值（排除 padding），click_mask: [B, 200] 1=有效 0=padding
        # K=4 时 portrait 取 K 个 token 的均值作为画像整体方向
        click_mask_f = tf.cast(click_mask, tf.float32)                              # [B, 200]
        valid_count = tf.reduce_sum(click_mask_f, axis=-1, keepdims=True)            # [B, 1]
        click_seq_sum = tf.reduce_sum(
            user_click_emb * tf.expand_dims(click_mask_f, -1), axis=1)               # [B, D]
        click_seq_mean = click_seq_sum / (valid_count + 1e-9)                        # [B, D]
        portrait_vec = tf.reduce_mean(click_portrait, axis=1)                        # [B, D] K 个 token 均值
        portrait_unit = tf.nn.l2_normalize(portrait_vec, axis=-1)                    # [B, D]
        seq_mean_unit = tf.nn.l2_normalize(click_seq_mean, axis=-1)                   # [B, D]
        portrait_seq_sim = tf.reduce_sum(portrait_unit * seq_mean_unit, axis=-1)     # [B]
        print_tensor("qformer/portrait_seq_mean_sim", tf.reduce_mean(portrait_seq_sim))

        # === 5-E. 合并静态特征（身份锚点）+ 行为画像（多兴趣摘要）===
        enc_compressed = tf.concat([user_static_emb, click_portrait], axis=1)  # [B, 1+K, D]
        # 全部有效（静态 1 token + 画像 K tokens）
        src_mask_compressed = tf.ones([batch_size, 1, 1, 1 + K], dtype=tf.int8)

        # === 6. Transformer解码器 ===
        # 使用2层Transformer解码器生成序列表示
        # cross-attn 从 enc_compressed [B, 1+K, D]（静态1 + 画像K）而非原始 [B, 201, D]
        decoder_model = DecoderModel(num_layers=2, dim=self._dim, num_heads=8, dropout_rate=0.1, hidden_dim=self._dim*2, training=True)
        decoder_output = decoder_model.forward(decoder_input, enc_compressed, src_mask_compressed, training=True) # [batch_size, seq_len, dim]
        
        # 计算解码器各步输出的余弦相似度（用于调试）
        for i in range(len(self._vocab_sizes)):
            similarity = calc_sim_cos(decoder_output[:, i, :])
            print_tensor('decoder_sim/decoder_output_%d' % i, similarity)

        # === 7. 损失计算 ===
        losses = []
        # 创建损失掩码，对有效样本按播放时长进行对数加权
        # 加权公式: lg(2 + playing_time/1000)，播放时间越长权重越高
        # playing_time=0 时 lg(2)≈0.301，5s 时 lg(7)≈0.845，30s 时 lg(32)≈1.505，120s 时 lg(122)≈2.086，600s 时 lg(602)≈2.780
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
        # project_kv 对压缩后的 enc_compressed 投影 K/V，Tk 已从 201 降至 1+K（静态1 + 画像K）
        prm_K, prm_V = prm_model.project_kv(enc_compressed)   # [B, H, 1+K, Dh]
        sid_embeddings = tf.nn.embedding_lookup(self._vocab_embedding, photo_sid)  # [B, 3, dim]
        # === 5-F. 监控 QFormer 输出与目标 item 偏好的对齐程度 ===
        # click_portrait: [B, K, D] (K=4), QFormer 抽取的用户兴趣画像
        # sid_embeddings: [B, 3, D], 目标 item 的 3 级语义 ID embedding
        # target_pref_emb: 对 3 个 sid token 求和, 代表目标 item 的偏好向量
        # 该相似度衡量 QFormer 学到的兴趣与目标 item 语义的对齐程度
        # K=4 时 portrait 取 K 个 token 的均值作为画像整体方向
        target_pref_emb = tf.reduce_sum(sid_embeddings, axis=1)                          # [B, D]
        portrait_main = tf.reduce_mean(click_portrait, axis=1)                            # [B, D] K 个 token 均值
        portrait_norm = tf.nn.l2_normalize(portrait_main, axis=-1)                       # [B, D]
        target_norm   = tf.nn.l2_normalize(target_pref_emb, axis=-1)                     # [B, D]
        interest_target_sim = tf.reduce_sum(portrait_norm * target_norm, axis=-1)        # [B]
        print_tensor("qformer/interest_target_sim", tf.reduce_mean(interest_target_sim))
        prm_temperature = 1
        use_logq_correction = True
        prm_losses = []
        prm_H = prm_model.num_heads
        prm_Dh = self._dim // prm_H
        for step in range(len(self._vocab_sizes)):
            prefix_emb = sid_embeddings[:, :step + 1, :]                     # [B, step+1, dim]
            target_embedding = tf.reduce_sum(prefix_emb, axis=1)             # [B, dim]
            pair_target_embedding = tf.tile(tf.expand_dims(target_embedding, axis=0), [batch_size, 1, 1]) #[b,b,dim]
            pair_target_embedding = tf.reshape(pair_target_embedding, [batch_size * batch_size, 1, self._dim]) #[b*b,1,dim]
            # K/V 基于 enc_compressed（Tk=1+K），tile 到 B*B
            prm_K_tiled = tf.tile(tf.expand_dims(prm_K, axis=1), [1, batch_size, 1, 1, 1])   # [B, B, H, 1+K, Dh]
            prm_K_tiled = tf.reshape(prm_K_tiled, [batch_size * batch_size, prm_H, -1, prm_Dh])
            prm_V_tiled = tf.tile(tf.expand_dims(prm_V, axis=1), [1, batch_size, 1, 1, 1])
            prm_V_tiled = tf.reshape(prm_V_tiled, [batch_size * batch_size, prm_H, -1, prm_Dh])
            # src_mask 对应压缩后 1+K 个 token（静态1 + 画像K），全有效，tile 到 B*B
            # —— 兼容旧版 TF（TileOp 不支持 int8，如 1.12）：把 dtype 改成 tf.float32 ——
            pair_src_mask = tf.tile(
                tf.ones([1, 1, 1, 1 + K], dtype=tf.int8), [batch_size * batch_size, 1, 1, 1])

            prm_logits = prm_model.forward_with_kv(pair_target_embedding, prm_K_tiled, prm_V_tiled, pair_src_mask, training=True)
            prm_logits = tf.reshape(prm_logits, [batch_size, batch_size])

            # === logQ correction for in-batch negative sampling ===
            # in-batch 负采样下，路径 j 被采为负样本的概率 ≈ freq(j in batch) / B
            # corrected_logit(i, j) = logit(i, j) - log Q(j)，对所有 logit（含正样本对角线）做修正。
            # 参考: Yi et al. "Sampling-Bias-Corrected Neural Modeling for Retrievals" (WSDM 2019)
            prm_logits_for_loss = prm_logits

            # === 路径 hash（同时供 false negative mask 和 logQ 纠偏使用）===
            # 多项式 hash 把路径前缀 [s_0, ..., s_step] 映射到唯一 int64（无碰撞）
            # base = total_vocab_size + 1，3 层时 max_hash ≈ (24577)^3 ≈ 1.5e13 << int64 上限
            prefix_tokens = tf.cast(photo_sid[:, :step + 1], tf.int64)       # [B, step+1]
            base = tf.constant(self._total_vocab_size + 1, dtype=tf.int64)
            path_hash = tf.zeros([batch_size], dtype=tf.int64)
            for k in range(step + 1):
                path_hash = path_hash * base + prefix_tokens[:, k]

            # === False Negative Mask ===
            # in-batch 负采样下，batch 内走相同路径的不同用户互为 false negative（应当屏蔽）
            # same_path[i,j]=True 表示 path_i == path_j（含对角线正样本）
            same_path = tf.equal(
                tf.expand_dims(path_hash, 1),   # [B, 1]
                tf.expand_dims(path_hash, 0)    # [1, B]
            )                                   # [B, B]，broadcast 得到同路径矩阵
            diag_mask = tf.cast(tf.eye(batch_size), tf.bool)                        # 对角线 = 正样本，必须保留
            false_neg_mask = tf.logical_and(same_path, tf.logical_not(diag_mask))  # 非对角同路径 = false negative
            prm_logits_for_loss = tf.where(
                false_neg_mask,
                tf.fill(tf.shape(prm_logits_for_loss), -1e9),   # 屏蔽 false negative，令其对 softmax 无贡献
                prm_logits_for_loss
            )

            # 监控：每行 mask 后剩余的真负样本数（= B - 1 - 同路径的其他样本数）
            valid_neg_per_row = tf.reduce_sum(
                tf.cast(tf.logical_not(false_neg_mask), tf.float32) - tf.cast(diag_mask, tf.float32),
                axis=1)                                          # [B]
            print_tensor("prm/valid_neg_count_%d" % step, valid_neg_per_row)

            if use_logq_correction:
                # 统计 batch 内频率（复用上面的 path_hash）
                _, idx, count = tf.unique_with_counts(path_hash)                 # [U], [B], [U]
                freq = tf.gather(count, idx)                                     # [B]
                logQ = tf.math.log(
                    tf.cast(freq, tf.float32) / tf.cast(batch_size, tf.float32))  # [B]
                print_tensor("prm/logQ_%d" % step, logQ)
                print_tensor("prm/path_freq_%d" % step, tf.cast(freq, tf.float32))
                # 减去 log Q(j)（按列广播：row i 的所有候选 j 都减 logQ[j]）
                # 已被 mask 的位置（-1e9）减去 logQ 后仍极小，不影响 softmax
                prm_logits_for_loss = prm_logits_for_loss - tf.reshape(logQ, [1, -1])

            prm_label = tf.range(tf.shape(prm_logits)[0], dtype=tf.int32) # 对角线为正样本
            # prm_loss_i = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=prm_label, logits=prm_logits / prm_temperature)
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
        # Diversity loss 权重：起步 0.05，动态调（cos sim 不降则 0.1，主任务受损则 0.02）
        # 不乘 loss_mask：diversity 是 query 结构属性（非逐样本属性），全 batch 等权贡献
        diversity_loss_weight = 0.15
        loss = ntp_loss + prm_loss + diversity_loss_weight * diversity_loss
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
        # === 1-B. 静态特征 LayerNorm（与 QFormer 输出 click_portrait 尺度对齐 ~O(1)）===
        user_static_emb = layer_norm(user_static_emb, scope="static_ln")  # [B, 1, D]

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
        
        # === 3. 构建行为序列的 padding mask（供 QFormer 使用）===
        B          = tf.shape(used_len)[0]

        click_mask = tf.sequence_mask(
            lengths=used_len,         # [B]
            maxlen=max_len,           # 200
            dtype=tf.int8)            # [B, max_len]

        # debug
        self._print_ops.append(tf.print("seq_mask first sample:", click_mask[0], summarize=100))

        # QFormer src_mask
        click_src_mask = tf.reshape(click_mask, [B, 1, 1, max_len])   # [B, 1, 1, 200]

        # === 4. 行为特征直接送入 QFormer（不做外部 LayerNorm）===
        # 设计调整：与训练路径保持一致，QFormer 内部 RMSNorm + attention 1/sqrt(d_k) 缩放足够

        # === QFormer 压缩行为序列 [B, 200, D] → [B, K, D]（兴趣画像摘要）===
        # 静态特征（1 token）直接保留，行为序列（200 tokens）经 QFormer 压缩为 K 个画像 token
        # 设计调整：click 不做外部 LayerNorm，直接送 QFormer（与训练路径一致）
        with tf.variable_scope("user_qformer", reuse=tf.AUTO_REUSE):
            K = self._qformer_query_num
            qformer_queries = tf.tile(
                tf.reshape(self._qformer_queries, [1, K, self._dim]),
                [batch_size, 1, 1])  # [B, K, D]
            qformer_model = QFormer(
                num_layers=4, dim=self._dim, num_heads=8,
                dropout_rate=0.1, hidden_dim=self._dim * 2, training=False)
            click_portrait = qformer_model.forward(
                qformer_queries, user_click_emb, click_src_mask, training=False)  # [B, K, D]
        # 静态特征（1）+ 画像（K）拼接作为最终 user context
        enc_compressed = tf.concat([user_static_emb, click_portrait], axis=1)   # [B, 1+K, D]
        # Decoder/PRM cross-attn 的 src_mask：1+K 个 token 全有效
        src_mask_compressed = tf.ones([batch_size, 1, 1, 1 + K], dtype=tf.int8)

        # ---------- ② Beam 状态初始化 ----------
        start_tok = tf.fill([B, 1], self._total_vocab_size)   # global id of <START>
        seqs   = tf.expand_dims(start_tok, 1)                 # [B, 1, 1]
        # decoder 累计 log 概率（用于筛候选池 & 最终排序，不被 PRM 污染）
        dec_path_log_probs = tf.zeros([B, 1], dtype=tf.float32)

        cur_beam = 1  # 当前 beam 数
        cache = {}                    # 全层 KV

        prm_candidate_size = beam_size * 2
        prm_temperature = 1  # 0.2  与训练侧保持一致，用于 PRM logits 缩放

        # PRM 大张量延迟构建：仅在 step==1（真正需要 PRM 打分时）才创建，
        # 避免第一层额外占用 B * beam_size*2 的显存
        # Level 1：K/V 由 project_kv 在 step==0 一次性投影，step>=1 只 tile（无 FLOPs）
        prm_K = None
        prm_V = None
        prm_K_tiled = None
        prm_V_tiled = None
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

            # Decoder cross-attn 使用压缩后的 enc_compressed（Tk=1+K），而非原始 201 tokens
            dec_out, cache = decoder_model.step(
                dec_in, cur_beam, enc_compressed, src_mask_compressed, cache)  # 只算一步

            # ⚠️ 必须在 decoder.step() 之后、proj_0 之前调用，严格对齐训练侧建图顺序。
            # PRM project_kv 也使用 enc_compressed（Tk=1+K）
            if step == 0:
                prm_K, prm_V = prm_model.project_kv(enc_compressed)   # [B, H, K, Dh]

            last_h = tf.reshape(dec_out, [B, cur_beam, self._dim])
            
            with tf.variable_scope('proj_%d' % step):
                logits = tf.layers.dense(last_h, V, name=f'pred', reuse=tf.AUTO_REUSE)

            logp = tf.nn.log_softmax(logits / temperature)                     # [B, cur_beam, V]

            # ==============================================================
            #  第一层（step==0）：不做 PRM 打分，decoder 直接选 top beam_size
            #  - cur_beam=1，logp shape [B, 1, V]，直接从 V 选 beam_size 即可
            #  - 无需构建 cand_seqs 中间张量，直接拼出 seqs 送入下一层
            #  - 无需 PRM 大张量（prm_K_tiled / prm_V_tiled / prm_src_mask）
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
                # --- 延迟构建 PRM 大张量（仅 step==1 时执行一次）---
                # 这两个张量 shape 为 [B*beam_size*2, ...]，非常耗显存，
                # 第一层不需要，推迟到真正使用 PRM 时才构建
                if prm_K_tiled is None:
                    prm_enc_len = 1 + K  # 压缩后 Tk=1+K（静态1 + 画像K），而非原始 201
                    prm_H = prm_model.num_heads
                    prm_Dh = self._dim // prm_H
                    # K/V 已由 project_kv 预投影（基于 enc_compressed），tile 到每个候选
                    # [B, H, K, Dh] -> [B, cand, H, K, Dh] -> [B*cand, H, K, Dh]
                    prm_K_tiled = tf.tile(
                        tf.expand_dims(prm_K, axis=1),
                        [1, prm_candidate_size, 1, 1, 1]
                    )
                    prm_K_tiled = tf.reshape(
                        prm_K_tiled,
                        [B * prm_candidate_size, prm_H, prm_enc_len, prm_Dh]
                    )
                    prm_V_tiled = tf.tile(
                        tf.expand_dims(prm_V, axis=1),
                        [1, prm_candidate_size, 1, 1, 1]
                    )
                    prm_V_tiled = tf.reshape(
                        prm_V_tiled,
                        [B * prm_candidate_size, prm_H, prm_enc_len, prm_Dh]
                    )
                    # enc_compressed（1+K tokens）全有效，tile 到每个候选
                    # —— 兼容旧版 TF（TileOp 不支持 int8，如 1.12）：把 dtype 改成 tf.float32 ——
                    prm_src_mask = tf.ones(
                        [B * prm_candidate_size, 1, 1, prm_enc_len], dtype=tf.int8)

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
                # 论文 Eq.10：路径 -> 单 token 表示。方案 A：embedding_lookup + reduce_sum (sum-pool)
                # 每层 SID 通过全局 offset 互不相交，sum 保留位置信息且无合法路径碰撞
                # cand_seqs[:, :, 1:] 去掉 <START>，长度 = step+1，即当前 prefix 长度
                cand_sid_embeddings = tf.nn.embedding_lookup(
                    self._vocab_embedding, cand_seqs[:, :, 1:])                 # [B, cand, step+1, dim]
                prm_target_embedding = tf.reduce_sum(cand_sid_embeddings, axis=2)  # [B, cand, dim]
                prm_target_embedding = tf.reshape(
                    prm_target_embedding,
                    [B * prm_candidate_size, 1, self._dim]
                )

                prm_logits = prm_model.forward_with_kv(
                    prm_target_embedding,
                    prm_K_tiled,
                    prm_V_tiled,
                    prm_src_mask,
                    training=False
                )
                prm_logits = tf.reshape(prm_logits, [B, prm_candidate_size])

                # 第二阶段：PRM 只做剪枝（决定去留），不参与排序（决定先后）
                # 从 beam_size*2 条 decoder 候选中挑出 PRM 打分最高的 beam_size 条
                # 对 PRM logits 除以 temperature，与训练侧 softmax 前的缩放保持一致
                prm_logits_scaled = prm_logits / prm_temperature

                # ================================================================
                #  推理诊断：PRM 多样性 & 退化检测（不影响计算逻辑，仅打印）
                #  1. parent_beam 分布信息熵：衡量 beam 是否来自少数父 beam
                #     H = -sum(p * log(p))，上界 = log(beam_size)
                #     H 接近 0 → 候选全来自 1~2 个父 beam（退化）
                #  2. PRM logits 信息熵：衡量 PRM 打分是否退化成近均匀分布
                #     H 接近 log(cand) → PRM 无法区分候选（打分随机）
                #  3. PRM top-1 父 beam 集中度：最高分候选的父 beam 占比
                # ================================================================
                # --- 1. parent_beam 分布熵（per sample，取 batch mean）---
                # parent_beam [B, cand]，值域 0..cur_beam-1
                # 用 one-hot + reduce_sum 统计每个父 beam 被选为候选的次数
                parent_one_hot = tf.one_hot(parent_beam, depth=cur_beam, dtype=tf.float32)  # [B, cand, beam]
                parent_counts = tf.reduce_sum(parent_one_hot, axis=1)  # [B, beam]，每个父 beam 被选次数
                parent_freq = parent_counts / (tf.cast(prm_candidate_size, tf.float32) + 1e-9)  # 归一化
                parent_entropy = -tf.reduce_sum(
                    parent_freq * tf.math.log(parent_freq + 1e-9), axis=-1)  # [B]
                parent_entropy_mean = tf.reduce_mean(parent_entropy)
                parent_entropy_max = tf.math.log(tf.cast(cur_beam, tf.float32))  # 均匀分布时的上界
                tf.print("beam_diag/parent_entropy_step%d" % step, parent_entropy_mean,
                         "/ max", parent_entropy_max)

                # --- 2. PRM logits softmax 熵（衡量 PRM 分辨力）---
                prm_prob = tf.nn.softmax(prm_logits_scaled, axis=-1)  # [B, cand]
                prm_entropy = -tf.reduce_sum(
                    prm_prob * tf.math.log(prm_prob + 1e-9), axis=-1)  # [B]
                prm_entropy_mean = tf.reduce_mean(prm_entropy)
                prm_entropy_max = tf.math.log(tf.cast(prm_candidate_size, tf.float32))
                tf.print("beam_diag/prm_entropy_step%d" % step, prm_entropy_mean,
                         "/ max", prm_entropy_max)

                # --- 3. PRM top-1 父 beam 占比（prm_best_idx 选出 beam_size 条后看父 beam 来源）---
                # 先取 PRM 最高分的 beam_size 条的父 beam，统计最高频父 beam 占比
                _, prm_top_idx_tmp = tf.nn.top_k(prm_logits_scaled, k=beam_size)  # [B, beam]
                selected_parent = tf.gather(parent_beam, prm_top_idx_tmp, batch_dims=1)  # [B, beam_size]
                # —— 兼容旧版 TF（无 batch_dims，如 1.12）：用下面 3 行等价实现 ——
                # prm_top_batch_idx = tf.tile(tf.expand_dims(tf.range(B), 1), [1, beam_size])
                # prm_top_gather = tf.stack([prm_top_batch_idx, prm_top_idx_tmp], axis=2)
                # selected_parent = tf.gather_nd(parent_beam, prm_top_gather)
                sel_parent_oh = tf.one_hot(selected_parent, depth=cur_beam, dtype=tf.float32)
                sel_parent_cnt = tf.reduce_sum(sel_parent_oh, axis=1)  # [B, cur_beam]
                top1_parent_ratio = tf.reduce_max(sel_parent_cnt, axis=-1) / tf.cast(beam_size, tf.float32)
                tf.print("beam_diag/prm_top1_parent_ratio_step%d" % step,
                         tf.reduce_mean(top1_parent_ratio))

                # --- 4. 最终 beam 父 beam 来源（与 step0 父 beam 对应的 SID token）---
                # 打印留下的 beam_size 条路径中，unique 父 beam 数量
                # （unique 越少说明退化越严重）
                # ================================================================

                _, prm_best_idx = tf.nn.top_k(prm_logits_scaled, k=beam_size)
                # prm_best_idx 当前按 PRM 分数降序；为保持 PRM「只剪枝不重排」的语义，
                # 把索引按升序排序，恢复成 cand_seqs 的原始顺序（即 decoder 累积概率降序，
                # 因为 cand_seqs 来自 top_k(flat_scores) 的降序输出）。
                # 这样送入下一层的 seqs / dec_path_log_probs / cache 全部保持 decoder 概率序，
                # 末尾无需再用 final_order 重新排序。
                prm_best_idx = tf.sort(prm_best_idx, axis=-1)                      # [B, beam] 升序 = cand_seqs 原序
                # —— 兼容旧版 TF（无 tf.sort，如 1.12）：用下面 4 行等价实现 ——
                # _, prm_asc_idx = tf.nn.top_k(-prm_best_idx, k=beam_size)
                # prm_batch_idx = tf.tile(tf.expand_dims(tf.range(B), 1), [1, beam_size])
                # prm_asc_gather = tf.stack([prm_batch_idx, prm_asc_idx], axis=2)
                # prm_best_idx = tf.gather_nd(prm_best_idx, prm_asc_gather)
                beam_batch_idx = tf.tile(tf.expand_dims(tf.range(B), 1), [1, beam_size])
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
        # === 1-B. 静态特征 LayerNorm（与 QFormer 输出 click_portrait 尺度对齐 ~O(1)）===
        user_static_emb = layer_norm(user_static_emb, scope="static_ln")  # [B, 1, D]

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

        # === 3. 构建行为序列的 padding mask（供 QFormer 使用）===
        B          = tf.shape(used_len)[0]

        click_mask = tf.sequence_mask(
            lengths=used_len,         # [B]
            maxlen=max_len,           # 200
            dtype=tf.int8)            # [B, max_len]

        # debug
        self._print_ops.append(tf.print("seq_mask first sample:", click_mask[0], summarize=100))

        # QFormer src_mask
        click_src_mask = tf.reshape(click_mask, [B, 1, 1, max_len])   # [B, 1, 1, 200]

        # === 4. 行为特征直接送入 QFormer（不做外部 LayerNorm）===
        # 设计调整：与训练路径保持一致，QFormer 内部 RMSNorm + attention 1/sqrt(d_k) 缩放足够

        # === QFormer 压缩行为序列 [B, 200, D] → [B, K, D]（兴趣画像摘要）===
        # 设计调整：click 不做外部 LayerNorm，直接送 QFormer（与训练路径一致）
        with tf.variable_scope("user_qformer", reuse=tf.AUTO_REUSE):
            K = self._qformer_query_num
            qformer_queries = tf.tile(
                tf.reshape(self._qformer_queries, [1, K, self._dim]),
                [batch_size, 1, 1])  # [B, K, D]
            qformer_model = QFormer(
                num_layers=4, dim=self._dim, num_heads=8,
                dropout_rate=0.1, hidden_dim=self._dim * 2, training=False)
            click_portrait = qformer_model.forward(
                qformer_queries, user_click_emb, click_src_mask, training=False)  # [B, K, D]
        # 静态特征（1）+ 画像（K）拼接作为最终 user context
        enc_compressed = tf.concat([user_static_emb, click_portrait], axis=1)   # [B, 1+K, D]
        src_mask_compressed = tf.ones([batch_size, 1, 1, 1 + K], dtype=tf.int8)

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
                dec_in, cur_beam, enc_compressed, src_mask_compressed, cache)  # 只算一步

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
