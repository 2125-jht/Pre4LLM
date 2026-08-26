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
        # 创建损失掩码，只对有效的语义ID计算损失
        loss_mask = tf.where(
            photo_semantic_id_int > 0,  
            tf.ones_like(photo_semantic_id_int, dtype=tf.float32), 
            tf.zeros_like(photo_semantic_id_int, dtype=tf.float32)
        )
        loss_mask = tf.reshape(loss_mask, [-1])

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
        prm_losses = []
        for step in range(len(self._vocab_sizes)):
            # 路径 -> 单 token 表示：embedding_lookup + reduce_sum (sum-pool)
            # 每层 SID 通过全局 offset 位于互不相交的区间，sum 天然保留位置信息、无合法路径碰撞
            # 无参数投影层，梯度经 reduce_sum → embedding_lookup 回传至 _vocab_embedding（trainable=True）
            prefix_emb = sid_embeddings[:, :step + 1, :]                     # [B, step+1, dim]
            target_embedding = tf.reduce_sum(prefix_emb, axis=1)             # [B, dim]
            # pair_target_embedding 构造：将 batch 内所有 target embedding 两两配对
            pair_target_embedding = tf.tile(tf.expand_dims(target_embedding, axis=0), [batch_size, 1, 1]) #[b,b,dim]
            pair_target_embedding = tf.reshape(pair_target_embedding, [batch_size * batch_size, 1, self._dim]) #[b*b,1,dim]
            pair_encoder_output = tf.tile(tf.expand_dims(encoder_output, axis=1), [1, batch_size, 1, 1]) #[b,b,seq_len,dim]
            pair_encoder_output = tf.reshape(pair_encoder_output, [batch_size * batch_size, tf.shape(encoder_output)[1], self._dim]) #[b*b,seq_len,dim]
            pair_src_mask = tf.tile(tf.expand_dims(src_mask, axis=1), [1, batch_size, 1, 1, 1])
            pair_src_mask = tf.reshape(pair_src_mask, [batch_size * batch_size, 1, 1, tf.shape(encoder_output)[1]])

            prm_logits = prm_model.forward(pair_target_embedding, pair_encoder_output, pair_src_mask, training=True)
            prm_logits = tf.reshape(prm_logits, [batch_size, batch_size])

            # === logQ correction for in-batch negative sampling ===
            # in-batch 负采样下，路径 j 被采为负样本的概率 ≈ freq(j in batch) / B
            # corrected_logit(i, j) = logit(i, j) - log Q(j)，对所有 logit（含正样本对角线）做修正。
            # 参考: Yi et al. "Sampling-Bias-Corrected Neural Modeling for Retrievals" (WSDM 2019)
            prm_logits_for_loss = prm_logits
            if use_logq_correction:
                # 多项式 hash 把路径前缀 [s_0, ..., s_b] 映射到唯一 int64（无碰撞）
                # base = total_vocab_size + 1，3 层时 max_hash ≈ (24577)^3 ≈ 1.5e13 << int64 上限
                prefix_tokens = tf.cast(photo_sid[:, :step + 1], tf.int64)       # [B, step+1]
                base = tf.constant(self._total_vocab_size + 1, dtype=tf.int64)
                path_hash = tf.zeros([batch_size], dtype=tf.int64)
                for k in range(step + 1):
                    path_hash = path_hash * base + prefix_tokens[:, k]
                # 统计 batch 内频率
                _, idx, count = tf.unique_with_counts(path_hash)                 # [U], [B], [U]
                freq = tf.gather(count, idx)                                     # [B]
                logQ = tf.math.log(
                    tf.cast(freq, tf.float32) / tf.cast(batch_size, tf.float32))  # [B]
                print_tensor("prm/logQ_%d" % step, logQ)
                print_tensor("prm/path_freq_%d" % step, tf.cast(freq, tf.float32))
                # 减去 log Q(j)（按列广播：row i 的所有候选 j 都减 logQ[j]）
                prm_logits_for_loss = prm_logits - tf.reshape(logQ, [1, -1])

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
        loss = ntp_loss + prm_loss
        print_tensor("loss/ntp_loss", ntp_loss)
        print_tensor("loss/prm_loss", prm_loss)
        print_tensor("loss/total_loss", loss)
        return loss, result_dict

    def beam_search_fast(self, beam_size=512, temperature=1):
        """
        Decoder beam search + PRM rerank：
        * 每个 step 先用 decoder beam search 保留 beam_size * 2 条候选 prefix。
        * 第一层（step=0）不做 PRM 打分，直接用 decoder 累积概率选 top beam_size。
        * 第二层起，候选 prefix 按训练时的 path sum-pool 构造 PRM target embedding：
          [sid0], [sid0+sid1], [sid0+sid1+sid2]。
        * PRM 重新打分后保留 beam_size 条进入下一步。

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

        # PRM 大张量延迟构建：仅在 step==1（真正需要 PRM 打分时）才创建，
        # 避免第一层额外占用 B * beam_size*2 的显存
        prm_encoder_output = None
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
                # --- 延迟构建 PRM 大张量（仅 step==1 时执行一次）---
                # 这两个张量 shape 为 [B*beam_size*2, ...]，非常耗显存，
                # 第一层不需要，推迟到真正使用 PRM 时才构建
                if prm_encoder_output is None:
                    prm_enc_len = tf.shape(enc_out_base)[1]
                    # encoder 输出 tile 到每个候选：[B, enc_len, dim] -> [B*cand, enc_len, dim]
                    prm_encoder_output = tf.tile(
                        tf.expand_dims(enc_out_base, axis=1),
                        [1, prm_candidate_size, 1, 1]
                    )
                    prm_encoder_output = tf.reshape(
                        prm_encoder_output,
                        [B * prm_candidate_size, prm_enc_len, self._dim]
                    )
                    # encoder src_mask 同样 tile 到每个候选
                    prm_src_mask = tf.tile(
                        tf.expand_dims(src_mask, axis=1),
                        [1, prm_candidate_size, 1, 1, 1]
                    )
                    prm_src_mask = tf.reshape(
                        prm_src_mask,
                        [B * prm_candidate_size, 1, 1, prm_enc_len]
                    )

                # --- 第一阶段：decoder 选 top-(beam_size*2) 候选 ---
                k = prm_candidate_size
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

                prm_logits = prm_model.forward(
                    prm_target_embedding,
                    prm_encoder_output,
                    prm_src_mask,
                    training=False
                )
                prm_logits = tf.reshape(prm_logits, [B, prm_candidate_size])

                # 第二阶段：PRM 只做剪枝，不参与最终排序
                # 从 beam_size*2 条 decoder 候选中挑出 PRM 打分最高的 beam_size 条
                # 对 PRM logits 除以 temperature，与训练侧 softmax 前的缩放保持一致
                prm_logits_scaled = prm_logits / prm_temperature
                _, prm_best_idx = tf.nn.top_k(prm_logits_scaled, k=beam_size)
                beam_batch_idx = tf.tile(tf.expand_dims(tf.range(B), 1), [1, beam_size])
                gather_prm = tf.stack([beam_batch_idx, prm_best_idx], axis=2)  # [B, beam, 2]

                seqs  = tf.gather_nd(cand_seqs, gather_prm)                       # [B, beam, T+1]
                # 留下来的 beam 继承其原本的 decoder 累积 log 概率，PRM 分数不进入排序基准
                dec_path_log_probs = tf.gather_nd(cand_dec_log_probs, gather_prm) # [B, beam]

                # cache 只包含本轮已消费的 parent prefix，需按 PRM 最终留下的父 beam 重排。
                selected_parent_beam = tf.gather_nd(parent_beam, gather_prm)
                gather_parent_after_prm = tf.stack([beam_batch_idx, selected_parent_beam], axis=2)
                cache = gather_cache(cache, gather_parent_after_prm)

            cur_beam = beam_size            # 以后固定

        # 最终按 decoder 累积 log 概率降序排列，确保输出顺序按 decoder 概率排序
        final_order = tf.nn.top_k(dec_path_log_probs, k=beam_size).indices     # [B, beam]
        order_batch_idx = tf.tile(tf.expand_dims(tf.range(B), 1), [1, beam_size])
        gather_final = tf.stack([order_batch_idx, final_order], axis=2)
        seqs               = tf.gather_nd(seqs, gather_final)
        dec_path_log_probs = tf.gather_nd(dec_path_log_probs, gather_final)

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


# =====================================================================
# Lazy Decoder 模型
# 核心差异：Cross-Attention 不使用 w_k/w_v 投影，
# 而是使用 Context Processor 预计算的 K/V（所有层共享）。
# 其余（2 层 decoder、操作顺序、归一化）与 MultiInterestModel 完全一致。
# =====================================================================

class LazyMultiInterestModel(object):
    """
    Lazy Decoder 多兴趣推荐模型
    与 MultiInterestModel 的唯一差异：
    - Cross-Attention 不做 w_k/w_v 投影，直接使用 context LayerNorm 预计算 K/V
    - 所有 decoder 层共享同一组 context K/V（而非每层独立做 w_k/w_v）

    其余完全一致：2 层 decoder、Self→Cross→FFN 顺序、LayerNorm、PRM loss。
    """

    def __init__(self, feature_emb_dict, feature_emb_size_dict, dim=256,
                 vocab_sizes=[8192, 8192, 8192], print_ops=None):
        self._feature_emb_dict = feature_emb_dict
        self._feature_emb_size_dict = feature_emb_size_dict
        self._print_ops = print_ops
        self._vocab_sizes = vocab_sizes
        self._total_vocab_size = sum(self._vocab_sizes)

        # 共享同一个 vocab_embedding（与 MultiInterestModel 一致）
        self._vocab_embedding = tf.get_variable(
            shape=[self._total_vocab_size + 1, dim],
            name='vocab_embedding',
            initializer=tf.random_uniform_initializer(minval=-1.0 / dim, maxval=1.0 / dim),
            trainable=True
        )

        self._dim = dim

    # ------------------------------------------------------------------
    # 共用的特征预处理（与 MultiInterestModel 完全一致）
    # ------------------------------------------------------------------
    def _build_user_features(self):
        """
        Returns:
            encoder_input: [B, total_len, dim]  拼接后的用户特征
            encoder_output: [B, total_len, dim] layer_norm 后的结果（PRM 需要）
            src_mask: [B, 1, 1, total_len]       padding mask
            batch_size: int
            used_len: [B]
        """
        # === 1. 用户静态特征处理 ===
        user_static_fea = tf.concat(
            [self._feature_emb_dict[fea] for fea in user_static_fea_names], axis=1)
        user_static_emb = mlp('user_static_emb', user_static_fea, [2 * self._dim], self._dim,
                              activation=tf.nn.leaky_relu)
        batch_size = tf.shape(user_static_emb)[0]
        user_static_emb = tf.reshape(user_static_emb, [batch_size, 1, self._dim])

        # === 2. 用户点击行为特征处理 ===
        user_click_fea = tf.concat(
            [self._feature_emb_dict[fea] for fea in user_click_fea_names], axis=2)  # [B, L, dim]

        max_len = 200
        raw_len = tf.cast(
            self._feature_emb_size_dict['user_profile_v1_click_pid_list'], tf.int32)
        valid_len = tf.reshape(raw_len, [-1])
        print_tensor("valid_len_lazy", valid_len)

        max_len_i = tf.constant(max_len, dtype=tf.int32)
        used_len = tf.minimum(valid_len, max_len_i)
        print_tensor("used_len_lazy", used_len)

        user_click_emb = mlp('user_click_emb', user_click_fea, [4 * self._dim], self._dim,
                             activation=tf.nn.leaky_relu)

        self._print_ops.append(
            tf.print("lazy user_click_emb first sample:", user_click_emb[0, :, 1], summarize=100))

        # === 3. 构建编码器输入 ===
        encoder_input = tf.concat([user_static_emb, user_click_emb], axis=1)

        encoder_input_sim = tf.reshape(encoder_input, [batch_size, -1])
        print_tensor("lazy_encoder_input_sim", calc_sim_cos(encoder_input_sim))

        # === 3-A. padding mask ===
        total_len = 1 + max_len
        B = tf.shape(used_len)[0]

        click_mask = tf.sequence_mask(lengths=used_len, maxlen=max_len, dtype=tf.int8)
        user_tok = tf.ones([B, 1], dtype=tf.int8)
        seq_mask = tf.concat([user_tok, click_mask], axis=1)

        self._print_ops.append(tf.print("lazy_seq_mask first sample:", seq_mask[0], summarize=100))

        src_mask = tf.reshape(seq_mask, [B, 1, 1, total_len])

        # === LayerNorm（保留，PRM 仍需要） ===
        encoder_output = layer_norm(encoder_input, scope="enc_ln")

        return encoder_input, encoder_output, src_mask, batch_size, used_len

    # ------------------------------------------------------------------
    # 预计算 Context K/V（核心新增）
    # ------------------------------------------------------------------
    def _precompute_context_kv(self, encoder_input, num_heads=8):
        """
        用 LayerNorm 替代每层 cross-attention 的 w_k/w_v 投影。
        所有 decoder 层共享同一组 K/V。

        Args:
            encoder_input: [B, ctx_len, dim]
            num_heads: int
        Returns:
            context_k: [B, H, ctx_len, Dh]
            context_v: [B, H, ctx_len, Dh]  (S_kv=1: k=v 共享)
        """
        def split_heads_ctx(x, num_heads):
            B = tf.shape(x)[0]
            L = tf.shape(x)[1]
            d = x.get_shape().as_list()[-1] // num_heads
            return tf.transpose(tf.reshape(x, [B, L, num_heads, d]), [0, 2, 1, 3])

        # 核心差异：用全局 LayerNorm 替代每层的 w_k/w_v 投影
        context_kv = layer_norm(encoder_input, scope="context_k_ln")
        context_k = split_heads_ctx(context_kv, num_heads)  # [B, H, ctx_len, Dh]
        context_v = context_k  # S_kv=1: K=V 共享

        return context_k, context_v

    # ------------------------------------------------------------------
    # 训练前向传播
    # ------------------------------------------------------------------
    def model(self, photo_sid, label, photo_semantic_id_int):
        """
        与 MultiInterestModel.model() 的差异：
        1. 预计算 context K/V（替代每层 cross-attn 的 w_k/w_v 投影）
        2. 使用 LazyDecoderModel 替代 DecoderModel
        3. 其余（特征预处理、loss 计算、PRM）完全一致
        """
        print_tensor("lazy_user_click_list_length",
                     self._feature_emb_size_dict['user_profile_v1_click_pid_list'])

        # === 1~3. 特征预处理（与 MultiInterestModel 完全一致）===
        encoder_input, encoder_output, src_mask, batch_size, used_len = \
            self._build_user_features()

        # === 4. 构建解码器输入（与 MultiInterestModel 完全一致）===
        start_token_indice = tf.tile(
            tf.constant(self._total_vocab_size, shape=(1, 1), dtype=tf.int32),
            [batch_size, 1])
        photo_with_start_token = tf.concat([start_token_indice, photo_sid], axis=1)
        decoder_input = tf.nn.embedding_lookup(self._vocab_embedding, photo_with_start_token)

        # === 5. 预计算 Context K/V（核心新增）===
        num_heads = 8
        context_k, context_v = self._precompute_context_kv(encoder_input, num_heads)

        # === 6. Lazy Decoder（替代原 DecoderModel）===
        decoder_model = LazyDecoderModel(
            num_layers=2, dim=self._dim, num_heads=num_heads,
            dropout_rate=0.1, hidden_dim=self._dim * 2, training=True)
        decoder_output = decoder_model.forward(
            decoder_input, context_k, context_v, src_mask, training=True)

        # debug: decoder 各步输出相似度
        for i in range(len(self._vocab_sizes)):
            similarity = calc_sim_cos(decoder_output[:, i, :])
            print_tensor('lazy_decoder_sim/decoder_output_%d' % i, similarity)

        # === 7. 损失计算（与 MultiInterestModel 完全一致）===
        losses = []
        loss_mask = tf.where(
            photo_semantic_id_int > 0,
            tf.ones_like(photo_semantic_id_int, dtype=tf.float32),
            tf.zeros_like(photo_semantic_id_int, dtype=tf.float32))
        loss_mask = tf.reshape(loss_mask, [-1])

        # === 7-A. PRM loss ===
        prm_model = PRMModel(dim=self._dim, num_heads=8, dropout_rate=0.1, training=True)
        sid_embeddings = tf.nn.embedding_lookup(self._vocab_embedding, photo_sid)
        prm_temperature = 1
        use_logq_correction = True
        prm_losses = []
        for step in range(len(self._vocab_sizes)):
            prefix_emb = sid_embeddings[:, :step + 1, :]
            target_embedding = tf.reduce_sum(prefix_emb, axis=1)
            pair_target_embedding = tf.tile(
                tf.expand_dims(target_embedding, axis=0), [batch_size, 1, 1])
            pair_target_embedding = tf.reshape(
                pair_target_embedding, [batch_size * batch_size, 1, self._dim])
            pair_encoder_output = tf.tile(
                tf.expand_dims(encoder_output, axis=1), [1, batch_size, 1, 1])
            pair_encoder_output = tf.reshape(
                pair_encoder_output, [batch_size * batch_size,
                                      tf.shape(encoder_output)[1], self._dim])
            pair_src_mask = tf.tile(
                tf.expand_dims(src_mask, axis=1), [1, batch_size, 1, 1, 1])
            pair_src_mask = tf.reshape(
                pair_src_mask, [batch_size * batch_size, 1, 1,
                                tf.shape(encoder_output)[1]])

            prm_logits = prm_model.forward(
                pair_target_embedding, pair_encoder_output, pair_src_mask, training=True)
            prm_logits = tf.reshape(prm_logits, [batch_size, batch_size])

            prm_logits_for_loss = prm_logits
            if use_logq_correction:
                prefix_tokens = tf.cast(photo_sid[:, :step + 1], tf.int64)
                base = tf.constant(self._total_vocab_size + 1, dtype=tf.int64)
                path_hash = tf.zeros([batch_size], dtype=tf.int64)
                for k in range(step + 1):
                    path_hash = path_hash * base + prefix_tokens[:, k]
                _, idx, count = tf.unique_with_counts(path_hash)
                freq = tf.gather(count, idx)
                logQ = tf.math.log(
                    tf.cast(freq, tf.float32) / tf.cast(batch_size, tf.float32))
                print_tensor("lazy_prm/logQ_%d" % step, logQ)
                print_tensor("lazy_prm/path_freq_%d" % step, tf.cast(freq, tf.float32))
                prm_logits_for_loss = prm_logits - tf.reshape(logQ, [1, -1])

            prm_label = tf.range(tf.shape(prm_logits)[0], dtype=tf.int32)
            prm_loss_i = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=prm_label, logits=prm_logits_for_loss / prm_temperature)
            prm_losses.append(prm_loss_i)
            print_tensor("lazy_loss/prm_loss_%d" % step,
                         tf.reduce_sum(prm_loss_i * loss_mask) / (tf.reduce_sum(loss_mask) + 1e-9))
            print_tensor("lazy_prm/pos_score_%d" % step,
                         tf.reduce_sum(tf.linalg.diag_part(prm_logits) * loss_mask) /
                         (tf.reduce_sum(loss_mask) + 1e-9))

        result_dict = {}

        for step in range(len(self._vocab_sizes)):
            with tf.variable_scope('proj_%d' % step):
                pred_logit = tf.layers.dense(
                    decoder_output[:, step, :], self._vocab_sizes[step], name='pred')
                print_tensor("lazy_logits/pred_logit_%d" % step, pred_logit)

                temperature = 1
                pred_prob = tf.nn.softmax(pred_logit / temperature, axis=-1)

                batch_idx = tf.range(tf.shape(pred_prob)[0], dtype=tf.int32)
                indices = tf.stack([batch_idx, label[:, step]], axis=1)
                correct_p = tf.gather_nd(pred_prob, indices)
                print_tensor("lazy_probs/correct_token_prob_%d" % step,
                             tf.reduce_sum(correct_p * loss_mask) /
                             (tf.reduce_sum(loss_mask) + 1e-9))
                result_dict['truth%d_probs' % step] = correct_p

                max_probs, _ = tf.nn.top_k(pred_prob, k=1)
                print_tensor("lazy_probs/max_token_prob_%d" % step,
                             tf.reduce_sum(tf.squeeze(max_probs, -1) * loss_mask) /
                             (tf.reduce_sum(loss_mask) + 1e-9))

                max_16_probs, max_16_indices = tf.nn.top_k(pred_prob, k=16, sorted=True)
                result_dict["sid%d_probs" % step] = max_16_probs
                result_dict["sid%d_indices" % step] = max_16_indices

                one_hot_labels = tf.one_hot(label[:, step], self._vocab_sizes[step])
                loss_i = tf.nn.softmax_cross_entropy_with_logits(
                    labels=one_hot_labels, logits=pred_logit / temperature)
                losses.append(loss_i)

                print_tensor("lazy_loss/loss_%d" % step,
                             tf.reduce_sum(loss_i * loss_mask) /
                             (tf.reduce_sum(loss_mask) + 1e-9))
                recall_at_k(pred_logit, label[:, step], loss_mask, self._print_ops,
                            top_k=[1, 16, 128], name="lazy_predict_recall_%d" % step)

        print_tensor("lazy_loss_mask", loss_mask)
        ntp_loss = tf.reduce_sum(tf.add_n(losses) * loss_mask) / \
                   (tf.reduce_sum(loss_mask) + 1e-9)
        prm_loss = tf.reduce_sum(tf.add_n(prm_losses) * loss_mask) / \
                   (tf.reduce_sum(loss_mask) + 1e-9)
        loss = ntp_loss + prm_loss
        print_tensor("lazy_loss/ntp_loss", ntp_loss)
        print_tensor("lazy_loss/prm_loss", prm_loss)
        print_tensor("lazy_loss/total_loss", loss)
        return loss, result_dict

    # ------------------------------------------------------------------
    # Beam Search 推理（Step 2）
    # ------------------------------------------------------------------
    def beam_search_lazy(self, beam_size=512, temperature=1):
        """
        Lazy Decoder beam search + PRM rerank

        与 beam_search_fast 的核心差异：
        - Context K/V 预计算一次，所有 decoder 层共享
        - Cache 中不再有 per-layer k_enc/v_enc，改为全局 k_context/v_context
        - PRM 仍使用 encoder_output（layer_norm(encoder_input, "enc_ln")）
        """
        offsets = [0,
                  self._vocab_sizes[0],
                  self._vocab_sizes[0] + self._vocab_sizes[1]]

        decoder_model = LazyDecoderModel(
            num_layers=2, dim=self._dim, num_heads=8,
            dropout_rate=0.1, hidden_dim=self._dim * 2)
        prm_model = PRMModel(dim=self._dim, num_heads=8, dropout_rate=0.1, training=False)

        # ---------- ① 预处理：编码用户 ----------
        encoder_input, encoder_output, src_mask, B, used_len = \
            self._build_user_features()

        # ---------- ①.5 预计算 Context K/V（核心新增）----------
        num_heads = 8
        context_k, context_v = self._precompute_context_kv(encoder_input, num_heads)

        # 存入 cache 格式：[B, 1, H, ctx_len, Dh]（dim-1 给 beam broadcast 用）
        cache = {}
        cache["k_context"] = context_k[:, None]   # [B, 1, H, ctx_len, Dh]
        cache["v_context"] = context_v[:, None]

        # ---------- ② Beam 状态初始化 ----------
        start_tok = tf.fill([B, 1], self._total_vocab_size)
        seqs = tf.expand_dims(start_tok, 1)   # [B, 1, 1]
        dec_path_log_probs = tf.zeros([B, 1], dtype=tf.float32)

        cur_beam = 1

        prm_candidate_size = beam_size * 2
        prm_temperature = 1

        prm_encoder_output = None
        prm_src_mask = None

        # ---------- Cache 辅助函数 ----------
        def gather_cache_lazy(old_cache, gp):
            """按 beam 选择索引重排 self-attention KV；
               context K/V 全局共享，原样保留。"""
            new_cache = {}
            for ck, v in old_cache.items():
                if ck.startswith(("k_self_", "v_self_")):
                    new_cache[ck] = tf.gather_nd(v, gp)
                # k_context / v_context 原样保留
                else:
                    new_cache[ck] = v
            return new_cache

        def tile_cache_for_first_step_lazy(old_cache, num_beams):
            """第一层从 1 beam 扩展到 num_beams beams；
               self-attention KV tile，context KV 原样保留。"""
            new_cache = {}
            for ck, v in old_cache.items():
                if ck.startswith(("k_self_", "v_self_")):
                    new_cache[ck] = tf.tile(v, [1, num_beams, 1, 1, 1])
                else:
                    new_cache[ck] = v  # k_context / v_context 原样保留
            return new_cache

        # ---------- ③ 逐层解码 ----------
        for step, V in enumerate(self._vocab_sizes):
            dec_in = tf.nn.embedding_lookup(self._vocab_embedding, seqs[:, :, -1])
            dec_in = tf.reshape(dec_in, [B * cur_beam, 1, self._dim])

            # ✅ 核心差异：传 context_k/v 替代 enc_output
            dec_out, cache = decoder_model.step(
                dec_in, cur_beam,
                context_k=cache["k_context"],
                context_v=cache["v_context"],
                context_mask=src_mask,
                cache=cache)

            if step == 0:
                prm_model.build_variables()

            last_h = tf.reshape(dec_out, [B, cur_beam, self._dim])

            with tf.variable_scope('proj_%d' % step):
                logits = tf.layers.dense(last_h, V, name=f'pred', reuse=tf.AUTO_REUSE)

            logp = tf.nn.log_softmax(logits / temperature)   # [B, cur_beam, V]

            # ==============================================================
            #  第一层（step==0）：不做 PRM，decoder 直接选 top beam_size
            # ==============================================================
            if step == 0:
                topk_logp, topk_tok = tf.nn.top_k(logp[:, 0, :], k=beam_size)

                dec_path_log_probs = topk_logp

                start_tiled = tf.tile(seqs, [1, beam_size, 1])
                next_tok_glb = topk_tok + offsets[0]
                seqs = tf.concat(
                    [start_tiled, tf.expand_dims(next_tok_glb, -1)], axis=-1)

                cache = tile_cache_for_first_step_lazy(cache, beam_size)

            # ==============================================================
            #  后续层（step>=1）：decoder 选 beam_size*2 候选 → PRM 剪枝
            # ==============================================================
            else:
                if prm_encoder_output is None:
                    prm_enc_len = tf.shape(encoder_output)[1]
                    prm_encoder_output = tf.tile(
                        tf.expand_dims(encoder_output, axis=1),
                        [1, prm_candidate_size, 1, 1])
                    prm_encoder_output = tf.reshape(
                        prm_encoder_output,
                        [B * prm_candidate_size, prm_enc_len, self._dim])
                    prm_src_mask = tf.tile(
                        tf.expand_dims(src_mask, axis=1),
                        [1, prm_candidate_size, 1, 1, 1])
                    prm_src_mask = tf.reshape(
                        prm_src_mask,
                        [B * prm_candidate_size, 1, 1, prm_enc_len])

                k = prm_candidate_size
                topk_logp, topk_tok = tf.nn.top_k(logp, k=k)

                cand_scores = tf.expand_dims(dec_path_log_probs, -1) + topk_logp

                flat_scores = tf.reshape(cand_scores, [B, -1])
                cand_dec_log_probs, best_idx = tf.nn.top_k(
                    flat_scores, k=prm_candidate_size)

                parent_beam = best_idx // k
                tok_rank = best_idx % k

                batch_idx = tf.tile(
                    tf.expand_dims(tf.range(B), 1), [1, prm_candidate_size])

                gather_parent = tf.stack([batch_idx, parent_beam], axis=2)
                parent_seq = tf.gather_nd(seqs, gather_parent)

                tok_gather = tf.stack([batch_idx, parent_beam, tok_rank], axis=2)
                next_tok = tf.gather_nd(topk_tok, tok_gather)
                next_tok_glb = next_tok + offsets[step]

                cand_seqs = tf.concat(
                    [parent_seq, tf.expand_dims(next_tok_glb, -1)], axis=-1)

                # PRM 打分
                cand_sid_embeddings = tf.nn.embedding_lookup(
                    self._vocab_embedding, cand_seqs[:, :, 1:])
                prm_target_embedding = tf.reduce_sum(cand_sid_embeddings, axis=2)
                prm_target_embedding = tf.reshape(
                    prm_target_embedding,
                    [B * prm_candidate_size, 1, self._dim])

                prm_logits = prm_model.forward(
                    prm_target_embedding, prm_encoder_output, prm_src_mask,
                    training=False)
                prm_logits = tf.reshape(prm_logits, [B, prm_candidate_size])

                prm_logits_scaled = prm_logits / prm_temperature
                _, prm_best_idx = tf.nn.top_k(prm_logits_scaled, k=beam_size)
                beam_batch_idx = tf.tile(
                    tf.expand_dims(tf.range(B), 1), [1, beam_size])
                gather_prm = tf.stack([beam_batch_idx, prm_best_idx], axis=2)

                seqs = tf.gather_nd(cand_seqs, gather_prm)
                dec_path_log_probs = tf.gather_nd(cand_dec_log_probs, gather_prm)

                selected_parent_beam = tf.gather_nd(parent_beam, gather_prm)
                gather_parent_after_prm = tf.stack(
                    [beam_batch_idx, selected_parent_beam], axis=2)
                cache = gather_cache_lazy(cache, gather_parent_after_prm)

            cur_beam = beam_size

        # 最终排序
        final_order = tf.nn.top_k(dec_path_log_probs, k=beam_size).indices
        order_batch_idx = tf.tile(tf.expand_dims(tf.range(B), 1), [1, beam_size])
        gather_final = tf.stack([order_batch_idx, final_order], axis=2)
        seqs = tf.gather_nd(seqs, gather_final)
        dec_path_log_probs = tf.gather_nd(dec_path_log_probs, gather_final)

        seqs = seqs[:, :, 1:]  # 去掉 <START>
        probs = tf.exp(dec_path_log_probs)

        offsets_t = tf.constant(offsets, dtype=seqs.dtype)
        gen_part_loc = seqs - offsets_t

        return gen_part_loc, probs

    def beam_search_lazy_no_prm(self, beam_size=512, temperature=1):
        """
        Lazy Decoder beam search（无 PRM rerank）

        与 beam_search_fast_no_prm 的核心差异：
        - Context K/V 预计算一次，所有层共享
        - Cache 中用 k_context/v_context 替代 per-layer k_enc/v_enc
        """
        offsets = [0,
                  self._vocab_sizes[0],
                  self._vocab_sizes[0] + self._vocab_sizes[1]]

        decoder_model = LazyDecoderModel(
            num_layers=2, dim=self._dim, num_heads=8,
            dropout_rate=0.1, hidden_dim=self._dim * 2)
        prm_model = PRMModel(dim=self._dim, num_heads=8, dropout_rate=0.1, training=False)

        # ---------- ① 预处理 ----------
        encoder_input, encoder_output, src_mask, B, used_len = \
            self._build_user_features()

        # ---------- ①.5 预计算 Context K/V ----------
        num_heads = 8
        context_k, context_v = self._precompute_context_kv(encoder_input, num_heads)

        cache = {}
        cache["k_context"] = context_k[:, None]
        cache["v_context"] = context_v[:, None]

        # ---------- ② Beam 初始化 ----------
        start_tok = tf.fill([B, 1], self._total_vocab_size)
        seqs = tf.expand_dims(start_tok, 1)
        probs = tf.ones_like(seqs, dtype=tf.float32)
        scores = tf.zeros([B, 1], dtype=tf.float32)

        cur_beam = 1

        # ---------- ③ 逐层解码 ----------
        for step, V in enumerate(self._vocab_sizes):
            dec_in = tf.nn.embedding_lookup(self._vocab_embedding, seqs[:, :, -1])
            dec_in = tf.reshape(dec_in, [B * cur_beam, 1, self._dim])

            dec_out, cache = decoder_model.step(
                dec_in, cur_beam,
                context_k=cache["k_context"],
                context_v=cache["v_context"],
                context_mask=src_mask,
                cache=cache)

            if step == 0:
                prm_model.build_variables()

            last_h = tf.reshape(dec_out, [B, cur_beam, self._dim])

            with tf.variable_scope('proj_%d' % step):
                logits = tf.layers.dense(last_h, V, name=f'pred', reuse=tf.AUTO_REUSE)

            logp = tf.nn.log_softmax(logits / temperature)
            topk_logp, topk_tok = tf.nn.top_k(logp, k=beam_size)
            topk_prob = tf.exp(topk_logp)

            cand_scores = tf.expand_dims(scores, -1) + topk_logp

            flat_scores = tf.reshape(cand_scores, [B, -1])
            best_scores, best_idx = tf.nn.top_k(flat_scores, k=beam_size)

            parent_beam = best_idx // beam_size
            tok_rank = best_idx % beam_size

            batch_idx = tf.tile(tf.expand_dims(tf.range(B), 1), [1, beam_size])

            gather_parent = tf.stack([batch_idx, parent_beam], axis=2)
            parent_seq = tf.gather_nd(seqs, gather_parent)
            parent_prob = tf.gather_nd(probs, gather_parent)

            def gather_cache_lazy_no_prm(old_cache, gp):
                new_cache = {}
                for ck, v in old_cache.items():
                    if ck.startswith(("k_self_", "v_self_")):
                        new_cache[ck] = tf.gather_nd(v, gp)
                    else:
                        new_cache[ck] = v
                return new_cache

            tok_gather = tf.stack([batch_idx, parent_beam, tok_rank], axis=2)
            next_tok = tf.gather_nd(topk_tok, tok_gather)
            next_prob = tf.gather_nd(topk_prob, tok_gather)

            next_tok_glb = next_tok + offsets[step]

            cand_seqs = tf.concat(
                [parent_seq, tf.expand_dims(next_tok_glb, -1)], axis=-1)
            cand_probs = tf.concat(
                [parent_prob, tf.expand_dims(next_prob, -1)], axis=-1)

            seqs = cand_seqs
            probs = cand_probs
            scores = best_scores

            beam_batch_idx = tf.tile(
                tf.expand_dims(tf.range(B), 1), [1, beam_size])
            gather_parent_after_decoder = tf.stack(
                [beam_batch_idx, parent_beam], axis=2)
            cache = gather_cache_lazy_no_prm(cache, gather_parent_after_decoder)

            cur_beam = beam_size

        seqs = seqs[:, :, 1:]
        probs = probs[:, :, 1:]

        offsets_t = tf.constant(offsets, dtype=seqs.dtype)
        gen_part_loc = seqs - offsets_t

        return gen_part_loc, probs