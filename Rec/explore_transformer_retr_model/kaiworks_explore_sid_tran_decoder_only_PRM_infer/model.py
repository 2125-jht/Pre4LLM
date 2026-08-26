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
            dtype=tf.float16,
            initializer=tf.random_uniform_initializer(minval=-1.0/dim, maxval=1.0/dim), 
            trainable=False
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
        sid_embeddings = tf.nn.embedding_lookup(self._vocab_embedding, photo_sid)  # [B, 3, dim]
        target_embeddings = tf.cumsum(sid_embeddings, axis=1)  # path sum-pool: [sid0, sid0+sid1, sid0+sid1+sid2]
        prm_losses = []
        for step in range(len(self._vocab_sizes)):
            target_embedding = target_embeddings[:, step, :]  # [B, dim]
            pair_target_embedding = tf.tile(tf.expand_dims(target_embedding, axis=0), [batch_size, 1, 1]) #[b,b,dim]
            pair_target_embedding = tf.reshape(pair_target_embedding, [batch_size * batch_size, 1, self._dim]) #[b*b,1,dim]
            pair_encoder_output = tf.tile(tf.expand_dims(encoder_output, axis=1), [1, batch_size, 1, 1]) #[b,b,seq_len,dim]
            pair_encoder_output = tf.reshape(pair_encoder_output, [batch_size * batch_size, tf.shape(encoder_output)[1], self._dim]) #[b*b,seq_len,dim]
            pair_src_mask = tf.tile(tf.expand_dims(src_mask, axis=1), [1, batch_size, 1, 1, 1])
            pair_src_mask = tf.reshape(pair_src_mask, [batch_size * batch_size, 1, 1, tf.shape(encoder_output)[1]])

            prm_logits = prm_model.forward(pair_target_embedding, pair_encoder_output, pair_src_mask, training=True)
            prm_logits = tf.reshape(prm_logits, [batch_size, batch_size])
            prm_label = tf.range(tf.shape(prm_logits)[0], dtype=tf.int32)
            prm_loss_i = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=prm_label, logits=prm_logits)
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
        * 候选 prefix 按训练时的 path sum-pool 构造 PRM target embedding：
          [sid0], [sid0+sid1], [sid0+sid1+sid2]。
        * PRM 重新打分后保留 beam_size 条进入下一步。

        返回：
            gen_part_loc  – shape [B, beam_size, seq_len] 的推荐 sid 局部 id 序列
            probs         – shape [B, beam_size]，整条 path 的 PRM 概率
        """
        # ------------- 常量 & 子模型 -------------
        offsets = [0,
                self._vocab_sizes[0],
                self._vocab_sizes[0] + self._vocab_sizes[1]]       # 局部→全局 id 偏移

        decoder_model = DecoderModel(num_layers=2, dim=self._dim, num_heads=8,
                                    dropout_rate=0.1, hidden_dim=self._dim * 2)
        prm_model = PRMModel(dim=self._dim, num_heads=8, dropout_rate=0.1, training=False)

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
        prm_path_probs = tf.ones([B, 1], dtype=tf.float32)     # PRM 对当前完整 path 的概率
        prm_path_log_probs = tf.zeros([B, 1], dtype=tf.float32)  # log(PRM path prob)，用于乘 decoder 当前 token prob
        prm_path_emb_sums = tf.zeros([B, 1, self._dim], dtype=enc_out_base.dtype)

        cur_beam = 1  # 当前 beam 数
        cache = {}                    # 全层 KV

        prm_candidate_size = beam_size
        candidate_batch_idx = tf.tile(tf.expand_dims(tf.range(B), 1), [1, prm_candidate_size])
        beam_batch_idx = tf.tile(tf.expand_dims(tf.range(B), 1), [1, beam_size])
        prm_k_enc_cache = None
        prm_v_enc_cache = None

        # ---------- ③ 逐层解码 ----------
        for step, V in enumerate(self._vocab_sizes):
            # 只 embed 当前 token
            dec_in = tf.nn.embedding_lookup(self._vocab_embedding, seqs[:, :, -1])  # [B,beam,1,C]
            dec_in = tf.reshape(dec_in, [B*cur_beam, 1, self._dim])

            dec_out, cache = decoder_model.step(
                dec_in, cur_beam, enc_out_base, src_mask, cache)            # 只算一步

            # 训练图里 PRM 参数先于 proj_0 创建。推理图也保持同样 dense 参数顺序，
            # 避免线上 dense bin 按顺序加载时把投影层和 PRM 层权重错位。
            if step == 0:
                prm_model.build_variables()
                prm_k_enc_cache, prm_v_enc_cache = prm_model.build_encoder_cache(enc_out_base)

            last_h = tf.reshape(dec_out, [B, cur_beam, self._dim])
            
            with tf.variable_scope('proj_%d' % step):
                logits = tf.layers.dense(last_h, V, name=f'pred', reuse=tf.AUTO_REUSE,trainable=False)

            # --- 本轮候选：decoder 先取 beam_size * 2 条，再交给 PRM rerank
            k = prm_candidate_size
            scaled_logits = tf.cast(logits, tf.float32) / temperature          # [B, cur_beam, V]
            topk_logits, topk_tok = tf.nn.top_k(scaled_logits, k=k)            # [B, cur_beam, k]
            log_norm = tf.reduce_logsumexp(scaled_logits, axis=-1, keepdims=True)
            topk_logp = topk_logits - log_norm                                 # [B, cur_beam, k]

            # 用“上一段 path 的 PRM 概率 * 当前 token 的 decoder 概率”筛候选池。
            # 这里在 log 空间相加，等价于概率相乘。
            if step == 0:
                best_idx = tf.tile(
                    tf.expand_dims(tf.range(prm_candidate_size, dtype=tf.int32), 0),
                    [B, 1]
                )
            else:
                cand_scores = tf.expand_dims(prm_path_log_probs, -1) + topk_logp    # [B, cur_beam, k]
                # --- decoder 选全局 top-(beam_size * 2) ---
                flat_scores = tf.reshape(cand_scores, [B, -1])                     # [B, cur_beam*k]
                _, best_idx = tf.nn.top_k(flat_scores, k=prm_candidate_size)

            parent_beam = best_idx // k                                        # index in 0..cur_beam‑1
            tok_rank    = best_idx %  k                                        # index in 0..k‑1

            # gather 父路径
            gather_parent = tf.stack([candidate_batch_idx, parent_beam], axis=2)         # [B, cand, 2]
            parent_seq   = tf.gather_nd(seqs,  gather_parent)                  # [B, cand, T]
            parent_prm_path_emb_sum = tf.gather_nd(prm_path_emb_sums, gather_parent)

            def gather_cache(old_cache, gp):
                new_cache = {}
                for k, v in old_cache.items():
                    if k.startswith(("k_self_", "v_self_")):
                        new_cache[k] = tf.gather_nd(v, gp)   # [B, beam, H, T, Dh] → 重新排序
                    else:
                        new_cache[k] = v                     # k_enc / v_enc 原样保留
                return new_cache
            
            # gather 新 token
            tok_gather = tf.stack([candidate_batch_idx, parent_beam, tok_rank], axis=2)
            next_tok   = tf.gather_nd(topk_tok,  tok_gather)                   # [B, cand]
            
            # map 到全局 id
            next_tok_glb = next_tok + offsets[step]

            # decoder 候选 prefix，先不进入下一步，交给 PRM 做二次筛选
            cand_seqs  = tf.concat([parent_seq, tf.expand_dims(next_tok_glb, -1)], axis=-1)  # [B, cand, T+1]

            # 当前候选只包含到本 step 的 prefix，增量维护 sum-pool reward target。
            next_tok_embedding = tf.nn.embedding_lookup(self._vocab_embedding, next_tok_glb)
            prm_target_embedding = parent_prm_path_emb_sum + next_tok_embedding              # [B, cand, C]
            prm_target_embedding = tf.reshape(
                prm_target_embedding,
                [B * prm_candidate_size, 1, self._dim]
            )

            prm_logits = prm_model.forward_with_encoder_cache(
                prm_target_embedding,
                prm_k_enc_cache,
                prm_v_enc_cache,
                src_mask,
                prm_candidate_size,
                training=False
            )
            prm_logits = tf.reshape(prm_logits, [B, prm_candidate_size])

            # --- PRM 选全局 top-beam_size，进入下一层解码 ---
            prm_logits_fp32 = tf.cast(prm_logits, tf.float32)
            prm_top_logits, prm_best_idx = tf.nn.top_k(prm_logits_fp32, k=beam_size)
            prm_log_norm = tf.reduce_logsumexp(prm_logits_fp32, axis=-1, keepdims=True)
            prm_path_log_probs = prm_top_logits - prm_log_norm
            prm_path_probs = tf.exp(prm_path_log_probs)
            gather_prm = tf.stack([beam_batch_idx, prm_best_idx], axis=2)

            seqs  = tf.gather_nd(cand_seqs, gather_prm)                       # [B, beam, T+1]
            prm_path_log_probs = tf.maximum(
                prm_path_log_probs,
                tf.log(tf.constant(1e-12, dtype=tf.float32))
            )     # [B, beam]

            if step < len(self._vocab_sizes) - 1:
                prm_path_emb_sums = tf.gather_nd(tf.reshape(prm_target_embedding, [B, prm_candidate_size, self._dim]), gather_prm)

                # cache 只包含本轮已消费的 parent prefix，需按 PRM 最终留下的父 beam 重排。
                selected_parent_beam = tf.gather_nd(parent_beam, gather_prm)
                gather_parent_after_prm = tf.stack([beam_batch_idx, selected_parent_beam], axis=2)
                cache = gather_cache(cache, gather_parent_after_prm)

                cur_beam = beam_size            # 以后固定

        # 去掉 <START>
        seqs  = seqs[:, :, 1:]
        probs = prm_path_probs

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
        prm_model = PRMModel(dim=self._dim, num_heads=8, dropout_rate=0.1, training=False)

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
        beam_batch_idx = tf.tile(tf.expand_dims(tf.range(B), 1), [1, beam_size])

        # ---------- ③ 逐层解码 ----------
        for step, V in enumerate(self._vocab_sizes):
            # 只 embed 当前 token
            dec_in = tf.nn.embedding_lookup(self._vocab_embedding, seqs[:, :, -1])  # [B,beam,1,C]
            dec_in = tf.reshape(dec_in, [B*cur_beam, 1, self._dim])

            dec_out, cache = decoder_model.step(
                dec_in, cur_beam, enc_out_base, src_mask, cache)            # 只算一步

            # 训练图里 PRM 参数先于 proj_0 创建。no_prm 虽然不用 PRM 计算，
            # 也要按同样顺序建变量，保证 dense 参数顺序和训练一致。
            if step == 0:
                prm_model.build_variables()

            last_h = tf.reshape(dec_out, [B, cur_beam, self._dim])

            with tf.variable_scope('proj_%d' % step):
                logits = tf.layers.dense(last_h, V, name=f'pred', reuse=tf.AUTO_REUSE,trainable=False)

            # --- 本轮候选：decoder-only beam search
            k = beam_size
            scaled_logits = tf.cast(logits, tf.float32) / temperature          # [B, cur_beam, V]
            topk_logits, topk_tok = tf.nn.top_k(scaled_logits, k=k)            # [B, cur_beam, k]
            log_norm = tf.reduce_logsumexp(scaled_logits, axis=-1, keepdims=True)
            topk_logp = topk_logits - log_norm                                 # [B, cur_beam, k]
            topk_prob = tf.exp(topk_logp)

            # 累积得分，并选全局 top-beam_size。
            # step 0 只有一个 parent beam，topk 结果已经是全局最优，跳过一次等价 top_k。
            if step == 0:
                best_idx = tf.tile(
                    tf.expand_dims(tf.range(beam_size, dtype=tf.int32), 0),
                    [B, 1]
                )
                best_scores = tf.squeeze(topk_logp, axis=1)
            else:
                cand_scores = tf.expand_dims(scores, -1) + topk_logp           # [B, cur_beam, k]
                flat_scores = tf.reshape(cand_scores, [B, -1])                 # [B, cur_beam*k]
                best_scores, best_idx = tf.nn.top_k(flat_scores, k=beam_size)

            parent_beam = best_idx // k                                        # index in 0..cur_beam‑1
            tok_rank    = best_idx %  k                                        # index in 0..k‑1

            # gather 父路径
            gather_parent = tf.stack([beam_batch_idx, parent_beam], axis=2)    # [B, cand, 2]
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
            tok_gather = tf.stack([beam_batch_idx, parent_beam, tok_rank], axis=2)
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

            if step < len(self._vocab_sizes) - 1:
                # cache 按 decoder 最终留下的父 beam 重排。
                gather_parent_after_decoder = tf.stack([beam_batch_idx, parent_beam], axis=2)
                cache = gather_cache(cache, gather_parent_after_decoder)

                cur_beam = beam_size            # 以后固定

        # 去掉 <START>
        seqs  = seqs[:, :, 1:]
        probs = tf.exp(scores)

        # 转回局部 id
        offsets_t = tf.constant(offsets, dtype=seqs.dtype)          # [3]
        gen_part_loc = seqs - offsets_t

        return gen_part_loc, probs
