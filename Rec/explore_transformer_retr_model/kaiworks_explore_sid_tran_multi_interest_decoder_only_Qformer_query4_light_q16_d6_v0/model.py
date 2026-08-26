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
        self._query_token_numb=4
        
        # 创建统一的嵌入矩阵，包含所有语义ID的嵌入向量
        # 使用均匀分布初始化，范围为[-1/dim, 1/dim]
        self._embedding = tf.get_variable(
            shape=[self._total_vocab_size+self._query_token_numb, dim], 
            name='embedding',
            initializer=tf.random_uniform_initializer(minval=-1.0/dim, maxval=1.0/dim), 
            trainable=True
        )
        
        self._dim = dim

    def _clip_or_pad_lengths(self, raw_len, batch_size, max_len):
        max_len_i = tf.constant(max_len, dtype=tf.int32)
        valid_len = tf.reshape(tf.cast(raw_len, tf.int32), [-1])
        valid_len = tf.minimum(tf.maximum(valid_len, 0), max_len_i)
        valid_len = valid_len[:batch_size]
        pad_num = tf.maximum(batch_size - tf.size(valid_len), 0)
        return tf.concat([valid_len, tf.fill([pad_num], max_len_i)], axis=0)

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
        
        # 2.2 统计左对齐点击序列的有效长度，超过 max_len 时截断到 max_len
        max_len  = 200                                             
        raw_len  = tf.cast(
            self._feature_emb_size_dict['user_profile_v1_click_pid_list'],
            tf.int32)                 # 可能是 [B,1] 也可能是 [B]
        valid_len = tf.reshape(raw_len, [-1])      # 强制展平成 [B]
        print_tensor("valid_len", valid_len)
        
        max_len_i  = tf.constant(max_len, dtype=tf.int32)           # 200
        used_len   = tf.minimum(valid_len, max_len_i)               # [B] 小于 200 保留原值，>=200 置 200
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

        # ① 为点击序列生成左对齐的 0/1 mask：左侧有效=1，右侧 padding=0
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

        encoder_output = encoder_input
                
        # photo sid embedding
        sid_embedding = tf.nn.embedding_lookup(self._embedding, photo_sid) #[b,code_size,dim]
        preference_embedding = tf.stop_gradient(sid_embedding[:,0,:]) #[b,dim]


        # === 5 构建query token ===
        query_token_indice_1d = tf.range(start=self._total_vocab_size, limit=self._total_vocab_size + self._query_token_numb, delta=1, dtype=tf.int32)#[q]
        query_token_indice_2d = tf.expand_dims(query_token_indice_1d, axis=0)#[1,q]
        query_token_indice = tf.tile(query_token_indice_2d, multiples=[batch_size, 1])#[b,q]
        coarse_interest_input = tf.nn.embedding_lookup(self._embedding, query_token_indice)#[b,q,dim]

        #计算input query token之间的相似度
        print_tensor('query_token_sim/input', calc_sim_cos_btd(coarse_interest_input))
        
                
        # === 6. 粗粒度解码器 ===
        # 使用4层Transformer解码器生成序列表示
        coarse_interest_model = QFormer(num_layers=12, dim=self._dim, num_heads=8, dropout_rate=0.1, hidden_dim=self._dim*2, training=True)
        coarse_interest = coarse_interest_model.forward(coarse_interest_input, encoder_output, src_mask, training=True) # [b, q, dim]
        
        #计算output query token之间的相似度
        print_tensor('query_token_sim/output', calc_sim_cos_btd(coarse_interest))




        #构造fine_item_input的输入，
        coarse_expand = tf.expand_dims(coarse_interest, axis=2)  # [b, q, 1, dim]
        sid_tiled = tf.tile(tf.expand_dims(sid_embedding[:, :-1, :], axis=1), multiples=[1, self._query_token_numb, 1, 1])  # [b, q, code_size-1, dim]
        fine_item_input = tf.concat([coarse_expand, sid_tiled], axis=2)  # [b, q, code_size, dim]

        fine_item_outputs = []


        for i in range(self._query_token_numb):
            fine_item_input_i = fine_item_input[:, i, :, :]  # [b, code_size, dim]


            fine_item_model = DecoderModel(num_layers=4, dim=self._dim, num_heads=8, dropout_rate=0.1, hidden_dim=self._dim*2, training=True)
            out_i = fine_item_model.forward(fine_item_input_i, encoder_output, src_mask, training=True) # [b, code_size, dim]
            fine_item_outputs.append(out_i)

        fine_item_output = tf.stack(fine_item_outputs, axis=1) #[b, q, code_size, dim]
        
        
        #计算偏好向量和模型输出的coarse_interest之间的余弦相似度找出余弦相似度
        coarse_norm = tf.nn.l2_normalize(coarse_interest, axis=-1)        # [b, q, dim]
        pref_norm   = tf.nn.l2_normalize(preference_embedding, axis=-1)   # [b, dim]
        pref_norm   = tf.expand_dims(pref_norm, axis=1)                   # [b, 1, dim]
        cos_sim = tf.reduce_sum(coarse_norm * pref_norm, axis=-1)         # [b, q]
        best_head = tf.argmax(cos_sim, axis=1, output_type=tf.int32)      # [b]


        #取出余弦相似度最大对应的fine_item_output
        gather_idx = tf.stack([tf.range(B, dtype=tf.int32), best_head], axis=1)#[b,2]
        decoder_output = tf.gather_nd(fine_item_output, gather_idx) #[b, code_size, dim]

        
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

                # print_tensor("probs/pred_prob_%d" % step, pred_prob)

                # 2. 取出正确 label 的概率
                #    先拼出索引对 (batch_idx, label_id)
                batch_idx = tf.range(tf.shape(pred_prob)[0], dtype=tf.int32)
                indices   = tf.stack([batch_idx, label[:, step]], axis=1)  # [B, 2]
                correct_p = tf.gather_nd(pred_prob, indices)               # [B]
                # 3. 打印
                print_tensor("probs/correct_token_prob_%d" % step, tf.reduce_sum(correct_p * loss_mask) / (tf.reduce_sum(loss_mask) + 1e-9))
                result_dict['truth%d_probs' % step] = correct_p

                greater = tf.cast(pred_prob > tf.expand_dims(correct_p, 1),tf.float32)  # [B,V]
                correct_token_rank = 1 + tf.reduce_sum(greater, axis=1)      # [B], 1=top1
                trim_rank = masked_trimmed_mean(correct_token_rank, loss_mask, trim_ratio=0.05,name="trim_rank_%d" % step)
                print_tensor("probs/correct_token_rank_%d" % step,trim_rank)

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

    def _hungarian_max_assignment(self, scores, row_num, col_num):
        B = tf.shape(scores)[0]
        inf = tf.constant(1e9, dtype=tf.float32)
        cost = -scores

        def gather_col(values, col_idx):
            batch_idx = tf.range(tf.shape(values)[0], dtype=tf.int32)
            return tf.gather_nd(values, tf.stack([batch_idx, col_idx], axis=1))

        def set_col(values, col_idx, col_value):
            depth = tf.shape(values)[1]
            mask = tf.one_hot(col_idx, depth=depth, dtype=values.dtype)
            return values * (1 - mask) + tf.expand_dims(col_value, 1) * mask

        u = tf.zeros([B, row_num + 1], dtype=tf.float32)
        v = tf.zeros([B, col_num + 1], dtype=tf.float32)
        p = tf.zeros([B, col_num + 1], dtype=tf.int32)
        way = tf.zeros([B, col_num + 1], dtype=tf.int32)

        for row in range(1, row_num + 1):
            p = tf.concat([tf.fill([B, 1], tf.constant(row, dtype=tf.int32)), p[:, 1:]], axis=1)
            minv = tf.ones([B, col_num + 1], dtype=tf.float32) * inf
            used = tf.zeros([B, col_num + 1], dtype=tf.bool)
            j0 = tf.zeros([B], dtype=tf.int32)

            def search_cond(j0, minv, used, way, u, v, p, loop_idx):
                del minv, used, way, u, v
                return tf.logical_and(
                    tf.less(loop_idx, col_num + 1),
                    tf.reduce_any(tf.not_equal(gather_col(p, j0), 0)))

            def search_body(j0, minv, used, way, u, v, p, loop_idx):
                active = tf.not_equal(gather_col(p, j0), 0)
                active_all = tf.tile(tf.expand_dims(active, 1), [1, col_num + 1])
                active_cols = active_all[:, 1:]
                used_j0 = tf.cast(tf.one_hot(j0, depth=col_num + 1, dtype=tf.int32), tf.bool)
                used = tf.logical_or(used, tf.logical_and(used_j0, active_all))

                i0 = gather_col(p, j0)
                row_idx = tf.maximum(i0 - 1, 0)
                batch_idx = tf.range(B, dtype=tf.int32)
                cost_i = tf.gather_nd(cost, tf.stack([batch_idx, row_idx], axis=1))
                u_i = gather_col(u, i0)

                cur = cost_i - tf.tile(tf.expand_dims(u_i, 1), [1, col_num]) - v[:, 1:]
                searchable = tf.logical_and(tf.logical_not(used[:, 1:]), active_cols)
                better = tf.logical_and(searchable, tf.less(cur, minv[:, 1:]))
                way_from = tf.tile(tf.expand_dims(j0, 1), [1, col_num])
                minv_tail = tf.where(better, cur, minv[:, 1:])
                way_tail = tf.where(better, way_from, way[:, 1:])
                minv = tf.concat([minv[:, :1], minv_tail], axis=1)
                way = tf.concat([way[:, :1], way_tail], axis=1)

                delta_candidates = tf.where(searchable, minv_tail, tf.ones_like(minv_tail) * inf)
                delta = tf.reduce_min(delta_candidates, axis=1)
                delta = tf.where(active, delta, tf.zeros_like(delta))
                j1 = tf.argmin(delta_candidates, axis=1, output_type=tf.int32) + 1
                j1 = tf.where(active, j1, j0)

                used_active = tf.logical_and(used, active_all)
                used_active_float = tf.cast(used_active, tf.float32)
                delta_used = used_active_float * tf.tile(tf.expand_dims(delta, 1), [1, col_num + 1])
                u_add = tf.reduce_sum(
                    tf.one_hot(p, depth=row_num + 1, dtype=tf.float32) * tf.expand_dims(delta_used, -1),
                    axis=1)
                u = u + u_add
                v = v - delta_used

                unused_active_cols = tf.logical_and(tf.logical_not(used[:, 1:]), active_cols)
                minv_tail = tf.where(
                    unused_active_cols,
                    minv[:, 1:] - tf.tile(tf.expand_dims(delta, 1), [1, col_num]),
                    minv[:, 1:])
                minv = tf.concat([minv[:, :1], minv_tail], axis=1)

                return j1, minv, used, way, u, v, p, loop_idx + 1

            j0, minv, used, way, u, v, p, _ = tf.while_loop(
                search_cond,
                search_body,
                [j0, minv, used, way, u, v, p, tf.constant(0, dtype=tf.int32)],
                shape_invariants=[
                    tf.TensorShape([None]),
                    tf.TensorShape([None, col_num + 1]),
                    tf.TensorShape([None, col_num + 1]),
                    tf.TensorShape([None, col_num + 1]),
                    tf.TensorShape([None, row_num + 1]),
                    tf.TensorShape([None, col_num + 1]),
                    tf.TensorShape([None, col_num + 1]),
                    tf.TensorShape([])])

            def augment_cond(j0, p, loop_idx):
                return tf.logical_and(tf.less(loop_idx, col_num + 1), tf.reduce_any(tf.not_equal(j0, 0)))

            def augment_body(j0, p, loop_idx):
                active = tf.not_equal(j0, 0)
                j1 = gather_col(way, j0)
                p_j1 = gather_col(p, j1)
                p_j0 = gather_col(p, j0)
                p = set_col(p, j0, tf.where(active, p_j1, p_j0))
                j0 = tf.where(active, j1, j0)
                return j0, p, loop_idx + 1

            _, p, _ = tf.while_loop(
                augment_cond,
                augment_body,
                [j0, p, tf.constant(0, dtype=tf.int32)],
                shape_invariants=[
                    tf.TensorShape([None]),
                    tf.TensorShape([None, col_num + 1]),
                    tf.TensorShape([])])

        assigned_cols = []
        matched_rows = p[:, 1:]
        for row in range(1, row_num + 1):
            assigned_cols.append(tf.argmax(tf.cast(tf.equal(matched_rows, row), tf.int32), axis=1, output_type=tf.int32))
        return tf.stack(assigned_cols, axis=1)



    def _match_top_coarse_interest(self, coarse_interest, user_sid0, top_n=None):
        if top_n is None:
            top_n = user_sid0.get_shape().as_list()[1]
            if top_n is None:
                raise ValueError("user_sid0 second dimension must be statically known")
        top_n = min(top_n, self._query_token_numb)
        user_sid0 = tf.cast(user_sid0[:, :top_n], tf.int32)
        valid_sid0 = tf.greater(user_sid0, 0)
        user_sid0 = tf.minimum(tf.maximum(user_sid0, 0), self._vocab_sizes[0] - 1)
        sid0_embedding = tf.nn.embedding_lookup(self._embedding, user_sid0)  # [B, top_n, dim]
        sid0_norm = tf.nn.l2_normalize(sid0_embedding, axis=-1)
        coarse_norm = tf.nn.l2_normalize(coarse_interest, axis=-1)
        sim = tf.matmul(sid0_norm, coarse_norm, transpose_b=True)  # [B, top_n, Q]
        valid_sid0_mask = tf.tile(tf.expand_dims(valid_sid0, -1), [1, 1, self._query_token_numb])
        sim = tf.where(valid_sid0_mask, sim, tf.ones_like(sim) * -1e6)

        return self._hungarian_max_assignment(sim, top_n, self._query_token_numb)

    def beam_search_fast_share_head(self, user_sid0, beam_size=16, temperature=1, hot_beam_size=128):
        """
        O(batch·beam·logV) 近似复杂度的束搜索（显存与 beam_size 线性）

        改进版本：
        * **step=0** 仅用 1 条 beam，从 |V_0| 里直接选 top‑k 形成不同路径，
        避免所有 beam 被同一起点锁死。
        * step>0 时保持固定 beam_size。

        返回：
            gen_part_loc  – shape [B, Q*beam_size, seq_len] 的推荐 sid 局部 id 序列
            probs         – 同形状，逐 token 的 softmax 概率（便于做温度/多样性分析）
        """
        # ------------- 常量 & 子模型 -------------
        offsets = [0,
                self._vocab_sizes[0],
                self._vocab_sizes[0] + self._vocab_sizes[1]]       # 局部→全局 id 偏移
        

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
        
        # 2.2 统计左对齐点击序列的有效长度，超过 max_len 时截断到 max_len
        max_len  = 200                                             
        raw_len  = tf.cast(
            self._feature_emb_size_dict['user_profile_v1_click_pid_list'],
            tf.int32)                 # 可能是 [B,1] 也可能是 [B]
        used_len = self._clip_or_pad_lengths(raw_len, batch_size, max_len)

        # 通过MLP处理点击特征
        user_click_emb = mlp('user_click_emb', user_click_fea, [4*self._dim], self._dim, activation=tf.nn.leaky_relu)
        
        # === 3. 构建编码器输入 ===
        # 将静态特征和点击行为特征拼接作为编码器输入
        encoder_input = tf.concat([user_static_emb, user_click_emb], axis=1)
        
        # === 3-A. 构建 Encoder/Decoder 的 padding mask =============================
        # 整个序列长度 = 1（user token）+ max_len（点击序列）
        total_len  = 1 + max_len                       # int, e.g. 6 when max_len=5
        B          = batch_size                       # batch_size 动态

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
        
        enc_out_base = encoder_input  # [B, L_enc, C]

        # === 5 构建query token ===
        query_token_indice_1d = tf.range(start=self._total_vocab_size, limit=self._total_vocab_size + self._query_token_numb, delta=1, dtype=tf.int32)#[q]
        query_token_indice_2d = tf.expand_dims(query_token_indice_1d, axis=0)#[1,q]
        query_token_indice = tf.tile(query_token_indice_2d, multiples=[batch_size, 1])#[b,q]
        coarse_interest_input = tf.nn.embedding_lookup(self._embedding, query_token_indice)#[b,q,dim]
                 
        # === 6. 粗粒度解码器 ===
        # 使用4层Transformer解码器生成序列表示
        coarse_interest_model = QFormer(num_layers=12, dim=self._dim, num_heads=8, dropout_rate=0.1, hidden_dim=self._dim*2)
        coarse_interest = coarse_interest_model.forward(coarse_interest_input, enc_out_base, src_mask, training=False) # [b, q, dim]

        def beam_search_group(query_token_embed, group_head_num, group_beam_size,
                              forced_first_tokens=None, forced_first_valid=None):
            # ========== 0) 把 decoder 维折叠进 batch：B*Q ==========
            query_token_embed_all = tf.reshape(query_token_embed, [B * group_head_num, -1]) # [B*q, dim],实现并行推理
            ## enc_out_base: [B, Lenc, D] -> [B*Q, Lenc, D]（每个 decoder复用同一个 enc_out）
            Lenc = tf.shape(enc_out_base)[1]
            enc_out_all = tf.reshape(tf.tile(tf.expand_dims(enc_out_base, 1), [1, group_head_num, 1, 1]),[B * group_head_num, Lenc, -1])
            src_mask_all = tf.reshape(tf.tile(tf.expand_dims(src_mask, 1), [1, group_head_num, 1, 1, 1]),[B * group_head_num, 1, 1, -1])

            # ---------- ② Beam 状态初始化 ----------
            BQ = B * group_head_num
            start_tok = tf.fill([BQ, 1], self._total_vocab_size)   # global id of <START> [b,1]
            initial_seqs   = tf.expand_dims(start_tok, 1)                 # [B, 1, 1]  每条路径，第一维b，第二维是beam_size，第三维是输出的seq_len(输出每一个具体的token),
            initial_probs  = tf.ones_like(initial_seqs, dtype=tf.float32)         # [B, 1, 1]  每条路径得分，第一维b，第二维是beam_size，第三维是输出的seq_len(输出每一个token的概率),
            initial_scores = tf.zeros([BQ, 1], dtype=tf.float32)           # [B, 1]  初始得分为0，第一位B，第二维是beam_size(每条路径的概率和)，相当于reduce_mean(probs,axis=-1)

            seqs=initial_seqs
            probs=initial_probs
            scores=initial_scores
            cur_beam = 1  # 当前 beam 数
            cache = {}                    # 全层 KV

            fine_item_model = DecoderModel(num_layers=4, dim=self._dim, num_heads=8, dropout_rate=0.1, hidden_dim=self._dim*2)

            # ---------- ③ 逐层解码 ----------
            for step, V in enumerate(self._vocab_sizes):



                fine_item_input = query_token_embed_all if step==0 else tf.nn.embedding_lookup(self._embedding, seqs[:, :, -1])  # [B,cur_beam,1,dim]
                fine_item_input = tf.reshape(fine_item_input, [BQ*cur_beam, 1, self._dim])


                dec_out, cache = fine_item_model.step(fine_item_input, cur_beam, enc_out_all, src_mask_all, cache) #[b*beam,1,dim]

                last_h = tf.reshape(dec_out, [BQ, cur_beam, self._dim]) #[b,beam,dim]

                with tf.variable_scope('proj_%d' % step):
                    logits = tf.layers.dense(last_h, V, name=f'pred', reuse=tf.AUTO_REUSE)

                logp = tf.nn.log_softmax(logits / temperature)                     # [B, cur_beam, V]

                # --- 本轮候选：parent_beam × top‑V → (cur_beam*V)
                k = group_beam_size                                                # 第 0 步从 |V| 里挑 beam_size
                topk_logp, topk_tok = tf.nn.top_k(logp, k=k)                       # [B, cur_beam, k]

                if step == 0 and forced_first_tokens is not None:
                    logp_first = tf.squeeze(logp, axis=1)                           # [B*Q, V]
                    topk_logp_first = tf.squeeze(topk_logp, axis=1)                 # [B*Q, k]
                    topk_tok_first = tf.squeeze(topk_tok, axis=1)                   # [B*Q, k]

                    forced_tok = tf.reshape(tf.cast(forced_first_tokens, tf.int32), [BQ])
                    forced_tok = tf.minimum(tf.maximum(forced_tok, 0), tf.constant(V - 1, dtype=tf.int32))
                    if forced_first_valid is None:
                        use_forced = tf.ones([BQ], dtype=tf.bool)
                    else:
                        use_forced = tf.reshape(tf.cast(forced_first_valid, tf.bool), [BQ])

                    already_in_beam = tf.reduce_any(
                        tf.equal(topk_tok_first, tf.expand_dims(forced_tok, 1)),
                        axis=1)
                    inject_forced = tf.logical_and(use_forced, tf.logical_not(already_in_beam))
                    forced_logp = tf.gather_nd(logp_first, tf.stack([tf.range(BQ), forced_tok], axis=1))

                    last_tok = tf.where(inject_forced, forced_tok, topk_tok_first[:, -1])
                    last_logp = tf.where(inject_forced, forced_logp, topk_logp_first[:, -1])
                    topk_tok_first = tf.concat([topk_tok_first[:, :-1], tf.expand_dims(last_tok, 1)], axis=1)
                    topk_logp_first = tf.concat([topk_logp_first[:, :-1], tf.expand_dims(last_logp, 1)], axis=1)

                    topk_tok = tf.expand_dims(topk_tok_first, axis=1)
                    topk_logp = tf.expand_dims(topk_logp_first, axis=1)

                topk_prob = tf.exp(topk_logp)                                      #下一个token的预测分数
                cand_scores = tf.expand_dims(scores, -1) + topk_logp               # [B, cur_beam, k] ，总分数

                # --- 选全局 top‑beam_size ---
                flat_scores = tf.reshape(cand_scores, [BQ, -1])                    # [B, cur_beam*k]
                best_scores, best_idx = tf.nn.top_k(flat_scores, k=group_beam_size)  # 取新的 beam

                parent_beam = best_idx // k                                        # index in 0..cur_beam‑1
                tok_rank    = best_idx %  k                                        # index in 0..k‑1

                batch_idx = tf.tile(tf.expand_dims(tf.range(BQ), 1), [1, group_beam_size]) #batch中的每条取beam size个

                # gather 父路径
                gather_parent = tf.stack([batch_idx, parent_beam], axis=2)         # [B, beam, 2]
                parent_seq   = tf.gather_nd(seqs,  gather_parent)                  # [B, beam, T]
                parent_prob  = tf.gather_nd(probs, gather_parent)

                def gather_cache(old_cache, gp):
                    new_cache = {}
                    for kk, vv in old_cache.items():
                        if kk.startswith(("k_self_", "v_self_")):
                            new_cache[kk] = tf.gather_nd(vv, gp)   # [B, beam, H, T, Dh] → 重新排序
                        else:
                            new_cache[kk] = vv                   # k_enc / v_enc 原样保留
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
                probs = tf.concat([parent_prob, tf.expand_dims(next_prob,  -1)], axis=-1)   #
                scores = best_scores                                                # [B, beam]

                cur_beam = group_beam_size            # 以后固定


            # 去掉 <START>
            seqs  = seqs[:, :, 1:]  #[b,beam,seq]
            probs = probs[:, :, 1:]  #[b,beam,seq]

            # 转回局部 id
            offsets_t = tf.constant(offsets, dtype=seqs.dtype)          # [3]
            gen_part_loc = seqs - offsets_t

            all_seqs = tf.reshape(gen_part_loc, [B, group_head_num * group_beam_size, -1])   # [B, Q*beam, Seq]
            all_probs = tf.reshape(probs, [B, group_head_num * group_beam_size, -1])          # [B, Q*beam, Seq]
            all_scores = tf.reshape(scores, [B, group_head_num * group_beam_size])            # [B, Q*beam]
            return all_seqs, all_probs, all_scores

        hot_head_num = user_sid0.get_shape().as_list()[1]
        if hot_head_num is None:
            raise ValueError("user_sid0 second dimension must be statically known")
        hot_head_num = min(hot_head_num, self._query_token_numb)
        if hot_head_num <= 0:
            raise ValueError("user_sid0 second dimension must be greater than 0")
        cold_head_num = self._query_token_numb - hot_head_num
        hot_sid0 = tf.cast(user_sid0[:, :hot_head_num], tf.int32)
        hot_sid0_valid = tf.greater(hot_sid0, 0)
        hot_sid0 = tf.minimum(tf.maximum(hot_sid0, 0), self._vocab_sizes[0] - 1)
        hot_query_indices = self._match_top_coarse_interest(coarse_interest, user_sid0, top_n=hot_head_num)

        hot_batch_idx = tf.tile(tf.expand_dims(tf.range(B), 1), [1, hot_head_num])
        hot_coarse_interest = tf.gather_nd(coarse_interest, tf.stack([hot_batch_idx, hot_query_indices], axis=2))

        # padding 的 sid0 仍保留静态 hot slot，但 forced_first_valid 会避免强制生成 sid0=0。
        hot_seqs, hot_probs, hot_scores = beam_search_group(
            hot_coarse_interest, hot_head_num, hot_beam_size, hot_sid0, hot_sid0_valid)
        if cold_head_num > 0:
            all_query_indices = tf.tile(tf.expand_dims(tf.range(self._query_token_numb, dtype=tf.int32), 0), [B, 1])
            hot_mask = tf.reduce_any(tf.equal(tf.expand_dims(all_query_indices, -1), tf.expand_dims(hot_query_indices, 1)), axis=-1)
            cold_scores = tf.where(hot_mask, tf.ones_like(all_query_indices) * -1, all_query_indices)
            _, cold_query_indices = tf.nn.top_k(cold_scores, k=cold_head_num)
            cold_batch_idx = tf.tile(tf.expand_dims(tf.range(B), 1), [1, cold_head_num])
            cold_coarse_interest = tf.gather_nd(coarse_interest, tf.stack([cold_batch_idx, cold_query_indices], axis=2))
            cold_seqs, cold_probs, cold_scores = beam_search_group(cold_coarse_interest, cold_head_num, beam_size)
            all_seqs = tf.concat([hot_seqs, cold_seqs], axis=1)
            all_probs = tf.concat([hot_probs, cold_probs], axis=1)
            all_scores = tf.concat([hot_scores, cold_scores], axis=1)
        else:
            all_seqs = hot_seqs
            all_probs = hot_probs
            all_scores = hot_scores
        output_beam_size = hot_head_num * hot_beam_size + cold_head_num * beam_size


        K = output_beam_size                 # K = Q*beam
        sorted_scores, sorted_idx = tf.nn.top_k(all_scores, k=K)  # desc, [B,K]

        # 用 sorted_idx 重排 seqs/probs
        batch_idx = tf.tile(tf.expand_dims(tf.range(B), 1), [1, K])             # [B,K]
        gather_2d = tf.stack([batch_idx, sorted_idx], axis=2)                   # [B,K,2]

        all_seqs  = tf.gather_nd(all_seqs,  gather_2d)   # [B,K,seq]
        all_probs = tf.gather_nd(all_probs, gather_2d)   # [B,K,seq]
        all_scores = sorted_scores                        # [B,K]

        
        return all_seqs,all_probs
