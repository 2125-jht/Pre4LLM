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
        
    def model(self, click_semantic_id_list, photo_sid, label, photo_semantic_id_int):
        """
        主训练模型前向传播
        
        Args:
            click_semantic_id_list:用户点击的视频SID int列表 [batch_size, 200] int64
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
        # 拼接原始点击特征
        user_click_fea = tf.concat(
            [self._feature_emb_dict[fea] for fea in user_click_fea_names], axis=2)   # 最新->最旧 [B, L, dim]

        # 打印有效长度                                       
        raw_len  = tf.cast(self._feature_emb_size_dict['user_profile_v1_click_pid_list'], tf.int32) # 可能是 [B,1] 也可能是 [B]
        valid_len = tf.reshape(raw_len, [-1])      # 强制展平成 [B]
        print_tensor("valid_len", valid_len)
        max_len = 200
        used_len = valid_len

        # 调整序列
        user_click_sid = tf.reverse(click_semantic_id_list, axis=[1]) # reverse 使得 最旧→最新
        user_click_fea = tf.reverse(user_click_fea, axis=[1]) # reverse 使得 最旧→最新

        # 通过MLP处理点击特征
        user_click_emb = mlp('user_click_emb', user_click_fea, [4*self._dim], self._dim, activation=tf.nn.leaky_relu)
        
        # MLP 结果屏蔽
        click_mask_left = tf.sequence_mask(lengths=used_len, maxlen=max_len, dtype=tf.int8)  # [B,L]  左对齐
        click_mask      = tf.reverse(click_mask_left, axis=[1])                              # [B,L]  与 user_click_emb 对齐
        m = tf.cast(click_mask, tf.float32)[..., None]     # [B,L,1]
        user_click_emb = user_click_emb * m
        
        # debug
        self._print_ops.append(tf.print("user_click_sid first sample:", user_click_sid[0], summarize=100))
        self._print_ops.append(tf.print("user_click_emb first sample:", user_click_emb[0,:,1], summarize=100))
        
        # === 3. 构建编码器输入 ===
        # 将静态特征和点击行为特征拼接作为编码器输入
        encoder_input = tf.concat([user_static_emb, user_click_emb], axis=1)
        
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
        
        # === 3-A. 构建 Encoder/Decoder 的 padding mask =============================
        # 整个序列长度 = 1（user token）+ max_len（点击序列）
        total_len  = 1 + max_len                       # int, e.g. 6 when max_len=5
        B          = tf.shape(used_len)[0]            # batch_size 动态

        # 有效位（与 encoder_input 对齐： user 静态 token + 反转后的 click 序列）
        click_mask_left = tf.sequence_mask(lengths=used_len, maxlen=max_len, dtype=tf.int8)  # [B,L]  左对齐
        click_mask      = tf.reverse(click_mask_left, axis=[1])                              # [B,L]  与 user_click_emb 对齐
        user_tok   = tf.ones([B, 1], dtype=tf.int8)                                          # [B,1]
        # === 所有视频都可见的 mask（给第1级用） ===
        seq_mask_all = tf.concat([user_tok, click_mask], axis=1)                             # [B,S]

        # === 依据第1级 SID 进行“同类选择”的 mask（给第2/3级用） ===
        # 提取点击序列中每个视频的第1级 SID（与 user_click_sid 对齐：最旧->最新）
        lvl1_click = tf.bitwise.bitwise_and(tf.bitwise.right_shift(user_click_sid, 30), 0x7FFF) # [B,L] int64
        lvl1_click = tf.cast(lvl1_click, tf.int32)
        
        # 训练：用 GT 的第1级 label 进行筛选（teacher-forcing）
        target_lvl1 = tf.cast(label[:, 0], tf.int32)                                         # [B]
        same_lvl1   = tf.equal(lvl1_click, target_lvl1[:, None])                             # [B,L] bool
        same_lvl1   = tf.cast(same_lvl1, tf.int8)

        # ① 标记哪些历史位的 sid_int 有效（非 0）
        valid_sid_mask = tf.cast(tf.not_equal(user_click_sid, 0), tf.int8)  # [B,L] 1=有效, 0=缺失

        # ② 组合：必须同时满足「该位有效」∧「与第1级同类」∧「在有效长度内」
        select_click_mask = tf.bitwise.bitwise_and(click_mask, valid_sid_mask)   # [B,L]
        select_click_mask = tf.bitwise.bitwise_and(select_click_mask, same_lvl1) # [B,L]
        
        # ③ 拼成第2/3级用的cross attention mask
        seq_mask_sel = tf.concat([user_tok, select_click_mask], axis=1)          # [B,S]
        
        # === 组装按解码步位的跨注意力 mask: [B,1,Tq,S] ===
        Tq = tf.shape(decoder_input)[1]                                                      # 解码输入序列长度 (= 3, BOS S0 S1)
        rest_len = tf.maximum(Tq - 1, 0)

        # t=0（用于预测第1级）：用 seq_mask_all
        pos0  = tf.expand_dims(seq_mask_all, axis=1)                                         # [B,1,S]
        # t>=1（用于预测第2/3级）：用 seq_mask_sel
        rest  = tf.tile(tf.expand_dims(seq_mask_sel, axis=1), [1, rest_len, 1])              # [B,Tq-1,S]

        seq_mask_per_t = tf.concat([pos0, rest], axis=1)                                     # [B,Tq,S]
        src_mask = tf.expand_dims(seq_mask_per_t, axis=1) 

        # debug
        print_tensor("valid_sid_mask",
                     tf.reduce_sum(tf.cast(valid_sid_mask, tf.int32), axis=1))
        print_tensor("select_click_mask",
                     tf.reduce_sum(tf.cast(select_click_mask, tf.int32), axis=1))

        # debug
        self._print_ops.append(tf.print("lvl1_click[0]:", lvl1_click[0], summarize=100))
        self._print_ops.append(tf.print("label[0]:", label[0], summarize=100))
        self._print_ops.append(tf.print("mask/valid_sid_mask[0]:", valid_sid_mask[0], summarize=100))
        self._print_ops.append(tf.print("mask/seq_mask_all[0]:", seq_mask_all[0], summarize=100))
        self._print_ops.append(tf.print("mask/seq_mask_sel[0]:", seq_mask_sel[0], summarize=100))
        self._print_ops.append(tf.print("mask/num_selected_clicks:", 
                                        tf.reduce_sum(tf.cast(select_click_mask, tf.int32), axis=1)))
        # === 5. no enc ===
        encoder_output = layer_norm(encoder_input, scope="enc_ln")
        # encoder_output = encoder_input
        
        # 计算余弦相似度（用于调试）
        encoder_output_sim = tf.reshape(encoder_output, [batch_size, -1])
        print_tensor("encoder_output_sim", calc_sim_cos(encoder_output_sim))
        
        # 序列总掩码（含 user 静态 token）
        seq_mask_all_f = tf.cast(seq_mask_all, tf.float32)[..., None]  # [B,S,1]

        enc_in_masked  = encoder_input  * seq_mask_all_f
        enc_out_masked = encoder_output * seq_mask_all_f

        # 有效 token 加权平均得到每条样本的全局表征
        denom   = tf.reduce_sum(seq_mask_all_f, axis=1) + 1e-6         # [B,1]
        enc_in_pool  = tf.reduce_sum(enc_in_masked,  axis=1) / denom   # [B,D]
        enc_out_pool = tf.reduce_sum(enc_out_masked, axis=1) / denom   # [B,D]

        print_tensor("encoder_input_sim(masked-pooled)",  calc_sim_cos(enc_in_pool))
        print_tensor("encoder_output_sim(masked-pooled)", calc_sim_cos(enc_out_pool))
        
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
                #    先拼出索引对 (batch_idx, label_id)
                batch_idx = tf.range(tf.shape(pred_prob)[0], dtype=tf.int32)
                indices   = tf.stack([batch_idx, label[:, step]], axis=1)  # [B, 2]
                true_p = tf.gather_nd(pred_prob, indices)               # [B]

                # 3. 打印
                print_tensor("probs/true_token_prob_%d" % step, tf.reduce_sum(true_p * loss_mask) / (tf.reduce_sum(loss_mask) + 1e-9))
                max_probs, _ = tf.nn.top_k(pred_prob, k=1)
                print_tensor("probs/max_token_prob_%d" % step, tf.reduce_sum(tf.squeeze(max_probs, -1) * loss_mask) / (tf.reduce_sum(loss_mask) + 1e-9))

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
        return loss

    def beam_search_fast(self, click_semantic_id_list, beam_size=512, temperature=1):
        """
        O(batch·beam·logV) 近似复杂度的束搜索（显存与 beam_size 线性）
        * step=0：全部历史可见
        * step>=1：只允许 attend 到「与第1级预测同类 & sid!=0」的视频位
        """
        offsets = [0,
                self._vocab_sizes[0],
                self._vocab_sizes[0] + self._vocab_sizes[1]]  # 局部→全局 id 偏移

        decoder_model = DecoderModel(num_layers=4, dim=self._dim, num_heads=8,
                                    dropout_rate=0.1, hidden_dim=self._dim * 2)

        # ---------- ① 编码用户 ----------
        # 1) 静态特征
        user_static_fea = tf.concat([self._feature_emb_dict[fea] for fea in user_static_fea_names], axis=1)
        user_static_emb = mlp('user_static_emb', user_static_fea, [2*self._dim], self._dim, activation=tf.nn.leaky_relu)
        B = tf.shape(user_static_emb)[0]
        user_static_emb = tf.reshape(user_static_emb, [B, 1, self._dim])  # [B,1,C]

        # 2) 点击特征（与 train 对齐：反转为 最旧→最新）
        user_click_fea = tf.concat(
            [self._feature_emb_dict[fea] for fea in user_click_fea_names], axis=2)   # [B,L,C]
        max_len  = 200
        raw_len  = tf.cast(self._feature_emb_size_dict['user_profile_v1_click_pid_list'], tf.int32)
        valid_len = tf.reshape(raw_len, [-1])                      # [B]
        used_len  = tf.minimum(valid_len, tf.constant(max_len, tf.int32))
        
        # 反转序列方向对齐 train
        user_click_fea = tf.reverse(user_click_fea, axis=[1])                    # [B,L,C] 最旧→最新
        user_click_sid = tf.reverse(click_semantic_id_list, axis=[1])  # [B,L]

        # MLP 投影
        user_click_emb = mlp('user_click_emb', user_click_fea, [4*self._dim], self._dim, activation=tf.nn.leaky_relu)

        # encoder 输入
        encoder_input = tf.concat([user_static_emb, user_click_emb], axis=1)     # [B,S,C], S=1+L
        enc_out_base  = layer_norm(encoder_input, scope="enc_ln")                 # [B,S,C]
        # enc_out_base = encoder_input

        # ---------- ② 基础 mask（与 train 对齐） ----------
        # 点击位的有效性（长度）
        click_mask_left = tf.sequence_mask(lengths=used_len, maxlen=max_len, dtype=tf.int8)  # [B,L] 左对齐
        click_mask      = tf.reverse(click_mask_left, axis=[1])                               # [B,L] 与 emb 对齐
        user_tok        = tf.ones([B, 1], dtype=tf.int8)                                      # [B,1]
        seq_mask_all    = tf.concat([user_tok, click_mask], axis=1)   

        # level-1 of 历史点击（与 train 同位宽：高 15bit）
        lvl1_click = user_click_sid
        lvl1_click = tf.cast(lvl1_click, tf.int32)

        # sid 是否有效（非 0）
        valid_sid_mask = tf.cast(tf.not_equal(user_click_sid, 0), tf.int8)        # [B,L]

        # ---------- ③ Beam 状态 ----------
        start_tok = tf.fill([B, 1], self._total_vocab_size)   # <START> 的全局 id
        seqs   = tf.expand_dims(start_tok, 1)                 # [B,1,1]
        probs  = tf.ones_like(seqs, dtype=tf.float32)         # [B,1,1]
        scores = tf.zeros([B, 1], dtype=tf.float32)           # [B,1]
        cur_beam = 1
        cache = {}

        # 小工具：把 [B,*,S] 的 0/1 mask 展平到 [B*beam,1,1,S]
        def expand_src_mask(mask_BxBeamxS, beam_count):
            # mask_BxBeamxS: [B,beam,S] 或 [B,1,S]
            if beam_count == 1 and tf.shape(mask_BxBeamxS)[1] == 1:
                m = tf.reshape(mask_BxBeamxS, [B, 1, tf.shape(mask_BxBeamxS)[2]])   # [B,1,S]
            else:
                m = mask_BxBeamxS  # [B,beam,S]
            m = tf.reshape(m, [B*beam_count, 1, 1, tf.shape(m)[-1]])                # [B*beam,1,1,S]
            return m
            
        # ---------- ④ 逐级解码 ----------
        for step, V in enumerate(self._vocab_sizes):
            # ---- 构造本 step 的 cross-attention mask ----
            if step == 0:
                # t=0：全量视频可见
                # 先加一个 beam 维，便于统一处理
                mask_b = tf.expand_dims(seq_mask_all, axis=1)           # [B,1,S]
                src_mask_step = expand_src_mask(mask_b, cur_beam)       # [B*beam,1,1,S]
            else:
                # t>=1：依据每条 beam 的第1级预测构造“同类选择” mask
                # 取每条 beam 的第1级局部 id：seqs 现为 [B,beam,T]，T>=2（含<START>和第1级）
                lvl1_loc_beam = seqs[:, :, 1] - offsets[0]              # [B,beam]

                # equal: [B,beam,L] 与 lvl1_click [B,L] 比较（广播到 beam）
                same_lvl1_b = tf.equal(lvl1_click[:, None, :], lvl1_loc_beam[:, :, None])   # [B,beam,L] bool
                same_lvl1_b = tf.cast(same_lvl1_b, tf.int8)

                # 三条件：长度有效 ∧ sid!=0 ∧ 同类
                click_mask_b = tf.tile(click_mask[:, None, :], [1, cur_beam, 1])            # [B,beam,L]
                valid_sid_b  = tf.tile(valid_sid_mask[:, None, :], [1, cur_beam, 1])        # [B,beam,L]

                select_click_b = tf.bitwise.bitwise_and(click_mask_b, valid_sid_b)
                select_click_b = tf.bitwise.bitwise_and(select_click_b, same_lvl1_b)        # [B,beam,L]

                # 拼到带 user token 的 S 维
                user_tok_b = tf.ones([B, cur_beam, 1], dtype=tf.int8)                       # [B,beam,1]
                seq_mask_sel_b = tf.concat([user_tok_b, select_click_b], axis=2)            # [B,beam,S]

                src_mask_step = expand_src_mask(seq_mask_sel_b, cur_beam)                   # [B*beam,1,1,S]

            # ---- 单步前向（维持与原实现一致） ----
            dec_in = tf.nn.embedding_lookup(self._vocab_embedding, seqs[:, :, -1])          # [B,beam,1,C]
            dec_in = tf.reshape(dec_in, [B*cur_beam, 1, self._dim])

            dec_out, cache = decoder_model.step(dec_in, cur_beam, enc_out_base, src_mask_step, cache)
            last_h = tf.reshape(dec_out, [B, cur_beam, self._dim])

            with tf.variable_scope('proj_%d' % step):
                logits = tf.layers.dense(last_h, V, name='pred', reuse=tf.AUTO_REUSE)       # [B,beam,V]

            logp = tf.nn.log_softmax(logits / temperature)                                   # [B,beam,V]
            k = beam_size if step == 0 else beam_size                          # 第 0 步从 |V| 里挑 beam_size
            topk_logp, topk_tok = tf.nn.top_k(logp, k=k)                                     # [B,beam,k]
            topk_prob = tf.exp(topk_logp)

            # 累积分数并选全局 top-beam
            cand_scores = tf.expand_dims(scores, -1) + topk_logp                             # [B,beam,k]
            flat_scores = tf.reshape(cand_scores, [B, -1])                                   # [B, beam*k]
            best_scores, best_idx = tf.nn.top_k(flat_scores, k=beam_size)                    # [B,beam]

            parent_beam = best_idx // k
            tok_rank    = best_idx %  k
            batch_idx   = tf.tile(tf.expand_dims(tf.range(B), 1), [1, beam_size])

            # 重新排序 beam（seq/prob/cache）
            gather_parent = tf.stack([batch_idx, parent_beam], axis=2)                       # [B,beam,2]
            parent_seq   = tf.gather_nd(seqs,  gather_parent)                                # [B,beam,T]
            parent_prob  = tf.gather_nd(probs, gather_parent)

            def gather_cache(old_cache, gp):
                new_cache = {}
                for kname, val in old_cache.items():
                    if kname.startswith(("k_self_", "v_self_")):
                        new_cache[kname] = tf.gather_nd(val, gp)  # [B,beam,H,T,Dh]
                    else:
                        new_cache[kname] = val                    # enc KV 不依 beam 变化
                return new_cache
            cache = gather_cache(cache, gather_parent)

            tok_gather = tf.stack([batch_idx, parent_beam, tok_rank], axis=2)
            next_tok   = tf.gather_nd(topk_tok,  tok_gather)                                  # [B,beam] (局部 id)
            next_prob  = tf.gather_nd(topk_prob, tok_gather)                                  # [B,beam]

            # 映射到全局 id & 更新路径
            next_tok_glb = next_tok + offsets[step]
            seqs  = tf.concat([parent_seq, tf.expand_dims(next_tok_glb, -1)], axis=-1)       # [B,beam,T+1]
            probs = tf.concat([parent_prob, tf.expand_dims(next_prob,  -1)], axis=-1)
            scores = best_scores
            cur_beam = beam_size  # 以后固定

        # 去掉 <START>，转回局部 id
        seqs  = seqs[:, :, 1:]
        probs = probs[:, :, 1:]
        gen_part_loc = seqs - tf.constant(offsets, dtype=seqs.dtype)  # 广播减

        return gen_part_loc, probs
