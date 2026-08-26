import tensorflow as tf
from feature_attr_extract import *
from modules_ import *

class StackedTransformerModel():
    def __init__(self, name, num_layers, dim, num_heads, dk, dropout_rate, training=False):
        '''
        dim: query 的维度
        dk: key 投影矩阵的维度
        '''
        super(StackedTransformerModel, self).__init__()
        self.num_layers = num_layers
        self.dim = dim
        self.layers = [EncoderLayer(f"{name}_transformer_layer_{i}", dim, num_heads, dk, dropout_rate) for i in range(num_layers)]
        
    def forward(self, hidden_states, training, causal_mask=False):
        for i in range(self.num_layers):
            hidden_states = self.layers[i].forward(
                hidden_states,
                training=training,
                causal_mask=causal_mask,
            )
        return hidden_states
    
class EvaluatorModel:
    def __init__(self, parameters_dict, print_ops, list_size, candidates_size, list_num,
                 swim_segment_width_ratios, training=True):
        self._parameters_dict = parameters_dict
        self._list_size = list_size
        self._candidates_size = candidates_size
        self._list_num = list_num
        self._swim_segment_width_ratios = list(swim_segment_width_ratios)
        self._swim_num_segments = len(self._swim_segment_width_ratios)
        self._training = training
        self.position_embedding = tf.get_variable(
            name='list_value_position_embedding',
            shape=[self._list_size, 64],
            initializer=tf.random_normal_initializer(stddev=0.02)
        )
        self.segment_position_embedding = tf.get_variable(
            name='swim_segment_position_embedding',
            shape=[self._swim_num_segments, 32],
            initializer=tf.random_normal_initializer(stddev=0.02)
        )
        self.print_ops = print_ops

    def _get_list_features(self, input_dicts) -> tuple:
        """构建完全由 List loss 训练的独立特征底座。"""
        with tf.variable_scope("list_backbone", reuse=tf.AUTO_REUSE):
            user_embs     = tf.concat([input_dicts[k] for k in input_dicts if k in user_fea_names], axis=-1)
            user_embs     = tf.tile(tf.expand_dims(user_embs, axis=1), [1,self._candidates_size,1]) if self._training else user_embs
            photo_embs    = tf.concat([input_dicts[k] for k in input_dicts if k in photo_fea_names], axis=-1)

            # 用户历史序列：不做 tile，直接保持 (batch, seq_len, dim) 形状
            # cross-attention 时用 matmul(query, key^T) 避免将序列复制 cand_size 份导致 OOM
            ft_click_list = self._parameters_dict['user_fountain_profile_click_pid_list']
            ft_click_aid_list = self._parameters_dict['user_fountain_profile_click_aid_list']
            ft_lv_list = self._parameters_dict['user_fountain_profile_long_view_pid_list']
            ft_lv_aid_list = self._parameters_dict['user_fountain_profile_long_view_aid_list']
            if not self._training:
                '''
                    infer 时需要注意实际请求的 batch size = items 长度, 但计算图 batch 的 shape 由 uni_predict_fused 中 executor_batchsizes 决定,
                    如60。为了通过计算图编译, 需要 reshape (1, -1, dim), 实际 -1 为请求端发送 items 长度, 即 60
                '''
                user_embs = tf.reshape(user_embs, [1, -1, user_embs.shape[-1]])
                photo_embs = tf.reshape(photo_embs, [1, -1, photo_embs.shape[-1]])
                ft_click_list = tf.reshape(ft_click_list, [1, -1, ft_click_list.shape[-2], ft_click_list.shape[-1]])[:, 0, :, :]
                ft_click_aid_list = tf.reshape(ft_click_aid_list, [1, -1, ft_click_aid_list.shape[-2], ft_click_aid_list.shape[-1]])[:, 0, :, :]
                ft_lv_list = tf.reshape(ft_lv_list, [1, -1, ft_lv_list.shape[-2], ft_lv_list.shape[-1]])[:, 0, :, :]
                ft_lv_aid_list = tf.reshape(ft_lv_aid_list, [1, -1, ft_lv_aid_list.shape[-2], ft_lv_aid_list.shape[-1]])[:, 0, :, :]
            common_embs = photo_embs
            common_embs   = tf.layers.dense(common_embs, 64, activation=tf.nn.leaky_relu) # (?, cand_size, 96)
            # user seq X candidate cross attn
            print("ft_click_list: ", ft_click_list, "ft_click_aid_list: ", ft_click_aid_list)
            ft_click_mha = self.attention_layer_4d("ft_click_mha", common_embs, tf.concat([ft_click_list, ft_click_aid_list], axis=-1)) # (?,cand_size,d)
            ft_lv_mha = self.attention_layer_4d("ft_lv_mha", common_embs, tf.concat([ft_lv_list, ft_lv_aid_list], axis=-1), use_gate=True) # (?,cand_size,d)
            history_embs = tf.concat([ft_click_mha, ft_lv_mha], axis=-1)

            # candidates aware by transformer
            transformer = StackedTransformerModel(name="candidates_aware", num_layers=1, dim=64, num_heads=2, dk=64, dropout_rate=0.0, training=self._training)
            candidates_aware_out = transformer.forward(common_embs, training=self._training) # (?,cand_size,128)
            # user_embs = tf.layers.dense(tf.concat([user_embs, source_embs], axis=-1), 32, activation=tf.nn.leaky_relu)
            user_embs = tf.layers.dense(user_embs, 32, activation=tf.nn.leaky_relu)
            common_embs = tf.concat([user_embs, history_embs, candidates_aware_out], axis=-1) # (?,cand_size,d)
            common_embs = tf.layers.dense(common_embs, 128, activation=tf.nn.leaky_relu)
            return common_embs
    
    def attention_layer_4d(self, name, query, key, use_gate=False, seq_len=None, nh=4, dim_in=16):
        '''
        query: (batch, cand_size, query_dim)
        key:   (batch, seq_len, key_dim)
        MHA cross-attention，无 tile，支持 gate 和 seq_len padding mask。
        '''
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            batch_size = tf.shape(query)[0]
            # 投影：直接用 dense，简洁且等价
            Q = tf.layers.dense(query, nh * dim_in, use_bias=False, name="Q_w")  # (batch, cand, nh*dim_in)
            K = tf.layers.dense(key,   nh * dim_in, use_bias=False, name="K_w")  # (batch, seq, nh*dim_in)
            V = tf.layers.dense(key,   nh * dim_in, use_bias=False, name="V_w")  # (batch, seq, nh*dim_in)
            # reshape 成多头形式
            Q = tf.reshape(Q, [batch_size, tf.shape(query)[1], nh, dim_in])  # (batch, cand, h, d)
            K = tf.reshape(K, [batch_size, tf.shape(key)[1],   nh, dim_in])  # (batch, seq,  h, d)
            V = tf.reshape(V, [batch_size, tf.shape(key)[1],   nh, dim_in])  # (batch, seq,  h, d)
            # scores: (batch, h, cand, seq)
            scores = tf.einsum("bqhd,bshd->bhqs", Q, K) / (dim_in ** 0.5)
            # seq_len padding mask
            if seq_len is not None:
                padding_mask = tf.sequence_mask(tf.reshape(seq_len, [-1]), maxlen=tf.shape(key)[1], dtype=tf.bool)
                padding_mask = tf.reshape(padding_mask, [batch_size, 1, 1, tf.shape(key)[1]])  # (batch, 1, 1, seq)
                scores = tf.where(tf.broadcast_to(padding_mask, tf.shape(scores)), scores, tf.ones_like(scores) * tf.cast(-1e9, scores.dtype))
            att_weights = tf.nn.softmax(scores, axis=-1)  # (batch, h, cand, seq)
            # context: (batch, h, cand, dim_in)
            context = tf.einsum("bhqs,bshd->bhqd", att_weights, V)
            # gate per head
            if use_gate:
                gate = tf.nn.sigmoid(
                    tf.layers.dense(query, nh, use_bias=True, bias_initializer=tf.constant_initializer(6.0), name="head_gate"))  # (batch, cand, h)
                gate = tf.transpose(gate, [0, 2, 1])    # (batch, h, cand)
                context = context * tf.expand_dims(gate, axis=-1)  # (batch, h, cand, dim_in)

            # 合并多头并输出投影
            result = tf.transpose(context, [0, 2, 1, 3])             # (batch, cand, h, dim_in)
            result = tf.reshape(result, [batch_size, tf.shape(query)[1], nh * dim_in])
            result = tf.layers.dense(result, query.shape[-1], use_bias=True, name="output_proj")
            return result
    
    def _build_list_value_outputs(
            self, list_item_emb, list_duration_s, batch_size, list_num):
        """构建 SWIM 的两层消费过程与 WT/EVV List Value。

        List 层 hazard 预测看完第 k 个 item 后是否继续；视频层 hazard
        预测进入某个视频后是否看完各时长片段。视频层 survival 积分得到
        单 item 期望时长，沿 List 位置累加成 PrefixWT，再由 P(K) 聚合。
        两层均只读取对应的事实前缀，不使用未来 List item。
        """
        with tf.variable_scope("list_value_branch", reuse=tf.AUTO_REUSE):
            # 加入位置编码后，将每条候选 List 展平到 batch 维送入因果 Transformer。
            position_emb = tf.reshape(
                self.position_embedding,
                [1, 1, self._list_size, 64],
            )
            causal_input = tf.reshape(
                list_item_emb + position_emb,
                [batch_size * list_num, self._list_size, 64],
            )
            prefix_transformer = StackedTransformerModel(
                name="causal_prefix",
                num_layers=1,
                dim=64,
                num_heads=2,
                dk=64,
                dropout_rate=0.0,
                training=self._training,
            )
            prefix_hidden = prefix_transformer.forward(
                causal_input,
                training=self._training,
                causal_mask=True,
            )
            prefix_hidden = tf.reshape(
                prefix_hidden,
                [batch_size, list_num, self._list_size, 64],
            )

            # Hazard 长度头：位置 k 预测用户消费完第 k 个 item 后是否继续。
            # 只需要前 LIST_SIZE-1 个决策；到达最后一位表示 K=LIST_SIZE，
            # 该样本是右截断的，不虚构“看完最后一位后停止”的标签。
            continue_hidden = tf.layers.dense(
                prefix_hidden[:, :, :-1, :],
                64,
                activation=tf.nn.leaky_relu,
                name="continue_hidden",
            )
            continue_logits = tf.squeeze(
                tf.layers.dense(
                    continue_hidden,
                    1,
                    activation=None,
                    name="continue_logits",
                ),
                axis=-1,
            )
            continue_probs = tf.nn.sigmoid(continue_logits, name="continue_probs")

            # 将逐位置继续概率转换为完整长度分布：
            # P(K=k)=prod_{i<k}q_i*(1-q_k)，
            # P(K=LIST_SIZE)=prod_{i<LIST_SIZE}q_i。
            survival_before_stop = tf.concat(
                [
                    tf.ones_like(continue_probs[:, :, :1]),
                    tf.math.cumprod(continue_probs[:, :, :-1], axis=-1),
                ],
                axis=-1,
            )
            stopped_length_probs = survival_before_stop * (1.0 - continue_probs)
            full_length_prob = tf.reduce_prod(
                continue_probs,
                axis=-1,
                keepdims=True,
            )
            length_probs_raw = tf.concat(
                [stopped_length_probs, full_length_prob],
                axis=-1,
            )
            length_probs = tf.identity(
                length_probs_raw / tf.maximum(
                    tf.reduce_sum(length_probs_raw, axis=-1, keepdims=True),
                    1e-8,
                ),
                name="length_probs",
            )
            # 第 k 个位置被消费的概率 P(K>=k)。直接由归一化后的 P(K)
            # 反向累加，既补齐最后一个位置，也保证下面两种 List 聚合写法
            # 在浮点计算中保持同源：sum P(K=k)*PrefixValue(k)
            # == sum P(K>=k)*ItemValue(k)。
            list_reach_probs = tf.reverse(
                tf.cumsum(tf.reverse(length_probs, axis=[-1]), axis=-1),
                axis=[-1],
                name="list_reach_probs",
            )
            # SWIM 视频内生成式时长头。每个 List prefix hidden 被扩展为
            # 一条 segment 序列；显式加入原视频 duration、segment 位置与
            # 累计时长比例。segment causal Transformer 并行地产生条件
            # continue 概率，不需要逐片段 decoder。
            segment_width_ratios = tf.reshape(
                tf.constant(
                    self._swim_segment_width_ratios,
                    dtype=tf.float32,
                ),
                [1, 1, 1, self._swim_num_segments],
            )
            segment_end_ratios = tf.reshape(
                tf.cumsum(
                    tf.constant(
                        self._swim_segment_width_ratios,
                        dtype=tf.float32,
                    )
                ),
                [1, 1, 1, self._swim_num_segments, 1],
            )
            segment_position_emb = tf.reshape(
                self.segment_position_embedding,
                [1, 1, 1, self._swim_num_segments, 32],
            )
            tiled_prefix_hidden = tf.tile(
                tf.expand_dims(prefix_hidden, axis=3),
                [1, 1, 1, self._swim_num_segments, 1],
            )
            duration_log = tf.math.log1p(
                tf.maximum(tf.cast(list_duration_s, tf.float32), 0.0)
            )
            duration_feature = tf.tile(
                tf.expand_dims(tf.expand_dims(duration_log, axis=-1), axis=-1),
                [1, 1, 1, self._swim_num_segments, 1],
            )
            segment_position_emb = tf.tile(
                segment_position_emb,
                [batch_size, list_num, self._list_size, 1, 1],
            )
            segment_end_ratios = tf.tile(
                segment_end_ratios,
                [batch_size, list_num, self._list_size, 1, 1],
            )
            segment_input = tf.concat(
                [
                    tiled_prefix_hidden,
                    segment_position_emb,
                    duration_feature,
                    segment_end_ratios,
                ],
                axis=-1,
            )
            segment_input = tf.layers.dense(
                segment_input,
                64,
                activation=tf.nn.leaky_relu,
                name="segment_input_projection",
            )
            flat_item_count = batch_size * list_num * self._list_size
            segment_input = tf.reshape(
                segment_input,
                [flat_item_count, self._swim_num_segments, 64],
            )
            segment_transformer = StackedTransformerModel(
                name="causal_watch_segments",
                num_layers=1,
                dim=64,
                num_heads=2,
                dk=64,
                dropout_rate=0.0,
                training=self._training,
            )
            segment_hidden = segment_transformer.forward(
                segment_input,
                training=self._training,
                causal_mask=True,
            )
            segment_continue_logits = tf.reshape(
                tf.squeeze(
                    tf.layers.dense(
                        segment_hidden,
                        1,
                        activation=None,
                        name="segment_continue_logits",
                    ),
                    axis=-1,
                ),
                [
                    batch_size,
                    list_num,
                    self._list_size,
                    self._swim_num_segments,
                ],
            )
            segment_continue_probs = tf.nn.sigmoid(
                segment_continue_logits,
                name="segment_continue_probs",
            )
            segment_reach_probs = tf.math.cumprod(
                segment_continue_probs,
                axis=-1,
                name="segment_reach_probs",
            )
            segment_duration_s = tf.expand_dims(
                tf.maximum(tf.cast(list_duration_s, tf.float32), 0.0),
                axis=-1,
            ) * segment_width_ratios
            item_watch_time = tf.reduce_sum(
                segment_reach_probs * segment_duration_s,
                axis=-1,
                name="item_watch_time",
            )
            prefix_watch_time = tf.cumsum(
                item_watch_time,
                axis=-1,
                name="prefix_watch_time",
            )

            # EVV：单 item 是否达到 duration-conditioned 有效播放阈值。
            # 概率累加后 PrefixEVV 天然单调，不需要额外的单调约束。
            evv_hidden = tf.layers.dense(
                prefix_hidden,
                64,
                activation=tf.nn.leaky_relu,
                name="evv_hidden",
            )
            evv_logits = tf.squeeze(
                tf.layers.dense(
                    evv_hidden,
                    1,
                    activation=None,
                    name="evv_logits",
                ),
                axis=-1,
            )
            evv_probs = tf.nn.sigmoid(evv_logits, name="evv_probs")
            prefix_effective_vv = tf.cumsum(
                evv_probs,
                axis=-1,
                name="prefix_effective_vv",
            )

            # 互动预测头仍停用。保留位置张量用于 expected consume length。
            prefix_position = tf.reshape(
                tf.cast(tf.range(1, self._list_size + 1), tf.float32),
                [1, 1, self._list_size],
            )

            return {
                "continue_logits": continue_logits,
                "continue_probs": continue_probs,
                "length_probs": length_probs,
                "list_reach_probs": list_reach_probs,
                "segment_continue_logits": segment_continue_logits,
                "segment_continue_probs": segment_continue_probs,
                "segment_reach_probs": segment_reach_probs,
                "segment_duration_s": segment_duration_s,
                "item_watch_time": item_watch_time,
                "prefix_watch_time": prefix_watch_time,
                "evv_logits": evv_logits,
                "evv_probs": evv_probs,
                "prefix_effective_vv": prefix_effective_vv,
                "expected_list_watch_time": tf.reduce_sum(
                    length_probs * prefix_watch_time,
                    axis=-1,
                    name="expected_watch_time",
                ),
                "expected_list_watch_time_by_reach": tf.reduce_sum(
                    list_reach_probs * item_watch_time,
                    axis=-1,
                    name="expected_watch_time_by_reach",
                ),
                "expected_list_effective_vv": tf.reduce_sum(
                    length_probs * prefix_effective_vv,
                    axis=-1,
                    name="expected_effective_vv",
                ),
                "expected_list_effective_vv_by_reach": tf.reduce_sum(
                    list_reach_probs * evv_probs,
                    axis=-1,
                    name="expected_effective_vv_by_reach",
                ),
                "expected_consume_length": tf.reduce_sum(
                    length_probs * prefix_position,
                    axis=-1,
                    name="expected_consume_length",
                ),
            }

    def model(self, list_index, list_duration_s, list_num=None):
        """仅构建 standalone List 模型，不创建任何 point-wise 参数或输出。

        ``list_num`` 默认使用线上候选 List 数；训练时可追加少量合成 List，
        共享一次 backbone 计算并复用同一套 List Value 参数。
        """
        with tf.variable_scope("evaluator", reuse=tf.AUTO_REUSE):
            if list_num is None:
                list_num = self._list_num
            self._list_index = list_index
            hidden_states = self._get_list_features(self._parameters_dict)
            print("list_backbone_hidden_states:", hidden_states)

            batch_size = tf.shape(hidden_states)[0]
            batch_idx = tf.reshape(tf.range(batch_size), [batch_size, 1, 1])
            batch_idx = tf.tile(
                batch_idx,
                [1, list_num, self._list_size],
            )
            indices = tf.stack([batch_idx, self._list_index], axis=-1)
            zeros = tf.zeros(
                shape=[batch_size, 1, hidden_states.shape[-1]],
                dtype=tf.float32,
            )
            list_item_emb = tf.gather_nd(
                tf.concat([zeros, hidden_states], axis=1),
                indices,
            )
            with tf.variable_scope(
                "list_value_input_adapter",
                reuse=tf.AUTO_REUSE,
            ):
                list_item_emb = tf.layers.dense(
                    list_item_emb,
                    64,
                    activation=tf.nn.leaky_relu,
                    name="projection",
                )
            print("list_item_emb ", list_item_emb)

            return self._build_list_value_outputs(
                list_item_emb,
                list_duration_s,
                batch_size,
                list_num,
            )
