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
                 point_wise_tasks, training=True):
        self._point_wise_tasks = point_wise_tasks
        self._parameters_dict = parameters_dict
        self._list_size = list_size
        self._candidates_size = candidates_size
        self._list_num = list_num
        self._training = training
        self.cls_embedding = tf.get_variable(
            name='cls_embedding',
            shape=[1, 64],
            initializer=tf.random_normal_initializer()
        )
        self.position_embedding = tf.get_variable(
            name='list_value_position_embedding',
            shape=[self._list_size, 64],
            initializer=tf.random_normal_initializer(stddev=0.02)
        )
        self.print_ops = print_ops

    def _get_shared_features(self, input_dicts) -> tuple:
        with tf.variable_scope("share_bottom", reuse=tf.AUTO_REUSE):
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
    
    def self_attention_4d(self, name, x):
        with tf.variable_scope(f"{name}", reuse=tf.AUTO_REUSE):
            batch_size, cand_size, len, dim = tf.shape(x)[0], x.shape[1], x.shape[2], x.shape[3]
            x = tf.reshape(x, [batch_size * cand_size, len, dim])
            attn_out, attention_weights = scaled_dot_product_attention(x, x, x, mask=None)
            attn_out = tf.reshape(attn_out, [batch_size, cand_size, len, dim]) # (?, cand_size, len, dim1)
            return attn_out

    def _build_list_value_outputs(self, list_item_emb, batch_size):
        """构建新增的因果 List Value 分支。

        输入 list_item_emb 的形状为 [batch, list_num, list_size, 64]。
        第 k 个位置只允许读取前 k 个 item，分别预测消费长度分布 P(K)
        以及每个前缀的累计观看时长、累计有效播放数。
        """
        with tf.variable_scope("list_value_branch", reuse=tf.AUTO_REUSE):
            # 加入位置编码后，将每条候选 List 展平到 batch 维送入因果 Transformer。
            position_emb = tf.reshape(
                self.position_embedding,
                [1, 1, self._list_size, 64],
            )
            causal_input = tf.reshape(
                list_item_emb + position_emb,
                [batch_size * self._list_num, self._list_size, 64],
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
                [batch_size, self._list_num, self._list_size, 64],
            )

            # 长度头：第 k 个 logit 只读取 Prefix[1:k]，再在所有 k 上做 softmax，
            # 得到用户最终消费 K 个 item 的概率 P(K=k)。
            length_hidden = tf.layers.dense(
                prefix_hidden,
                64,
                activation=tf.nn.leaky_relu,
                name="length_hidden",
            )
            length_logits = tf.squeeze(
                tf.layers.dense(
                    length_hidden,
                    1,
                    activation=None,
                    name="length_logits",
                ),
                axis=-1,
            )
            length_probs = tf.nn.softmax(length_logits, axis=-1, name="length_probs")

            # 价值头：在每个 k 上预测 Prefix[1:k] 的累计 WT 和累计有效 VV。
            # 两个任务共享 value_hidden，但保留独立输出头。
            value_hidden = tf.layers.dense(
                prefix_hidden,
                64,
                activation=tf.nn.leaky_relu,
                name="prefix_value_hidden",
            )
            # WT 在 log1p 空间训练。softplus 保证解码前非负，截断避免 expm1 溢出。
            prefix_watch_time_log = tf.nn.softplus(
                tf.squeeze(
                    tf.layers.dense(
                        value_hidden,
                        1,
                        activation=None,
                        name="prefix_watch_time_logit",
                    ),
                    axis=-1,
                )
            )
            max_watch_time_log = tf.math.log(tf.constant(36001.0, dtype=tf.float32))
            prefix_watch_time = tf.math.expm1(
                tf.minimum(prefix_watch_time_log, max_watch_time_log)
            )

            # Prefix[1:k] 的有效 VV 理论范围是 [0, k]：
            # sigmoid 给出 [0, 1] 的比例，再乘当前位置 k。
            prefix_position = tf.reshape(
                tf.cast(tf.range(1, self._list_size + 1), tf.float32),
                [1, 1, self._list_size],
            )
            prefix_effective_vv = tf.nn.sigmoid(
                tf.squeeze(
                    tf.layers.dense(
                        value_hidden,
                        1,
                        activation=None,
                        name="prefix_effective_vv_logit",
                    ),
                    axis=-1,
                )
            ) * prefix_position

            # 最终用于 List 比较的是期望价值：
            # E[V] = sum_k P(K=k) * V(Prefix[1:k])。
            return {
                "length_logits": length_logits,
                "length_probs": length_probs,
                "prefix_watch_time_log": prefix_watch_time_log,
                "prefix_watch_time": prefix_watch_time,
                "prefix_effective_vv": prefix_effective_vv,
                "expected_watch_time": tf.reduce_sum(
                    length_probs * prefix_watch_time,
                    axis=-1,
                    name="expected_watch_time",
                ),
                "expected_effective_vv": tf.reduce_sum(
                    length_probs * prefix_effective_vv,
                    axis=-1,
                    name="expected_effective_vv",
                ),
                "expected_consume_length": tf.reduce_sum(
                    length_probs * prefix_position,
                    axis=-1,
                    name="expected_consume_length",
                ),
            }

    def model(self, list_index) -> tuple:
        with tf.variable_scope("evaluator", reuse=tf.AUTO_REUSE):
            def multi_task_module(name, point_wise_input, loss_names, shared_key, cand_size):
                with tf.variable_scope(f"task_{name}", reuse=tf.AUTO_REUSE):
                    ple_layer = PLE(loss_names, shared_key=shared_key, cgc_layers = 1, task_expert_num=1, shared_expert_num=1,
                                        expert_tower_dim = [64], gate_tower_dim = [64], print_ops = self.print_ops)
                    input_feature_dict = {x: point_wise_input for x in loss_names}
                    output_fea_dict = ple_layer(input_feature_dict, input_feature_dict)  # (?,cand_size,64)
                    output_list = []
                    key_output_list = []
                    for key in output_fea_dict.keys():
                        key_output_list.append(key)
                        output_list.append(output_fea_dict[key])
                    output_list = tf.stack(output_list, axis=2) # (?,cand_size,n,64)
                    output_list = output_attention("output_cross_attention", output_list, output_list, 64, values=output_list, need_initialize_values=False)
                    output_list = tf.split(output_list, len(output_fea_dict), axis=2)
                    for j in range(len(output_fea_dict)):
                        output_fea_dict[key_output_list[j]] = tf.squeeze(output_list[j], axis=2)
                    output_dict = {}
                    for loss_name, output in output_fea_dict.items():
                        with tf.variable_scope(f"output_mlp_{loss_name}", reuse=tf.AUTO_REUSE):
                            output  = tf.layers.dense(output, 32, activation=tf.nn.leaky_relu)
                            output = tf.layers.dense(output, 1, activation=tf.nn.sigmoid)  # (?,cand_size,1)
                            output = tf.reshape(output, [-1, cand_size]) # (?, cand_size)
                        output_dict[loss_name] = output
                    return output_dict
            input_dicts = self._parameters_dict
            self._list_index = list_index
            com_embs = self._get_shared_features(input_dicts) # (?, cand_size, 512)
            print("com_embs:", com_embs)
            hidden_states = com_embs

            batch_size = tf.shape(hidden_states)[0]
            # list index: [-1, LIST_NUM, LIST_SIZE]
            batch_idx = tf.reshape(tf.range(batch_size), [batch_size, 1, 1])
            batch_idx = tf.tile(batch_idx, [1, self._list_num, self._list_size])
            print("batch_idx ", batch_idx, " _list_index ", self._list_index)
            indices = tf.stack([batch_idx, self._list_index], axis=-1) # (?, list_num, list_size, 2)
            zeros = tf.zeros(shape=[batch_size, 1, hidden_states.shape[-1]], dtype=tf.float32)
            list_item_emb = tf.gather_nd(tf.concat([zeros, hidden_states], axis=1), indices) # (?, list_num, list_size, dim)
            list_item_emb = tf.layers.dense(list_item_emb, 64, activation=tf.nn.leaky_relu)
            print("list_item_emb ", list_item_emb)

            # -------- 原有 point-wise 基线分支 --------
            cls_embedding = tf.tile(tf.expand_dims(self.cls_embedding, axis=0), [batch_size, self._list_num, 1]) #(?,list_num,dim)
            cls_embedding = tf.expand_dims(cls_embedding, axis=2) #(?,list_num,1,dim)
            print("cls_embedding ", cls_embedding)
            list_emb = tf.concat([cls_embedding,list_item_emb],axis=2) #(?,list_num,list_size+1,dim)
            print("list_emb concat cls ", list_emb)
            list_emb_dim = list_emb.shape[-1]
            list_emb = self.self_attention_4d("list_aware_attention", list_emb)
            point_wise_input = tf.reshape(list_emb[:, :, 1:, :], [batch_size * self._list_num, self._list_size, list_emb_dim]) # (?*list_num, list_size, dim)
            print("point_wise_input ", point_wise_input)
            point_wise_output_dict = multi_task_module("point_wise", point_wise_input, self._point_wise_tasks, shared_key="vtr", cand_size=self._list_size) # (?*list_num,list_size)

            # -------- 新增因果 List Value 分支 --------
            list_value_output_dict = self._build_list_value_outputs(
                list_item_emb,
                batch_size,
            )
            return point_wise_output_dict, list_value_output_dict
