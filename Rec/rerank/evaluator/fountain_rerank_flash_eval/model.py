from numpy import dtype
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
        self.decoder_layers = [DecoderLayer(f"{name}_position_layer_{i}", dim, num_heads, dk, dropout_rate) for i in range(num_layers)]
        
    def forward(self, hidden_states, training):
        for i in range(self.num_layers):
            hidden_states = self.layers[i].forward(hidden_states, training=training)
        return hidden_states
    
    def forward_decoder(self, hidden_states, item_embedding, training):
        for i in range(self.num_layers):
            item_embedding = self.decoder_layers[i].forward(item_embedding, hidden_states, training=training)
        return item_embedding

class EvaluatorModel:
    def __init__(self, parameters_dict, feature_emb_size_dict, label_value_dict, print_ops, list_size, candidates_size, list_num,
                 point_wise_tasks, list_wise_tasks, dim=32, extra_param_dict= None, training=True):
        self._list_wise_tasks = list_wise_tasks
        self._point_wise_tasks = point_wise_tasks
        self._pxtr_names = [
            "context_pctr",
            "context_pltr",
            "context_pwtr", # 关注
            "context_pftr", # 分享
            "context_plvtr",
            "context_pvtr",
            "context_pptr",
            "context_pcmtr",
            "context_pepstr",
            "context_pcpr",
            "context_pcltr",
            "context_psvr",
            "context_pwtd",
        ]
        self._photo_attr_names = [
            "photo_id",
            "photo_author_id",
            # "photo_author_gender",
            # "photo_hetu_tag_level1_list",
            # "photo_hetu_tag_level2_list",
            # "photo_hetu_tag_level3_list",
            # "photo_hetu_tag_level5_list",
            # "photo_tag",
            "photo_duration_ms",
            "photo_upload_type",
            # "photo_city_id",
            # "photo_music",
        ]
        self._photo_emp_explore_names = [
            # "emp_explore_show_count",
            "emp_explore_click_count",
            "photo_emp_explore_ctr",
            "photo_emp_explore_ltr",
            # "photo_emp_explore_avg_time",
        ]
        self._photo_emp_fountain_names = [
            "emp_fountain_show_count",
            "emp_fountain_like_count",
            # "emp_fountain_follow_count",
            "emp_fountain_long_play_count",
            # "photo_emp_fountain_svtr",
            # "photo_emp_fountain_lvtr",
            "photo_emp_fountain_ltr",
            # "photo_emp_fountain_wtr",
            # "photo_emp_fountain_avg_fintr",
        ]
        self._parameters_dict = parameters_dict
        self._feature_emb_size_dict = feature_emb_size_dict
        self._label_value_dict = label_value_dict
        self._list_size = list_size
        self._candidates_size = candidates_size
        self._list_num = list_num
        self.dim = dim
        self._training = training
        self._bucket_emb_conf = {}
        # for k, v in dense_features_config.items():
        #     self._bucket_emb_conf[v['name']] = {
        #         'value': self._label_value_dict[k],
        #         'boundaries': v['boundaries'],
        #         'norm_type': v['norm_type'] if 'norm_type' in v.keys() else 'none',
        #         'embedding': tf.get_variable(
        #             name=f'bucket_emb_{v["name"]}',
        #             shape=[len(v['boundaries']) + 1, 4],
        #             initializer=tf.random_normal_initializer()
        #         )
        #     }
        # self.dense_features_config = dense_features_config
        self.cls_embedding = tf.get_variable(
            name='cls_embedding',
            shape=[1, 64],
            initializer=tf.random_normal_initializer()
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
    
    def get_bucket_emb_from_sorted_boundaries(self, name):
        value = self._bucket_emb_conf[name]['value']
        norm_type = self._bucket_emb_conf[name]['norm_type']
        value = tf.cast(value, tf.float32)
        if norm_type == "x^0.7":
            value = tf.pow(value, 0.7)
        boundaries = self._bucket_emb_conf[name]['boundaries']
        embeddings = self._bucket_emb_conf[name]['embedding']
        print(f"bucket emb conf name: {name}, bucket_size: {len(boundaries) + 1}, embeddings: {embeddings}")
        boundaries = tf.constant(boundaries, dtype=tf.float32)
        boundaries = tf.tile(tf.expand_dims(boundaries, axis=0), [tf.shape(value)[0], 1])
        # print("boundaries ", boundaries, " value ", value)
        bucket_id = tf.searchsorted(boundaries, values=value, out_type=tf.int32)
        emb = tf.nn.embedding_lookup(embeddings, bucket_id) # (?, cand_size, dim)
        return emb, bucket_id

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

    def model(self, list_index) -> tuple:
        with tf.variable_scope("evaluator", reuse=tf.AUTO_REUSE):
            def multi_task_module(name, point_wise_input, loss_names, shared_key, cand_size):
                with tf.variable_scope(f"task_{name}", reuse=tf.AUTO_REUSE):
                    ple_layer = PLE(loss_names, shared_key=shared_key, cgc_layers = 1, task_expert_num=1, shared_expert_num=1,
                                        expert_tower_dim = [64], gate_tower_dim = [64], print_ops = self.print_ops)
                    # continue 任务用 prefix_emb 作为输入：pos i 的特征为 mean(emb[0:i+1])，体现顺序依赖
                    # shape 与其他任务保持一致 (?, list_size, dim)，PLE 结构无需修改
                    if "continue" in loss_names:
                        prefix_emb = tf.stack([tf.reduce_mean(point_wise_input[:, :i+1, :], axis=1) for i in range(cand_size)], axis=1)  # (?, list_size, dim)
                        input_feature_dict = {x: (prefix_emb if x == "continue" else point_wise_input) for x in loss_names}
                    else:
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
            list_emb = tf.gather_nd(tf.concat([zeros, hidden_states], axis=1), indices) # (?, list_num, list_size, dim)
            list_emb = tf.layers.dense(list_emb, 64, activation=tf.nn.leaky_relu)
            print("list_emb ", list_emb)
            # cls token
            cls_embedding = tf.tile(tf.expand_dims(self.cls_embedding, axis=0), [batch_size, self._list_num, 1]) #(?,list_num,dim)
            cls_embedding = tf.expand_dims(cls_embedding, axis=2) #(?,list_num,1,dim)
            print("cls_embedding ", cls_embedding)
            list_emb = tf.concat([cls_embedding,list_emb],axis=2) #(?,list_num,list_size+1,dim)
            print("list_emb concat cls ", list_emb)
            list_emb_dim = list_emb.shape[-1]
            # transformer = StackedTransformerModel(name="list_aware", num_layers=1, dim=64, num_heads=2, dk=64, dropout_rate=0.0, training=self._training)
            # list_emb = transformer.forward(tf.reshape(list_emb, [batch_size * self._list_num, self._list_size + 1, list_emb_dim]), training=self._training)
            # list_emb = tf.reshape(list_emb, [batch_size, self._list_num, self._list_size + 1, list_emb_dim])
            list_emb = self.self_attention_4d("list_aware_attention", list_emb)
            point_wise_input = tf.reshape(list_emb[:, :, 1:, :], [batch_size * self._list_num, self._list_size, list_emb_dim]) # (?*list_num, list_size, dim)
            print("point_wise_input ", point_wise_input)
            # point-wise task
            point_wise_output_dict = multi_task_module("point_wise", point_wise_input, self._point_wise_tasks, shared_key="vtr", cand_size=self._list_size) # (?*list_num,list_size)

            if len(self._list_wise_tasks) > 0:
                # list-wise module
                list_cls = list_emb[:, :, 0, :] # (?,list_num, dim)
                transformer = StackedTransformerModel(name="lists_aware", num_layers=1, dim=64, num_heads=2, dk=64, dropout_rate=0.1, training=self._training)
                list_hidden = transformer.forward(list_cls, training=self._training) # (?,list_num, dim)
                # list_wise_input = tf.layers.dense(tf.concat([user_hidden[:, :self._list_num, :], list_hidden], axis=-1), 128, activation=tf.nn.relu) # (?, 50, 128)
                list_wise_input = tf.layers.dense(list_hidden, 128, activation=tf.nn.relu) # (?, 50, 128)
                print("list_wise_input ", list_wise_input)
                # list_wise_output_dict = multi_task_module("list_wise", list_wise_input, self._list_wise_tasks, shared_key="list_ltr", cand_size=self._list_num) # (?,list_num)
                list_dnn = tf.layers.dense(list_wise_input, 64, activation=tf.nn.leaky_relu)
                list_output = tf.layers.dense(list_dnn, 1, activation=tf.nn.sigmoid)
                list_wise_output_dict = {"listwise_wtd": tf.squeeze(list_output, axis=-1)}  # (?,list_num)
            else:
                list_wise_output_dict = {}

            return point_wise_output_dict, list_wise_output_dict

