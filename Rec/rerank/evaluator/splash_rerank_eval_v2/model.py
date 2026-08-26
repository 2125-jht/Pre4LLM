from math import tau
from numpy import dtype
import sys
import tensorflow as tf
from modules_ import *
from feature_attr_extract import user_fea_names,photo_fea_names,source_fea_names,explore_profile_fea_names,fountain_seq_pid_names,fountain_seq_aid_names,dense_features_config


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

class Evaluator():
    def __init__(self, num_layers, dim, num_heads, dk, dropout_rate, k, training=False):
        super(Evaluator, self).__init__()
        self.num_layers = num_layers
        self.k = k
        self.dim = dim
        self.layers = [EncoderLayer(f"transformer_layer_{i}", dim, num_heads, dk, dropout_rate, training=training) for i in range(num_layers)]
        
    def forward(self, hidden_states, training):
        for i in range(self.num_layers):
            hidden_states = self.layers[i].forward(hidden_states, training=training)
        return hidden_states

class EvaluatorModel:
    def __init__(self, parameters_dict, label_value_dict, print_ops, list_size, candidates_size, list_num,
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
            "context_fountain_related_score_v2",
        ]
        self._photo_attr_names = [
            "photo_id",
            "photo_author_id",
            "photo_author_gender",
            "photo_hetu_tag_level1_list",
            "photo_hetu_tag_level2_list",
            "photo_hetu_tag_level3_list",
            "photo_hetu_tag_level5_list",
            "photo_tag",
            "photo_duration_ms",
            "photo_upload_type",
            # "photo_city_id",
            # "photo_music",
        ]
        self._photo_emp_explore_names = [
            "emp_explore_show_count",
            "emp_explore_click_count",
            "photo_emp_explore_ctr",
            "photo_emp_explore_ltr",
            "photo_emp_explore_avg_time",
        ]
        self._photo_emp_fountain_names = [
            "emp_fountain_show_count",
            "emp_fountain_like_count",
            "emp_fountain_follow_count",
            "emp_fountain_long_play_count",
            "photo_emp_fountain_svtr",
            "photo_emp_fountain_lvtr",
            "photo_emp_fountain_ltr",
            "photo_emp_fountain_wtr",
            "photo_emp_fountain_avg_fintr",
        ]
        self._parameters_dict = parameters_dict
        self._label_value_dict = label_value_dict
        self._list_size = list_size
        self._candidates_size = candidates_size
        self._list_num = list_num
        self.dim = dim
        self._training = training
        self._bucket_emb_conf = {}
        for k, v in dense_features_config.items():
            self._bucket_emb_conf[v['name']] = {
                'value': self._label_value_dict[k],
                'boundaries': v['boundaries'],
                'norm_type': v['norm_type'] if 'norm_type' in v.keys() else 'none',
                'embedding': tf.get_variable(
                    name=f'bucket_emb_{v["name"]}',
                    shape=[len(v['boundaries']) + 1, 4],
                    initializer=tf.random_normal_initializer()
                )
            }
        self.dense_features_config = dense_features_config
        self.cls_embedding = tf.get_variable(
            name='cls_embedding',
            shape=[1, 256],
            initializer=tf.random_normal_initializer()
        )
        # self._list_index = tf.cast(self._label_value_dict['rerank_list_item_idx_flat_list'], tf.int32)
        self.print_ops = print_ops

    def _mlp_layer(self,
                  scope_name,
                  hidden_states: tf.Tensor,
                  hidden_units: list,
                  activation=tf.nn.relu) -> tf.Tensor:
        with tf.variable_scope(f"{scope_name}_mlp_layer", reuse=tf.AUTO_REUSE):
            for i, hidden_unit in enumerate(hidden_units):
                hidden_states = tf.layers.dense(hidden_states, hidden_unit, activation=activation, use_bias=True)
        return hidden_states

    def _get_shared_features(self, input_dicts) -> tuple:
        with tf.variable_scope("share_bottom", reuse=tf.AUTO_REUSE):
            user_embs     = tf.concat([input_dicts[k] for k in input_dicts if k in user_fea_names], axis=-1)
            user_embs     = tf.tile(tf.expand_dims(user_embs, axis=1), [1,self._candidates_size,1]) if self._training else user_embs
            photo_embs    = tf.concat([input_dicts[k] for k in input_dicts if k in photo_fea_names], axis=-1)
            source_embs   = tf.concat([input_dicts[k] for k in input_dicts if k in source_fea_names], axis=-1)
            source_embs   = tf.tile(tf.expand_dims(source_embs, axis=1), [1,self._candidates_size,1]) if self._training else source_embs

            common_embs   = tf.concat([user_embs, photo_embs, source_embs], axis=-1)
            common_embs   = tf.layers.dense(common_embs, 256, activation=tf.nn.leaky_relu) # (?, cand_size, 256)

            photo_attr_embs    = tf.concat([self._parameters_dict[k] for k in self._photo_attr_names], axis=-1) # (?,cand_size,dim) infer (?,dim)
            if self._training:
                pxtr_list_embs    = tf.concat([tf.expand_dims(self._parameters_dict[k], axis=2) for k in self._parameters_dict if k in self._pxtr_names], axis=2) # (?,cand_size,n,dim)
            else:
                pxtr_list_embs    = tf.concat([tf.expand_dims(self._parameters_dict[k], axis=1) for k in self._parameters_dict if k in self._pxtr_names], axis=1) # (?,n,dim)
            pxtr_embs    = tf.concat([self._parameters_dict[k] for k in self._parameters_dict if k in self._pxtr_names], axis=-1) # (?,cand_size,dim) infer (?,dim)
            pxtr_embs = tf.layers.dense(pxtr_embs, 128, activation=tf.nn.leaky_relu, use_bias=True)
            # user seq model
            up_click_list = tf.tile(tf.expand_dims(self._parameters_dict['user_profile_v1_click_pid_list'], axis=1), [1, self._candidates_size, 1, 1]) \
                if self._training else self._parameters_dict['user_profile_v1_click_pid_list'] # (?, cand_size, seq_len, d) infer (?, seq_len, d)
            up_click_aid_list = tf.tile(tf.expand_dims(self._parameters_dict['user_profile_v1_click_aid_list'], axis=1), [1, self._candidates_size, 1, 1]) \
                if self._training else self._parameters_dict['user_profile_v1_click_aid_list']
            ft_like_list = tf.tile(tf.expand_dims(self._parameters_dict['user_fountain_profile_like_pid_list'], axis=1), [1, self._candidates_size, 1, 1]) \
                if self._training else self._parameters_dict['user_fountain_profile_like_pid_list']
            ft_like_aid_list = tf.tile(tf.expand_dims(self._parameters_dict['user_fountain_profile_like_aid_list'], axis=1), [1, self._candidates_size, 1, 1]) \
                if self._training else self._parameters_dict['user_fountain_profile_like_aid_list']
            ft_ev_list = tf.tile(tf.expand_dims(self._parameters_dict['user_fountain_profile_effective_view_pid_list'], axis=1), [1, self._candidates_size, 1, 1]) \
                if self._training else self._parameters_dict['user_fountain_profile_effective_view_pid_list']
            ft_ev_aid_list = tf.tile(tf.expand_dims(self._parameters_dict['user_fountain_profile_effective_view_aid_list'], axis=1), [1, self._candidates_size, 1, 1]) \
                if self._training else self._parameters_dict['user_fountain_profile_effective_view_aid_list']
            ft_lv_list = tf.tile(tf.expand_dims(self._parameters_dict['user_fountain_profile_long_view_pid_list'], axis=1), [1, self._candidates_size, 1, 1]) \
                if self._training else self._parameters_dict['user_fountain_profile_long_view_pid_list']
            ft_lv_aid_list = tf.tile(tf.expand_dims(self._parameters_dict['user_fountain_profile_long_view_aid_list'], axis=1), [1, self._candidates_size, 1, 1]) \
                if self._training else self._parameters_dict['user_fountain_profile_long_view_aid_list']
            if not self._training:
                '''
                    infer 时需要注意实际请求的 batch size = items 长度, 但计算图 batch 的 shape 由 uni_predict_fused 中 executor_batchsizes 决定,
                    如600。为了通过计算图编译, 需要 rashape (1, -1, dim), 实际 -1 为请求端发送 items 长度
                '''
                emb_dim = common_embs.shape[-1]
                common_embs = tf.reshape(common_embs, [1, -1, emb_dim])
                user_embs = tf.reshape(user_embs, [1, -1, user_embs.shape[-1]])
                source_embs = tf.reshape(source_embs, [1, -1, source_embs.shape[-1]])
                photo_attr_embs = tf.reshape(photo_attr_embs, [1, -1, photo_attr_embs.shape[-1]])
                photo_embs = tf.reshape(photo_embs, [1, -1, photo_embs.shape[-1]])
                pxtr_embs = tf.reshape(pxtr_embs, [1, -1, pxtr_embs.shape[-1]])
                print("pxtr_list_embs ", pxtr_list_embs)
                pxtr_list_embs = tf.reshape(pxtr_list_embs, [1, -1, pxtr_list_embs.shape[-2], pxtr_list_embs.shape[-1]])
                up_click_list = tf.reshape(up_click_list, [1, -1, up_click_list.shape[-2], up_click_list.shape[-1]])
                up_click_aid_list = tf.reshape(up_click_aid_list, [1, -1, up_click_aid_list.shape[-2], up_click_aid_list.shape[-1]])
                ft_like_list = tf.reshape(ft_like_list, [1, -1, ft_like_list.shape[-2], ft_like_list.shape[-1]])
                ft_like_aid_list = tf.reshape(ft_like_aid_list, [1, -1, ft_like_aid_list.shape[-2], ft_like_aid_list.shape[-1]])
                ft_ev_list = tf.reshape(ft_ev_list, [1, -1, ft_ev_list.shape[-2], ft_ev_list.shape[-1]])
                ft_ev_aid_list = tf.reshape(ft_ev_aid_list, [1, -1, ft_ev_aid_list.shape[-2], ft_ev_aid_list.shape[-1]])
                ft_lv_list = tf.reshape(ft_lv_list, [1, -1, ft_lv_list.shape[-2], ft_lv_list.shape[-1]])
                ft_lv_aid_list = tf.reshape(ft_lv_aid_list, [1, -1, ft_lv_aid_list.shape[-2], ft_lv_aid_list.shape[-1]])
            photo_attr_embs = tf.layers.dense(photo_attr_embs, 24, activation=tf.nn.leaky_relu)
            pxtr_list_embs = tf.layers.dense(pxtr_list_embs, 24, activation=tf.nn.leaky_relu)
            print("photo_attr_embs ", photo_attr_embs, "pxtr_list_embs ", pxtr_list_embs)
            pxtr_mha_0 = self.attention_layer_4d("pxtr_mha_0", photo_attr_embs, pxtr_list_embs) # (?,cand_size,d)
            query_emb = tf.concat([user_embs, source_embs], axis=-1) # (?,cand_size,d)
            query_emb = tf.layers.dense(query_emb, 24, activation=tf.nn.leaky_relu)
            print("query_emb ", query_emb)
            pxtr_mha_1 = self.attention_layer_4d("pxtr_mha_1", query_emb, pxtr_list_embs) # (?,cand_size,d)
            # user seq
            photo_embs = tf.layers.dense(photo_embs, 64, activation=tf.nn.leaky_relu)
            up_click_mha = self.attention_layer_4d("up_click_mha", photo_embs, up_click_list) # (?,cand_size,d)
            up_click_aid_mha = self.attention_layer_4d("up_click_aid_mha", photo_embs, up_click_aid_list) # (?,cand_size,d)
            ft_like_mha = self.attention_layer_4d("ft_like_mha", photo_embs, ft_like_list) # (?,cand_size,d)
            ft_like_aid_mha = self.attention_layer_4d("ft_like_aid_mha", photo_embs, ft_like_aid_list) # (?,cand_size,d)
            ft_ev_mha = self.attention_layer_4d("ft_ev_mha", photo_embs, ft_ev_list) # (?,cand_size,d)
            ft_ev_aid_mha = self.attention_layer_4d("ft_ev_aid_mha", photo_embs, ft_ev_aid_list) # (?,cand_size,d)
            ft_lv_mha = self.attention_layer_4d("ft_lv_mha", photo_embs, ft_lv_list) # (?,cand_size,d)
            ft_lv_aid_mha = self.attention_layer_4d("ft_lv_aid_mha", photo_embs, ft_lv_aid_list) # (?,cand_size,d)

            history_embs = tf.concat([up_click_mha, up_click_aid_mha, ft_like_mha, ft_like_aid_mha, ft_ev_mha, ft_ev_aid_mha, ft_lv_mha, ft_lv_aid_mha], axis=-1)
            history_embs = tf.layers.dense(history_embs, 256, activation=tf.nn.relu) # (?,cand_size,d)
            common_embs = tf.concat([common_embs, pxtr_mha_0, pxtr_mha_1, history_embs], axis=-1)
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

    def attention_layer_4d(self, name, query, key):
        '''
        query: (?, cand_size, dim1)
        key: (?, cand_size, key_len, dim2)
        '''
        with tf.variable_scope(f"{name}", reuse=tf.AUTO_REUSE):
            batch_size, cand_size, key_len, key_dim, query_dim = \
                tf.shape(key)[0], tf.shape(key)[1], key.shape[2], key.shape[3], query.shape[-1]
            assert query_dim == key_dim
            query = tf.expand_dims(query, axis=2) # (?, cand_size, 1, dim1)
            query = tf.reshape(query, [batch_size * cand_size, 1, query_dim])
            key = tf.reshape(key, [batch_size * cand_size, key_len, key_dim])
            attn_out, attention_weights = scaled_dot_product_attention(query, key, key, mask=None)
            attn_out = tf.reshape(attn_out, [batch_size, cand_size, query_dim]) # (?, cand_size, dim1)
            return attn_out
    
    def self_attention_4d(self, name, x):
        with tf.variable_scope(f"{name}", reuse=tf.AUTO_REUSE):
            batch_size, cand_size, len, dim = \
                tf.shape(x)[0], x.shape[1], x.shape[2], x.shape[3]
            x = tf.reshape(x, [batch_size * cand_size, len, dim])
            attn_out, attention_weights = scaled_dot_product_attention(x, x, x, mask=None)
            attn_out = tf.reshape(attn_out, [batch_size, cand_size, len, dim]) # (?, cand_size, dim1)
            return attn_out

    def model(self):
        with tf.variable_scope("evaluator", reuse=tf.AUTO_REUSE):
            def multi_task_module(name, point_wise_input, loss_names, shared_key, cand_size):
                with tf.variable_scope(f"task_{name}", reuse=tf.AUTO_REUSE):
                    ple_layer = PLE(loss_names, shared_key=shared_key, cgc_layers = 1, task_expert_num=1, shared_expert_num=2,
                                        expert_tower_dim = [128,64], gate_tower_dim = [128,64], print_ops = self.print_ops)
                    input_feature_dict = {x: point_wise_input for x in loss_names}
                    output_fea_dict = ple_layer(input_feature_dict, input_feature_dict)  # (?,cand_size,64)
                    output_list = []
                    key_output_list = []
                    for key in output_fea_dict.keys():
                        key_output_list.append(key)
                        output_list.append(output_fea_dict[key])
                    output_list = tf.stack(output_list, axis=2) # (?,cand_size,n,64)
                    # output_list = self.self_attention_4d("output_cross_attention", output_list)
                    output_list = output_attention("output_cross_attention", output_list, output_list, 64, values=output_list, need_initialize_values=False)
                    output_list = tf.split(output_list, len(output_fea_dict), axis=2)
                    for j in range(len(output_fea_dict)):
                        output_fea_dict[key_output_list[j]] = tf.squeeze(output_list[j], axis=2)
                    output_dict = {}
                    for loss_name, output in output_fea_dict.items():
                        with tf.variable_scope(f"output_mlp_{loss_name}", reuse=tf.AUTO_REUSE):
                            output  = tf.layers.dense(output, 64, activation=tf.nn.leaky_relu)
                            output = tf.layers.dense(output, 1, activation=tf.nn.sigmoid)  # (?,cand_size,1)
                            output = tf.reshape(output, [-1, cand_size]) # (?, cand_size)
                        output_dict[loss_name] = output
                    return output_dict
            input_dicts = self._parameters_dict
            com_embs = self._get_shared_features(input_dicts) # (?, cand_size, 512)
            print("com_embs:", com_embs)
            hidden_states = com_embs
            hidden_states = tf.layers.dense(hidden_states, 256, activation=tf.nn.leaky_relu)
            # candidates aware by transformer
            transformer = StackedTransformerModel(name="candidates_aware", num_layers=1, dim=256, num_heads=4, dk=256, dropout_rate=0.1, training=self._training)
            hidden_states = transformer.forward(hidden_states, training=self._training) # (?,cand_size,256)

            point_wise_input = hidden_states  # (?,cand_size,256)
            print("point_wise_input ", point_wise_input)
            point_wise_output_dict = multi_task_module("point_wise", point_wise_input, self._point_wise_tasks, shared_key="vtr", cand_size=self._candidates_size) # (?*list_num,list_size)

            return point_wise_output_dict, {}