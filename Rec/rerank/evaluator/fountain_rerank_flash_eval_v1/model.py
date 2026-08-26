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
        self.cls_embedding = tf.get_variable(name='cls_embedding', shape=[1, 64], initializer=tf.random_normal_initializer())
        self.page_embedding = tf.get_variable(name='page_embedding', shape=[201, 16], initializer=tf.random_normal_initializer())
        # self.real_show_index_embedding = tf.get_variable(name='real_show_index_embedding', shape=[501, 16], initializer=tf.random_normal_initializer())
        self.position_embedding = tf.get_variable(name='position_embedding', shape=[self._list_size, 32], initializer=tf.random_normal_initializer())
        self.print_ops = print_ops

    def _get_shared_features(self, input_dicts) -> tuple:
        """
        返回 (common_embs, pattern_bias_emb):
          common_embs:       (?, cand_size, 128)  主任务特征
          pattern_bias_emb:  (?, cand_size, bias_dim)  辅助任务 bias，由 pid/aid/did copy emb + page/rsi 词表拼成
        train 和 infer 在此函数内统一对齐 shape，外部无需区分。
        """
        with tf.variable_scope("share_bottom", reuse=tf.AUTO_REUSE):
            user_embs     = tf.concat([input_dicts[k] for k in input_dicts if k in user_fea_names], axis=-1)
            user_embs     = tf.tile(tf.expand_dims(user_embs, axis=1), [1,self._candidates_size,1]) if self._training else user_embs
            photo_embs    = tf.concat([input_dicts[k] for k in input_dicts if k in photo_fea_names], axis=-1)

            if self._training:
                pxtr_list_embs    = tf.concat([tf.expand_dims(input_dicts[k], axis=2) for k in input_dicts if k in self._pxtr_names], axis=2) # (?,cand_size,n,dim)
            else:
                pxtr_list_embs    = tf.concat([tf.expand_dims(input_dicts[k], axis=1) for k in input_dicts if k in self._pxtr_names], axis=1) # (?,n,dim)
            pxtr_embs    = tf.concat([input_dicts[k] for k in input_dicts if k in self._pxtr_names], axis=-1) # (?,cand_size,dim) infer (?,dim)
            pxtr_embs = tf.layers.dense(pxtr_embs, 128, activation=tf.nn.leaky_relu, use_bias=True)

            ft_click_list = tf.tile(tf.expand_dims(self._parameters_dict['user_fountain_profile_click_pid_list'], axis=1), [1, self._candidates_size, 1, 1]) \
                if self._training else self._parameters_dict['user_fountain_profile_click_pid_list']
            ft_click_aid_list = tf.tile(tf.expand_dims(self._parameters_dict['user_fountain_profile_click_aid_list'], axis=1), [1, self._candidates_size, 1, 1]) \
                if self._training else self._parameters_dict['user_fountain_profile_click_aid_list']
            ft_lv_list = tf.tile(tf.expand_dims(self._parameters_dict['user_fountain_profile_long_view_pid_list'], axis=1), [1, self._candidates_size, 1, 1]) \
                if self._training else self._parameters_dict['user_fountain_profile_long_view_pid_list']
            ft_lv_aid_list = tf.tile(tf.expand_dims(self._parameters_dict['user_fountain_profile_long_view_aid_list'], axis=1), [1, self._candidates_size, 1, 1]) \
                if self._training else self._parameters_dict['user_fountain_profile_long_view_aid_list']

            # --- pattern bias emb：pid(64d)/aid(64d) 各切3x16，did(32d) 切3x8，page/rsi 词表16维共享 ---
            pid_copy = input_dicts["pid_copy"]   # (?, cand_size, 64) train / (?, 64) infer
            aid_copy = input_dicts["aid_copy"]   # (?, cand_size, 64) train / (?, 64) infer
            did_copy = tf.tile(tf.expand_dims(input_dicts["did_copy"], axis=1), [1, self._candidates_size, 1]) if self._training else input_dicts["did_copy"]  # train:(?,cand_size,32) infer:(?,32)
            # context_info__page 在 infer 时是 common attr [1, 1] train 为 [?, 1]
            page_val = tf.cast(tf.clip_by_value(self._parameters_dict["context_info__page"], 0, 200), tf.int32)
            page_val = tf.tile(page_val, [1, self._candidates_size]) if not self._training else page_val # (?,1)
            page_emb = tf.tile(tf.nn.embedding_lookup(self.page_embedding, page_val), [1, self._candidates_size, 1]) if self._training else tf.nn.embedding_lookup(self.page_embedding, page_val)  # train:(?,cand_size,16) infer:(?,1,16)->squeeze later
            # rsi_val  = tf.cast(tf.clip_by_value(self._label_value_dict["context_info__real_show_index"], 0, 500), tf.int32)# (?, cand_size) train / (?,1) infer
            # rsi_emb  = tf.nn.embedding_lookup(self.real_show_index_embedding, rsi_val)  # train:(?,cand_size,16) infer:(?,1,16)

            if not self._training:
                '''
                    infer 时需要注意实际请求的 batch size = items 长度, 但计算图 batch 的 shape 由 uni_predict_fused 中 executor_batchsizes 决定,
                    如600。为了通过计算图编译, 需要 rashape (1, -1, dim), 实际 -1 为请求端发送 items 长度
                '''
                user_embs = tf.reshape(user_embs, [1, -1, user_embs.shape[-1]])
                photo_embs = tf.reshape(photo_embs, [1, -1, photo_embs.shape[-1]])
                pxtr_embs = tf.reshape(pxtr_embs, [1, -1, pxtr_embs.shape[-1]])
                print("pxtr_list_embs ", pxtr_list_embs)
                pxtr_list_embs = tf.reshape(pxtr_list_embs, [1, -1, pxtr_list_embs.shape[-2], pxtr_list_embs.shape[-1]])
                ft_click_list = tf.reshape(ft_click_list, [1, -1, ft_click_list.shape[-2], ft_click_list.shape[-1]])
                ft_click_aid_list = tf.reshape(ft_click_aid_list, [1, -1, ft_click_aid_list.shape[-2], ft_click_aid_list.shape[-1]])
                ft_lv_list = tf.reshape(ft_lv_list, [1, -1, ft_lv_list.shape[-2], ft_lv_list.shape[-1]])
                ft_lv_aid_list = tf.reshape(ft_lv_aid_list, [1, -1, ft_lv_aid_list.shape[-2], ft_lv_aid_list.shape[-1]])
                # infer 时 pid/aid/did/page/rsi 均为 2D，先 reshape 成 (1, -1, dim) 再做 3D 切片
                pid_copy = tf.reshape(pid_copy, [1, -1, pid_copy.shape[-1]])
                aid_copy = tf.reshape(aid_copy, [1, -1, aid_copy.shape[-1]])
                did_copy = tf.reshape(did_copy, [1, -1, did_copy.shape[-1]])
                page_emb = tf.reshape(page_emb, [1, -1, page_emb.shape[-1]])
                # rsi_emb  = tf.reshape(rsi_emb,  [1, -1, rsi_emb.shape[-1]])

            # 三个任务各自 concat 好，每个 (?,cand_size, 16+16+8+16+16=72)
            print("pid_copy ", pid_copy, "aid_copy", aid_copy, "did_copy", did_copy, "page_emb", page_emb)
            pattern_bias_embs = [tf.concat([pid_copy[:,:,t*16:t*16+16], aid_copy[:,:,t*16:t*16+16], did_copy[:,:,t*8:t*8+8], page_emb], axis=-1) for t in range(3)]
            print("pattern_bias_embs ", pattern_bias_embs)
            pxtr_embs = tf.layers.dense(pxtr_embs, 64, activation=tf.nn.leaky_relu, use_bias=True)
            # pxtr 投影到统一维度后做 target attention
            pxtr_list_embs_proj = tf.layers.dense(pxtr_list_embs, 64, activation=tf.nn.leaky_relu)  # (?,cand_size,11,64) or (1,N,11,64)
            photo_embs_for_pxtr = tf.layers.dense(photo_embs, 64, activation=tf.nn.leaky_relu, name="photo_proj_pxtr")
            # photo_emb queries pxtr sequence: 每个候选item从上游pxtr信号中自适应提取信息
            pxtr_mha = self.mha_layer_4d("pxtr_target_attn", photo_embs_for_pxtr, pxtr_list_embs_proj, dim_in=64, num_heads=4, dropout_rate=0.1, training=self._training)
            common_embs = photo_embs
            common_embs   = tf.layers.dense(common_embs, 64, activation=tf.nn.leaky_relu) # (?, cand_size, 96)
            # user seq X candidate cross attn
            ft_click_mha = self.attention_layer_4d("ft_click_mha", common_embs, ft_click_list) # (?,cand_size,d)
            ft_click_aid_mha = self.attention_layer_4d("ft_click_aid_mha", common_embs, ft_click_aid_list) # (?,cand_size,d)
            ft_lv_mha = self.attention_layer_4d("ft_lv_mha", common_embs, ft_lv_list) # (?,cand_size,d)
            ft_lv_aid_mha = self.attention_layer_4d("ft_lv_aid_mha", common_embs, ft_lv_aid_list) # (?,cand_size,d)
            history_embs = tf.concat([ft_click_mha, ft_click_aid_mha, ft_lv_mha, ft_lv_aid_mha], axis=-1)

            # candidates aware by transformer
            transformer = StackedTransformerModel(name="candidates_aware", num_layers=1, dim=64, num_heads=2, dk=64, dropout_rate=0.0, training=self._training)
            candidates_aware_out = transformer.forward(common_embs, training=self._training) # (?,cand_size,128)
            user_embs = tf.layers.dense(user_embs, 32, activation=tf.nn.leaky_relu)
            common_embs = tf.concat([user_embs, pxtr_embs, pxtr_mha, history_embs, candidates_aware_out], axis=-1) # (?,cand_size,d)
            common_embs = tf.layers.dense(common_embs, 128, activation=tf.nn.leaky_relu)
            return common_embs, pattern_bias_embs
    
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
    
    def mha_layer_4d(self, name, query, key, dim_in=64, num_heads=4, dropout_rate=0.0, training=False, causal_mask=False):
        with tf.variable_scope(f"{name}", reuse=tf.AUTO_REUSE):
            batch_size = tf.shape(key)[0]
            cand_size = tf.shape(key)[1]
            key_len = key.shape[2]
            query_dim = query.shape[-1]
            key_dim = key.shape[-1]
            # 将4D key reshape为3D用于multi_head_attention
            query_expanded = tf.expand_dims(query, axis=2)  # (?, cand_size, 1, query_dim)
            query_3d = tf.reshape(query_expanded, [batch_size * cand_size, 1, query_dim])  # (?*cand_size, 1, query_dim)
            key_3d = tf.reshape(key, [batch_size * cand_size, key_len, key_dim])  # (?*cand_size, key_len, key_dim)
            # multi_head_attention: Q=query, K=key, V=key
            attn_out = multi_head_attention(f"{name}_mha", query_3d, key_3d, key_3d, dim_in=dim_in, num_heads=num_heads,
                                            dropout_rate=dropout_rate, training=training, causal_mask=causal_mask)  # (?*cand_size, 1, dim_in)
            attn_out = tf.reshape(attn_out, [batch_size, cand_size, dim_in])  # (?, cand_size, dim_in)
        return attn_out

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
            attn_out = tf.reshape(attn_out, [batch_size, cand_size, len, dim]) # (?, cand_size, len, dim1)
            return attn_out

    def model(self, list_index, selected_list_index=None) -> tuple:
        """
        list_index: (?, list_num, list_size) 全部候选 list 的 item index，用于 list context awareness
        selected_list_index: (?, list_size) 训练时用于产生 loss 的目标 list（如 max_score list），
                             推理时为 None（所有 list 都需要输出）。
                             当提供时，PLE 只在该 1 个 list 的 emb 上运行，避免 30x 无效计算。
        """
        with tf.variable_scope("evaluator", reuse=tf.AUTO_REUSE):
            def multi_task_module(name, point_wise_input, loss_names, shared_key, cand_size):
                with tf.variable_scope(f"task_{name}", reuse=tf.AUTO_REUSE):
                    ple_layer = PLE(loss_names, shared_key=shared_key, cgc_layers = 1, task_expert_num=1, shared_expert_num=2,
                                        expert_tower_dim = [128, 64], gate_tower_dim = [128, 64], print_ops = self.print_ops)
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
                            # vtr label 值域为 [0, +inf)（wt_encode 编码，允许超长播放 > 1）
                            # 使用 softplus 代替 sigmoid，保证输出非负且无上界截断
                            if loss_name == "vtr":
                                output = tf.layers.dense(output, 1, activation=tf.nn.softplus)
                            else:
                                output = tf.layers.dense(output, 1, activation=tf.nn.sigmoid)
                            output = tf.reshape(output, [-1, cand_size]) # (?, cand_size)
                        output_dict[loss_name] = output
                    return output_dict
            def pattern_module(name, x, bias_embs):
                """
                bias_embs: list of 3 x (?,list_size,72)
                  [0]=continue bias, [1]=pair_vtr_gain bias, [2]=tri_vtr_gain bias
                  每个: pid16 + aid16 + did8 + page16 + rsi16 = 72
                """
                with tf.variable_scope(f"task_{name}", reuse=tf.AUTO_REUSE):
                    x = tf.stop_gradient(x)  # 阻断辅助任务梯度流向主任务 list_emb
                    def tower(tower_name, fea, act=None):
                        with tf.variable_scope(tower_name, reuse=tf.AUTO_REUSE):
                            pattern_dnn = tf.layers.dense(fea, 64, activation=tf.nn.leaky_relu)
                            pattern_dnn = tf.layers.dense(pattern_dnn, 64, activation=tf.nn.leaky_relu)
                            pattern_dnn = tf.layers.dense(pattern_dnn, 1, activation=act)
                            return tf.squeeze(pattern_dnn, axis=-1)
                    # position embedding: index [0, list_size)，shape (list_size, 32) -> (?, list_size, 32)
                    pos_indices = tf.range(self._list_size)  # (list_size,)
                    pos_emb = tf.nn.embedding_lookup(self.position_embedding, pos_indices)  # (list_size, 32)
                    pos_emb = tf.tile(tf.expand_dims(pos_emb, axis=0), [tf.shape(x)[0], 1, 1])  # (?, list_size, 32)
                    # continue（task 0）
                    xb0 = tf.concat([x, bias_embs[0], pos_emb], axis=-1)
                    prefix_emb = tf.stack([tf.reduce_mean(xb0[:, :i+1, :], axis=1) for i in range(self._list_size - 1)], axis=1)
                    # pair_vtr_gain（task 1）
                    xb1 = tf.concat([x, bias_embs[1], pos_emb], axis=-1)
                    pair_fea = tf.concat([xb1[:, :-1, :], xb1[:, 1:, :], x[:, :-1, :] * x[:, 1:, :], tf.abs(x[:, :-1, :] - x[:, 1:, :])], axis=-1)
                    # tri_vtr_gain（task 2）
                    xb2 = tf.concat([x, bias_embs[2], pos_emb], axis=-1)
                    tri_fea  = tf.concat([xb2[:, :-2, :], xb2[:, 1:-1, :], xb2[:, 2:, :], x[:, :-2, :] * x[:, 1:-1, :], x[:, 1:-1, :] * x[:, 2:, :], tf.abs(x[:, :-2, :] - x[:, 2:, :])], axis=-1)
                    return {"continue": tower("continue", prefix_emb, tf.nn.sigmoid), "pair_vtr_gain": tower("pair_vtr_gain", pair_fea), "tri_vtr_gain": tower("tri_vtr_gain", tri_fea)}
            input_dicts = self._parameters_dict
            self._list_index = list_index
            com_embs, pattern_bias_embs = self._get_shared_features(input_dicts)  # (?,cand_size,128), list of 3x(?,cand_size,72)
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
            cls_embedding = tf.tile(tf.expand_dims(self.cls_embedding, axis=0),
                                [batch_size, self._list_num, 1]) #(?,list_num,dim)
            cls_embedding = tf.expand_dims(cls_embedding, axis=2) #(?,list_num,1,dim)
            print("cls_embedding ", cls_embedding)
            list_emb = tf.concat([cls_embedding,list_emb],axis=2) #(?,list_num,list_size+1,dim)
            print("list_emb concat cls ", list_emb)
            list_emb_dim = list_emb.shape[-1]
            # transformer = StackedTransformerModel(name="list_aware", num_layers=1, dim=64, num_heads=2, dk=64, dropout_rate=0.0, training=self._training)
            # list_emb = transformer.forward(tf.reshape(list_emb, [batch_size * self._list_num, self._list_size + 1, list_emb_dim]), training=self._training)
            # list_emb = tf.reshape(list_emb, [batch_size, self._list_num, self._list_size + 1, list_emb_dim])
            list_emb = self.self_attention_4d("list_aware_attention", list_emb)
            # list_emb: (?, list_num, list_size+1, dim)，其中 [:, :, 0, :] 是 CLS，[:, :, 1:, :] 是 item emb

            if selected_list_index is not None:
                # 训练时：只对 selected_list_index 指定的 1 个 list 跑 PLE，避免 30x 无效计算
                # selected_list_index: (?, ) int32，取值范围 [0, list_num)
                # 取出目标 list 的 item emb (位置1开始，跳过CLS): (?, list_size, dim)
                selected_list_index = tf.reshape(selected_list_index, [-1, 1])
                selected_list_emb = tf.squeeze(tf.gather(list_emb[:, :, 1:, :], selected_list_index, axis=1, batch_dims=1), axis=1)  # (?, list_size, dim)
                print("selected_list_emb for PLE ", selected_list_emb)
                # 从 pattern_bias_embs 中取 selected list 对应的 item bias
                # 与 list_emb 保持一致：先垫 zeros（index=0），再用 indices gather_nd，再按 selected_list_index 取目标 list
                bias_dim = pattern_bias_embs[0].shape[-1]
                bias_zeros = tf.zeros([batch_size, 1, bias_dim], dtype=tf.float32)
                selected_bias_embs = [
                    tf.squeeze(tf.gather(tf.gather_nd(tf.concat([bias_zeros, b], axis=1), indices), selected_list_index, axis=1, batch_dims=1), axis=1)          # (?,list_size,72)
                    for b in pattern_bias_embs
                ]
                # PLE 在单个 list 上运行：cand_size = list_size
                point_wise_output_dict = multi_task_module("point_wise", selected_list_emb, self._point_wise_tasks, shared_key="vtr", cand_size=self._list_size)
                pattern_output_dict = pattern_module("pattern", selected_list_emb, selected_bias_embs)
            else:
                # 推理时：对所有 list 跑 PLE
                point_wise_input = tf.reshape(list_emb[:, :, 1:, :], [batch_size * self._list_num, self._list_size, list_emb_dim])
                print("point_wise_input ", point_wise_input)
                point_wise_output_dict = multi_task_module("point_wise", point_wise_input, self._point_wise_tasks, shared_key="vtr", cand_size=self._list_size)
                # 推理时同样按 indices 取各 list 对应 item 的 bias，reshape 成 (?*list_num, list_size, 72)
                bias_dim = pattern_bias_embs[0].shape[-1]
                infer_bias_embs = [
                    tf.reshape(tf.gather_nd(tf.concat([tf.zeros([batch_size, 1, bias_dim], dtype=tf.float32), b], axis=1), indices), [batch_size * self._list_num, self._list_size, bias_dim])
                    for b in pattern_bias_embs
                ]
                pattern_output_dict = pattern_module("pattern", point_wise_input, infer_bias_embs)

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
            list_wise_output_dict.update(pattern_output_dict)

            return point_wise_output_dict, list_wise_output_dict

