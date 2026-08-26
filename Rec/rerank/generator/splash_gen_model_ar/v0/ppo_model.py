import sys
from numpy import dtype
import tensorflow as tf
from modules_ import *
from feature_attr_extract import user_fea_names,photo_fea_names,source_fea_names,dense_features_config

    
class StackedTransformerModel():
    def __init__(self, num_layers, dim, num_heads, dk, dropout_rate, k, training=False):
        '''
        dim: query 的维度
        dk: key 投影矩阵的维度
        '''
        super(StackedTransformerModel, self).__init__()
        self.num_layers = num_layers
        self.k = k
        self.dim = dim
        self.layers = [EncoderLayer(f"transformer_layer_{i}", dim, num_heads, dk, dropout_rate) for i in range(num_layers)]
        self.decoder_layers = [DecoderLayer(f"position_layer_{i}", dim, num_heads, dk, dropout_rate) for i in range(num_layers)]
        
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


class GenModel:
    def __init__(self, parameters_dict, label_value_dict, dense_value_dict, print_ops, list_size, candidates_size, list_num, point_wise_tasks, list_wise_tasks, dim=32, extra_param_dict= None, training=True):
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
            "photo_author_gender",
            "photo_hetu_tag_level1_list",
            "photo_hetu_tag_level2_list",
            "photo_hetu_tag_level3_list",
            "photo_hetu_tag_level5_list",
            "photo_tag",
            "photo_duration_ms",
            "photo_upload_type",
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
            "emp_fountain_follow_count",
            "emp_fountain_long_play_count",
            "photo_emp_fountain_ltr",
            "photo_emp_fountain_wtr",
            "photo_emp_fountain_avg_fintr",
        ]
        self._parameters_dict = parameters_dict
        self._label_value_dict = label_value_dict
        self._dense_value_dict = dense_value_dict
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
            shape=[1, 64],
            initializer=tf.random_normal_initializer()
        )
        self.position_embeddings = tf.get_variable(
            name='position_embeddings', 
            shape=[list_size, 64], 
            initializer=tf.random_normal_initializer()
        )

        # Create [sos] and [eos] embeddings
        self.sos_embedding = tf.get_variable(
            "sos_embedding", shape=[1, 64], initializer=tf.random_uniform_initializer()
        )
        self.eos_embedding = tf.get_variable(
            "eos_embedding", shape=[1, 64], initializer=tf.random_uniform_initializer()
        )
        self.pad_embedding = tf.get_variable(
            "pad_embedding", shape=[1, 64], initializer=tf.random_uniform_initializer()
        )
        self.print_ops = print_ops

    def _z_score(self, x):
        mean, std = tf.reduce_mean(x), tf.math.reduce_std(x)
        x = (x - mean) / (std + 1e-7)
        return x
    def _min_max_score(self, x):
        min, max = tf.reduce_min(x), tf.math.reduce_max(x)
        x = (x - min) / (max - min + 1e-7)
        return x

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

            photo_attr_embs    = tf.concat([self._parameters_dict[k] for k in self._photo_attr_names], axis=-1) # (?,cand_size,dim) infer (?,dim)
            photo_emp_embs = []
            for x in self._photo_emp_explore_names + self._photo_emp_fountain_names:
                emb, bucket_id = self.get_bucket_emb_from_sorted_boundaries(x) # (?, cand_size, 4)
                photo_emp_embs.append(emb)
            photo_emp_embs = tf.concat(photo_emp_embs, axis=-1) #  (?, cand_size, dim)
            if self._training:
                pxtr_list_embs    = tf.concat([tf.expand_dims(self._parameters_dict[k], axis=2) for k in self._parameters_dict if k in self._pxtr_names], axis=2) # (?,cand_size,n,dim)
            else:
                pxtr_list_embs    = tf.concat([tf.expand_dims(self._parameters_dict[k], axis=1) for k in self._parameters_dict if k in self._pxtr_names], axis=1) # (?,n,dim)
            pxtr_embs    = tf.concat([self._parameters_dict[k] for k in self._parameters_dict if k in self._pxtr_names], axis=-1) # (?,cand_size,dim) infer (?,dim)
            pxtr_embs = tf.layers.dense(pxtr_embs, 128, activation=tf.nn.leaky_relu, use_bias=True)

            ft_click_list = tf.tile(tf.expand_dims(self._parameters_dict['user_fountain_profile_click_pid_list'], axis=1), [1, self._candidates_size, 1, 1]) \
                if self._training else self._parameters_dict['user_fountain_profile_click_pid_list']
            ft_click_aid_list = tf.tile(tf.expand_dims(self._parameters_dict['user_fountain_profile_click_aid_list'], axis=1), [1, self._candidates_size, 1, 1]) \
                if self._training else self._parameters_dict['user_fountain_profile_click_aid_list']
            ft_ev_list = tf.tile(tf.expand_dims(self._parameters_dict['user_fountain_profile_effective_view_pid_list'], axis=1), [1, self._candidates_size, 1, 1]) \
                if self._training else self._parameters_dict['user_fountain_profile_effective_view_pid_list']
            ft_ev_aid_list = tf.tile(tf.expand_dims(self._parameters_dict['user_fountain_profile_effective_view_aid_list'], axis=1), [1, self._candidates_size, 1, 1]) \
                if self._training else self._parameters_dict['user_fountain_profile_effective_view_aid_list']
            if not self._training:
                '''
                    infer 时需要注意实际请求的 batch size = items 长度, 但计算图 batch 的 shape 由 uni_predict_fused 中 executor_batchsizes 决定,
                    如600。为了通过计算图编译, 需要 rashape (1, -1, dim), 实际 -1 为请求端发送 items 长度
                '''
                user_embs = tf.reshape(user_embs, [1, -1, user_embs.shape[-1]])
                source_embs = tf.reshape(source_embs, [1, -1, source_embs.shape[-1]])
                photo_attr_embs = tf.reshape(photo_attr_embs, [1, -1, photo_attr_embs.shape[-1]])
                photo_embs = tf.reshape(photo_embs, [1, -1, photo_embs.shape[-1]])
                photo_emp_embs = tf.reshape(photo_emp_embs, [1, -1, photo_emp_embs.shape[-1]])
                pxtr_embs = tf.reshape(pxtr_embs, [1, -1, pxtr_embs.shape[-1]])
                print("pxtr_list_embs ", pxtr_list_embs)
                pxtr_list_embs = tf.reshape(pxtr_list_embs, [1, -1, pxtr_list_embs.shape[-2], pxtr_list_embs.shape[-1]])
                ft_click_list = tf.reshape(ft_click_list, [1, -1, ft_click_list.shape[-2], ft_click_list.shape[-1]])
                ft_click_aid_list = tf.reshape(ft_click_aid_list, [1, -1, ft_click_aid_list.shape[-2], ft_click_aid_list.shape[-1]])
                ft_ev_list = tf.reshape(ft_ev_list, [1, -1, ft_ev_list.shape[-2], ft_ev_list.shape[-1]])
                ft_ev_aid_list = tf.reshape(ft_ev_aid_list, [1, -1, ft_ev_aid_list.shape[-2], ft_ev_aid_list.shape[-1]])
            pxtr_list_embs = tf.layers.dense(pxtr_list_embs, 24, activation=tf.nn.leaky_relu)
            query_emb = tf.layers.dense(photo_emp_embs, 24, activation=tf.nn.leaky_relu) # (?,cand_size,d)
            pxtr_mha_2 = self.linear_attention("pxtr_mha_2", query_emb, pxtr_list_embs, nh=2, dim=16) # (?,cand_size,d)
            common_embs = tf.concat([photo_embs, pxtr_mha_2], axis=-1)
            common_embs   = tf.layers.dense(common_embs, 64, activation=tf.nn.leaky_relu) # (?, cand_size, 96)
            # user seq X candidate cross attn
            ft_click_mha = self.linear_attention("ft_click_mha", common_embs, ft_click_list, nh=1, dim=32) # (?,cand_size,d)
            ft_click_aid_mha = self.linear_attention("ft_click_aid_mha", common_embs, ft_click_aid_list, nh=1, dim=32) # (?,cand_size,d)
            ft_ev_mha = self.linear_attention("ft_ev_mha", common_embs, ft_ev_list, nh=1, dim=32) # (?,cand_size,d)
            ft_ev_aid_mha = self.linear_attention("ft_ev_aid_mha", common_embs, ft_ev_aid_list, nh=1, dim=32) # (?,cand_size,d)
            history_embs = tf.concat([ft_click_mha, ft_click_aid_mha, ft_ev_mha, ft_ev_aid_mha], axis=-1)

            # candidates aware by transformer
            transformer = StackedTransformerModel(name="candidates_aware", num_layers=1, dim=64, num_heads=2, dk=64, dropout_rate=0.0, training=self._training)
            candidates_aware_out = transformer.forward(common_embs, training=self._training) # (?,cand_size,128)
            user_embs = tf.layers.dense(tf.concat([user_embs, source_embs], axis=-1), 32, activation=tf.nn.leaky_relu)
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

    def linear_attention(self, name, query, key, nh, dim):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            n = key.shape[2]
            batch_size = tf.shape(key)[0]
            Q = tf.layers.dense(query, nh * dim, activation=tf.nn.elu)  # [batch_size, query_length, hidden_dim]
            dense_q = Q
            Q = tf.nn.l2_normalize(tf.stack(tf.split(Q, nh, axis=2)), axis=3)
            K = tf.layers.dense(key, nh * dim, activation=tf.nn.elu)  # [batch_size, sequence_length, hidden_dim]
            K = tf.nn.l2_normalize(tf.stack(tf.split(K, nh, axis=2)), axis=3)
            V = tf.layers.dense(key, nh * dim)  # [batch_size, sequence_length, n_classes]
            V = tf.stack(tf.split(V, nh, axis=2))  # (head_num, batch_size, sequence_length, att_embedding_size)
            attention = tf.matmul(K, V, transpose_a=True)  # [batch_size, sequence_length, sequence_length]

            output = tf.matmul(Q, attention)  # [head_num, batch_size, query_length, n_classes]
            output = tf.transpose(output, perm=[1, 2, 0, 3])  # (batch_size, query_length ,hn, att_embedding_sizev)
            output = tf.reshape(output, [batch_size, n, nh * dim])
            return output, dense_q
    
    def self_attention_4d(self, name, x):
        with tf.variable_scope(f"{name}", reuse=tf.AUTO_REUSE):
            batch_size, cand_size, len, dim = \
                tf.shape(x)[0], x.shape[1], x.shape[2], x.shape[3]
            x = tf.reshape(x, [batch_size * cand_size, len, dim])
            attn_out, attention_weights = scaled_dot_product_attention(x, x, x, mask=None)
            attn_out = tf.reshape(attn_out, [batch_size, cand_size, len, dim]) # (?, cand_size, dim1)
            return attn_out
    
    def gumbel_softmax(self, logits, tau=1.0, hard=False, dim=-1):
        def sample_gumbel(shape):
            """Sample from Gumbel(0, 1)"""
            uniform_samples = tf.random_uniform(shape, minval=0, maxval=1)
            return -tf.log(-tf.log(uniform_samples + 1e-20) + 1e-20)
        
        # Sample Gumbel noise
        gumbels = sample_gumbel(tf.shape(logits))
        gumbels = (logits + gumbels) / tau 
        y_soft = tf.nn.softmax(gumbels, axis=dim)

        if hard:
            # Straight through.
            index = tf.argmax(y_soft, axis=dim)
            y_hard = tf.one_hot(index, depth=tf.shape(logits)[dim], dtype=logits.dtype)
            y_hard = tf.reshape(y_hard, tf.shape(logits))
            ret = tf.stop_gradient(y_hard - y_soft) + y_soft
        else:
            ret = y_soft
        return ret

    def choose_item(self, decoder_emb, vocab_emb, method=0, use_gumbel_softmax=False, tau=1.0, hard=True):
        # 端到端的情况下需要打开 gumbel softmax，具备采样能力
        if not self._training:
            infer_batch_size = tf.shape(decoder_emb)[0]
            infer_beam_size = decoder_emb.shape[1]
            infer_list_size = tf.shape(decoder_emb)[2]
            infer_vocab_size = tf.shape(vocab_emb)[2]
            infer_dim = vocab_emb.shape[-1]
            decoder_emb = tf.reshape(decoder_emb, [infer_batch_size * infer_beam_size, infer_list_size, infer_dim])
            vocab_emb = tf.reshape(vocab_emb, [infer_batch_size * infer_beam_size, infer_vocab_size, infer_dim])
        if method == 0:
            # MLP 预测每个step对应选择哪个item [0, 1, 2, 3, EOT]
            with tf.variable_scope("predict_token_nn", reuse=tf.AUTO_REUSE):
                batch_size = tf.shape(decoder_emb)[0]
                list_size = tf.shape(decoder_emb)[1]
                vocab_size = tf.shape(vocab_emb)[1]
                dim = vocab_emb.shape[-1]
                decoder_emb = tf.expand_dims(decoder_emb, axis=2) # (?,list_size,1,dim)
                vocab_emb = tf.expand_dims(vocab_emb, axis=1) # (?,1,vocab_size,dim)
                decoder_emb = tf.broadcast_to(decoder_emb, [batch_size, list_size, vocab_size, dim])
                vocab_emb = tf.broadcast_to(vocab_emb, [batch_size, list_size, vocab_size, dim])
                concat_emb = tf.concat([decoder_emb, vocab_emb], axis=-1) # (?,list_size,vocab_size,64)
                predict = tf.layers.dense(concat_emb, 128, activation=tf.nn.relu)
                predict = tf.layers.dense(predict, 64, activation=tf.nn.relu)
                predict = tf.layers.dense(predict, 1, activation=tf.nn.sigmoid, name="prob_layer") # (?,list_size,vocab_size,1)
                predict = tf.squeeze(predict, axis=-1) # (?,list_size,vocab_size)
        elif method == 1:
            # cosine 选取
            with tf.variable_scope("predict_cosine", reuse=tf.AUTO_REUSE):
                predict = tf.matmul(decoder_emb, tf.transpose(vocab_emb,  perm=[0, 2, 1])) # (?, list_size, vocab_size)
                # predict = tf.nn.softmax(predict, axis=-1)
        if use_gumbel_softmax:
            predict = self.gumbel_softmax(predict, tau=tau, hard=hard)
        else:
            predict = tf.nn.softmax(predict, axis=-1) # (?,list_size,candidates_size+3)
        if not self._training:
            predict = tf.reshape(predict, [infer_batch_size, infer_beam_size, infer_list_size, infer_vocab_size])
            print("xxx ", predict)
        return predict

    def cal_batch_advantage(self, reward, mask):
        mask = tf.cast(mask, reward.dtype)
        valid_cnt = tf.reduce_sum(mask)
        mean = tf.reduce_sum(reward * mask) / (valid_cnt + 1e-8)
        variance = (reward - mean) ** 2 * mask
        std = tf.sqrt(tf.reduce_sum(variance) / (valid_cnt + 1e-8))
        advantages = (reward - mean) / (std + 1e-8)
        return advantages

    def bpr_loss(self, output, score, threshold, mask):
        with tf.variable_scope("bpr_loss", reuse=tf.AUTO_REUSE):
            # 生成配对矩阵
            output_i = tf.expand_dims(output, 2)
            output_j = tf.expand_dims(output, 1)
            score_i = tf.expand_dims(score, 2)
            score_j = tf.expand_dims(score, 1)
            score_diff = score_i - score_j
            pairwise_labels = tf.cast(score_diff >= threshold, tf.float32)
            pairwise_label_mask = tf.logical_or(score_diff > threshold, score_diff < -threshold) # 在阈值内的pair不计算loss
            logit_diff = tf.sigmoid(output_i - output_j)
            # 生成有效掩码
            mask_i = tf.expand_dims(mask, 2)
            mask_j = tf.expand_dims(mask, 1)
            valid_pair_mask = tf.logical_and(mask_i, mask_j)
            # 生成混合掩码
            # random_mask = tf.random.uniform(tf.shape(valid_pair_mask)) < 0.05
            # activated_mask = tf.logical_and(random_mask, tf.logical_not(valid_pair_mask))
            # final_mask = tf.logical_or(valid_pair_mask, activated_mask)
            # final_mask = tf.logical_and(final_mask, pairwise_label_mask)
            final_mask = valid_pair_mask
            # 计算BPR损失
            bpr_loss = -tf.log(logit_diff) * pairwise_labels
            print("bpr_loss", bpr_loss)
            bpr_loss = tf.where(final_mask, bpr_loss, tf.zeros_like(bpr_loss, dtype=tf.float32))
            return bpr_loss

    def model(self, training=True, decode_method="beam_search", beam_size=1, max_length=10):
        self._training = training
        with tf.variable_scope("prepare", reuse=tf.AUTO_REUSE):
            common_embs = self._get_shared_features(self._parameters_dict) # (?,60,d)
            batch_size = tf.shape(common_embs)[0]
            # 添加特殊token的embedding
            pad_embedding = tf.tile(tf.expand_dims(self.pad_embedding, axis=0), #(?,1,32)
                                [batch_size, 1, 1])
            sos_embedding = tf.tile(tf.expand_dims(self.sos_embedding, axis=0),  #(?,1,32)
                                [batch_size, 1, 1])
            eos_embedding = tf.tile(tf.expand_dims(self.eos_embedding, axis=0), #(?,1,32)
                                [batch_size, 1, 1])

            sos_token = tf.tile(tf.constant(1, shape=[1, 1], dtype=tf.int32), [batch_size, 1]) #(?,1)
            eos_token = tf.tile(tf.constant(self.candidates_size + 2, shape=[1, 1], dtype=tf.int32), [batch_size, 1])
            pad_token = tf.tile(tf.constant(0, shape=[1, 1], dtype=tf.int32), [batch_size, 1]) #(?,1) all 0
            self.item_embs = common_embs
            self.photo_embs = tf.concat([pad_embedding, sos_embedding, common_embs, eos_embedding], axis=1) # (?,candidates_size+3,32)

            if self._training:
                show_label = self._label_value_dict["show_label"][:,:self._list_size] # (?,list_size)，截断为 list_size 个
                like_label = self._label_value_dict["like_label"][:,:self._list_size] # (?,list_size)，截断为 list_size 个
                follow_label = self._label_value_dict["follow_label"][:,:self._list_size] # (?,list_size)，截断为 list_size 个
                forward_label = self._label_value_dict["forward_label"][:,:self._list_size] # (?,list_size)，截断为 list_size 个
                forward_label = self._label_value_dict["forward_label"][:,:self._list_size] # (?,list_size)，截断为 list_size 个
                finish_label = self._label_value_dict["finish_label"][:,:self._list_size] # (?,list_size)，截断为 list_size 个
                slide_label = self._label_value_dict["slide_label"][:,:self._list_size] # (?,list_size)，截断为 list_size 个
                evtr_label, lvtr_label = self._label_value_dict["evtr_label"][:,:self._list_size], self._label_value_dict["lvtr_label"][:,:self._list_size]
                playtime = self._label_value_dict["play_time_s"][:,:self._list_size] # (?,list_size)

                indices_shape = tf.shape(realshow_label)
                col_indices = tf.tile(tf.expand_dims(tf.range(indices_shape[1]),0),[indices_shape[0],1]) + 2 # (?,list_size) 从第2个起
                realshow_indices = tf.cast(col_indices * realshow_label,dtype=tf.int32) # label为0 1，过滤了未曝光的index
                inputs = tf.concat([sos_token, realshow_indices], axis=1) # (?,list_size+1)
                outputs = tf.concat([realshow_indices, eos_token], axis=1) # (?,list_size+1)
                batch_indices = tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), [1, self._list_size+1]) # (?, self._list_size+1)
                gather_indices = tf.stack([batch_indices, inputs], axis=-1) # (?, self._list_size+1, 2)

        with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
            # 初始化transformer模型
            model = StackedTransformerModel(num_layers=1, dim=64, num_heads=2, dk=64, dropout_rate=0.1, k=6)
            hidden_states = model.forward(common_embs, training=True)
            encoder_output = common_embs
            print("encoder output shape ",encoder_output.shape) # (?,candidates_size,32)
            if self._training:
                item_embeddings = tf.gather_nd(self.photo_embs, gather_indices) # (?,candidates_size+3,32) 中查找对应 list idx 的 emb
                item_embedding = model.forward_decoder(hidden_states, item_embeddings, training=True) # (?,list_size+1,32)
                print("item_embedding shape ", item_embedding.shape)

                # 从候选集选取 item, 0: nn, 1: cosine; 是否进行采样
                predict = self.choose_item(item_embedding, self.photo_embs, method=0, use_gumbel_softmax=False, tau=0.1, hard=True) # (?,list_size+1,candidates_size+3)
                output_indices = tf.expand_dims(outputs, axis=2) # (?,list_size+1,1)
                pos_output = tf.batch_gather(predict, output_indices) # (?,list_size+1,1) 拿到真实index对应的score, 非全局emb Matrix 需要使用batch_gather
                pos_output = tf.squeeze(pos_output, axis=-1) #(?,list_size+1)

                zeros = tf.zeros([batch_size, 1], dtype=tf.int32) # mask EOS token
                realshow_label = tf.concat([realshow_label, zeros], axis=1)
                print("outputs shape", outputs.shape)
                realshow_label = tf.cast(realshow_label,dtype=tf.float32)
                valid_pos_output = -tf.log(tf.clip_by_value(pos_output, 1e-10, 1.0)) * realshow_label #(?,list_size+1)
                gen_loss = tf.reduce_mean(tf.reduce_sum(valid_pos_output, axis=-1))
                # reward_loss = tf.reduce_mean(tf.reduce_sum(valid_pos_output * weight, axis=-1))
            
            else:
                def beam_search(model, encoder_output, sos_token, eos_token, pad_token, beam_size=3, max_length=6):
                    """
                    实现 Beam Search 的自回归解码，同时添加每条生成路径中 token 不能重复的约束。
                    Args:
                        model: 生成模型，包含 encoder 和 decoder。
                        encoder_output: 编码器的输出，形状 [batch_size, vocab_size, dim]。
                        sos_token: 起始 token 的 ID，形状 [batch_size, 1]。
                        eos_token: 结束 token 的 ID，形状 [batch_size, 1]。
                        pad_token: 填充 token 的 ID，形状 [batch_size, 1]。
                        beam_size: Beam Search 的宽度，表示保留的候选路径数量。
                        max_length: 最大生成长度。

                    Returns:
                        best_sequences: 形状 [batch_size, max_length]，表示生成的序列。
                    """
                    batch_size, vocab_size, vocab_dim = tf.shape(self.photo_embs)[0], tf.shape(self.photo_embs)[1], self.photo_embs.shape[-1]
                    
                    # 初始化每束的生成序列、分数和完成状态
                    sequences = tf.tile(tf.expand_dims(sos_token, axis=1), [1, beam_size, 1])  # [batch_size, beam_size, 1]
                    scores = tf.zeros([batch_size, beam_size])  # [batch_size, beam_size]
                    reward = tf.zeros([batch_size, beam_size])  # [batch_size, beam_size]
                    
                    # repeat encoder_output for each beam 
                    encoder_output = tf.tile(tf.expand_dims(encoder_output, axis=1), [1, beam_size, 1, 1])  # [batch_size, beam_size, candidate_size, dim])
                    item_embs = tf.tile(tf.expand_dims(self.item_embs, axis=1), [1, beam_size, 1, 1])  # [batch_size, beam_size, vocab_size, dim])
                    photo_embs = tf.tile(tf.expand_dims(self.photo_embs, axis=1), [1, beam_size, 1, 1])  # [batch_size, beam_size, vocab_size, dim])
                    probs = []
                    # 开始 Beam Search
                    for step in range(max_length):
                        # position_reward = max_length - step
                        # position_reward = tf.concat([sos_reward, pad_reward, ones * position_reward, eos_reward], axis=-1)
                        # 当前所有序列的 embedding
                        batch_indices = tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), [1, beam_size]) # [batch_size, beam_size]
                        beam_indices = tf.tile(tf.expand_dims(tf.range(beam_size), axis=0), [batch_size, 1]) # [batch_size, beam_size]
                        # 扩展batch和beam维度以匹配sequences的每个位置
                        batch_indices = tf.expand_dims(batch_indices, axis=2)  # [batch_size, beam_size, 1]
                        beam_indices = tf.expand_dims(beam_indices, axis=2)    # [batch_size, beam_size, 1]
                        # 复制到已生成序列长度
                        seq_length = tf.shape(sequences)[2]
                        batch_indices = tf.tile(batch_indices, [1, 1, seq_length])  # [batch_size, beam_size, seq_length]
                        beam_indices = tf.tile(beam_indices, [1, 1, seq_length])    # [batch_size, beam_size, seq_length]
                        # 提取已生成序列的embedding
                        gather_indices = tf.stack([batch_indices, beam_indices, sequences], axis=-1) # [batch_size, beam_size, seq_length, 3]
                        # decoder_input = tf.gather_nd(encoder_output, gather_indices)  # [batch_size, beam_size, candidates_size+3, dim]
                        decoder_input = tf.gather_nd(photo_embs, gather_indices) # (?, beam_size,candidates_size+3,32) 中查找对应 list idx 的 emb
                        decoder_dim = decoder_input.shape[-1]
                        # decoder forward
                        enc_output_3d = tf.reshape(encoder_output, [batch_size * beam_size, tf.shape(encoder_output)[-2], vocab_dim])
                        dec_input_3d = tf.reshape(decoder_input, [batch_size * beam_size, seq_length, decoder_dim])
                        # print("enc_output_3d ", enc_output_3d)
                        # print("dec_input_3d ", dec_input_3d)
                        decoder_output = model.forward_decoder(enc_output_3d, dec_input_3d, training=self._training)  # [batch_size * beam_size, seq_length, dim]
                        decoder_output = tf.reshape(decoder_output, [batch_size, beam_size, seq_length, decoder_dim])
                        # print("decoder_output ", decoder_output)
                        # # nce loss 不实际计算
                        nce_loss = self.nce_loss(decoder_output[:, 0, :-1, :], item_embs[:, 0, :, :])
                        # 计算 logits
                        # logits = tf.matmul(self.photo_embs, tf.transpose(decoder_output, perm=[0, 1, 3, 2])) # [batch_size, beam_size, vocab_size, seq_length]
                        logits = self.choose_item(decoder_output, photo_embs, method=0, use_gumbel_softmax=False, tau=0.1, hard=True) # [batch_size, beam_size, seq_length, vocab_size]
                        next_token_logits = logits[:, :, -1, :]  # [batch_size, beam_size, vocab_size]
                        
                        # 选择下一个token

                        tau = 1.0
                        next_token_probs = tf.nn.softmax(next_token_logits/tau, axis=-1)  # [batch_size, beam_size, vocab_size]
                        probs.append(next_token_probs)
                        log_probs = tf.math.log(next_token_probs+1e-9)  # 转换为 log 概率
                        # cur_reward = next_token_probs * tf.tile(tf.expand_dims(preward, axis=1), [1, beam_size, 1])  # 转换为 log 概率 [batch_size, beam_size, vocab_size]

                        # 初始化 used_tokens，每个束内的 token 初始状态为未使用
                        used_token = tf.zeros([batch_size, beam_size, vocab_size], dtype=tf.bool)  # [batch_size, beam_size, vocab_size]
                        # 将每束的sos、eos、pad token 的 used_tokens 置为 True
                        batch_indices, beam_indices = tf.repeat(tf.range(batch_size), beam_size), tf.tile(tf.range(beam_size), [batch_size])  # [batch_size * beam_size]

                        # 特殊token索引
                        sos_indices = tf.repeat(tf.squeeze(sos_token, axis=1), beam_size)  # [batch_size * beam_size]
                        eos_indices = tf.repeat(tf.squeeze(eos_token, axis=1), beam_size)  # [batch_size * beam_size]
                        pad_indices = tf.repeat(tf.squeeze(pad_token, axis=1), beam_size)  # [batch_size * beam_size]
                        
                        # 更新used_tokens
                        used_token = tf.tensor_scatter_nd_update(
                            used_token,
                            tf.stack([batch_indices, beam_indices, sos_indices], axis=1),
                            tf.ones([batch_size * beam_size], dtype=tf.bool)
                        )
                        used_token = tf.tensor_scatter_nd_update(
                            used_token,
                            tf.stack([batch_indices, beam_indices, eos_indices], axis=1),
                            tf.ones([batch_size * beam_size], dtype=tf.bool)
                        )
                        used_token = tf.tensor_scatter_nd_update(
                            used_token,
                            tf.stack([batch_indices, beam_indices, pad_indices], axis=1),
                            tf.ones([batch_size * beam_size], dtype=tf.bool)
                        ) # (?, 1, ?)
                        # print("used_token shape",used_token.shape)

                        one_hot = tf.one_hot(sequences, vocab_size, on_value=True, off_value=False)  # shape: [batch_size, beam_size, seq_len, vocab_size]
                        used_token_tmp = tf.reduce_any(one_hot, axis=2)  # shape: [batch_size, beam_size, vocab_size]
                        used_token = tf.logical_or(used_token_tmp, used_token)
                        
                        # 根据used_token将已生成tokn的分数设为-inf
                        log_probs = tf.where(used_token, tf.fill(tf.shape(log_probs), float('-inf')), log_probs)  # [batch_size, beam_size, vocab_size]
                        # used_reward = tf.where(used_token, tf.fill(tf.shape(log_probs), float('-inf')), cur_reward)  # [batch_size, beam_size, vocab_size]
                        # 计算总分数  (当前路径分数 + 新 token 的分数)
                        scores = tf.expand_dims(scores, axis=-1) + log_probs  # [batch_size, beam_size, vocab_size]
                        # reward = tf.expand_dims(reward, axis=-1) + cur_reward + used_reward  # [batch_size, beam_size, vocab_size]
                        
                        # topk最高分数 
                        if step == 0:
                            # 第一步直接取 topk 不同 idx
                            top_k_scores, top_k_indices = tf.math.top_k(scores[:, 0, :], k=beam_size, sorted=True)  # [batch_size,beam_size)
                            # top_k_reward, top_k_indices = tf.math.top_k(reward[:, 0, :], k=beam_size, sorted=True)  # [batch_size,beam_size)
                        else:
                            scores_flat = tf.reshape(scores, [batch_size, -1]) # [batch_size, beam_size * vocab_size]
                            top_k_scores, top_k_indices = tf.math.top_k(scores_flat, k=beam_size, sorted=True)  # [batch_size,beam_size)
                            # reward_flat = tf.reshape(reward, [batch_size, -1]) # [batch_size, beam_size * vocab_size]
                            # top_k_reward, top_k_indices = tf.math.top_k(reward_flat, k=beam_size, sorted=True)  # [batch_size,beam_size)
                        # 更新序列和分数
                        beam_indices = top_k_indices // vocab_size  # [batch_size, beam_size]# 当前topk路径上一步的idx
                        token_indices = top_k_indices % vocab_size  # [batch_size, beam_size]# 当前topk路径当前步的idx

                        # 更新生成的序列的beam索引
                        batch_indices = tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), [1, beam_size])  # [batch_size, beam_size]
                        gather_indices = tf.stack([batch_indices, beam_indices], axis=-1)  # [batch_size, beam_size, 2]
                        # 获取要更新的beam
                        selected_sequences = tf.gather_nd(sequences, gather_indices)  # [batch_size, beam_size, seq_length]
                        # 将新token添加到beam序列末尾
                        sequences = tf.concat([selected_sequences, tf.expand_dims(token_indices, axis=-1)], axis=-1) # [batch_size, beam_size, seq_length + 1]
                        scores = top_k_scores
                        # reward = top_k_reward

                    # 从 beam_size 条路径中选择分数最高的路径
                    best_sequence_indices = tf.expand_dims(tf.argmax(scores, axis=1), axis=-1)  # [batch_size,1]
                    # best_sequence_indices = tf.expand_dims(tf.argmax(reward, axis=1), axis=-1)  # [batch_size,1]
                    best_sequences = tf.gather_nd(sequences, tf.concat([tf.expand_dims(tf.cast(tf.range(batch_size),dtype=tf.int64), axis=-1), best_sequence_indices], axis=-1)) # [batch_size, max_length]
                    best_sequences = best_sequences[:, 1:]
                    generated_sequence = sequences[:,:,1:] # [batch_size, beam_size, seq_length]
                    # print("beam search end!")
                    # self.print_ops.append(tf.print("used_token ", used_token, summarize=10, output_stream=sys.stdout))
                    return logits, generated_sequence, None, best_sequences, probs
                # 初始化解码过程
                logits, generated_sequence, _, best_sequences, probs = beam_search(
                    model,
                    encoder_output,
                    sos_token,
                    eos_token,
                    pad_token,
                    beam_size,
                    max_length,
                )
                print("logits shape",logits.shape)
                print("generated_sequence shape",generated_sequence.shape)
                return logits, generated_sequence, None, best_sequences, probs

        with tf.variable_scope("evaluator", reuse=tf.AUTO_REUSE): # 仅训练使用
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
            def get_eval_logit(item_emb):
                position_ids = tf.range(self._list_size, dtype=tf.int32)
                position_ids = tf.expand_dims(position_ids, 0)
                position_embeddings = tf.nn.embedding_lookup(self.position_embeddings, position_ids)
                position_embeddings = tf.tile(position_embeddings, [tf.shape(hidden_states)[0], 1, 1])
                print("position_embeddings ", position_embeddings)
                item_hidden = item_emb + position_embeddings # (?,list_size,d)

                # cls token
                cls_embedding = tf.tile(tf.expand_dims(self.cls_embedding, axis=0), [batch_size, 1]) #(?, dim)
                cls_embedding = tf.expand_dims(cls_embedding, axis=1) #(?,1,dim)
                list_emb = tf.concat([cls_embedding,item_hidden],axis=1) #(?,list_size+1,dim)
                transformer = StackedTransformerModel(name="list_aware", num_layers=1, dim=64, num_heads=2, dk=64, dropout_rate=0.0, training=self._training)
                list_emb = transformer.forward(list_emb, training=self._training)
                point_wise_input = list_emb[:, 1:, :] # (?, list_size, dim)
                print("point_wise_input ", point_wise_input)
                # point-wise task
                point_wise_output_dict = multi_task_module("point_wise", point_wise_input, self._point_wise_tasks, shared_key="vtr", cand_size=self._list_size) # (?*list_num,list_size)

                if len(self._list_wise_tasks) > 0:
                    # list-wise module
                    list_cls = list_emb[:, :1, :] # (?, 1, dim)
                    list_wise_input = tf.layers.dense(list_cls, 128, activation=tf.nn.relu) # (?, 1, 128)
                    list_wise_output_dict = multi_task_module("list_wise", list_wise_input, self._list_wise_tasks, shared_key="list_ltr", cand_size=1) # (?,1)
                    # list_dnn = tf.layers.dense(list_wise_input, 64, activation=tf.nn.leaky_relu)
                    # list_output = tf.layers.dense(list_dnn, 1, activation=tf.nn.sigmoid)
                    # list_wise_output_dict = {"listwise_wtd": tf.squeeze(list_output, axis=-1)}  # (?,list_num)
                else:
                    list_wise_output_dict = {}
                return point_wise_output_dict, list_wise_output_dict

            if self.training:
                # 训练 Evaluator
                common_embs = tf.concat([pad_embedding, sos_embedding, common_embs, eos_embedding], axis=1) # (?,candidates_size+3,32)
                item_embeddings = tf.gather_nd(common_embs, gather_indices) # (?,list+1,32) 查找对应 list idx 的 emb
                point_wise_output_dict, list_wise_output_dict = get_eval_logit(item_emb=item_embeddings[:, :-1,]) # (?, 1) (?, list_size)

                # gen-eval 不更新Evaluator
                gen_emb = tf.matmul(predict[:, :-1, :], common_embs) # (?,list_size,32) generator选择evaluator阶段对应item的emb，前向每个位置只选择一个emb
                gen_eval_logits, _ = get_eval_logit(item_emb=gen_emb) # (?, 1)
                gen_eval_logits = tf.tile(tf.expand_dims(gen_eval_logits, axis=-1), [1, self._list_size, self.candidates_size + 3])
                gen_eval_logits = tf.stop_gradient(gen_eval_logits - predict[:, :-1, :]) + predict[:, :-1, :] # gen-eval loss只更新predict
                gen_eval_logits = tf.reshape(gen_eval_logits[:, 0, 0], [-1, 1])
                gen_eval_labels = tf.ones_like(gen_eval_logits, dtype=tf.float32)

                return predict[:, :-1, 2:-1], gen_loss, eval_loss, eval_logits, eval_weight, eval_label, list_eval_logits, list_label, gen_eval_logits, gen_eval_labels