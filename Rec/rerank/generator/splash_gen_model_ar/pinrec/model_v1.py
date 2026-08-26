import tensorflow as tf
from modules_ import *
from feature_attr_extract import user_fea_names,photo_fea_names,source_fea_names,dense_features_config

# ===== 模块级开关 =====
BAD_LIST_CORRECTION = False  # 改为 True 启用 Bad List Label Correction


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

class MTPModel():
    def __init__(self, num_future_tokens, num_layers, dim, num_heads, dk, dropout_rate):
        '''
        dim: query 的维度
        dk: key 投影矩阵的维度
        '''
        super(MTPModel, self).__init__()
        self.num_future_tokens = num_future_tokens
        self.num_layers = num_layers
        self.dim = dim
        self.layers = [EncoderLayer(f"encoder_layer_{i}", dim, num_heads, dk, dropout_rate) for i in range(num_layers)]
        self.decoder_layers = [DecoderLayer(f"main_decoder_layer_{i}", dim, num_heads, dk, dropout_rate) for i in range(num_layers)] # 主干
        self.extra_heads = [DecoderLayer(f"extra{i}_decoder_layer_0", dim, num_heads, dk, dropout_rate) for i in range(num_future_tokens)] # 独立预测头 1 layer

    def forward(self, hidden_states, training):
        for i in range(self.num_layers):
            hidden_states = self.layers[i].forward(hidden_states, training=training)
        return hidden_states

    # use_casual_chain 是否使用残差链接，参考deepseek, https://arxiv.org/pdf/2412.19437
    def mtp_forward_decoder(self, hidden_states, item_embedding, use_casual_chain, training):
        for layers in self.decoder_layers[:-1]:
            item_embedding = layers.forward(item_embedding, hidden_states, training=training) # (?, list_size + 1, dim)
        latents = []
        prediction_heads = [self.decoder_layers[-1]] + self.extra_heads
        for i in range(len(prediction_heads)):
            current_input = item_embedding
            if use_casual_chain and len(latents) > 0:
                current_input = item_embedding + latents[i - 1]
            h = prediction_heads[i].forward(current_input, hidden_states, training=training)
            latents.append(h)
        if not training:
            return latents[0] 
        h = tf.stack(latents, axis=1)
        return h
    
    def mtp_forward_decoder_deepseek(self, hidden_states, item_embedding, future_embeddings=None, training=False):
        """
        参数:
        - hidden_states: Encoder 层的输出
        - item_embedding: Decoder 的当前输入 embedding
        - future_embeddings: 训练时提供，形状应为包含 num_future_tokens 个 tensor 的列表，
                             每个 tensor 对应向后偏移 i 位的目标 token 的 embedding。
                             例如 future_embeddings[0] 对应 target_token_{t+1} 的 embedding。
        """
        for layer in self.decoder_layers[:-1]:
            item_embedding = layer.forward(item_embedding, hidden_states, training=training)
        main_latent = self.decoder_layers[-1].forward(item_embedding, hidden_states, training=training)
        if not training:
            return main_latent 
        assert future_embeddings is not None, "训练 MTP 阶段必须提供 future_embeddings"
        assert len(future_embeddings) == self.num_future_tokens, "future_embeddings 的长度必须等于预测头的数量"
        latents = [main_latent]
        # 依次计算独立的 MTP 预测头
        for i in range(self.num_future_tokens):
            prev_latent = latents[-1]
            current_future_emb = future_embeddings[i] 
            with tf.variable_scope(f"mtp_projection_{i}", reuse=tf.AUTO_REUSE):
                # DeepSeek 使用 RMSNorm
                normed_latent = layer_norm(f"mtp_ln_{i}", prev_latent)
                # 拼接：在最后一个维度拼接 [Norm(Latent) ; Future_Embedding]
                # 假设维度原本都是 dim，拼接后变为 2 * dim
                concat_input = tf.concat([normed_latent, current_future_emb], axis=-1)
                # 线性投影：将 2 * dim 降维回原来的 dim，并融合信息
                projected_input = tf.layers.dense(inputs=concat_input, units=self.dim, activation=None, name=f"mtp_linear_{i}")
            # 送入当前的预测头 (DecoderLayer)
            h = self.extra_heads[i].forward(projected_input, hidden_states, training=training)
            latents.append(h)
        return tf.stack(latents, axis=1) # (?, num_future_tokens + 1, seq_len, dim)

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


class FountainDeepLtrMultiTaskModel:
    def __init__(self, parameters_dict, label_value_dict, print_ops, list_size, candidates_size, training=True):
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
            "photo_hetu_tag_level1_list",
            "photo_hetu_tag_level2_list",
            "photo_hetu_tag_level5_list",
        ]
        self._photo_emp_explore_names = [
            "emp_explore_click_count",
            "photo_emp_explore_ctr",
            "photo_emp_explore_ltr",
        ]
        self._photo_emp_fountain_names = [
            "emp_fountain_show_count",
            "emp_fountain_like_count",
            "emp_fountain_follow_count",
            "photo_emp_fountain_ltr",
            "photo_emp_fountain_wtr",
            "photo_emp_fountain_avg_fintr",
        ]
        self._parameters_dict = parameters_dict
        self._label_value_dict = label_value_dict
        self._list_size = list_size
        self._candidates_size = candidates_size
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
            shape=[1, 32],
            initializer=tf.random_normal_initializer()
        )
        self.position_embeddings = tf.get_variable(
            name='position_embeddings', 
            shape=[list_size, 32], 
            initializer=tf.random_normal_initializer()
        )
        # Create [sos] and [eos] embeddings
        self.sos_embedding = tf.get_variable(
            "sos_embedding", shape=[1, 32], initializer=tf.random_uniform_initializer()
        )
        self.eos_embedding = tf.get_variable(
            "eos_embedding", shape=[1, 32], initializer=tf.random_uniform_initializer()
        )
        self.pad_embedding = tf.get_variable(
            "pad_embedding", shape=[1, 32], initializer=tf.random_uniform_initializer()
        )
        self.evtr_embedding = tf.get_variable("evtr_embedding", shape=[1, 32], initializer=tf.random_uniform_initializer())
        self.lvtr_embedding = tf.get_variable("lvtr_embedding", shape=[1, 32], initializer=tf.random_uniform_initializer())
        self.ltr_embedding = tf.get_variable("ltr_embedding", shape=[1, 32], initializer=tf.random_uniform_initializer())
        self.show_embedding = tf.get_variable("show_embedding", shape=[1, 32], initializer=tf.random_uniform_initializer())
        self.print_ops = print_ops
        self._bad_list_correction = BAD_LIST_CORRECTION

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
                  hidden_states,
                  hidden_units,
                  activation=tf.nn.relu):
        with tf.variable_scope(f"{scope_name}_mlp_layer", reuse=tf.AUTO_REUSE):
            for i, hidden_unit in enumerate(hidden_units):
                hidden_states = tf.layers.dense(hidden_states, hidden_unit, activation=activation, use_bias=True)
        return hidden_states

    def _get_shared_features(self, input_dicts) -> tuple:
        with tf.variable_scope("share_bottom", reuse=tf.AUTO_REUSE):
            user_embs     = tf.concat([input_dicts[k] for k in input_dicts if k in user_fea_names], axis=-1) # (?, dim), infer: (?, cand_size, dim)
            source_embs     = tf.concat([input_dicts[k] for k in input_dicts if k in source_fea_names], axis=-1) # (?, dim), infer: (?, cand_size, dim)
            photo_embs    = tf.concat([input_dicts[k] for k in input_dicts if k in photo_fea_names], axis=-1) # (?, cand_size, dim), infer: (cand_size, dim)
            photo_emp_embs = []
            for x in self._photo_emp_explore_names + self._photo_emp_fountain_names:
                emb, bucket_id = self.get_bucket_emb_from_sorted_boundaries(x) # (?, cand_size, 4)
                photo_emp_embs.append(emb)
            photo_embs = tf.concat(photo_emp_embs + [photo_embs if self._training else tf.reshape(photo_embs, [-1, self._candidates_size, photo_embs.shape[-1]])], axis=-1)

            ft_click_list = self._parameters_dict['user_fountain_profile_click_pid_list'] #  train: (?, 200, 1), infer: (?, cand_size, 200, 1)
            ft_click_aid_list = self._parameters_dict['user_fountain_profile_click_aid_list']
            ft_lv_list = self._parameters_dict['user_fountain_profile_effective_view_pid_list'] # 有效播放序列
            ft_lv_aid_list = self._parameters_dict['user_fountain_profile_effective_view_aid_list']
            if not self._training:
                '''
                    infer 时需要注意实际请求的 batch size = items 长度, 但计算图 batch 的 shape 由 uni_predict_fused 中 executor_batchsizes 决定,
                    如600。为了通过计算图编译, 需要 rashape (1, -1, dim), 实际 -1 为请求端发送 items 长度
                '''
                user_embs = tf.reshape(user_embs, [1, -1, user_embs.shape[-1]])
                source_embs = tf.reshape(source_embs, [1, -1, source_embs.shape[-1]])
                photo_embs = tf.reshape(photo_embs, [1, -1, photo_embs.shape[-1]])
                ft_click_list = tf.reshape(ft_click_list, [1, -1, ft_click_list.shape[-2], ft_click_list.shape[-1]])
                ft_click_aid_list = tf.reshape(ft_click_aid_list, [1, -1, ft_click_aid_list.shape[-2], ft_click_aid_list.shape[-1]])
                ft_lv_list = tf.reshape(ft_lv_list, [1, -1, ft_lv_list.shape[-2], ft_lv_list.shape[-1]])
                ft_lv_aid_list = tf.reshape(ft_lv_aid_list, [1, -1, ft_lv_aid_list.shape[-2], ft_lv_aid_list.shape[-1]])
            common_embs = photo_embs
            common_embs   = tf.layers.dense(common_embs, 32, activation=tf.nn.leaky_relu) # (?, cand_size, 96)
            print("common_embs ", common_embs)
            # print("ft_click_list[:, 0, :, :] ", ft_click_list[:, 0, :, :])
            # user seq X candidate cross attn
            ft_click_mha = self.linear_attention("ft_click_mha", common_embs, ft_click_list if self._training else ft_click_list[:, 0, :, :], nh=2, dim=16)
            ft_click_aid_mha = self.linear_attention("ft_click_aid_mha", common_embs, ft_click_aid_list if self._training else ft_click_aid_list[:, 0, :, :], nh=2, dim=16)
            ft_lv_mha = self.linear_attention("ft_lv_mha", common_embs, ft_lv_list if self._training else ft_lv_list[:, 0, :, :], nh=2, dim=16)
            ft_lv_aid_mha = self.linear_attention("ft_lv_aid_mha", common_embs, ft_lv_aid_list if self._training else ft_lv_aid_list[:, 0, :, :], nh=2, dim=16)
            history_embs = tf.concat([ft_click_mha, ft_click_aid_mha, ft_lv_mha, ft_lv_aid_mha], axis=-1)

            # candidates aware by transformer
            transformer = StackedTransformerModel(name="candidates_aware", num_layers=1, dim=32, num_heads=2, dk=32, dropout_rate=0.0, training=self._training)
            candidates_aware_out = transformer.forward(common_embs, training=self._training) # (?,cand_size,32)
            source_embs = tf.layers.dense(source_embs, 32, activation=tf.nn.tanh, use_bias=True)
            user_embs = tf.layers.dense(tf.concat([user_embs, source_embs], axis=-1), 32, activation=tf.nn.leaky_relu)
            user_embs = tf.tile(tf.expand_dims(user_embs, axis=1), [1, self._candidates_size, 1]) if self._training else user_embs # (?,cand_size,32)
            common_embs = tf.concat([user_embs, history_embs, candidates_aware_out], axis=-1) # (?,cand_size,d)
            common_embs = tf.layers.dense(common_embs, 32, activation=tf.nn.leaky_relu)
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
        print("boundaries ", boundaries, " value ", value)
        bucket_id = tf.searchsorted(boundaries, values=value, out_type=tf.int32)
        emb = tf.nn.embedding_lookup(embeddings, bucket_id) # (?, cand_size, dim)
        return emb, bucket_id

    def linear_attention(self, name, query, key, nh, dim):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            query_len = tf.shape(query)[1]
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
            output = tf.reshape(output, [batch_size, query_len, nh * dim])
            return output

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
            print("attention_layer_4d query ", query, "key ", key)
            attn_out, attention_weights = scaled_dot_product_attention(query, key, key, mask=None)
            attn_out = tf.reshape(attn_out, [batch_size, cand_size, query_dim]) # (?, cand_size, dim1)
            return attn_out
        
    def _contrastive_loss(self, score_matrix, margin=0.5, seqlen=6):
       gold_score = tf.linalg.diag_part(score_matrix)
       gold_score = tf.expand_dims(gold_score, axis=2)

       difference_matrix = gold_score - score_matrix
       loss_matrix = margin - difference_matrix
       loss_matrix = tf.nn.relu(loss_matrix)

       base_mask = tf.ones((seqlen, seqlen)) - tf.linalg.diag(tf.ones(seqlen))
       base_mask = tf.expand_dims(base_mask, axis=0)
       base_mask = tf.tile(base_mask,[tf.shape(score_matrix)[0],1,1])

       cl_loss = tf.reduce_mean(loss_matrix*base_mask)
       
       return cl_loss
    
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

    def choose_item(self, decoder_emb, vocab_emb, method=0, use_gumbel_softmax=False, tau=1.0, hard=True, logit_bias=None):
        # 端到端的情况下需要打开 gumbel softmax，具备采样能力
        if not self._training:
            infer_batch_size = tf.shape(decoder_emb)[0]
            infer_beam_size = decoder_emb.shape[1]
            infer_list_size = tf.shape(decoder_emb)[2]
            infer_vocab_size = tf.shape(vocab_emb)[2]
            infer_dim = vocab_emb.shape[-1]
            decoder_emb = tf.reshape(decoder_emb, [infer_batch_size * infer_beam_size, infer_list_size, infer_dim])
            vocab_emb = tf.reshape(vocab_emb, [infer_batch_size * infer_beam_size, infer_vocab_size, infer_dim])
            if logit_bias is not None:
                logit_bias = tf.tile(tf.expand_dims(logit_bias, axis=1), [1, infer_beam_size, 1, 1])
                logit_bias = tf.reshape(logit_bias, [infer_batch_size * infer_beam_size, 1, -1])
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
                concat_emb = tf.concat([decoder_emb, vocab_emb], axis=-1) # (?,list_size,vocab_size,32)
                logits = tf.layers.dense(concat_emb, 32, activation=tf.nn.relu)
                logits = tf.layers.dense(logits, 32, activation=tf.nn.relu)
                logits = tf.layers.dense(logits, 1, activation=None) # (?,list_size,vocab_size,1)
                logits = tf.squeeze(logits, axis=-1) # (?,list_size,vocab_size)
        elif method == 1:
            # cosine 选取
            with tf.variable_scope("predict_cosine", reuse=tf.AUTO_REUSE):
                logits = tf.matmul(decoder_emb, tf.transpose(vocab_emb,  perm=[0, 2, 1])) # (?, list_size, vocab_size)
        elif method == 2:
            with tf.variable_scope("predict_dot", reuse=tf.AUTO_REUSE):
                q = tf.layers.dense(decoder_emb, 64, use_bias=False, name="Wq")            # [B,L,proj]
                k = tf.layers.dense(vocab_emb, 64, use_bias=False, name="Wk")            # [B,V,proj]
                q = tf.nn.l2_normalize(q, axis=-1)
                k = tf.nn.l2_normalize(k, axis=-1)
                logits = tf.matmul(q, tf.transpose(k, perm=[0, 2, 1]))                           # logits: [B,L,V]
                logits = logits / 0.5
        if logit_bias is not None:
            logits += logit_bias
        if use_gumbel_softmax:
            predict = self.gumbel_softmax(logits, tau=tau, hard=hard)
        else:
            predict = tf.nn.softmax(logits, axis=-1) # (?,list_size,candidates_size+3)
        if not self._training:
            predict = tf.reshape(predict, [infer_batch_size, infer_beam_size, infer_list_size, infer_vocab_size])
            logits = tf.reshape(logits, [infer_batch_size, infer_beam_size, infer_list_size, infer_vocab_size])
            print("xxx ", predict)
        return predict, logits

    def nce_loss(self, decoder_emb, item_embs, num_sampled=2048, use_dot=True, temperature=0.8):
        """
        decoder_emb: [batch_size, list_size, dim] 
        item_embs: [batch_size, vocab_size, dim]
        """
        with tf.variable_scope("predict_token_nce", reuse=tf.AUTO_REUSE):
            batch_size = tf.shape(decoder_emb)[0]
            list_size = decoder_emb.shape[1] if self._training else tf.shape(decoder_emb)[1]
            dim = item_embs.shape[-1]
            vocab_size = item_embs.shape[1] if self._training else tf.shape(item_embs)[1]
            flat_queries = tf.reshape(decoder_emb, [-1, dim]) # [B * L, dim]
            flat_item_pool = tf.reshape(item_embs, [-1, dim]) # [B * V, dim]
            pos_embs = item_embs[:, :list_size, :]
            neg_indices = tf.random_uniform([num_sampled], minval=0, maxval=batch_size * vocab_size, dtype=tf.int32)
            neg_embs = tf.gather(flat_item_pool, neg_indices) # [num_sampled, dim]
            if use_dot:
                pos_logits = tf.reduce_sum(decoder_emb * pos_embs, axis=-1, keepdims=True) # [B, L, 1]
                neg_logits = tf.matmul(flat_queries, neg_embs, transpose_b=True) # [B * L, num_sampled]
                neg_logits = tf.reshape(neg_logits, [batch_size, list_size, num_sampled])
                logits = tf.concat([pos_logits, neg_logits], axis=-1) # [B, L, 1 + num_sampled]
                logits /= temperature
                logits = tf.reshape(logits, [-1, 1 + num_sampled]) # [B * L, 1 + num_sampled]
            else:
                flat_pos_embs = tf.reshape(pos_embs, [-1, 1, dim]) # [B * L, 1, dim]
                flat_neg_embs = tf.broadcast_to(tf.expand_dims(neg_embs, 0), [batch_size * list_size, num_sampled, dim]) # [B * L, num_sampled, dim]
                candidates = tf.concat([flat_pos_embs, flat_neg_embs], axis=1) # [B * L, 1 + num_sampled, dim]
                queries_expanded = tf.broadcast_to(tf.expand_dims(flat_queries, 1), [batch_size * list_size, 1 + num_sampled, dim])
                mlp_input = tf.concat([queries_expanded, candidates], axis=-1) # [B * L, 1 + num_sampled, 2 * dim]
                hidden = tf.layers.dense(mlp_input, 32, activation=tf.nn.relu, name="nce_hidden")
                logits = tf.layers.dense(hidden, 1, activation=None, name="nce_logits") 
                logits = tf.squeeze(logits, axis=-1) # [B * L, 1 + num_sampled]
            # 在前面拼接时，把正样本放在了第 0 个位置，所以所有 Query 的 label 都是 0
            labels = tf.zeros([batch_size * list_size], dtype=tf.int32)
            # 使用 TensorFlow 原生 API 计算，既稳定又快 (自带 log-sum-exp 优化，防止 NaN)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            
            return tf.reduce_sum(loss)

    def cal_batch_advantage(self, reward, mask):
        mask = tf.cast(mask, reward.dtype)
        valid_cnt = tf.reduce_sum(mask)
        mean = tf.reduce_sum(reward * mask) / (valid_cnt + 1e-8)
        variance = (reward - mean) ** 2 * mask
        std = tf.sqrt(tf.reduce_sum(variance) / (valid_cnt + 1e-8))
        advantages = (reward - mean) / (std + 1e-8)
        return advantages
    
    def bpr_loss(self, output, score, threshold, mask):
        """
        output: (?, L)
        score: (?, L)
        threshold: (?, L)
        mask: (?, L)
        """
        with tf.variable_scope("bpr_loss", reuse=tf.AUTO_REUSE):
            # 生成配对矩阵 (?, L, 1) - (?, 1, L)
            output_diff = tf.expand_dims(output, 2) - tf.expand_dims(output, 1) # (?, L, L)
            score_diff = tf.expand_dims(score, 2) - tf.expand_dims(score, 1)
            pairwise_labels = tf.cast(score_diff >= tf.expand_dims(threshold, 2), tf.float32)
            individual_loss = tf.nn.softplus(-output_diff)
            mask_2d = tf.cast(tf.expand_dims(mask, 2) * tf.expand_dims(mask, 1), tf.float32)
            bpr_loss = individual_loss * pairwise_labels * mask_2d
            return bpr_loss

    def _rank_bias(self, x, offset=10.0):
        order = tf.argsort(x, axis=-1, direction="DESCENDING")          # [B, cand]
        rank = tf.argsort(order, axis=-1, direction="ASCENDING")       # [B, cand]
        return 1.0 / (tf.cast(rank, tf.float32) + offset)

    def model(self, training=True, beam_size=1, max_length=10):
        self._training = training
        with tf.variable_scope("prepare", reuse=tf.AUTO_REUSE):
            common_embs = self._get_shared_features(self._parameters_dict) # (?,60,d)
            batch_size = tf.shape(common_embs)[0]
            dim = common_embs.shape[-1]
            # 添加特殊token的embedding
            pad_embedding = tf.broadcast_to(tf.reshape(self.pad_embedding, [1, 1, dim]), [batch_size, 1, dim])
            sos_embedding = tf.broadcast_to(tf.reshape(self.sos_embedding, [1, 1, dim]), [batch_size, 1, dim])
            eos_embedding = tf.broadcast_to(tf.reshape(self.eos_embedding, [1, 1, dim]), [batch_size, 1, dim])

            sos_token = tf.fill([batch_size, 1], 1)
            eos_token = tf.fill([batch_size, 1], self._candidates_size + 2)
            pad_token = tf.zeros([batch_size, 1], dtype=tf.int32)
            self.item_embs = common_embs
            self.photo_embs = tf.concat([pad_embedding, sos_embedding, common_embs, eos_embedding], axis=1) # (?,candidates_size+3,32)

            if self._training:
                show_label = self._label_value_dict["show_label"][:,:self._list_size] # (?,list_size)，截断为 list_size 个
                like_label = self._label_value_dict["like_label"][:,:self._list_size] # (?,list_size)，截断为 list_size 个
                follow_label = self._label_value_dict["follow_label"][:,:self._list_size] # (?,list_size)，截断为 list_size 个
                comment_label = self._label_value_dict["comment_label"][:,:self._list_size] # (?,list_size)，截断为 list_size 个
                forward_label = self._label_value_dict["forward_label"][:,:self._list_size] # (?,list_size)，截断为 list_size 个
                wtd_label = self._label_value_dict["wtd_label"][:,:self._list_size] # (?,list_size)，截断为 list_size 个
                finish_label = self._label_value_dict["finish_label"][:,:self._list_size] # (?,list_size)，截断为 list_size 个
                next_label = self._label_value_dict["slide_label"][:,:self._list_size] # (?,list_size)，截断为 list_size 个
                evtr_label, lvtr_label = self._label_value_dict["evtr_label"][:,:self._list_size], self._label_value_dict["lvtr_label"][:,:self._list_size]
                svtr_label = self._label_value_dict["svtr_label"][:,:self._list_size]
                playtime = tf.clip_by_value(self._label_value_dict["play_time_s"][:,:self._list_size], 0, 1000) # (?,list_size)

                indices_shape = tf.shape(show_label)
                col_indices = tf.expand_dims(tf.range(tf.shape(show_label)[1]), 0) + 2
                realshow_indices = tf.cast(col_indices * tf.cast(show_label, dtype=tf.int32),dtype=tf.int32) # label为0 1，过滤了未曝光的index
                inputs = tf.concat([sos_token, realshow_indices], axis=1) # (?,list_size+1)
                outputs = tf.concat([realshow_indices, eos_token], axis=1) # (?,list_size+1)
                print("inputs ", inputs)
                print("outputs ", outputs)
                batch_indices = tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), [1, self._list_size+1]) # (?, self._list_size+1)
                gather_indices = tf.stack([batch_indices, inputs], axis=-1) # (?, self._list_size+1, 2)

        with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
            # fusion action emb
            # TODO: 对各个emb都计算一遍 loss
            if self._training:
                show_emb = tf.reshape(self.show_embedding, [1, 1, -1])
                # evtr_mask = tf.expand_dims(tf.cast(evtr_label > 0, tf.float32), -1)
                evtr_mask = tf.expand_dims(tf.cast((1 - svtr_label) > 0, tf.float32), -1)
                action_emb = tf.broadcast_to(show_emb, [batch_size, self._list_size, dim]) # 兜底逻辑
                action_emb = evtr_mask * tf.reshape(self.evtr_embedding, [1, 1, -1]) + (1.0 - evtr_mask) * action_emb
            else:
                action_emb = self.evtr_embedding
                action_emb = tf.broadcast_to(tf.reshape(action_emb, [1, 1, -1]), [batch_size, self._list_size, dim])
            action_emb = tf.concat([action_emb, tf.zeros([batch_size, 1, dim], dtype=tf.float32)], axis=1) # (?,list_size+1,32)

            # 初始化transformer模型
            num_future_tokens = 1
            model = MTPModel(num_future_tokens=num_future_tokens, num_layers=1, dim=32, num_heads=2, dk=32, dropout_rate=0.1)
            hidden_states = model.forward(common_embs, training=self._training)
            encoder_output = hidden_states
            print("encoder output shape ",encoder_output.shape) # (?,candidates_size,32)

            pad_sos = tf.zeros([batch_size, 2], dtype=tf.float32)
            eos = tf.zeros([batch_size, 1], dtype=tf.float32)
            self.pwtd = self._label_value_dict["pwtd"] if self._training else tf.reshape(self._label_value_dict["pwtd"], [1, -1]) # (?, cand_size)
            self.pctr = self._label_value_dict["pctr"] if self._training else tf.reshape(self._label_value_dict["pctr"], [1, -1]) # (?, cand_size)
            self.pltr = self._label_value_dict["pltr"] if self._training else tf.reshape(self._label_value_dict["pltr"], [1, -1]) # (?, cand_size)
            self.pcmtr = self._label_value_dict["pcmtr"] if self._training else tf.reshape(self._label_value_dict["pcmtr"], [1, -1]) # (?, cand_size)
            self.pwtr = self._label_value_dict["pwtr"] if self._training else tf.reshape(self._label_value_dict["pwtr"], [1, -1]) # (?, cand_size)

            bias_item = self._rank_bias(self.pwtd) + self._rank_bias(self.pctr) * 0.0 + self._rank_bias(self.pltr) * 0.1 + self._rank_bias(self.pcmtr) * 0.0 + self._rank_bias(self.pwtr) * 0.0

            vocab_bias = tf.concat([pad_sos, bias_item, eos], axis=1)  # [B, cand_size+3]
            logit_bias =  0.0 * tf.reshape(vocab_bias, [batch_size, 1, -1])
            if self._training:
                # === Bad List Label Correction ===
                if self._bad_list_correction:
                    list_quality = tf.reduce_sum((playtime + (like_label + follow_label + comment_label) * 10.0) * show_label, axis=-1, keepdims=True)  # (?, 1)
                    is_bad_list = tf.less(list_quality, 20.0)  # (?, 1)
                    random_trigger = tf.less(tf.random_uniform(tf.stack([batch_size, 1])), 0.5)
                    apply_correction = tf.logical_and(is_bad_list, random_trigger)  # (?, 1)
                    tf.summary.scalar('bad_list_ratio', tf.reduce_mean(tf.cast(is_bad_list, tf.float32)))
                    tf.summary.scalar('correction_ratio', tf.reduce_mean(tf.cast(apply_correction, tf.float32)))

                    _, pwtd_top_k_idx = tf.math.top_k(self.pwtd, k=self._list_size)  # (?, list_size)
                    pwtd_vocab_indices = tf.cast(pwtd_top_k_idx + 2, tf.int32)  # vocab offset: PAD=0,SOS=1,items=2..61
                    apply_tiled = tf.tile(apply_correction, [1, self._list_size])  # (?, list_size)
                    corrected_realshow = tf.where(apply_tiled, pwtd_vocab_indices, realshow_indices)
                    show_label = tf.where(apply_tiled, tf.ones_like(show_label), show_label)  # 替换样本全部曝光
                    inputs = tf.concat([sos_token, corrected_realshow], axis=1)    # 重构 inputs
                    outputs = tf.concat([corrected_realshow, eos_token], axis=1)   # 重构 outputs

                    # 替换样本的 advantage 近似：用 pwtd 预估值替代真实 playtime（单位对齐到秒）
                    pwtd_top_k_values = tf.gather(self.pwtd, pwtd_top_k_idx, batch_dims=1)  # (?, list_size)
                    playtime = tf.where(apply_tiled, tf.clip_by_value(pwtd_top_k_values, 0.0, 400.0), playtime)
                    # 替换样本互动标签清零（反事实，互动行为未发生）
                    like_label    = tf.where(apply_tiled, tf.zeros_like(like_label),    like_label)
                    follow_label  = tf.where(apply_tiled, tf.zeros_like(follow_label),  follow_label)
                    comment_label = tf.where(apply_tiled, tf.zeros_like(comment_label), comment_label)
                    svtr_label    = tf.where(apply_tiled, tf.zeros_like(svtr_label),    svtr_label)

                # 从hidden_states查对应emb表示
                item_embeddings = tf.gather(self.photo_embs, inputs, batch_dims=1)
                item_embeddings = item_embeddings + action_emb # (?,list_size+1,32)

                item_embedding = model.mtp_forward_decoder(hidden_states, item_embeddings, use_casual_chain=False, training=True) # (?,num_future_tokens + 1,list_size+1,32)
                print("mtp item_embedding shape", item_embedding.shape) # (?,list_size+1,32)

                # 从候选集选取 item, 0: nn, 1: cosine; 是否进行采样
                predict, logits = self.choose_item(
                    tf.reshape(item_embedding, [batch_size, (self._list_size + 1) * (num_future_tokens + 1), item_embedding.shape[-1]]), self.photo_embs, method=2,
                    use_gumbel_softmax=False, tau=0.1, hard=True, logit_bias=logit_bias,
                )
                print("predict ", predict)
                predict = tf.reshape(predict, [batch_size, num_future_tokens + 1, self._list_size + 1, predict.shape[-1]]) # (?,num_future_tokens+1,list_size+1,candidates_size+3)
                mtp_outputs = [tf.concat([outputs[:,i+1:-1], tf.tile(eos_token, [1, i+2])], axis=1) for i in range(num_future_tokens)]
                mtp_outputs = tf.stack([outputs] + mtp_outputs, axis=1)
                print("mtp_outputs ", mtp_outputs) # (?, num_future_tokens + 1, list_size+1)
                mtp_mask = [tf.concat([tf.ones_like(outputs[:,i+1:-2], dtype=tf.float32), tf.zeros([batch_size, i+3], dtype=tf.float32)], axis=-1) for i in range(num_future_tokens)]
                print("mtp_mask ", mtp_mask) # (?, num_future_tokens, list_size+1)
                mtp_mask = tf.stack([tf.concat([tf.ones([batch_size, self._list_size], dtype=tf.float32), tf.zeros([batch_size, 1], dtype=tf.float32)], axis=-1)] + mtp_mask, axis=1)
                print("mtp_mask ", mtp_mask) # (?, num_future_tokens + 1, list_size+1)
                output_indices = tf.expand_dims(mtp_outputs, axis=-1) # (?,num_future_tokens + 1, list_size+1,1)
                pos_output = tf.batch_gather(predict, output_indices) # 需要前 n 维相同
                pos_output = tf.squeeze(pos_output, axis=-1) #(?,num_future_tokens + 1, list_size+1)
                print("pos_output shape", pos_output.shape)

                act_label = tf.cast((like_label + follow_label + comment_label + forward_label) > 0, tf.float32)
                # advantage_reward = playtime + lvtr_act_label * 2.0 + finish_label * 2.0 + evtr_like_label * 100.0 + evtr_forward_label * 50.0 + evtr_follow_label * 500 + evtr_comment_label * 200
                # advantage_reward = playtime + evtr_label * 1.0 + finish_label * 3.0 + like_label * 50.0 + follow_label * 200.0 + comment_label * 200.0 + forward_label * 100
                advantage_reward = tf.clip_by_value(playtime, 0, 400) + like_label * 20.0 + follow_label * 20.0 + comment_label * 20.0
                advantage_reward = tf.clip_by_value(advantage_reward, 0.0, 200.0)
                advantage = self.cal_batch_advantage(advantage_reward, mask=show_label)
                advantage = tf.nn.relu(tf.clip_by_value(advantage, 0.0, 20.0)) * (1 - svtr_label) + 1.0

                # 主 head bpr loss
                bpr_loss = tf.zeros([], dtype=tf.float32)
                # for i in range(self._list_size):
                #     ith_prediction = predict[:, 0, i, 2:8] # (?,6)
                #     bpr_threshold = tf.where(playtime < 20.0, tf.ones_like(playtime), tf.ones_like(playtime) * 3)
                #     bpr_threshold = tf.where(playtime > 60.0, tf.ones_like(playtime) * 5, bpr_threshold)
                #     bpr_loss2 = self.bpr_loss(ith_prediction, advantage_reward, bpr_threshold, show_label > 0)
                #     bpr_loss += tf.reduce_sum(bpr_loss2) / self._list_size

                zeros = tf.zeros([batch_size, 1], dtype=tf.float32) # mask EOS token
                ones = tf.ones([batch_size, 1], dtype=tf.float32)
                mask = tf.concat([show_label, zeros], axis=1) # (?, list_size + 1)

                valid_counts = tf.reduce_sum(mask, axis=-1)+1e-9
                item_weight = tf.concat([advantage_reward, ones], axis=1) # (?, list_size + 1) 单点reward
                seq_weight = tf.reduce_sum(item_weight * mask, axis=-1) / valid_counts # (?,)
                seq_advantage = self.cal_batch_advantage(seq_weight, mask=tf.ones_like(seq_weight, dtype=tf.float32)) # (?,)
                seq_advantage = tf.where(seq_advantage > 0.0, seq_advantage + 1.0, tf.ones_like(seq_advantage, dtype=tf.float32))
                seq_advantage = tf.clip_by_value(seq_advantage, 0.0, 20.0) # (?,)
                # advantage += interact_bonus # 互动加权
                advantage = tf.concat([advantage, ones], axis=1) # (?, list_size)
                print("advantage", advantage)
                print("seq_advantage", seq_advantage)
                seq_advantage = tf.expand_dims(seq_advantage, axis=1)
                # weight = 1.0 * advantage + 0.0 * seq_advantage
                weight = 1.0 * advantage
                weight *= mask # realshow mask
                mtp_weights = [tf.concat([weight[:, i+1:-1], tf.zeros([batch_size, i+2])], axis=1) for i in range(num_future_tokens)]
                weight = tf.stack([weight] + mtp_weights, axis=1) # [batch, num_future_tokens + 1, list_size + 1]
                valid_pos_output = -tf.log(pos_output + 1e-9) * mtp_mask * tf.cast(weight > 0, tf.float32) # (?, num_future_tokens + 1, list_size + 1)
                valid_pos_reward = -tf.log(pos_output + 1e-9) * mtp_mask * weight # (?, num_future_tokens + 1, list_size + 1)
                print("valid_pos_output ", valid_pos_output)
                gen_loss = tf.reduce_mean(tf.reduce_sum(valid_pos_output, axis=-1)) + tf.reduce_mean(tf.reduce_sum(valid_pos_reward, axis=-1)) * 2.0
                print("gen_loss shape", gen_loss.shape)

                # 主 head nce loss
                nce_loss = self.nce_loss(item_embedding[:, 0, :-1, :], self.item_embs)

                return predict[:,0,:,:], nce_loss, gen_loss, bpr_loss
            
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
                    # scores = tf.zeros([batch_size, beam_size])  # [batch_size, beam_size]
                    reward = tf.zeros([batch_size, beam_size])  # [batch_size, beam_size]

                    preward = self._rank_bias(self.pwtd) * 1.0 + self._rank_bias(self.pctr) * 0.0 + self._rank_bias(self.pltr) * 0.0 \
                        + self._rank_bias(self.pcmtr) * 0.0 + self._rank_bias(self.pwtr) * 0.0
                    preward = tf.concat([tf.zeros([batch_size, 2], dtype=tf.float32), preward, tf.zeros([batch_size, 1], dtype=tf.float32)], axis=-1) # (?, vocab_size)
                    
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
                        decoder_output = model.mtp_forward_decoder(tf.reshape(encoder_output, [batch_size * beam_size, tf.shape(encoder_output)[-2], vocab_dim]),
                                                               tf.reshape(decoder_input, [batch_size * beam_size, seq_length, decoder_dim]),
                                                               use_casual_chain=False,
                                                               training=self._training)  # [batch_size * beam_size, seq_length, dim]
                        decoder_output = tf.reshape(decoder_output, [batch_size, beam_size, seq_length, decoder_dim])
                        # # nce loss 不实际计算
                        nce_loss = self.nce_loss(decoder_output[:, 0, :-1, :], item_embs[:, 0, :, :])
                        # 计算 logits
                        # logits = tf.matmul(self.photo_embs, tf.transpose(decoder_output, perm=[0, 1, 3, 2])) # [batch_size, beam_size, vocab_size, seq_length]
                        predict, logits = self.choose_item(decoder_output, photo_embs, method=2, use_gumbel_softmax=False, tau=0.1, hard=True, logit_bias=logit_bias,) # [batch_size, beam_size, seq_length, vocab_size]
                        next_token_logits = logits[:, :, -1, :]  # [batch_size, beam_size, vocab_size]
                        
                        # 选择下一个token

                        tau = 1.0
                        next_token_probs = tf.nn.softmax(next_token_logits/tau, axis=-1)  # [batch_size, beam_size, vocab_size]
                        probs.append(next_token_probs)
                        log_probs = tf.math.log(next_token_probs+1e-9)  # 转换为 log 概率
                        # reward_coeff = 1.0 / (step + 1.0) * 2.0
                        reward_coeff = 0.0
                        cur_reward = tf.tile(tf.expand_dims(preward, axis=1), [1, beam_size, 1]) * reward_coeff + log_probs  # 转换为 log 概率 [batch_size, beam_size, vocab_size]

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
                        # log_probs = tf.where(used_token, tf.fill(tf.shape(log_probs), float('-inf')), log_probs)  # [batch_size, beam_size, vocab_size]
                        cur_reward = tf.where(used_token, tf.fill(tf.shape(cur_reward), float('-inf')), cur_reward)  # [batch_size, beam_size, vocab_size]
                        # 计算总分数  (当前路径分数 + 新 token 的分数)
                        # scores = tf.expand_dims(scores, axis=-1) + log_probs  # [batch_size, beam_size, vocab_size]
                        reward = tf.expand_dims(reward, axis=-1) + cur_reward  # [batch_size, beam_size, vocab_size]
                        
                        # topk最高分数 
                        if step == 0:
                            # 第一步直接取 topk 不同 idx
                            # top_k_scores, top_k_indices = tf.math.top_k(scores[:, 0, :], k=beam_size, sorted=True)  # [batch_size,beam_size)
                            top_k_reward, top_k_indices = tf.math.top_k(reward[:, 0, :], k=beam_size, sorted=True)  # [batch_size,beam_size)
                        else:
                            # scores_flat = tf.reshape(scores, [batch_size, -1]) # [batch_size, beam_size * vocab_size]
                            # top_k_scores, top_k_indices = tf.math.top_k(scores_flat, k=beam_size, sorted=True)  # [batch_size,beam_size)
                            reward_flat = tf.reshape(reward, [batch_size, -1]) # [batch_size, beam_size * vocab_size]
                            top_k_reward, top_k_indices = tf.math.top_k(reward_flat, k=beam_size, sorted=True)  # [batch_size,beam_size)
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
                        # scores = top_k_scores
                        reward = top_k_reward

                    # 从 beam_size 条路径中选择分数最高的路径
                    # best_sequence_indices = tf.expand_dims(tf.argmax(scores, axis=1), axis=-1)  # [batch_size,1]
                    best_sequence_indices = tf.expand_dims(tf.argmax(reward, axis=1), axis=-1)  # [batch_size,1]
                    best_sequences = tf.gather_nd(sequences, tf.concat([tf.expand_dims(tf.cast(tf.range(batch_size),dtype=tf.int64), axis=-1), best_sequence_indices], axis=-1)) # [batch_size, max_length]
                    best_sequences = best_sequences[:, 1:]
                    generated_sequence = sequences[:,:,1:] # [batch_size, beam_size, seq_length]
                    # print("beam search end!")
                    return logits, generated_sequence, preward, best_sequences, probs
                # 初始化解码过程
                logits, generated_sequence, preward, best_sequences, probs = beam_search(
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
                return logits, generated_sequence, preward, best_sequences, probs