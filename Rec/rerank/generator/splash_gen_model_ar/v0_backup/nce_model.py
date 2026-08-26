import sys
import tensorflow as tf
from modules_ import *
from feature_attr_extract import user_fea_names,photo_fea_names,source_fea_names,explore_profile_fea_names,fountain_seq_pid_names,fountain_seq_aid_names

    
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


class FountainDeepLtrMultiTaskModel:
    def __init__(self, parameters_dict, label_value_dict, dense_value_dict, print_ops, list_size, candidates_size, dim=32, extra_param_dict= None, training=True):
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
            "photo_hetu_tag_level1_list",
            "photo_hetu_tag_level2_list",
            "photo_hetu_tag_level3_list",
            "photo_hetu_tag_level5_list",
            "photo_duration_ms",
        ]
        self._parameters_dict = parameters_dict
        self._label_value_dict = label_value_dict
        self._dense_value_dict = dense_value_dict
        self.list_size = list_size
        self.candidates_size = candidates_size
        self.dim = dim
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
            user_embs_origin     = tf.concat([input_dicts[k] for k in input_dicts if k in user_fea_names], axis=-1)
            user_embs     = tf.tile(tf.expand_dims(user_embs_origin, axis=1), [1,self.candidates_size,1]) if self._training else user_embs_origin
            photo_embs    = tf.concat([input_dicts[k] for k in input_dicts if k in photo_fea_names], axis=-1)
            source_embs_origin   = tf.concat([input_dicts[k] for k in input_dicts if k in source_fea_names], axis=-1)
            source_embs   = tf.tile(tf.expand_dims(source_embs_origin, axis=1), [1,self.candidates_size,1]) if self._training else source_embs_origin

            photo_attr_embs    = tf.concat([input_dicts[k] for k in self._photo_attr_names], axis=-1) # (?,cand_size,dim)
            if self._training:
                pxtr_embs    = tf.concat([tf.expand_dims(input_dicts[k], axis=2) for k in input_dicts if k in self._pxtr_names], axis=2) # (?,cand_size,n,dim)
            else:
                pxtr_embs = tf.concat([tf.expand_dims(input_dicts[k], axis=1) for k in input_dicts if k in self._pxtr_names], axis=1) # (?,n,dim)
            print("photo_attr_embs", photo_attr_embs.shape)
            print("pxtr_embs", pxtr_embs.shape)

            common_embs   = tf.concat([user_embs, photo_embs, source_embs], axis=-1)
            if not self._training:
                '''
                    infer 时需要注意实际请求的 batch size = items 长度, 但计算图 batch 的 shape 由 uni_predict_fused 中 executor_batchsizes 决定,
                    如600。为了通过计算图编译, 需要 rashape (1, -1, dim), 实际 -1 为请求端发送 items 长度
                '''
                common_embs = tf.reshape(common_embs, [1, -1, common_embs.shape[-1]])
                user_embs = tf.reshape(user_embs, [1, -1, user_embs.shape[-1]])
                source_embs = tf.reshape(source_embs, [1, -1, source_embs.shape[-1]])
                photo_embs = tf.reshape(photo_embs, [1, -1, photo_embs.shape[-1]])
                photo_attr_embs = tf.reshape(photo_attr_embs, [1, -1, photo_attr_embs.shape[-1]])
                pxtr_embs = tf.reshape(pxtr_embs, [1, -1, pxtr_embs.shape[-2], pxtr_embs.shape[-1]])
            return common_embs, user_embs, source_embs, photo_embs, photo_attr_embs, pxtr_embs
        
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
    
    def batch_negative_sampling(query_emb, item_emb, labels, item_popularity, temperature=0.75):
        """
        query_emb: query向量 [batch_size, emb_dim]
        item_emb: 物品向量 [batch_size, emb_dim]
        labels: 正样本标签 [batch_size, 1] (1表示正例)
        item_popularity: 物品热度向量 [num_items]
        """
        batch_size = tf.shape(query_emb)[0]
        
        # 1. 计算所有样本相似度矩阵
        similarity = tf.matmul(query_emb, item_emb, transpose_b=True)  # [batch_size, batch_size]
        
        # 2. 生成负采样权重 (热门物品降权)
        pop_weights = tf.pow(item_popularity + 1, temperature)  # 热度调整
        norm_weights = pop_weights / tf.reduce_sum(pop_weights)  # 归一化
        neg_weights = tf.gather(norm_weights, tf.argmax(labels, axis=1))  # 提取batch物品权重
        
        # 3. 构造负采样Mask (排除自身正样本)
        pos_mask = tf.cast(tf.eye(batch_size), tf.bool)  # 对角线为正例
        neg_mask = tf.logical_not(pos_mask)  # 非对角线为候选负例
        
        # 4. 加权负采样
        neg_similarity = tf.where(neg_mask, similarity, -np.inf * tf.ones_like(similarity))
        neg_probs = tf.nn.softmax(neg_similarity * neg_weights, axis=-1)  # 加权采样概率
        sampled_neg_indices = tf.multinomial(tf.log(neg_probs), 1)  # 采样负例索引
        
        # 5. 计算损失函数
        pos_scores = tf.reduce_sum(query_emb * item_emb, axis=1)  # 正样本得分
        neg_scores = tf.gather_nd(similarity, sampled_neg_indices)  # 负样本得分
        loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(pos_scores),
                logits=pos_scores
            ) + tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(neg_scores),
                logits=neg_scores
            )
        )
        return loss
    
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

    def get_unique_vocab_with_counts(self, items):
        # 展平为二维张量 [batch_size * list_size, dim]
        reshaped = tf.reshape(items, [-1, tf.shape(items)[-1]])  # [N, dim]
        # 生成唯一指纹标识（FarmHash64 算法）
        fingerprints = tf.fingerprint(reshaped)  # 默认输出形状 [N, 8]
        fingerprints_flat = tf.strings.reduce_join(
            tf.as_string(fingerprints), axis=-1)  # 合并为单字符串 [N]
        # 获取唯一指纹及频次
        unique_fp, idx, counts = tf.unique_with_counts(fingerprints_flat)
        # 提取每个唯一指纹对应的首个原始向量
        indices = tf.range(tf.shape(reshaped)[0])  # 生成索引 [0, 1, ..., N-1]
        first_occur_idx = tf.math.unsorted_segment_min(
            indices, idx, tf.shape(unique_fp)[0])  # 每组最小索引
        unique_vectors = tf.gather(reshaped, first_occur_idx)  # [x, dim]
        return unique_vectors, counts

    def nce_loss(self, decoder_emb, item_embs):
        # MLP 预测每个step对应选择哪个item
        # TODO: logQ 纠偏
        with tf.variable_scope("predict_token_nce", reuse=tf.AUTO_REUSE):
            batch_size = tf.shape(decoder_emb)[0]
            list_size = tf.shape(decoder_emb)[1]
            vocab_size = tf.shape(item_embs)[1]
            dim = item_embs.shape[-1]
            vocab_emb = tf.reshape(item_embs, (-1, dim)) # (? * candidates_size,dim)
            concat_emb = tf.concat([
                tf.broadcast_to(tf.expand_dims(decoder_emb, 2), 
                            [batch_size, list_size, batch_size*vocab_size, dim]),
                tf.broadcast_to(tf.expand_dims(vocab_emb, 0), 
                            [batch_size, list_size, batch_size*vocab_size, dim])
            ], axis=-1) # [batch_size, list_size, batch_size*vocab_size, dim*2]
            predict = tf.layers.dense(concat_emb, 64, activation=tf.nn.relu)
            predict = tf.layers.dense(predict, 32, activation=tf.nn.relu)
            predict = tf.layers.dense(predict, 1, activation=tf.nn.sigmoid, name="prob_layer")
            predict = tf.squeeze(predict, axis=-1) # [batch_size, list_size, batch_size*vocab_size]
            # label, 样本组织形式为 vocab top 6 的item 为曝光样本。
            indices = tf.range(batch_size * vocab_size)
            indices = tf.reshape(indices, (batch_size, vocab_size))[:, :list_size] # (?,list_size)
            labels = tf.one_hot(indices, depth=batch_size * vocab_size) # (?,list_size,batch_size*vocab_size)
            labels = tf.cast(labels, dtype=tf.float32)
            probs = tf.nn.softmax(predict, axis=-1) # 也可以采用hinge loss
            loss = - tf.reduce_sum(tf.log(probs + 1e-9) * labels)
            return loss
    
    def get_view_label(self, playing_time, duration_ms):
        playing_time = tf.cast(playing_time, dtype=tf.int32)
        duration_ms = tf.cast(duration_ms, dtype=tf.int32)
        ones = tf.ones_like(duration_ms)
        eff_threshold = ones * 12700
        long_threshold = ones * 79700

        long_threshold = tf.where(duration_ms <= 195000, ones * 92500, long_threshold)
        eff_threshold = tf.where(duration_ms <= 195000, ones * 17600, eff_threshold)
        
        long_threshold = tf.where(duration_ms <= 118200, ones * 74900, long_threshold)
        eff_threshold = tf.where(duration_ms <= 118200, ones * 18300, eff_threshold)
        
        long_threshold = tf.where(duration_ms <= 71800, ones * 46600, long_threshold)
        eff_threshold = tf.where(duration_ms <= 71800, ones * 13100, eff_threshold)
        
        long_threshold = tf.where(duration_ms <= 38800, ones * 28800, long_threshold)
        eff_threshold = tf.where(duration_ms <= 38800, ones * 11400, eff_threshold)
        
        long_threshold = tf.where(duration_ms <= 20300, ones * 18400, long_threshold)
        eff_threshold = tf.where(duration_ms <= 20300, ones * 9900, eff_threshold)
        
        long_threshold = tf.where(duration_ms <= 12700, ones * 13600, long_threshold)
        eff_threshold = tf.where(duration_ms <= 12700, ones * 8700, eff_threshold)
        
        long_threshold = tf.where(duration_ms <= 8700, ones * 12000, long_threshold)
        eff_threshold = tf.where(duration_ms <= 8700, ones * 7200, eff_threshold)

        long_threshold = tf.where(tf.equal(duration_ms, 0), ones * 13100, long_threshold)
        # eff_threshold = tf.where(tf.equal(duration_ms, 0), ones * 4500, eff_threshold)
        eff_threshold = tf.where(tf.equal(duration_ms, 0), ones * 7000, eff_threshold)

        effective_view = playing_time >= eff_threshold
        effective_view = tf.cast(effective_view, dtype=tf.float32)
        long_view = playing_time >= long_threshold
        long_view = tf.cast(long_view, dtype=tf.float32)

        return effective_view, long_view

    def mha_layer_4d(self, name, query, key, dim_in=64, num_heads=4, dropout_rate=0.0, training=False, causal_mask=False):
        '''
        query: (?, cand_size, dim1)
        key: (?, cand_size, key_len, dim2)
        '''
        batch_size, cand_size, key_len = tf.shape(key)[0], tf.shape(key)[1], key.shape[2]
        query = tf.expand_dims(query, axis=2) # (?, cand_size, 1, dim1)
        query_dim = query.shape[-1]
        key_dim = key.shape[-1]
        query = tf.reshape(query, [batch_size * cand_size, 1, query_dim])
        key = tf.reshape(key, [batch_size * cand_size, key_len, key_dim])
        attn_out = multi_head_attention(name, query, key, key, dim_in=dim_in, num_heads=num_heads,
                                        dropout_rate=dropout_rate, training=training, causal_mask=causal_mask) # (?*cand_size,1,dim)
        attn_out = tf.reshape(attn_out, [batch_size, cand_size, dim_in]) # (?, cand_size, dim1)
        return attn_out

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
            common_embs, user_embs, source_embs, photo_embs, photo_attr_embs, pxtr_embs = self._get_shared_features(self._parameters_dict) # (?,60,d)
            pxtr_mha_0 = self.mha_layer_4d("pxtr_mha_0", photo_attr_embs, pxtr_embs, dim_in=64, num_heads=4, dropout_rate=0.1, training=self._training) # (?,cand_size,d)
            print("photo_embs ", photo_embs.shape)
            print("pxtr_mha_0", pxtr_mha_0.shape)
            print("common_embs", common_embs.shape)
            query_emb = tf.concat([user_embs, source_embs], axis=-1) # (?,cand_size,d)
            print("query_emb", query_emb.shape)
            pxtr_mha_1 = self.mha_layer_4d("pxtr_mha_1", query_emb, pxtr_embs, dim_in=64, num_heads=4, dropout_rate=0.1, training=self._training) # (?,cand_size,d)
            print("pxtr_mha_1", pxtr_mha_1.shape)
            photo_embs = tf.layers.dense(tf.concat([photo_embs, pxtr_mha_0, pxtr_mha_1], axis=-1), 64, activation=tf.nn.relu)
            common_embs = tf.layers.dense(tf.concat([common_embs, pxtr_mha_0, pxtr_mha_1], axis=-1), 64, activation=tf.nn.relu)
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
            self.item_embs = photo_embs
            self.photo_embs = tf.concat([pad_embedding, sos_embedding, photo_embs, eos_embedding], axis=1) # (?,candidates_size+3,32)

            if self._training:
                label_dicts = self._label_value_dict
                ltr_label = tf.cast(tf.reshape(label_dicts['fountain_ltr_label_list'], [-1, self.candidates_size]), tf.float32)
                ltr_label = ltr_label[:,:self.list_size] # (?,list_size)，截断为 list_size 个
                like_label = tf.cast(tf.reshape(label_dicts['fountain_ltr_label_list'], [-1, self.candidates_size]), tf.float32)
                like_label = like_label[:,:self.list_size] # (?,list_size)，截断为 list_size 个
                follow_label = tf.cast(tf.reshape(label_dicts['context_info__follow_list'], [-1, self.candidates_size]), tf.float32)
                follow_label = follow_label[:,:self.list_size]
                forward_label = tf.cast(tf.reshape(label_dicts['context_info__forward_list'], [-1, self.candidates_size]), tf.float32)
                forward_label = forward_label[:,:self.list_size]
                comment_label = tf.cast(tf.reshape(label_dicts['context_info__comment_list'], [-1, self.candidates_size]), tf.float32)
                comment_label = comment_label[:,:self.list_size]
                next_label = tf.cast(tf.reshape(label_dicts['context_info__fountain_slide_to_next_list'],  [-1, self.candidates_size]), tf.float32)
                next_label = next_label[:,:self.list_size]
                realshow_label = tf.reshape(label_dicts['context_info__real_show_list'], [-1, self.candidates_size])
                realshow_label = realshow_label[:,:self.list_size] # (?,list_size)，截断为 list_size 个
                realshow_label = tf.cast(realshow_label,dtype=tf.int32) #(?,list_size)
                click_label = tf.cast(tf.reshape(label_dicts["fountain_click_label_list"], [-1, self.candidates_size]),dtype=tf.float32)
                click_label = click_label[:, :self.list_size] # (?,list_size)，截断为 list_size 个
                click_mask = tf.cast(click_label > 0, dtype=tf.float32)
                wtd_label = tf.cast(tf.reshape(label_dicts["fountain_wtd_label_list"], [-1, self.candidates_size]),dtype=tf.float32)
                wtd_label = wtd_label[:, :self.list_size] # (?,list_size)，截断为 list_size 个
                playtime = tf.reshape(label_dicts['context_info__playing_time_list'], [-1, self.candidates_size])
                # evtr, lvtr = self.get_view_label(playtime, label_dicts["photo_info__duration_ms"])
                playtime = playtime[:,:self.list_size] # (?,list_size)
                playtime = tf.cast(tf.clip_by_value(playtime / 1000, 0, 900), dtype=tf.float32)
                svtr_label = tf.cast(tf.math.logical_and(playtime < 3.0, playtime > 0.0), dtype=tf.float32) # (?,list_size)

                indices_shape = tf.shape(realshow_label)
                col_indices = tf.tile(tf.expand_dims(tf.range(indices_shape[1]),0),[indices_shape[0],1]) + 2 # (?,list_size) 从第2个起
                print("realshow_label shape ",realshow_label.shape)
                print("col indices shape ",col_indices.shape)
                realshow_indices = tf.cast(col_indices * realshow_label,dtype=tf.int32) # label为0 1，过滤了未曝光的index
                inputs = tf.concat([sos_token, realshow_indices], axis=1) # (?,list_size+1)
                outputs = tf.concat([realshow_indices, eos_token], axis=1) # (?,list_size+1)
                print("inputs ", inputs)
                print("outputs ", outputs)
                batch_indices = tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), [1, self.list_size+1]) # (?, self.list_size+1)
                # batch_indices = tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), [1, self.candidates_size+1]) # after shuffle
                gather_indices = tf.stack([batch_indices, inputs], axis=-1) # (?, self.list_size+1, 2)

        with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
            
            # 初始化transformer模型
            model = StackedTransformerModel(num_layers=1, dim=64, num_heads=4, dk=64, dropout_rate=0.1, k=6)
            hidden_states = model.forward(common_embs, training=True)
            encoder_output = hidden_states
            print("encoder output shape ",encoder_output.shape) # (?,candidates_size,32)
            if self._training:
                # 从hidden_states查对应emb表示
                # item_embeddings = tf.gather_nd(hidden_states, gather_indices) # (?,candidates_size+3,32) 中查找对应 list idx 的 emb
                item_embeddings = tf.gather_nd(self.photo_embs, gather_indices) # (?,candidates_size+3,32) 中查找对应 list idx 的 emb
                print("item_embeddings shape", item_embeddings.shape) # (?,list_size+1,32)

                item_embedding = model.forward_decoder(hidden_states, item_embeddings, training=True) # (?,list_size+1,32)
                print("item_embedding shape ", item_embedding.shape)

                # nce loss
                nce_loss = self.nce_loss(item_embedding[:, :-1, :], self.item_embs)

                # 从候选集选取 item, 0: nn, 1: cosine; 是否进行采样
                predict = self.choose_item(item_embedding, self.photo_embs, method=0, use_gumbel_softmax=False, tau=0.1, hard=True) # (?,list_size+1,candidates_size+3)
                output_indices = tf.expand_dims(outputs, axis=2) # (?,list_size+1,1)
                pos_output = tf.batch_gather(predict, output_indices) # (?,list_size+1,1) 拿到真实index对应的score, 非全局emb Matrix 需要使用batch_gather
                pos_output = tf.squeeze(pos_output, axis=-1) #(?,list_size+1)
                print("pos_output shape", pos_output.shape)

                advantage_reward = next_label * 2.0 + ltr_label * 200.0 + playtime
                advantage_reward = tf.where(advantage_reward > 0.0, advantage_reward, tf.zeros_like(advantage_reward, dtype=tf.float32))
                advantage = self.cal_batch_advantage(advantage_reward, mask=realshow_label) # (?, list_size)
                advantage = tf.where(advantage > 0.0, advantage + 1.0, tf.ones_like(advantage, dtype=tf.float32))
                point_reward = ltr_label * 200.0 + playtime

                # bpr loss
                bpr_loss = tf.zeros([], dtype=tf.float32)
                bpr_reward = playtime + ltr_label * 200
                for i in range(self.list_size):
                    ith_prediction = predict[:, i, 2:8] # (?,list_size+1)
                    bpr_threshold = tf.where(playtime < 20.0, tf.ones_like(playtime), tf.ones_like(playtime) * 2)
                    bpr_threshold = tf.where(playtime > 60.0, tf.ones_like(playtime) * 5, bpr_threshold)
                    bpr_loss2 = self.bpr_loss(ith_prediction, bpr_reward, tf.expand_dims(bpr_threshold, axis=-1), realshow_label > 0)
                    bpr_loss += tf.reduce_sum(bpr_loss2) / self.list_size

                zeros = tf.zeros([batch_size, 1], dtype=tf.int32) # mask EOS token
                realshow_label = tf.concat([realshow_label, zeros], axis=1)
                print("outputs shape", outputs.shape)
                realshow_label = tf.cast(realshow_label,dtype=tf.float32)
                valid_pos_output = -tf.log(tf.clip_by_value(pos_output, 1e-10, 1.0)) * realshow_label #(?,list_size+1)
                # 子序列 loss
                # sub_seq_cumsum_reward = tf.cumsum(playtime, axis=1) + tf.cumsum(ltr_label * 400.0, axis=1)
                sub_seq_cumsum_reward = tf.cumsum(playtime, axis=1)
                sub_seq_probs = tf.cumsum(valid_pos_output, axis=1) * realshow_label
                sub_seq_loss = tf.reduce_mean(sub_seq_probs[:, :-1] * sub_seq_cumsum_reward)

                valid_counts = tf.reduce_sum(realshow_label, axis=-1)+1e-9
                # item_weight = playtime # (?,cand_size)
                item_weight = advantage_reward # (?,cand_size)
                ones = tf.ones([batch_size, 1], dtype=tf.float32)
                item_weight = tf.concat([item_weight, ones], axis=1) # (?, list_size + 1) 单点reward
                seq_weight = tf.reduce_sum(item_weight * realshow_label,axis=-1) / valid_counts # (?,)
                seq_advantage = self.cal_batch_advantage(seq_weight, mask=tf.ones_like(seq_weight, dtype=tf.float32)) # (?,)
                seq_advantage = tf.where(seq_advantage > 0.0, seq_advantage + 1.0, tf.ones_like(seq_advantage, dtype=tf.float32))
                # item_weight = tf.where(item_weight > 7, tf.log(item_weight) / tf.math.log(1.4) - 4.6, tf.ones_like(item_weight, dtype=tf.float32))
                # item_weight *= realshow_label
                # advantage += interact_bonus # 互动加权
                # advantage = tf.concat([advantage, ones], axis=1)
                seq_advantage = tf.expand_dims(seq_advantage, axis=1)
                # weight = 0.9 * advantage + 0.1 * seq_advantage
                weight = tf.concat([point_reward, ones], axis=1)
                # self.print_ops.append(tf.print("weight ", weight[2], summarize=8, output_stream=sys.stdout))
                gen_loss = tf.reduce_mean(tf.reduce_sum(valid_pos_output, axis=-1))
                reward_loss = tf.reduce_mean(tf.reduce_sum(valid_pos_output * weight, axis=-1))
                return predict, nce_loss, gen_loss, bpr_loss, reward_loss, sub_seq_loss
            
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