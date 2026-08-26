import sys
import tensorflow as tf
from modules_ import *
from feature_attr_extract import user_fea_names,explore_profile_fea_names,photo_fea_names,source_fea_names,fountain_profile_fea_names,playtime_fea_names


class StackedTransformerModel():
    def __init__(self, num_layers, dim, num_heads, dk, dropout_rate, k, training=False):
        super(StackedTransformerModel, self).__init__()
        self.num_layers = num_layers
        self.k = 6
        self.dim = dim
        self.position = tf.get_variable('s', shape=[self.k, self.dim], initializer=tf.random_normal_initializer())
        self.layers = [EncoderLayer(f"encoder_layer_{i}", dim, num_heads, dk, dropout_rate) for i in range(num_layers)]
        self.position_layers = [PositionLayer(f"position_layer_{i}", dim, num_heads, dk, dropout_rate, training=training) for i in range(num_layers)]
        
    def forward(self, hidden_states, training):
        pos_embedding = tf.reshape(self.position, [1, self.k, self.dim])
        pos_embedding = tf.tile(pos_embedding, [tf.shape(hidden_states)[0], 1, 1])
        for i in range(self.num_layers):
            hidden_states = self.layers[i].forward(hidden_states, training=training)
            pos_embedding = self.position_layers[i].forward(pos_embedding, hidden_states, training=training)
        return hidden_states, pos_embedding

class Evaluator():
    def __init__(self, num_layers, dim, num_heads, dk, dropout_rate, k, training=False):
        super(Evaluator, self).__init__()
        self.num_layers = num_layers
        self.k = k
        self.dim = dim
        self.position = tf.get_variable('s', shape=[self.k, self.dim], initializer=tf.random_normal_initializer())
        self.layers = [EncoderLayer(f"transformer_layer_{i}", dim, num_heads, dk, dropout_rate) for i in range(num_layers)]

    def forward(self, hidden_states, training):
        pos_embedding = tf.reshape(self.position, [1, self.k, self.dim])
        pos_embedding = tf.tile(pos_embedding, [tf.shape(hidden_states)[0], 1, 1])
        for i in range(self.num_layers):
            hidden_states = self.layers[i].forward(hidden_states, training=training)
        return hidden_states

class FountainDeepLtrMultiTaskModel:
    def __init__(self, parameters_dict, label_value_dict, dense_value_dict, print_ops, list_size, candidates_size, dim=32, extra_param_dict= None, training=True):
        self._parameters_dict = parameters_dict
        self._label_value_dict = label_value_dict
        self._dense_value_dict = dense_value_dict
        self.list_size = list_size
        self.candidates_size = candidates_size
        self.dim = dim
        self._training = training
        self.position_embeddings = tf.get_variable(
            name='position_embeddings', 
            shape=[6, 32], 
            initializer=tf.random_normal_initializer()
        )
        self.print_ops = print_ops

    def _calc_point_reward(self):
        # 计算单点价值
        pctr = tf.cast(self._dense_value_dict["context_info__pctr"], dtype=tf.float32)
        pvtr = tf.cast(self._dense_value_dict["context_info__pvtr"], dtype=tf.float32)
        pltr = tf.cast(self._dense_value_dict["context_info__pltr"], dtype=tf.float32)
        plvtr = tf.cast(self._dense_value_dict["context_info__plvtr"], dtype=tf.float32)
        pwtr = tf.cast(self._dense_value_dict["context_info__pwtr"], dtype=tf.float32)
        psvtr = tf.cast(self._dense_value_dict["context_info__psvtr"], dtype=tf.float32)
        preward = 1+ pctr + pvtr * 2 + pltr + plvtr * 0.8 + pwtr * 0.5 + psvtr * 0.1
        return preward

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
            user_embs     = tf.tile(tf.expand_dims(user_embs, axis=1), [1,self.candidates_size,1]) if self._training else user_embs
            photo_embs    = tf.concat([input_dicts[k] for k in input_dicts if k in photo_fea_names], axis=-1)
            source_embs   = tf.concat([input_dicts[k] for k in input_dicts if k in source_fea_names], axis=-1)
            source_embs   = tf.tile(tf.expand_dims(source_embs, axis=1), [1,self.candidates_size,1]) if self._training else source_embs
            explore_embs  = tf.concat([input_dicts[k] for k in input_dicts if k in explore_profile_fea_names], axis=-1)
            explore_embs  = tf.reduce_mean(explore_embs, axis=1)
            explore_embs = tf.tile(tf.expand_dims(explore_embs, axis=1),[1,self.candidates_size,1]) if self._training else explore_embs
            fountain_embs = tf.concat([input_dicts[k] for k in input_dicts if k in fountain_profile_fea_names], axis=-1)
            fountain_embs  = tf.reduce_mean(fountain_embs, axis=1)
            fountain_embs = tf.tile(tf.expand_dims(fountain_embs, axis=1), [1,self.candidates_size,1]) if self._training else fountain_embs

            playtime_embs = tf.concat([input_dicts[k] for k in input_dicts if k in playtime_fea_names], axis=-1)
            playtime_embs  = tf.reduce_mean(playtime_embs, axis=1) # (?, dim)
            uid_emb = tf.tile(tf.expand_dims(input_dicts["user_id"], axis=1), [1,self.candidates_size,1]) if self._training else input_dicts["user_id"]
            playtime_embs = tf.tile(tf.expand_dims(playtime_embs, axis=1), [1,self.candidates_size,1]) if self._training else playtime_embs
            print("photo_id_v2", input_dicts["photo_id_v2"])
            playtime_embs = tf.concat([input_dicts["photo_id_v2"], uid_emb, playtime_embs], axis=-1) # train:(?, cand_size, dim),infer:(?, dim)
            # common_embs   = tf.concat([user_embs, photo_embs, explore_embs, fountain_embs, playtime_embs, source_embs], axis=-1)
            common_embs   = tf.concat([user_embs, photo_embs, fountain_embs, playtime_embs, source_embs], axis=-1)
            photo_weights = tf.layers.dense(tf.concat([photo_embs, user_embs], axis=-1), 256, activation=tf.nn.sigmoid)

            common_embs   = tf.layers.dense(common_embs, 512, activation=tf.nn.leaky_relu)
            common_embs   = tf.layers.dense(common_embs, 256, activation=tf.nn.leaky_relu) * photo_weights # (?, cand_size, 256)
            if not self._training:
                '''
                    infer 时需要注意实际请求的 batch size = items 长度, 但计算图 batch 的 shape 由 uni_predict_fused 中 executor_batchsizes 决定,
                    如600。为了通过计算图编译, 需要 rashape (1, -1, dim), 实际 -1 为请求端发送 items 长度
                '''
                emb_dim = common_embs.shape[-1]
                common_embs = tf.reshape(common_embs, [1, -1, emb_dim])
            return common_embs
        
    def _contrastive_loss(self, score_matrix, margin=0.8, seqlen=6):
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

    def choose_item(self, decoder_emb, method=0, use_gumbel_softmax=False, tau=1.0, hard=True):
        # 端到端的情况下需要打开 gumbel softmax，具备采样能力
        if method == 0:
            # MLP 预测每个step对应选择哪个item [0, 1, 2, 3, EOT]
            with tf.variable_scope("predict_nn", reuse=tf.AUTO_REUSE):
                decoder_emb = tf.tile(tf.expand_dims(decoder_emb, axis=2), [1, 1, self.candidates_size + 3, 1]) # (?,list_size+1,candidates_size+3,32)
                vocab_emb = tf.tile(tf.expand_dims(self.photo_embs, axis=1), [1, tf.shape(decoder_emb)[1], 1, 1]) # (?,list_size+1,candidates_size+3,32)
                concat_emb = tf.concat([decoder_emb, vocab_emb], axis=-1) # (?,list_size+1,candidates_size+3,64)
                predict = self._mlp_layer("mlp_layer", concat_emb, [128, 64])
                predict = tf.layers.dense(predict, 1, activation=tf.nn.sigmoid, name="prob_layer") # (?,list_size+1,candidates_size+3,1)
                predict = tf.squeeze(predict, axis=-1) # (?,list_size+1,candidates_size+3)
        elif method == 1:
            # cosine 选取
            with tf.variable_scope("predict_cosine", reuse=tf.AUTO_REUSE):
                predict = tf.matmul(decoder_emb, tf.transpose(self.photo_embs,  perm=[0, 2, 1])) # (?, list_size+1, candidates_size+3)
                predict = tf.nn.softmax(predict, axis=-1)
        if use_gumbel_softmax:
            predict = self.gumbel_softmax(predict, tau=tau, hard=hard)
        else:
            predict = tf.nn.softmax(predict, axis=-1) # (?,list_size+1,candidates_size+3)
        return predict

    def weighted_log_loss(self, y_true, y_pred, weights):
        epsilon = 1e-15
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        log_loss = - weights * (y_true * tf.log(y_pred) + (1 - y_true) * tf.log(1 - y_pred))
        weighted_log_loss = log_loss
        return tf.reduce_mean(weighted_log_loss)

    def pairwise_bpr_loss_v2(self, output, score, threshold, mask):
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
        random_mask = tf.random.uniform(tf.shape(valid_pair_mask)) < 0.15
        activated_mask = tf.logical_and(random_mask, tf.logical_not(valid_pair_mask))
        final_mask = tf.logical_or(valid_pair_mask, activated_mask)
        final_mask = tf.logical_and(final_mask, pairwise_label_mask)
        # 计算BPR损失
        bpr_loss = -tf.log(logit_diff) * pairwise_labels
        print("bpr_loss", bpr_loss)
        bpr_loss = tf.where(final_mask, bpr_loss, tf.zeros_like(bpr_loss, dtype=tf.float32))
        return bpr_loss

    def model(self):
        with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
            common_embs = self._get_shared_features(self._parameters_dict) # (?,cand_size,256)
            batch_size = tf.shape(common_embs)[0]
            hidden_states = self._mlp_layer("mlp_layer_1", common_embs, [128, 64]) # (?,candidates_size,32)
            # 初始化transformer模型
            model = StackedTransformerModel(num_layers=1, dim=64, num_heads=4, dk=128, dropout_rate=0.0, k=6)
            hidden_states, pos_embedding = model.forward(hidden_states, training=self._training)
            pos_embedding_trans = tf.transpose(pos_embedding,  perm=[0, 2, 1])
            predict_ori = tf.matmul(hidden_states, pos_embedding_trans)
            # print("predict shape", predict.shape)

            tau = 0.05
            # tau = 0.1
            predict = self.gumbel_softmax(predict_ori, tau, hard=True, dim=1)
            predict_ori = tf.nn.softmax(predict_ori/tau, axis=1) #?,candidates_size,list_size
            norm_rep = pos_embedding / tf.norm(pos_embedding, axis=2, keepdims=True)
            cosine_scores_rep = tf.matmul(norm_rep, tf.transpose(norm_rep, perm=[0, 2, 1]))
            cl_loss_pad = self._contrastive_loss(cosine_scores_rep) #position cl loss

            norm_outputs = hidden_states / tf.norm(hidden_states, axis=2, keepdims=True)
            cosine_scores_outputs = tf.matmul(norm_outputs, tf.transpose(norm_outputs, perm=[0, 2, 1]))
            cl_loss_outputs = self._contrastive_loss(cosine_scores_outputs, seqlen=self.candidates_size) #candidates cl loss
            cl_loss = cl_loss_pad + cl_loss_outputs
            
        predict = tf.transpose(predict,  perm=[0, 2, 1]) #bs,6,60
        predict_ori = tf.transpose(predict_ori, perm=[0,2,1]) #?,6,60
        generator_embeding = tf.matmul(predict, common_embs) #bs,6,32 predict矩阵6个位置对应的candidates的embedding
        item_embedding_gen = hidden_states

        with tf.variable_scope("evaluator", reuse=tf.AUTO_REUSE):
            common_embs = self._get_shared_features(self._parameters_dict)
            batch_size = tf.shape(common_embs)[0]
            label_dicts = self._label_value_dict

            # rerank_wtd = label_dicts["fountain_wtd_label_list"]
            # rerank_wtd_label = tf.reshape(rerank_wtd, [-1, self.candidates_size])
            # rerank_wtd = rerank_wtd_label[:,:self.list_size]
            # wtd_label = tf.cast(tf.math.greater(rerank_wtd,0.05),tf.int32)
            # is_wtd = tf.cast(wtd_label, tf.float32)

            rerank_ltr = label_dicts["fountain_ltr_label_list"]
            rerank_ltr = tf.reshape(rerank_ltr, [-1, self.candidates_size])
            rerank_ltr = rerank_ltr[:,:self.list_size]
            ltr_label = tf.cast(rerank_ltr, tf.int32)

            rerank_label = label_dicts['context_info__real_show_list']
            rerank_label = tf.reshape(rerank_label, [-1, self.candidates_size])
            rerank_label = rerank_label[:,:self.list_size]
            indices_shape = tf.shape(rerank_label)
            
            # rerank_label = tf.math.logical_or(
            #     tf.math.equal(wtd_label, 1),
            #     tf.math.equal(ltr_label, 1)
            # )
            rerank_label = tf.cast(rerank_label,dtype=tf.int32)

            col_indices = tf.tile(tf.expand_dims(tf.range(indices_shape[1]),0),[indices_shape[0],1])
            rank_indices = tf.cast(col_indices*rerank_label,dtype=tf.int32) # (?, list_size)

            batch_indices = tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), [1, self.list_size]) 
            gather_indices = tf.stack([batch_indices, rank_indices], axis=-1)
            item_embeddings = tf.gather_nd(common_embs, gather_indices) #bs,6,32 ground truth exposured candidates embedding
            print("item_embeddings shape", item_embeddings.shape)

            rerank_weight = label_dicts["fountain_fulllink_rerank_realshow_label_weight_list"]
            rerank_weight = tf.reshape(rerank_weight, [-1, self.candidates_size])
            rerank_weight = rerank_weight[:,:self.list_size] # 截断 list 长度
            item_weight = tf.gather_nd(rerank_weight, gather_indices)
            click_thresh = 10.0
            click_label = tf.cast(tf.math.greater(item_weight - 1.0, click_thresh), tf.float32) # 非短播

            hidden_states = self._mlp_layer("mlp_layer_1", item_embeddings, [64, 32])
            position_ids = tf.range(6, dtype=tf.int32)
            position_ids = tf.expand_dims(position_ids, 0)
            position_embeddings = tf.nn.embedding_lookup(self.position_embeddings, position_ids)
            position_embeddings = tf.tile(position_embeddings, [tf.shape(hidden_states)[0], 1, 1])
            hidden_states = hidden_states+position_embeddings #grund truth can

            generator_embeding = self._mlp_layer("mlp_layer_1", generator_embeding, [64, 32])
            generator_embeding = generator_embeding+position_embeddings #generate can

            model = Evaluator(num_layers=3, dim=64, num_heads=4, dk=128, dropout_rate=0.1, k=self.list_size)
            hidden_states = model.forward(hidden_states, training=self._training) #ground truth -> evualator
            generator_embeding = model.forward(generator_embeding, training=self._training) #generator选出的embedding -> evaluator

            norm_states = hidden_states / tf.norm(hidden_states, axis=2, keepdims=True)
            cosine_scores_states = tf.matmul(norm_states, tf.transpose(norm_states, perm=[0, 2, 1]))
            cl_loss_states = self._contrastive_loss(cosine_scores_states)

            generator_logits = self._mlp_layer("mlp_layer_out", generator_embeding, [1], activation=tf.nn.sigmoid) #evualator对generator candidates的预估分
            logits = self._mlp_layer("mlp_layer_out", hidden_states, [1], activation=tf.nn.sigmoid) #evualator对ground truth candidates的预估分
            logits = tf.squeeze(logits, axis=-1)
            print("item_weight shape", item_weight.shape)
            print("logits shape", logits.shape)
            
            valid_label = tf.cast(rank_indices, tf.int64) #?,6
            masked_indices = tf.cast(tf.expand_dims(valid_label, axis=2),tf.int64) #?,6,1
            # 根据indice取出模型预估值
            pos_output = tf.batch_gather(predict_ori, masked_indices) #(?,6,1)
            # pos_output = tf.batch_gather(predict, masked_indices) #(?,6,1)
            pos_output = tf.squeeze(pos_output, axis=-1) #(?,6)
            real_show_top6 = tf.cast(rerank_label, dtype=tf.float32)
            # 计算loss
            valid_pos_output = tf.log(pos_output+1e-9)*real_show_top6 #(?,6)
            valid_counts = tf.reduce_sum(real_show_top6, axis=-1)+1e-9 #避免除0
            # 对每个样本，只计算有效位置的平均loss
            gen_loss = -tf.reduce_mean(tf.reduce_sum(valid_pos_output, axis=-1)/valid_counts) #(?,)


            # 计算 Evaluator label
            def cal_advantage(reward, mask):
                # mean_group_rewards = tf.reduce_mean(reward, axis=1)
                # std_group_rewards = tf.reduce_std(reward, axis=1)
                # advantages = (reward - mean_group_rewards) / (std_group_rewards + 1e-8)
                mask = tf.cast(mask, reward.dtype)
                valid_cnt = tf.reduce_sum(mask, axis=1, keepdims=True)
                mean = tf.reduce_sum(reward * mask, axis=1, keepdims=True) / (valid_cnt + 1e-8)
                variance = (reward - mean) ** 2 * mask
                std = tf.sqrt(tf.reduce_sum(variance, axis=1, keepdims=True) / (valid_cnt + 1e-8))
                advantages = (reward - mean) / (std + 1e-8)
                return advantages
            item_weight_clip = tf.clip_by_value(item_weight - 1.0, 0, 600)
            item_weight = item_weight_clip
            bound_neg_1 = tf.ones_like(item_weight, dtype=tf.float32) * 2.0 # [0, 3)
            bound_neg_2 = tf.ones_like(item_weight, dtype=tf.float32) * 1.0 # [3, 5)
            bound_neg_3 = tf.ones_like(item_weight, dtype=tf.float32) * 0.5 # [5, 7)
            bound_neg_4 = tf.ones_like(item_weight, dtype=tf.float32) * 0.2 # [7, 10)
            bound1 = tf.ones_like(item_weight, dtype=tf.float32) # [7, 12)
            bound2 = (item_weight - 10) * 0.05 + 1.0 # [12, 20) max = 1.5
            bound3 = (item_weight - 20) * 0.00125 + 1.5 # [20, 60) max = 2.0
            bound4 = tf.log(item_weight - 59) / tf.math.log(3.0) / 1.5 + 2.0 # [60, 1000)
            item_weight = tf.where(item_weight_clip >= 3, bound_neg_2, bound_neg_1)
            item_weight = tf.where(item_weight_clip >= 5, bound_neg_3, item_weight)
            item_weight = tf.where(item_weight_clip >= 7, bound_neg_4, item_weight)
            item_weight = tf.where(item_weight_clip >= 10, bound2, item_weight)
            item_weight = tf.where(item_weight_clip >= 20, bound3, item_weight)
            item_weight = tf.where(item_weight_clip >= 60, bound4, item_weight)
            # bound_neg_1 = tf.ones_like(item_weight, dtype=tf.float32) * 2.0 # [0, 3)
            # bound_neg_2 = tf.ones_like(item_weight, dtype=tf.float32) * 1.5 # [3, 5)
            # bound_neg_3 = tf.ones_like(item_weight, dtype=tf.float32) * 1.0 # [5, 7)
            # # bound_neg_4 = tf.ones_like(item_weight, dtype=tf.float32) * 0.2 # [7, 10)
            # bound1 = tf.ones_like(item_weight, dtype=tf.float32) # [7, 12)
            # bound2 = (item_weight - 10) * 0.1 + 1.0 # [12, 20) max = 2
            # bound3 = (item_weight - 20) * 0.025 + 2.0 # [20, 60) max = 3
            # # bound4 = tf.log(item_weight - 59) / tf.math.log(3.0) / 1.5 + 2.0 # [60, 1000)
            # bound4 = tf.ones_like(item_weight, dtype=tf.float32) * 2.5 # [60, 1000)
            # item_weight = tf.where(item_weight_clip >= 3, bound_neg_2, bound_neg_1)
            # item_weight = tf.where(item_weight_clip >= 5, bound_neg_3, item_weight)
            # item_weight = tf.where(item_weight_clip >= 7, bound1, item_weight)
            # item_weight = tf.where(item_weight_clip >= 10, bound2, item_weight)
            # item_weight = tf.where(item_weight_clip >= 20, bound3, item_weight)
            # item_weight = tf.where(item_weight_clip >= 60, bound4, item_weight)
            # # 增加互动权重
            # item_weight = item_weight + ltr_label

            item_weight = item_weight * tf.cast(rerank_label, dtype=tf.float32) # mask掉未曝光的item
            self.print_ops.append(tf.print(f"show_weight", rerank_weight[2], summarize = 8, output_stream=sys.stdout))
            self.print_ops.append(tf.print(f"item_weight", item_weight[2], summarize = 8, output_stream=sys.stdout))
            self.print_ops.append(tf.print(f"click_label [weight > 10]", click_label[2], summarize = 8, output_stream=sys.stdout))

            # gen-eval loss
            # generator_loss = -tf.reduce_mean(generator_logits-0.7)
            generator_loss = -tf.reduce_mean(tf.math.log(generator_logits)) # -logP

            #eval loss
            single_loss = self.weighted_log_loss(click_label, logits, tf.cast(rerank_label, dtype=tf.float32)) #evualator loss

            return logits, single_loss, item_weight, generator_loss, generator_logits, cl_loss_states, cl_loss, predict, item_embedding_gen, gen_loss, valid_label