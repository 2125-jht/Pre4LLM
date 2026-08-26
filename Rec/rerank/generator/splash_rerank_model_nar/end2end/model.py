import tensorflow as tf
from feature_attr_extract import *
from modules_ import *
import sys


def layer_norm(x, epsilon=1e-6):
    with tf.variable_scope("layer_norm", reuse=tf.AUTO_REUSE):
        gamma = tf.get_variable("gamma", [x.get_shape()[-1]], initializer=tf.ones_initializer())
        beta = tf.get_variable("beta", [x.get_shape()[-1]], initializer=tf.zeros_initializer())

        mean, variance = tf.nn.moments(x, axes=[-1], keep_dims=True)
        normalized = (x - mean) / tf.sqrt(variance + epsilon)
        output = gamma * normalized + beta
    return output


def multi_head_attention(name, queries, keys, values, num_heads, dropout_rate, training=False):
        def split_heads(x, num_heads):
            batch_size = tf.shape(x)[0]
            depth = x.get_shape().as_list()[-1] // num_heads
            reshaped = tf.reshape(x, [batch_size, -1, num_heads, depth])
            return tf.transpose(reshaped, [0, 2, 1, 3])

        def scaled_dot_product_attention(Q, K, V):
            matmul_qk = tf.matmul(Q, K, transpose_b=True)
            dk = tf.cast(tf.shape(K)[-1], tf.float32)
            scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
            attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
            output = tf.matmul(attention_weights, V)
            return output, attention_weights
        
        training = tf.constant(training, dtype=tf.bool)
        # with tf.variable_scope(f"multi_head_attention", reuse=tf.AUTO_REUSE):
        with tf.variable_scope(f"{name}", reuse=tf.AUTO_REUSE):
            depth = queries.get_shape().as_list()[-1]
            Q = tf.layers.dense(queries, depth, use_bias=False)
            K = tf.layers.dense(keys, depth, use_bias=False)
            V = tf.layers.dense(values, depth, use_bias=False)

            Q = split_heads(Q, num_heads)
            K = split_heads(K, num_heads)
            V = split_heads(V, num_heads)

            scaled_attention, attention_weights = scaled_dot_product_attention(Q, K, V)
            scaled_attention = tf.transpose(scaled_attention, [0, 2, 1, 3])

            concat_attention = tf.reshape(scaled_attention, [tf.shape(queries)[0], -1, depth])
            output = tf.layers.dense(concat_attention, depth)
            output = tf.cond(training, lambda: tf.nn.dropout(output, rate=dropout_rate), lambda: output)
        return output
    
def feed_forward_network(dim, hidden_dim, dropout_rate, training=False):
    def ffn(x, training=training):
        training = tf.constant(training, dtype=tf.bool)
        with tf.variable_scope(f"feed_forward_network", reuse=tf.AUTO_REUSE):
            x = tf.layers.dense(x, hidden_dim, activation=tf.nn.relu)
            x = tf.layers.dense(x, dim)
            # x = tf.nn.dropout(x, rate=dropout_rate)
            x = tf.cond(training, lambda: tf.nn.dropout(x, rate=dropout_rate), lambda: x)
            return x
    return ffn
    
class TransformerLayer:
    def __init__(self, name, dim, num_heads, hidden_dim, dropout_rate, training=False):
        super(TransformerLayer, self).__init__()
        self.name = name
        self.dim = dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
        self.mha = multi_head_attention
        self.ffn = feed_forward_network(dim, hidden_dim, dropout_rate)
        
    def forward(self, x, training):
        with tf.variable_scope(f"{self.name}", reuse=tf.AUTO_REUSE):
            attn_output = self.mha("self_atten", x, x, x, self.num_heads, self.dropout_rate, training=training)
            out1 = layer_norm(x + attn_output)
            
            ffn_output = self.ffn(out1, training=training)
            out2 = layer_norm(out1 + ffn_output)
        
        return out2

class PositionLayer:
    def __init__(self, name, dim, num_heads, hidden_dim, dropout_rate, training=False):
        super(PositionLayer, self).__init__()
        self.name = name
        self.dim = dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
        self.self_attention = multi_head_attention
        self.cross_attention = multi_head_attention
        self.ffn = feed_forward_network(dim, hidden_dim, dropout_rate)
        
    # def forward(self, x, enc_output, training):
    #     with tf.variable_scope(f"{self.name}", reuse=tf.AUTO_REUSE):
    #         cross_attn_output = self.cross_attention(f"cross_attn",x, enc_output, enc_output, self.num_heads, self.dropout_rate, training=training)
    #         out1 = layer_norm(x + cross_attn_output)
    #         attn_output = self.self_attention(f"self_atten",out1, out1, out1, self.num_heads, self.dropout_rate, training=training)
    #         out2 = layer_norm(out1 + attn_output)
        
    #         ffn_output = self.ffn(out2, training=training)
    #         out3 = layer_norm(out2 + ffn_output)
        
    #     return out3

    def forward(self, x, enc_output, training):
        with tf.variable_scope(f"{self.name}", reuse=tf.AUTO_REUSE):
            attn_output = self.self_attention(f"self_atten",x, x, x, self.num_heads, self.dropout_rate, training=training)
            out1 = layer_norm(x + attn_output)

            cross_attn_output = self.cross_attention(f"cross_attn",out1, enc_output, enc_output, self.num_heads, self.dropout_rate, training=training)
            out2 = layer_norm(out1 + cross_attn_output)
        
            ffn_output = self.ffn(out2, training=training)
            out3 = layer_norm(out2 + ffn_output)
        
        return out3
    
class StackedTransformerModel():
    def __init__(self, num_layers, dim, num_heads, hidden_dim, dropout_rate, k, training=False):
        super(StackedTransformerModel, self).__init__()
        self.num_layers = num_layers
        self.k = 6
        self.dim = dim
        self.position = tf.get_variable('s', shape=[self.k, self.dim], initializer=tf.random_normal_initializer())
        self.layers = [TransformerLayer(f"transformer_layer_{i}", dim, num_heads, hidden_dim, dropout_rate, training=training) for i in range(num_layers)]
        self.position_layers = [PositionLayer(f"position_layer_{i}", dim, num_heads, hidden_dim, dropout_rate, training=training) for i in range(num_layers)]
        
    def forward(self, hidden_states, training):
        pos_embedding = tf.reshape(self.position, [1, self.k, self.dim])
        pos_embedding = tf.tile(pos_embedding, [tf.shape(hidden_states)[0], 1, 1])
        for i in range(self.num_layers):
            hidden_states = self.layers[i].forward(hidden_states, training=training)
            pos_embedding = self.position_layers[i].forward(pos_embedding, hidden_states, training=training)
        return hidden_states, pos_embedding


class Evaluator():
    def __init__(self, num_layers, dim, num_heads, hidden_dim, dropout_rate, k, training=False):
        super(Evaluator, self).__init__()
        self.num_layers = num_layers
        self.k = 6
        self.dim = dim
        self.position = tf.get_variable('s', shape=[self.k, self.dim], initializer=tf.random_normal_initializer())
        self.layers = [TransformerLayer(f"transformer_layer_{i}", dim, num_heads, hidden_dim, dropout_rate, training=training) for i in range(num_layers)]
        
    def forward(self, hidden_states, training):
        pos_embedding = tf.reshape(self.position, [1, self.k, self.dim])
        pos_embedding = tf.tile(pos_embedding, [tf.shape(hidden_states)[0], 1, 1])
        for i in range(self.num_layers):
            hidden_states = self.layers[i].forward(hidden_states, training=training)
        return hidden_states


class FountainDeepLtrMultiTaskModel:
    def __init__(self, parameters_dict, label_value_dict, dim=32, cand_size=60, training=True):
        self._cand_size = cand_size
        self._parameters_dict = parameters_dict
        self._label_value_dict = label_value_dict
        self.dim = dim
        self._training = training
        self.position_embeddings = tf.get_variable(
            name='position_embeddings', 
            shape=[6, 32], 
            initializer=tf.random_normal_initializer()
        )
    
    def _mlp_layer(self,
                  scope_name,
                  hidden_states: Tensor,
                  hidden_units: list,
                  activation=tf.nn.relu) -> Tensor:
        with tf.variable_scope(f"{scope_name}_mlp_layer", reuse=tf.AUTO_REUSE):
            for i, hidden_unit in enumerate(hidden_units):
                hidden_states = tf.layers.dense(hidden_states, hidden_unit, activation=activation, use_bias=True)
        return hidden_states

    def _get_shared_features(self, input_dicts) -> tuple:
        with tf.variable_scope("share_bottom", reuse=tf.AUTO_REUSE):
            user_embs     = tf.concat([input_dicts[k] for k in input_dicts if k in user_fea_names], axis=-1)
            user_embs     = tf.tile(tf.expand_dims(user_embs, axis=1), [1,self._cand_size,1]) if self._training else user_embs
            photo_embs    = tf.concat([input_dicts[k] for k in input_dicts if k in photo_fea_names], axis=-1)
            source_embs   = tf.concat([input_dicts[k] for k in input_dicts if k in source_fea_names], axis=-1)
            source_embs   = tf.tile(tf.expand_dims(source_embs, axis=1), [1,self._cand_size,1]) if self._training else source_embs
            explore_embs  = tf.concat([input_dicts[k] for k in input_dicts if k in explore_profile_fea_names], axis=-1)
            explore_embs  = tf.reduce_mean(explore_embs, axis=1)
            explore_embs = tf.tile(tf.expand_dims(explore_embs, axis=1),[1,self._cand_size,1]) if self._training else explore_embs
            fountain_embs = tf.concat([input_dicts[k] for k in input_dicts if k in fountain_profile_fea_names], axis=-1)
            fountain_embs  = tf.reduce_mean(fountain_embs, axis=1)
            fountain_embs = tf.tile(tf.expand_dims(fountain_embs, axis=1), [1,self._cand_size,1]) if self._training else fountain_embs
            common_embs   = tf.concat([user_embs, photo_embs, fountain_embs, source_embs], axis=-1)
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
        gumbels = sample_gumbel(tf.shape(logits)) #采样Gumbel噪声   
        gumbels = (logits + gumbels) / tau #添加噪声
        y_soft = tf.nn.softmax(gumbels, axis=dim) #softmax归一化

        if hard:
            # Straight through.
            index = tf.argmax(y_soft, axis=dim) #根据y_soft选择最可能的index,argmax不可导
            y_hard = tf.one_hot(index, depth=tf.shape(logits)[dim], dtype=logits.dtype) #one-hot编码
            y_hard = tf.reshape(y_hard, tf.shape(logits)) #reshape为logits的形状
            ret = tf.stop_gradient(y_hard - y_soft) + y_soft #停止梯度更新，避免梯度消失
        else:
            ret = y_soft
        return ret
    
    # weight仅对正样本起作用
    def weighted_log_loss(self, y_true, y_pred, weights):
        """
        Compute weighted log loss.
        
        Parameters:
        y_true : Tensor
            True binary labels. Shape = [batch_size, num_classes]
        y_pred : Tensor
            Predicted probabilities. Shape = [batch_size, num_classes]
        weights : Tensor
            Weights for each class. Shape = [num_classes]

        Returns:
        loss : Tensor
            Weighted log loss.
        """
        # Ensure the predictions are within range [epsilon, 1 - epsilon] to avoid log(0)
        epsilon = 1e-15
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        log_loss = - weights * (y_true * tf.log(y_pred) + (1 - y_true) * tf.log(1 - y_pred))
        weighted_log_loss = log_loss
        return tf.reduce_mean(weighted_log_loss)

    def model(self):
        with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
            input_dicts = self._parameters_dict
            common_embs = self._get_shared_features(input_dicts)
            hidden_states = self._mlp_layer("mlp_layer_1", common_embs, [64, 32])
            dim = 32
            num_heads = 4
            hidden_dim = 128
            dropout_rate = 0.1
            sequence_length = 100
            num_layers = 3
            k = 6
            model = StackedTransformerModel(num_layers=3, dim=32, num_heads=4, hidden_dim=128, dropout_rate=0.1, k=6)
            hidden_states, pos_embedding = model.forward(hidden_states, training=True)
            pos_embedding_trans = tf.transpose(pos_embedding,  perm=[0, 2, 1])
            predict_ori = tf.matmul(hidden_states, pos_embedding_trans)
            # print("predict shape", predict.shape)

            tau = 0.05
            # tau = 0.1
            predict = self.gumbel_softmax(predict_ori, tau, hard=True, dim=1)
            predict_ori = tf.nn.softmax(predict_ori/tau, axis=1) #?,60,6
            norm_rep = pos_embedding / tf.norm(pos_embedding, axis=2, keepdims=True)
            cosine_scores_rep = tf.matmul(norm_rep, tf.transpose(norm_rep, perm=[0, 2, 1]))
            cl_loss_pad = self._contrastive_loss(cosine_scores_rep) #position cl loss

            norm_outputs = hidden_states / tf.norm(hidden_states, axis=2, keepdims=True)
            cosine_scores_outputs = tf.matmul(norm_outputs, tf.transpose(norm_outputs, perm=[0, 2, 1]))
            cl_loss_outputs = self._contrastive_loss(cosine_scores_outputs, seqlen=60) #candidates cl loss
            cl_loss = cl_loss_pad+cl_loss_outputs
            
        predict = tf.transpose(predict,  perm=[0, 2, 1]) #bs,6,60
        predict_ori = tf.transpose(predict_ori, perm=[0,2,1]) #?,6,60
        generator_embeding = tf.matmul(predict, common_embs) #bs,6,32 predict矩阵6个位置对应的candidates的embedding
        item_embedding_gen = hidden_states

        with tf.variable_scope("evaluator", reuse=tf.AUTO_REUSE):
            input_dicts = self._parameters_dict
            common_embs = self._get_shared_features(input_dicts)
            batch_size = tf.shape(common_embs)[0]
            label_dicts = self._label_value_dict

            rerank_wtd = label_dicts["fountain_wtd_label_list"]
            rerank_wtd_label = tf.reshape(rerank_wtd, [-1, 60])
            rerank_wtd = rerank_wtd_label[:,:6]
            wtd_label = tf.cast(tf.math.greater(rerank_wtd,0.05),tf.int32)
            is_wtd = tf.cast(wtd_label, tf.float32)

            rerank_ltr = label_dicts["fountain_ltr_label_list"]
            rerank_ltr = tf.reshape(rerank_ltr, [-1, 60])
            rerank_ltr = rerank_ltr[:,:6]
            ltr_label = tf.cast(rerank_ltr, tf.int32)

            rerank_label = label_dicts['context_info__real_show_list']
            rerank_label = tf.reshape(rerank_label, [-1, 60])
            rerank_label = rerank_label[:,:6]
            indices_shape = tf.shape(rerank_label)
            
            rerank_label = tf.math.logical_or(
                tf.math.equal(wtd_label, 1),
                tf.math.equal(ltr_label, 1)
            )
            rerank_label = tf.cast(rerank_label,dtype=tf.int32)
            
            col_indices = tf.tile(tf.expand_dims(tf.range(indices_shape[1]),0),[indices_shape[0],1])
            rank_indices = tf.cast(col_indices*rerank_label,dtype=tf.int32)

            batch_indices = tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), [1, 6]) 
            gather_indices = tf.stack([batch_indices, rank_indices], axis=-1)
            item_embeddings = tf.gather_nd(common_embs, gather_indices) #bs,6,32 ground truth exposured candidates embedding
            print("item_embeddings shape", item_embeddings.shape)

            rerank_weight = label_dicts["fountain_fulllink_rerank_realshow_label_weight_list"]
            rerank_weight = tf.reshape(rerank_weight, [-1, 60])
            item_weight = tf.gather_nd(rerank_weight, gather_indices)
            # mean_play_time = tf.reduce_sum(realshow_weight, axis=-1)-60     
            click_thresh = 10.0
            click_label = tf.cast(tf.math.greater(item_weight - 1.0, click_thresh), tf.float32) # 非短播       
            
            rerank_ltr = label_dicts["fountain_ltr_label_list"]
            rerank_ltr = tf.reshape(rerank_ltr, [-1, 60])
            rerank_ltr = tf.gather_nd(rerank_ltr, gather_indices)

            hidden_states = self._mlp_layer("mlp_layer_1", item_embeddings, [64, 32])
            position_ids = tf.range(6, dtype=tf.int32)
            position_ids = tf.expand_dims(position_ids, 0)
            position_embeddings = tf.nn.embedding_lookup(self.position_embeddings, position_ids)
            position_embeddings = tf.tile(position_embeddings, [tf.shape(hidden_states)[0], 1, 1])
            hidden_states = hidden_states+position_embeddings #grund truth can

            generator_embeding = self._mlp_layer("mlp_layer_1", generator_embeding, [64, 32])
            generator_embeding = generator_embeding+position_embeddings #generate can

            dim = 32
            num_heads = 4
            hidden_dim = 128
            dropout_rate = 0.1
            sequence_length = 100
            num_layers = 3
            k = 6
            model = Evaluator(num_layers, dim, num_heads, hidden_dim, dropout_rate, k)
            hidden_states = model.forward(hidden_states, training=True) #ground truth -> evualator
            generator_embeding = model.forward(generator_embeding, training=True) #generator选出的embedding -> evaluator

            norm_states = hidden_states / tf.norm(hidden_states, axis=2, keepdims=True)
            cosine_scores_states = tf.matmul(norm_states, tf.transpose(norm_states, perm=[0, 2, 1]))
            cl_loss_states = self._contrastive_loss(cosine_scores_states)

            generator_logits = self._mlp_layer("mlp_layer_out", generator_embeding, [1], activation=tf.nn.relu) #evualator对generator candidates的预估分
            logits = self._mlp_layer("mlp_layer_out", hidden_states, [1], activation=tf.nn.relu) #evualator对ground truth candidates的预估分
            logits = tf.squeeze(logits, axis=-1)
            print("item_weight shape", item_weight.shape)
            print("logits shape", logits.shape)
            
            # gen loss
            indice_label = rank_indices
            rerank_weight = rerank_wtd_label
            # gen_model_weight = tf.batch_gather(rerank_weight,is_exposed)
            # gen_model_weight = tf.batch_gather(rerank_weight,indice_label) #?,6
            # gen_model_weight = gen_model_weight*5 + 1

            gt_label = tf.cast(indice_label, tf.int64) #?,6
            masked_indices = tf.cast(tf.expand_dims(gt_label, axis=2),tf.int64) #?,6,1

            # 根据indice取出模型预估值
            pos_output = tf.batch_gather(predict_ori, masked_indices) #(?,6,1)
            # pos_output = tf.batch_gather(predict, masked_indices) #(?,6,1)
            pos_output = tf.squeeze(pos_output, axis=-1) #(?,6)
            real_show_top6 = tf.cast(rerank_label, dtype=tf.float32)
            # 计算loss
            valid_pos_output = tf.log(pos_output+1e-9)*real_show_top6 #(?,6)
            valid_counts = tf.reduce_sum(real_show_top6, axis=-1)+1e-9 #避免除0
            # 对每个样本，只计算有效位置的平均loss
            # gen_loss = -tf.reduce_mean(tf.reduce_sum(valid_pos_output*gen_model_weight, axis=-1)/valid_counts) #(?,)
            gen_loss = -tf.reduce_mean(tf.reduce_sum(valid_pos_output, axis=-1)/valid_counts) #(?,)

            item_weight = tf.clip_by_value(item_weight, 0, 100)/10.0 
            # gen-eval loss
            # generator_loss = -tf.reduce_mean(generator_logits-0.7)
            generator_loss = -tf.reduce_mean(tf.math.log(generator_logits + 1e-9)) # -logP

            #eval loss
            loss = self.weighted_log_loss(click_label, logits, item_weight) #evualator loss

            # return logits, loss, item_weight, generator_loss, generator_logits, cl_loss_states, cl_loss, predict, item_embedding_gen, gen_loss, gt_label
            return logits, loss, item_weight, generator_loss, generator_logits, cl_loss_states, cl_loss, predict_ori, item_embedding_gen, gen_loss, gt_label
        