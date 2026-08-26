import tensorflow as tf
from feature_attr_extract import *
from modules_ import *
import sys

user_fea_names = [
  "user_id",
  "user_device_id",
  "user_age_segment",
  "user_gender",
  "user_city_id",
  "user_client_id",
  "user_level",
  "user_active_days",
  "user_emp_ctr",
  "user_emp_ltr",
  "user_emp_wtr",
  "user_emp_lvtr",
  "user_request_province_id",
  "user_request_city_id",
  "user_request_poi_type",
  "user_region_type",
]

explore_profile_fea_names = [
  "user_realtime_click_list",
  "user_realtime_like_list",
  "user_realtime_follow_list",
  "user_realtime_forward_list",
]


photo_fea_names = [
  "photo_id",
  "photo_author_id",
  "photo_author_gender",
  "photo_author_age_segment",
  "photo_province_id",
  "photo_city_id",
  "photo_mod",
  "photo_upload_type",
  "photo_music",
  "photo_hetu_tag_level1_list",
  "photo_hetu_tag_level2_list",
  "photo_hetu_tag_level5_list",
  "photo_duration_ms",

  "photo_emp_ctr",
  "photo_emp_ltr",
  "photo_emp_wtr",
  "photo_emp_lvtr",
  "photo_emp_svtr",
  "context_pctr",
  "context_pltr",
  "context_pwtr",
  "context_pftr",
  "context_plvtr",
  "context_psvtr",
  "context_pvtr",
  "context_pptr",
  "context_pcmtr",
  "context_pwtd",
  "context_cascade_pctr",
  "context_cascade_pltr",
  "context_cascade_pwtr",
  "context_cascade_plvtr",
  "context_cascade_psvtr"
]

def layer_norm(x, epsilon=1e-6):
    with tf.variable_scope("layer_norm", reuse=tf.AUTO_REUSE):
        gamma = tf.get_variable("gamma", [x.get_shape()[-1]], initializer=tf.ones_initializer())
        beta = tf.get_variable("beta", [x.get_shape()[-1]], initializer=tf.zeros_initializer())

        mean, variance = tf.nn.moments(x, axes=[-1], keep_dims=True)
        normalized = (x - mean) / tf.sqrt(variance + epsilon)
        output = gamma * normalized + beta
    return output


def multi_head_attention(queries, keys, values, num_heads, dropout_rate, training=False):
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
        with tf.variable_scope(f"multi_head_attention", reuse=tf.AUTO_REUSE):
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
            attn_output = self.mha(x, x, x, self.num_heads, self.dropout_rate, training=training)
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
        
    def forward(self, x, enc_output, training):
        
        with tf.variable_scope(f"{self.name}", reuse=tf.AUTO_REUSE):
            cross_attn_output = self.cross_attention(x, enc_output, enc_output, self.num_heads, self.dropout_rate, training=training)
            out1 = layer_norm(x + cross_attn_output)
            attn_output = self.self_attention(out1, out1, out1, self.num_heads, self.dropout_rate, training=training)
            out2 = layer_norm(x + attn_output)
        
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
    def __init__(self, parameters_dict, label_value_dict, dim=32, extra_param_dict= None):
        self._parameters_dict = parameters_dict
        self._label_value_dict = label_value_dict
        self.dim = dim
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
            list_dim = 60
            print("user_id embeding shape", input_dicts['user_id'].shape)
            user_embs = tf.concat([input_dicts[k] for k in input_dicts if k in user_fea_names], axis=-1)
            user_embs = tf.tile(tf.expand_dims(user_embs, axis=1),[1,list_dim,1])
            print("user_embs shape ", user_embs.shape)
            print("photo_id ",input_dicts['photo_id'].shape)
            photo_embs    = tf.concat([input_dicts[k] for k in input_dicts if k in photo_fea_names], axis=2)
            print("photo_embs shape ", photo_embs.shape)
            explore_embs  = tf.concat([input_dicts[k] for k in input_dicts if k in explore_profile_fea_names], axis=2)
            explore_embs  = tf.reduce_mean(explore_embs, axis=1)
            explore_embs = tf.tile(tf.expand_dims(explore_embs, axis=1),[1,list_dim,1])

            print("explore_embs shape ", photo_embs.shape)
            common_embs   = tf.concat([user_embs, photo_embs, explore_embs], axis=-1)
            print("common_embs shape", common_embs.shape)
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
        
        # Compute the unweighted log loss
        log_loss = - (y_true * tf.log(y_pred) + (1 - y_true) * tf.log(1 - y_pred))
        
        # Apply the class weights
        # weighted_log_loss = log_loss * weights
        weighted_log_loss = log_loss
        
        # Reduce mean across the batch
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
            model = StackedTransformerModel(num_layers, dim, num_heads, hidden_dim, dropout_rate, k)
            hidden_states, pos_embedding = model.forward(hidden_states, training=True)
            print("hidden_states shape ", hidden_states.shape)
            print("pos_embedding shape ", pos_embedding.shape)
            pos_embedding_trans = tf.transpose(pos_embedding,  perm=[0, 2, 1])
            predict = tf.matmul(hidden_states, pos_embedding_trans)
            # print("predict shape", predict.shape)

            tau = 0.05
            predict = self.gumbel_softmax(predict, tau, hard=True, dim=1)
            print("ryx predict shape", predict.shape)
            # predict = tf.nn.softmax(predict/tau, axis=1)
            norm_rep = pos_embedding / tf.norm(pos_embedding, axis=2, keepdims=True)
            cosine_scores_rep = tf.matmul(norm_rep, tf.transpose(norm_rep, perm=[0, 2, 1]))
            cl_loss_pad = self._contrastive_loss(cosine_scores_rep)

            norm_common = common_embs / tf.norm(common_embs, axis=2, keepdims=True)
            print("norm_common shape", norm_common.shape)
            cosine_scores_common = tf.matmul(norm_common, tf.transpose(norm_common, perm=[0, 2, 1]))
            print("cosine_scores common shape", cosine_scores_common.shape)
            # cl_loss_common = self._contrastive_loss(cosine_scores_common)


            norm_outputs = hidden_states / tf.norm(hidden_states, axis=2, keepdims=True)
            cosine_scores_outputs = tf.matmul(norm_outputs, tf.transpose(norm_outputs, perm=[0, 2, 1]))
            print("cosine_scores_outputs shape", cosine_scores_outputs.shape)
            cl_loss_outputs = self._contrastive_loss(cosine_scores_outputs, seqlen=60)
            cl_loss = cl_loss_pad+cl_loss_outputs
            
            print("predict shape", predict.shape)
        predict = tf.transpose(predict,  perm=[0, 2, 1])
        generator_embeding = tf.matmul(predict, common_embs)
        print("generator_embeding shape", generator_embeding.shape)

        with tf.variable_scope("evaluator", reuse=tf.AUTO_REUSE):
            input_dicts = self._parameters_dict
            common_embs = self._get_shared_features(input_dicts)
            
            batch_size = tf.shape(common_embs)[0]
            label_dicts = self._label_value_dict
            # rerank_label = label_dicts["context_info__real_show_index_list"]
            # rerank_label = tf.reshape(rerank_label, [-1, 60])
            # _, rank_indices = tf.nn.top_k(rerank_label, k=6, sorted=True, name='top_k')
            # print("rank_indices shape", rank_indices.shape)

            rerank_label = label_dicts['context_info__real_show_list']
            rerank_label = tf.reshape(rerank_label, [-1, 60])
            rerank_label = rerank_label[:,:6]
            indices_shape = tf.shape(rerank_label)
            rerank_label = tf.cast(rerank_label,dtype=tf.int32)
            col_indices = tf.tile(tf.expand_dims(tf.range(indices_shape[1]),0),[indices_shape[0],1])
            rank_indices = tf.cast(col_indices*rerank_label,dtype=tf.int32)


            rank_indices = tf.reverse(rank_indices, axis=[1])
            batch_indices = tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), [1, 6]) 
            gather_indices = tf.stack([batch_indices, rank_indices], axis=-1)
            item_embeddings = tf.gather_nd(common_embs, gather_indices)
            print("item_embeddings shape", item_embeddings.shape)

            rerank_weight = label_dicts["fountain_fulllink_rerank_realshow_label_weight_list"]
            rerank_weight = tf.reshape(rerank_weight, [-1, 60])
            item_weight = tf.gather_nd(rerank_weight, gather_indices)
            print("item_weight shape", item_weight.shape)


            click_label = label_dicts['fountain_click_label_list']
            click_label = tf.reshape(click_label, [-1, 60])
            click_label = tf.gather_nd(click_label, gather_indices)
            print("click_label shape", click_label.shape)

            hidden_states = self._mlp_layer("mlp_layer_1", item_embeddings, [64, 32])
            position_ids = tf.range(6, dtype=tf.int32)
            position_ids = tf.expand_dims(position_ids, 0)
            position_embeddings = tf.nn.embedding_lookup(self.position_embeddings, position_ids)
            position_embeddings = tf.tile(position_embeddings, [tf.shape(hidden_states)[0], 1, 1])
            print("position_embeddings shape", position_embeddings.shape)
            hidden_states = hidden_states+position_embeddings

            generator_embeding = self._mlp_layer("mlp_layer_1", generator_embeding, [64, 32])
            generator_embeding = generator_embeding+position_embeddings

            dim = 32
            num_heads = 4
            hidden_dim = 128
            dropout_rate = 0.1
            sequence_length = 100
            num_layers = 3
            k = 6
            model = Evaluator(num_layers, dim, num_heads, hidden_dim, dropout_rate, k)
            hidden_states = model.forward(hidden_states, training=True)
            generator_embeding = model.forward(generator_embeding, training=True)

            norm_states = hidden_states / tf.norm(hidden_states, axis=2, keepdims=True)
            cosine_scores_states = tf.matmul(norm_states, tf.transpose(norm_states, perm=[0, 2, 1]))
            cl_loss_states = self._contrastive_loss(cosine_scores_states)

            # generator_logits = self._mlp_layer("mlp_layer_out", generator_embeding, [1], activation=tf.nn.relu)
            # logits = self._mlp_layer("mlp_layer_out", hidden_states, [1], activation=tf.nn.relu)
            # logits = tf.squeeze(logits, axis=-1)
            # print("item_weight shape", item_weight.shape)
            # print("logits shape", logits.shape)
            # item_weight = tf.clip_by_value(item_weight, 0, 300)/10.0
            # generator_loss = -tf.reduce_mean(generator_logits-2)
            # loss = tf.reduce_mean(tf.reduce_sum(tf.square(item_weight-logits),axis=-1)/tf.cast(tf.reduce_sum(rerank_label, axis=-1),tf.float32))

            generator_logits = self._mlp_layer("mlp_layer_out", generator_embeding, [1], activation=tf.nn.sigmoid)
            logits = self._mlp_layer("mlp_layer_out", hidden_states, [1], activation=tf.nn.sigmoid)
            logits = tf.squeeze(logits, axis=-1)
            print("item_weight shape", item_weight.shape)
            print("logits shape", logits.shape)
            print("rerank label shape", rerank_label.shape)
            # rerank_label = tf.cast(rerank_label, tf.float32)
            click_label = tf.cast(click_label, tf.float32)

            item_weight = tf.clip_by_value(item_weight, 0, 100)/10.0
            # generator_loss = -tf.reduce_mean(generator_logits-2)
            generator_loss = -tf.reduce_mean(generator_logits-0.5)
            loss = self.weighted_log_loss(click_label, logits, item_weight)

            print("loss shape", loss.shape)

            return logits, loss, item_weight, generator_loss, generator_logits, cl_loss_states, cl_loss
        