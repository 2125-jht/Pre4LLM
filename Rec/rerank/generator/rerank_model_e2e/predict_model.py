import tensorflow as tf
from feature_predict_attr_extract import *
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


class FountainDeepLtrPredictModel:
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
            print("user_embs shape ", user_embs.shape)
            photo_embs = tf.concat([input_dicts[k] for k in input_dicts if k in photo_fea_names], axis=-1)
            print("photo_embs shape ", photo_embs.shape)
            explore_embs  = tf.concat([input_dicts[k] for k in input_dicts if k in explore_profile_fea_names], axis=-1)
            explore_embs  = tf.reduce_mean(explore_embs, axis=1)
            print("explore_embs shape ", explore_embs.shape)

            emb_dim = 532
            
            common_embs = tf.concat([user_embs, photo_embs, explore_embs], axis=-1)
            common_embs = tf.reshape(common_embs, [1, -1, emb_dim])
            
            print("common_embs shape", common_embs.shape)
            return common_embs

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
            item_embedding, pos_embedding = model.forward(hidden_states, training=True)
            print("item_embedding shape ", item_embedding.shape)
            print("pos_embedding shape ", pos_embedding.shape)
            pos_embedding_trans = tf.transpose(pos_embedding,  perm=[0, 2, 1])
            predict = tf.matmul(item_embedding, pos_embedding_trans)
            print("predict shape", predict.shape)

            tau = 0.01
            predict = tf.nn.softmax(predict/tau, axis=1)
            print("predict shape", predict.shape)
        predict = tf.transpose(predict,  perm=[0, 2, 1])
        generator_embeding = tf.matmul(predict, common_embs)
        print("generator_embeding shape", generator_embeding.shape)

        with tf.variable_scope("evaluator", reuse=tf.AUTO_REUSE):
            input_dicts = self._parameters_dict
            common_embs = self._get_shared_features(input_dicts)
            
            batch_size = tf.shape(common_embs)[0]
            label_dicts = self._label_value_dict
            rerank_label = label_dicts["fountain_fulllink_rerank_index_list"]
            rerank_label = tf.reshape(rerank_label, [-1, 60])
            _, rank_indices = tf.nn.top_k(rerank_label, k=6, sorted=True, name='top_k')
            print("rank_indices shape", rank_indices.shape)

            rank_indices = tf.reverse(rank_indices, axis=[1])
            batch_indices = tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), [1, 6]) 
            gather_indices = tf.stack([batch_indices, rank_indices], axis=-1)
            item_embeddings = tf.gather_nd(common_embs, gather_indices)
            print("item_embeddings shape", item_embeddings.shape)

            rerank_weight = label_dicts["fountain_fulllink_rerank_index_weight_list"]
            rerank_weight = tf.reshape(rerank_weight, [-1, 60])
            item_weight = tf.gather_nd(rerank_weight, gather_indices)
            print("item_weight shape", item_weight.shape)

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

            generator_logits = self._mlp_layer("mlp_layer_out", generator_embeding, [1], activation=tf.nn.relu)
            logits = self._mlp_layer("mlp_layer_out", hidden_states, [1], activation=tf.nn.relu)
            # logits = tf.squeeze(logits, axis=-1)
            # print("item_weight shape", item_weight.shape)
            # print("logits shape", logits.shape)
            # item_weight = tf.clip_by_value(item_weight, 0, 300)
            # generator_loss = -tf.reduce_sum(generator_logits)
            # loss = tf.reduce_sum(tf.square(item_weight-logits))
            # print("loss shape", loss.shape)

            return predict, item_embedding
        
        
