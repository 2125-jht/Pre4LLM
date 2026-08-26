import tensorflow as tf
import numpy as np
from modules_ import *

# def multi_head_attention(queries, keys, values, is_training=False, causality=False, mask=None, num_heads=8, dropout_rate=0.1, dim=16):
#     with tf.variable_scope("multi_head_attention", reuse=tf.AUTO_REUSE):
#     # Set the fall back option for num_units
#         input_shape = tf.shape(queries)
#         is_4d = len(queries.get_shape().as_list()) == 4
#         batch_size, beam_size = None, None
#         if is_4d:
#             # [batch_size, beam_size, seq_len, dim]
#             batch_size = input_shape[0]
#             beam_size = input_shape[1]
#             query_len = tf.shape(queries)[2]
#             kv_len = tf.shape(keys)[2]
#             # dim = input_shape[3]

#             queries = tf.reshape(queries, [batch_size*beam_size, query_len, dim])
#             keys = tf.reshape(keys, [batch_size*beam_size, kv_len, dim])
#             values = tf.reshape(values, [batch_size*beam_size, kv_len, dim])
#         else:
#             batch_size = input_shape[0]
#             # dim = input_shape[2]
#         # Linear projections
#         Q = tf.layers.dense(queries, dim, activation=None, name='query_dense') # (N, T_q, C)
#         K = tf.layers.dense(keys, dim, activation=None, name='key_dense') # (N, T_k, C)
#         V = tf.layers.dense(values, dim, activation=None, name='value_dense') # (N, T_k, C)
        
#         # Split and concat
#         Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, C/h) 
#         K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 
#         V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 

#         # Multiplication
#         outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)
#         outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

#         if causality:
#             diag_vals = tf.ones_like(outputs[0, :, :]) # (T_q, T_k)
#             tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense() # (T_q, T_k)
#             cau_masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]) # (h*N, T_q, T_k)
#             paddings = tf.ones_like(cau_masks)*(-2**32+1)
#             outputs = tf.where(tf.equal(cau_masks, 0), paddings, outputs) # (h*N, T_q, T_k)
        
#         if mask is not None:
#             # origin mask # [bs, seq_len], transform mask #[bs, seq_len, seq_len]
#             mask = tf.tile(tf.expand_dims(mask, axis=1), [num_heads, tf.shape(mask)[1], 1])
#             paddings = tf.ones_like(mask)*(-2**32+1)
#             outputs = tf.where(tf.equal(mask, 0), paddings, outputs)

#         # Activation
#         outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)
#         outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
#         # Weighted sum
#         outputs = tf.matmul(outputs, V_) # ( h*N, T_q, C/h)
#         # Restore shape
#         outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2) # (N, T_q, C)

#         if is_4d:
#             outputs = tf.reshape(outputs, [batch_size, beam_size, query_len, dim])
        
#         outputs = tf.layers.dense(outputs, dim)

#     return outputs

# def feed_forward(x, dim=16, is_training=False, dropout_rate=0.1):
#     training = tf.constant(is_training, dtype=tf.bool)
#     with tf.variable_scope(f"feed_forward_network", reuse=tf.AUTO_REUSE):
#         x = tf.layers.dense(x, 2*dim, activation=tf.nn.relu)
#         x = tf.layers.dense(x, dim)
#         # x = tf.nn.dropout(x, rate=dropout_rate)
#         #x = tf.cond(training, lambda: tf.nn.dropout(x, rate=dropout_rate), lambda: x)
#         return x
    
# def layer_norm(x, epsilon=1e-6):
#     with tf.variable_scope("layer_norm", reuse=tf.AUTO_REUSE):
#         gamma = tf.get_variable("gamma", [x.get_shape()[-1]], initializer=tf.ones_initializer())
#         beta = tf.get_variable("beta", [x.get_shape()[-1]], initializer=tf.zeros_initializer())

#         mean, variance = tf.nn.moments(x, axes=[-1], keep_dims=True)
#         normalized = (x - mean) / tf.sqrt(variance + epsilon)
#         output = gamma * normalized + beta
#     return output

# def transformer_encoder_layer(seq_input_embeddings, num_layer, dropout_rate=0.1, mask=None, is_training=True, dim=16):
#     batch_size = tf.shape(seq_input_embeddings)[0]
#     for i in range(num_layer):
#         with tf.variable_scope("num_blocks_%d" % i, reuse=tf.AUTO_REUSE):
#             with tf.variable_scope("atten", reuse=tf.AUTO_REUSE):
#                 new_seq_input_embeddings = multi_head_attention(seq_input_embeddings, seq_input_embeddings, seq_input_embeddings,
#                 is_training=is_training, causality=False, mask=mask, dim=dim, dropout_rate=dropout_rate)
#                 seq_input_embeddings = layer_norm(new_seq_input_embeddings) + seq_input_embeddings
#                 print_tensor("encoder/atten_sim_%d" % i, calc_sim_cos(tf.reshape(seq_input_embeddings, [batch_size, -1])))
#             with tf.variable_scope("ffn", reuse=tf.AUTO_REUSE):
#                 new_seq_input_embeddings = feed_forward(new_seq_input_embeddings, dim=dim, dropout_rate=dropout_rate,is_training=is_training)
#                 seq_input_embeddings = layer_norm(new_seq_input_embeddings) + seq_input_embeddings
#                 print_tensor('encoder/ffn_sim_%d' % i, calc_sim_cos(tf.reshape(seq_input_embeddings, [batch_size, -1])))
#     return seq_input_embeddings

# def transformer_decoder_layer(encoder_output, decoder_input, num_layer, dropout_rate=0.1, mask=None, is_training=True, dim=16):
#     for i in range(num_layer):
#         with tf.variable_scope("num_blocks_%d" % i, reuse=tf.AUTO_REUSE):
#             with tf.variable_scope("self_atten", reuse=tf.AUTO_REUSE):
#                 new_decoder_input = multi_head_attention(decoder_input, decoder_input, decoder_input,
#                 is_training=is_training, causality=True, mask=mask, dim=dim, dropout_rate=dropout_rate)
#                 decoder_input = layer_norm(new_decoder_input) + decoder_input
#             with tf.variable_scope("cross_atten", reuse=tf.AUTO_REUSE):
#                 new_decoder_input = multi_head_attention(decoder_input, encoder_output, encoder_output,
#                 is_training=is_training, causality=False, mask=mask, dim=dim, dropout_rate=dropout_rate)
#                 decoder_input = layer_norm(new_decoder_input) + decoder_input
#             with tf.variable_scope("ffn", reuse=tf.AUTO_REUSE):
#                 new_decoder_input = feed_forward(decoder_input, dim=dim, dropout_rate=dropout_rate, is_training=is_training)
#                 decoder_input = layer_norm(new_decoder_input) + decoder_input
#     return decoder_input

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
        with tf.variable_scope("multi_head_attention", reuse=tf.AUTO_REUSE):
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

class EncoderLayer:
    def __init__(self, name, dim, num_heads, hidden_dim, dropout_rate, training=False):
        super(EncoderLayer, self).__init__()
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

# transformer_encoder_layer(encoder_input, 4, dim=self._dim)

class StackedTransformerModel():
    def __init__(self, num_layers, dim, num_heads, hidden_dim, dropout_rate, k=6, training=False):
        super(StackedTransformerModel, self).__init__()
        self.num_layers = num_layers
        # self.k = 6
        self.dim = dim
        # self.position = tf.get_variable('s', shape=[self.k, self.dim], initializer=tf.random_normal_initializer())
        self.layers = [EncoderLayer(f"encoder_layer_{i}", dim, num_heads, hidden_dim, dropout_rate, training=training) for i in range(num_layers)]
        
    def forward(self, hidden_states, training):
        # pos_embedding = tf.reshape(self.position, [1, self.k, self.dim])
        # pos_embedding = tf.tile(pos_embedding, [tf.shape(hidden_states)[0], 1, 1])
        for i in range(self.num_layers):
            hidden_states = self.layers[i].forward(hidden_states, training=training)
            # pos_embedding = self.position_layers[i].forward(pos_embedding, hidden_states, training=training)
        # return hidden_states, pos_embedding
        return hidden_states
