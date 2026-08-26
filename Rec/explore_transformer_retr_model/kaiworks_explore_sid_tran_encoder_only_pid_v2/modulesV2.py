import tensorflow as tf
import numpy as np
from modules_ import *

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

        training = tf.constant(training, dtype=tf.bool)
    
        def scaled_dot_product_attention(Q, K, V):
            matmul_qk = tf.matmul(Q, K, transpose_b=True)
            dk = tf.cast(tf.shape(K)[-1], tf.float32)
            scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
            attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
            
            attention_weights = tf.layers.dropout(attention_weights, rate=dropout_rate,
                                    training=training, name="attn_dropout")
            
            output = tf.matmul(attention_weights, V)
            return output, attention_weights
        
        with tf.variable_scope("multi_head_attention", reuse=tf.AUTO_REUSE):
            depth = queries.get_shape().as_list()[-1]
            Q = tf.layers.dense(queries, depth, use_bias=False, name="w_q")
            K = tf.layers.dense(keys,    depth, use_bias=False, name="w_k")
            V = tf.layers.dense(values,  depth, use_bias=False, name="w_v")

            Q = split_heads(Q, num_heads)
            K = split_heads(K, num_heads)
            V = split_heads(V, num_heads)

            scaled_attention, attention_weights = scaled_dot_product_attention(Q, K, V)
            scaled_attention = tf.transpose(scaled_attention, [0, 2, 1, 3])

            concat_attention = tf.reshape(scaled_attention, [tf.shape(queries)[0], -1, depth])
            output = tf.layers.dense(concat_attention, depth, name="w_o")

        return output
    
def multi_head_attention_causality(queries, keys, values, num_heads, dropout_rate, training=False):
        
        def apply_masks(atten_scores):
            """
            atten_scores: [B, H, T_q, T_k]
            """
            B = tf.shape(atten_scores)[0]
            T_q = tf.shape(atten_scores)[2]
            T_k = tf.shape(atten_scores)[3]

            # 1) 因果下三角掩码，先做 [T_q, T_k]
            causal_mask = tf.linalg.band_part(tf.ones([T_q, T_k]), -1, 0)

            # 2) 扩展到 [B, H, T_q, T_k]（或用 [1,1,T_q,T_k] 让广播去做）
            causal_mask = tf.reshape(causal_mask, [1, 1, T_q, T_k])          # 4 维
            causal_mask = tf.tile(causal_mask, [B, num_heads, 1, 1])         # 4 维一致

            paddings = tf.ones_like(atten_scores) * (-2**32 + 1)
            atten_scores  = tf.where(tf.equal(causal_mask, 0), paddings, atten_scores)

            return atten_scores
        
        def split_heads(x, num_heads):
            batch_size = tf.shape(x)[0]
            depth = x.get_shape().as_list()[-1] // num_heads
            reshaped = tf.reshape(x, [batch_size, -1, num_heads, depth])
            return tf.transpose(reshaped, [0, 2, 1, 3])
        
        training = tf.constant(training, dtype=tf.bool)
        
        def scaled_dot_product_attention(Q, K, V):
            matmul_qk = tf.matmul(Q, K, transpose_b=True)
            dk = tf.cast(tf.shape(K)[-1], tf.float32)
            scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
            scaled_attention_logits = apply_masks(scaled_attention_logits)
            attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
            
            attention_weights = tf.layers.dropout(attention_weights, rate=dropout_rate,
                                    training=training, name="attn_dropout")
            
            output = tf.matmul(attention_weights, V)
            return output, attention_weights
        
        with tf.variable_scope("multi_head_attention_causality", reuse=tf.AUTO_REUSE):
            depth = queries.get_shape().as_list()[-1]
            Q = tf.layers.dense(queries, depth, use_bias=False, name="w_q")
            K = tf.layers.dense(keys,    depth, use_bias=False, name="w_k")
            V = tf.layers.dense(values,  depth, use_bias=False, name="w_v")

            Q = split_heads(Q, num_heads)
            K = split_heads(K, num_heads)
            V = split_heads(V, num_heads)

            scaled_attention, attention_weights = scaled_dot_product_attention(Q, K, V)
            scaled_attention = tf.transpose(scaled_attention, [0, 2, 1, 3])

            concat_attention = tf.reshape(scaled_attention, [tf.shape(queries)[0], -1, depth])
            output = tf.layers.dense(concat_attention, depth, name="w_o")

        return output
    
    
def feed_forward_network(dim, hidden_dim, dropout_rate, training=False):
    def ffn(x, training=training):
        training = tf.constant(training, dtype=tf.bool)
        with tf.variable_scope(f"feed_forward_network", reuse=tf.AUTO_REUSE):
            x = tf.layers.dense(x, hidden_dim, activation=tf.nn.relu)
            x = tf.layers.dropout(x, rate=dropout_rate,
                                    training=training)
            x = tf.layers.dense(x, dim)
            # x = tf.nn.dropout(x, rate=dropout_rate)
            # x = tf.cond(training, lambda: tf.nn.dropout(x, rate=dropout_rate), lambda: x)
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

class DecoderLayer:
    def __init__(self, name, dim, num_heads, hidden_dim, dropout_rate, training=False):
        super(DecoderLayer, self).__init__()
        self.name = name
        self.dim = dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.self_attention = multi_head_attention_causality
        self.cross_attention = multi_head_attention
        self.ffn = feed_forward_network(dim, hidden_dim, dropout_rate)
        
    def forward(self, x, enc_output, training=False):
        with tf.variable_scope(f"{self.name}", reuse=tf.AUTO_REUSE):
            attn_output = self.self_attention(x, x, x, self.num_heads, self.dropout_rate, training=training)
            out1 = layer_norm(x + attn_output)
            cross_attn_output = self.cross_attention(out1, enc_output, enc_output, self.num_heads, self.dropout_rate, training=training)
            out2 = layer_norm(out1 + cross_attn_output)
            ffn_output = self.ffn(out2, training=training)
            out3 = layer_norm(out2 + ffn_output)
        
        return out3    


class EncoderModel():
    def __init__(self, num_layers, dim, num_heads, hidden_dim, dropout_rate, k=6, training=False):
        super(EncoderModel, self).__init__()
        self.num_layers = num_layers
        self.dim = dim
        self.layers = [EncoderLayer(f"encoder_layer_{i}", dim, num_heads, hidden_dim, dropout_rate, training=training) for i in range(num_layers)]
        
    def forward(self, encoder_embedding, training):
        for i in range(self.num_layers):
            encoder_embedding = self.layers[i].forward(encoder_embedding, training=training)
        return encoder_embedding

class DecoderModel():
    def __init__(self, num_layers, dim, num_heads, hidden_dim, dropout_rate, k=6, training=False):
        super(DecoderModel, self).__init__()
        self.num_layers = num_layers
        self.dim = dim
        self.layers = [DecoderLayer(f"decoder_layer_{i}", dim, num_heads, hidden_dim, dropout_rate, training=training) for i in range(num_layers)]
        
    def forward(self, decoder_embedding, hidden_states, training):
        for i in range(self.num_layers):
            decoder_embedding = self.layers[i].forward(decoder_embedding, hidden_states, training=training)
        return decoder_embedding
