import tensorflow as tf
import numpy as np
from modules_ import *

def layer_norm(x, scope, eps=1e-6):
    with tf.variable_scope(f"{scope}/layer_norm", reuse=tf.AUTO_REUSE):
        gamma = tf.get_variable("gamma", [x.get_shape()[-1]], initializer=tf.ones_initializer())
        beta = tf.get_variable("beta", [x.get_shape()[-1]], initializer=tf.zeros_initializer())

        mean, variance = tf.nn.moments(x, axes=[-1], keep_dims=True)
        normalized = (x - mean) / tf.sqrt(variance + eps)
        output = gamma * normalized + beta
    return output
    
def multi_head_attention_mask(queries, keys, values,
                              num_heads, atten_mask, dropout_rate,
                              training=False):
    """
    atten_mask: [B, 1, T_q, T_k]  允许位置 = 1，屏蔽 = 0
    """

    def split_heads(x, h):
        b = tf.shape(x)[0]; d = x.get_shape().as_list()[-1] // h
        return tf.transpose(tf.reshape(x, [b, -1, h, d]), [0, 2, 1, 3])  # [B,h,T,d]

    depth = queries.get_shape().as_list()[-1]
    Q = tf.layers.dense(queries, depth, use_bias=False, name="w_q")
    K = tf.layers.dense(keys,    depth, use_bias=False, name="w_k")
    V = tf.layers.dense(values,  depth, use_bias=False, name="w_v")

    Qh, Kh, Vh = map(lambda x: split_heads(x, num_heads), (Q, K, V))

    logits = tf.matmul(Qh, Kh, transpose_b=True)                  # [B,h,Tq,Tk]
    logits = logits / tf.sqrt(tf.cast(tf.shape(Kh)[-1], tf.float32))

    atten_mask = tf.tile(atten_mask, [1, num_heads, 1, 1])         # 4 维一致
    paddings = tf.cast(tf.ones_like(logits) * (-2**32 + 1), logits.dtype)
    logits  = tf.where(tf.equal(atten_mask, 0), paddings, logits)

    weights = tf.nn.softmax(logits)
    weights = tf.layers.dropout(weights, rate=dropout_rate,
                                training=tf.convert_to_tensor(training))

    ctx = tf.matmul(weights, Vh)                                  # [B,h,Tq,d]
    ctx = tf.transpose(ctx, [0, 2, 1, 3])
    ctx = tf.reshape(ctx, [tf.shape(queries)[0], -1, depth])
    return tf.layers.dense(ctx, depth, name="w_o")
    
    
def feed_forward_network(name, dim, hidden_dim, dropout_rate, training=False):
    def ffn(x, training=training):
        # training = tf.constant(training, dtype=tf.bool)
        with tf.variable_scope(f"{name}/ffn", reuse=tf.AUTO_REUSE):
            x = tf.layers.dense(x, hidden_dim, activation=tf.nn.relu)
            # x = tf.nn.dropout(x, rate=dropout_rate)
            # x = tf.cond(training, lambda: tf.nn.dropout(x, rate=dropout_rate), lambda: x)
            x = tf.layers.dropout(x, rate=dropout_rate,
                                training=tf.convert_to_tensor(training))
            x = tf.layers.dense(x, dim)
            return x
    return ffn


class DecoderOnlyLayer:
    def __init__(self, name, dim, num_heads, hidden_dim, dropout_rate, training=False):
        super(DecoderOnlyLayer, self).__init__()
        self.name = name
        self.dim = dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        # 不再内建
        self.self_attention = multi_head_attention_mask
        self.ffn            = feed_forward_network(self.name, dim, hidden_dim, dropout_rate)   # ✅ 补上

    def forward(self, x, atten_mask, training=False):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            attn_out = self.self_attention(x, x, x,
                                           self.num_heads,
                                           atten_mask,          # <<< 透传
                                           self.dropout_rate,
                                           training)
            out1 = layer_norm(x + attn_out, self.name)
            ffn_out = self.ffn(out1, training=training)
            out2 = layer_norm(out1 + ffn_out, self.name)
        return out2


class DecoderOnlyModel():
    def __init__(self, num_layers, dim, num_heads, hidden_dim, dropout_rate, training=False):
        super(DecoderOnlyModel, self).__init__()
        self.num_layers = num_layers
        self.dim = dim
        self.layers = [DecoderOnlyLayer(f"decoder_only_layer_{i}", dim, num_heads, hidden_dim, dropout_rate, training=training) for i in range(num_layers)]
        
    def forward(self, dec_embedding, atten_mask, training):
        for layer in self.layers:
            dec_embedding = layer.forward(dec_embedding, atten_mask, training)
        return dec_embedding
