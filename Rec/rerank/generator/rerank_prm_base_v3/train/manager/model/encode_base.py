# -*- coding: UTF8 -*-
from __future__ import absolute_import, division, print_function
import kuiba_utils

import tensorflow as tf

import numpy as np

def debug_log(name, tensor):
    print("debug_log:{},{}".format(name, str(tensor)))


class ScaledDotProductAttention():
    def __init__(self, d_model, attn_dropout=0.0):
        self.d_model = d_model
        self.attn_dropout = attn_dropout

    def shape_list(self, x):
        ps = x.get_shape().as_list()
        ts = tf.shape(x)
        return [ts[i] if ps[i] is None else ps[i] for i in range(len(ps))]

    def mask_attn_weights(self, w):
        n = self.shape_list(w)[-1]
        b = tf.matrix_band_part(tf.ones([n, n]), -1, 0)
        b = tf.reshape(b, [1, 1, n, n])
        w = w*b + -1e9*(1-b)
        return w

    def __call__(self, q, k, v, mask):
        """
        :param q: (batch_size, num_heads, seq_len_v, depth)
        :param k: (batch_size, num_heads, seq_len_v, depth)
        :param v: (batch_size, num_heads, seq_len_v, depth)
        :param mask: (batch_size, num_heads, seq_len_v, depth)
        :return: output:(batch_size, num_heads, seq_len_v, depth)
                 attn:(batch_size, num_heads, seq_len_v, seq_len_v)
        """
        with tf.variable_scope("ScaledDotProductAttention_call", reuse=kuiba_utils.reuse_variables()):
            debug_log("ScaledDotProductAttention.__call__.q", q)
            temper = tf.cast(tf.shape(k)[-1], tf.float32)
            debug_log("ScaledDotProductAttention.__call__.temper", temper)
            # attn = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2, 2]) / self.temper)([q, k])
            attn = tf.matmul(q, k, transpose_b=True)
            debug_log("ScaledDotProductAttention.__call__.matmul_qk.attn", attn)
            # 缩放 matmul_qk
            attn = attn / tf.math.sqrt(temper)
            debug_log("ScaledDotProductAttention.__call__.div.attn", attn)

            if mask:
              #attn += (mask * -1e9)
              attn = self.mask_attn_weights(attn)
            attn = tf.nn.softmax(attn, axis=-1)  # [n_head , batch_size, len_q, len_k]
            debug_log("ScaledDotProductAttention.__call__.softmax.attn", attn)

            # attn = tf.nn.dropout(attn,keep_prob=self.attn_dropout)
            output = tf.matmul(attn, v)  # (batch_size, num_heads, seq_len_v, depth)
            debug_log("ScaledDotProductAttention.__call__.output", output)

            return output, attn


class MultiHeadAttention():
    # mode 0 - big martixes, faster; mode 1 - more clear implementation
    def __init__(self, n_head, d_model, use_norm=True, layer_index=0):
        """
        :param n_head: head size
        :param d_model:
        :param dropout:
        :param mode:
        :param use_norm:
        """
        self.n_head = n_head
        self.d_model = d_model
        self.use_norm = use_norm
        self.layer_index = layer_index

    def __call__(self, q, k, v, mask=False):
        """
        :param q: [batch_size, len_q, d_model]
        :param k: [batch_size, len_q, d_model]
        :param v: [batch_size, len_q, d_model]
        :param mask:
        :return:
        """
        with tf.variable_scope("MultiHeadAttention_call", reuse=kuiba_utils.reuse_variables()):
            self.attention = ScaledDotProductAttention(self.d_model)
            batch_size = tf.shape(q)[0]

            qs = tf.layers.dense(inputs=q, units=self.d_model,
                                 use_bias=False)  # [batch_size, len_q, d_model]
            ks = tf.layers.dense(inputs=k, units=self.d_model,
                                 use_bias=False)  # [batch_size, len_q, d_model]
            vs = tf.layers.dense(inputs=v, units=self.d_model,
                                 use_bias=False)  # [batch_size, len_q, d_model]

            debug_log("MultiHeadAttention.__call__.qs", qs)
            debug_log("MultiHeadAttention.__call__.ks", ks)
            debug_log("MultiHeadAttention.__call__.vs", vs)

            qs = self.split_heads(qs,batch_size)  # (batch_size, num_heads, seq_len_q, depth)
            ks = self.split_heads(ks,batch_size)  # (batch_size, num_heads, seq_len_k, depth)
            vs = self.split_heads(vs,batch_size)  # (batch_size, num_heads, seq_len_v, depth)

            debug_log("MultiHeadAttention.__call__.split_heads.qs", qs)
            debug_log("MultiHeadAttention.__call__.split_heads.ks", ks)
            debug_log("MultiHeadAttention.__call__.split_heads.vs", vs)


            head, attn = self.attention(qs, ks, vs, mask=mask)  # head (batch_size, num_heads, seq_len_q, d_v)

            debug_log("MultiHeadAttention.__call__.head", head)

            head = tf.transpose(head, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, d_v)

            debug_log("MultiHeadAttention.__call__.transpose.head", head)
            # concat
            concat_attention = tf.reshape(head, (
                batch_size, -1, self.d_model))  # (batch_size, seq_len_q,  num_heads * d_v)
            debug_log("MultiHeadAttention.__call__.concat_attention", concat_attention)
            output = tf.layers.dense(concat_attention, self.d_model)  # (batch_size, seq_len_q, d_model)
            debug_log("MultiHeadAttention.__call__.output", output)
            return output, attn

    def split_heads(self, x, batch_size):
        """分拆最后一个维度到 (num_heads, depth).
        转置结果使得形状为 (batch_size, num_heads, seq_len, depth)
        """
        assert self.d_model%self.n_head==0, "d_model must be % self.n_head==0"
        x = tf.reshape(x, (batch_size, -1, self.n_head, self.d_model//self.n_head))
        return tf.transpose(x, perm=[0, 2, 1, 3])


class PositionwiseFeedForward():
    def __init__(self, d_model, d_inner_hid, dropout=0.1):
        """
        :param d_model: 输入的dim 和输出的dim
        :param d_inner_hid: 中间层
        :param dropout:
        :return output (batch_size, seq_len, d_model)
        """
        self.d_model = d_model
        self.d_inner_hid = d_inner_hid
        self.dropout = dropout

    def __call__(self, x):
        with tf.variable_scope("PositionwiseFeedForward_call", reuse=kuiba_utils.reuse_variables()):
            debug_log("PositionwiseFeedForward.__call__.x", x)
            output1 = tf.layers.dense(x, self.d_inner_hid, activation='relu')  # (batch_size, seq_len, dff)
            debug_log("PositionwiseFeedForward.__call__.output1", output1)
            output = tf.layers.dense(output1, self.d_model)  # (batch_size, seq_len, d_model)
            debug_log("PositionwiseFeedForward.__call__.output", output)
            #self.layer_norm = tf.keras.layers.LayerNormalization()

            output = tf.layers.dropout(output, self.dropout)  # (batch_size, seq_len, d_model)
            output = output + x  # (batch_size, seq_len, d_model)
            #output = self.layer_norm(output)
            debug_log("PositionwiseFeedForward.__call__.output", output)
            return output


class EncoderLayer():
    def __init__(self, d_model, d_inner_hid, n_head, dropout, index_layers):
        """
        :param d_model: ffn 输出 (batch_size, seq_len, d_model)
        :param d_inner_hid: ffn  中间层 size
        :param n_head: num head
        :param dropout: ffn 层drop out
        """
        self.d_inner_hid = d_inner_hid
        self.n_head = n_head
        self.dropout = dropout
        self.index_layers = index_layers
        self.d_model = d_model

    def __call__(self, enc_input, mask=False):
        """
        :param enc_input: 输入序列
        :param mask: None
        :return:
        """
        with tf.variable_scope("EncoderLayer_{}_call".format(self.index_layers), reuse=kuiba_utils.reuse_variables()):
            self.self_att_layer = MultiHeadAttention(n_head=self.n_head,
                                                     d_model=self.d_model,
                                                     )

            self.pos_ffn_layer = PositionwiseFeedForward(d_model=self.d_model,
                                                         d_inner_hid=self.d_inner_hid,
                                                         dropout=self.dropout)

            multi_output, slf_attn = self.self_att_layer(k=enc_input,
                                                         q=enc_input,
                                                         v=enc_input,
                                                         mask=mask)
            debug_log("EncoderLayer.__call__.multi_output", multi_output)

            ##  残差连接，然后进行层归一化。残差连接有助于避免深度网络中的梯度消失问题
            # multi_output = tf.nn.dropout(multi_output,keep_prob=self.dropout)
            multi_output = tf.contrib.layers.layer_norm(inputs=enc_input + multi_output, center=True, scale=True)

            ffn_output = self.pos_ffn_layer(multi_output)  # (batch_size, seq_len, d_model)
            debug_log("EncoderLayer.__call__.ffn_output", ffn_output)

            ## 残差连接，然后进行层归一化。残差连接有助于避免深度网络中的梯度消失问题
            # ffn_output = tf.nn.dropout(ffn_output, keep_prob=self.dropout)
            ffn_output = tf.contrib.layers.layer_norm(inputs=ffn_output + multi_output, center=True, scale=True)

        return ffn_output, slf_attn


class Encoder():
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, layers=2, dropout=0.9):
        """
        :param d_model: ffn 输出 (batch_size, seq_len, d_model)
        :param d_inner_hid: ffn 中间层 size
        :param n_head: num head
        :param d_k: key embeding size,废弃
        :param d_v: value embeding size，废弃
        :param layers: num_layers
        :param dropout: ffn 层drop out
        """
        self.d_model = d_model
        self.d_inner_hid = d_inner_hid
        self.n_head = n_head
        self.layers = layers
        self.dropout = dropout

    def __call__(self, x, return_att=False, mask=False):
        with tf.variable_scope("Encoder", reuse=kuiba_utils.reuse_variables()):
            layers = [EncoderLayer(d_model=self.d_model,
                                   d_inner_hid=self.d_inner_hid,
                                   n_head=self.n_head,
                                   dropout=self.dropout,
                                   index_layers=index_layers) for index_layers in range(self.layers)]
            # x = tf.nn.dropout(x, keep_prob=self.dropout)
            # if return_att:
            atts = []
            debug_log("Encoder.__call__.begin.x", x)
            for enc_layer_index in range(self.layers):
                enc_layer = layers[enc_layer_index]
                x, att = enc_layer(x, mask)
                debug_log("Encoder.__call__.done.x_{}".format(enc_layer_index), x)
                if return_att:
                    atts.append(att)
            return (x, atts) if return_att else x

