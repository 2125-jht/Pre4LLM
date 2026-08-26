# -*- coding: UTF8 -*-
# ----------
# @FILE    : model_base.py
# @ANNO    : 
# @TIME    : 2022/03/25 11:02:14
# @OWNER   : yuankun <yuankun@kuaishou.com>
# @VERS    : 1.0
# ----------

import tensorflow as tf

Tensor = tf.Tensor

class MultiHeadAttention():
    '''
        推荐专用，输入二维 Tensor, 升维后做 attention
    '''
    def __init__(self) -> None:
        self._atten_type = "multi_head"

    def _scaled_dot_product_attention(self,
                                      Q: Tensor,
                                      K: Tensor,
                                      V: Tensor,
                                      scope="scaled_dot_product_attention") -> Tensor:
        #Q (B, dq, da)
        #K (B, dk, da)
        #V (B, dk, da)
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            d_k = Q.get_shape().as_list()[-1]

            # dot product
            outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (B, dq, dk)

            # scale
            outputs /= d_k ** 0.5

            # softmax
            outputs = tf.nn.softmax(outputs)

            # weighted sum (context vectors)
            outputs = tf.matmul(outputs, V)  # (B, dq, da)

            return outputs

    def _muiti_haed_atten(self,
                          queries: Tensor,
                          keys: Tensor,
                          values: Tensor,
                          atten_num: int,
                          head_num=8,
                          scope="multi_head_atten") -> Tensor:
        atten_dim = queries.get_shape().as_list()[-1]
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # Linear projections
            Q = tf.reshape(tf.layers.dense(queries, atten_dim * atten_num * head_num, use_bias=True), [-1, atten_dim, atten_num * head_num])
            K = tf.reshape(tf.layers.dense(keys, atten_dim * atten_num * head_num, use_bias=True), [-1, atten_dim, atten_num * head_num])
            V = tf.reshape(tf.layers.dense(values, atten_dim * atten_num * head_num, use_bias=True), [-1, atten_dim, atten_num * head_num])

            # Split and concat
            Q_ = tf.concat(tf.split(Q, head_num, axis=2), axis=0) # (B*head_num, atten_dim, atten_num)
            K_ = tf.concat(tf.split(K, head_num, axis=2), axis=0) # (B*head_num, atten_dim, atten_num)
            V_ = tf.concat(tf.split(V, head_num, axis=2), axis=0) # (B*head_num, atten_dim, atten_num)

            # Attention
            outputs = self._scaled_dot_product_attention(Q_, K_, V_) # (B*head_num, atten_dim, atten_num)
            outputs = tf.concat(tf.split(outputs, head_num, axis=0), axis=2 ) # (B, atten_dim, atten_num*head_num)
            outputs = tf.squeeze(tf.layers.dense(outputs, 1, use_bias=True), axis=[2]) # (B, atten_dim)

            # Residual
            outputs = outputs + queries

            return outputs

    def _single_head_atten(self,
                           queries: Tensor,
                           keys: Tensor,
                           values: Tensor,
                           atten_num: int,
                           scope="single_head_atten") -> Tensor:
        atten_dim = queries.get_shape().as_list()[-1]
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # Linear projections
            Q = tf.reshape(tf.layers.dense(queries, atten_dim * atten_num, use_bias=True), [-1, atten_dim, atten_num])
            K = tf.reshape(tf.layers.dense(keys, atten_dim * atten_num, use_bias=True), [-1, atten_dim, atten_num])
            V = tf.reshape(tf.layers.dense(values, atten_dim * atten_num, use_bias=True), [-1, atten_dim, atten_num])

            # Attention
            outputs = self._scaled_dot_product_attention(Q, K, V) # (B, atten_dim, atten_num)
            outputs = tf.squeeze(tf.layers.dense(outputs, 1, use_bias=True), axis=[2]) # (B, atten_dim)

            # Residual
            outputs = outputs + queries

            return outputs


class BaseBlocks():
    def __init__(self) -> None:
        self._base_blocks = "base_block"

    def _residual_block(self,
                        input: Tensor,
                        mid_dim: int,
                        scope="resi_block") -> Tensor:
        out_dim = input.get_shape().as_list()[-1]
        with tf.name_scope(scope, reuse=tf.AUTO_REUSE):
            outputs = tf.layers.dense(input, mid_dim, activation=tf.nn.relu)
            outputs = tf.layers.dense(outputs, out_dim, activation=None)

            return outputs + input

    def _residual_block_v2(self,
                           input: Tensor,
                           mid_dim: int,
                           out_dim: int,
                           scope="resi_block_v2") -> Tensor:
        with tf.name_scope(scope, reuse=tf.AUTO_REUSE):
            bias = tf.layers.dense(input, mid_dim, activation=tf.nn.relu)
            bias = tf.layers.dense(bias, out_dim, activation=None)
            outputs = tf.layers.dense(input, out_dim, activation=None)

            return outputs + bias

    def _ln(self,
            inputs: Tensor,
            epsilon = 1e-8,
            scope="ln") -> Tensor:
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]
        
            mean, variance = tf.nn.moments(inputs, [-1], keepdims=True)
            beta= tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
            gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
            normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
            outputs = gamma * normalized + beta
            
            return outputs


