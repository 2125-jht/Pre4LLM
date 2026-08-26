from numpy import rate
import tensorflow as tf
from tensorflow import Tensor

# -*- coding: utf-8 -*-
'''
------------------------------------------------------------------------
@Description : reco transformer
@Author :  邓英杰
@Time :  2025/01/23 17:29:06
------------------------------------------------------------------------
'''

import tensorflow as tf

if tf.__version__ >= '2.0':
  tf = tf.compat.v1


def dnn_layer(inputs, hidden_units, activation=tf.nn.relu, batch_normalization=False, training=True,
              dropout=None, last_layer_no_activation=False, last_layer_no_batch_norm=False,
              last_layer_no_dropout=False, scope="mlp", **kwargs):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        x = inputs
        for i, units in enumerate(hidden_units):
            if (i == len(hidden_units) - 1) and last_layer_no_activation:
                activation = None
            x = tf.layers.dense(x, units, activation, name="layer_{}".format(i), **kwargs)

            if batch_normalization is True and ((i < len(hidden_units) - 1) or not last_layer_no_batch_norm):
                # 训练阶段，要保证均值和方差的正确更新；预测阶段，则要保证所有参数与训练阶段的一致，其实主要就4个，训练阶段全局的gamma beta 均值 方差
                x = tf.layers.batch_normalization(x, training=training, name="{}_bn_{}".format(i))

            if dropout is not None and ((i < len(hidden_units) - 1) or not last_layer_no_dropout):
                if training:
                    x = tf.nn.dropout(x, rate=dropout, name="layer_dropout_{}".format(i))
                else: x = x
    return x

def din_attn(queries, keys, keys_length, scope_pre, training, hidden_units=[64,32,1], activation=tf.nn.relu, use_prelu=True):
    '''
        queries:     [B, H]    [batch_size,embedding_size]
        keys:        [B, T, H]   [batch_size,T,embedding_size]
        keys_length: [B]        [batch_size] 真实长度
        # T为历史行为序列长度
        return: B * 1 * H
    '''
    with tf.variable_scope(f"{scope_pre}_din_attn", reuse=tf.AUTO_REUSE):
        def prelu(_x):
            alphas = tf.get_variable('prelu_alpha', _x.get_shape()[-1],
                                initializer=tf.constant_initializer(0.0),
                                dtype=tf.float32)
            pos = tf.nn.relu(_x)
            neg = alphas * (_x - abs(_x)) * 0.5
            return pos + neg

        T = tf.shape(keys)[1]
        H = queries.get_shape().as_list()[-1]
        queries = tf.tile(queries, [1, T])
        queries = tf.reshape(queries, [-1, T, H])
        din_all = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1) # B*T*4H

        activation = prelu if use_prelu else activation
        din_output = dnn_layer(din_all, hidden_units=hidden_units,activation=activation, training=training,
                            last_layer_no_activation=True, last_layer_no_batch_norm=True) # B*T*1

        # 为了让outputs维度和keys的维度一致
        outputs = tf.reshape(din_output, [-1, 1, T]) # B*1*T
        
        key_masks = tf.sequence_mask(keys_length, T) # B*T
        key_masks = tf.expand_dims(key_masks,1) # B*1*T
        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(key_masks,outputs,paddings) # B * 1 * T

        # Scale（缩放）
        # outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)
        outputs = tf.nn.softmax(outputs) # B * 1 * T
        # Weighted Sum outputs=g(Vi,Va)   keys=Vi
        #这步为公式中的g(Vi*Va)*Vi
        outputs = tf.matmul(outputs,keys) # B * 1 * H 三维矩阵相乘，相乘发生在后两维，即 B * (( 1 * T ) * ( T * H ))

    return outputs

def layer_norm(name, x, epsilon=1e-6):
    with tf.variable_scope(f"{name}_layer_norm", reuse=tf.AUTO_REUSE):
        gamma = tf.get_variable("gamma", [x.get_shape()[-1]], initializer=tf.ones_initializer())
        beta = tf.get_variable("beta", [x.get_shape()[-1]], initializer=tf.zeros_initializer())

        mean, variance = tf.nn.moments(x, axes=[-1], keep_dims=True)
        normalized = (x - mean) / tf.sqrt(variance + epsilon)
        output = gamma * normalized + beta
    return output

def scaled_dot_product_attention(Q, K, V, mask=None):
    matmul_qk = tf.matmul(Q, K, transpose_b=True)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    if mask is not None:
        scaled_attention_logits = scaled_attention_logits + (mask * -1e9)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, V)
    return output, attention_weights

def multi_head_attention(name, queries, keys, values, dk=256, num_heads=8, dropout_rate=0.0, training=False, causal_mask=False):
        def split_heads(x, num_heads, dim_in):
            # 判断输入维度
            input_shape = tf.shape(x)
            # 三维输入 [batch_size, seq_len, dim]
            batch_size = input_shape[0]
            depth = x.get_shape().as_list()[-1] // num_heads
            reshaped = tf.reshape(x, [batch_size, -1, num_heads, depth])

            # 转置为 [batch_size(*beam_size), num_heads, seq_len, depth]
            return tf.transpose(reshaped, [0, 2, 1, 3])

        # def create_causal_mask(batch_size, num_heads, seq_len):
        def create_causal_mask(batch_size, seq_len, num_heads):
            """
            Creates a causal mask to prevent attention to future tokens.
            Args:
                seq_len: Length of the sequence.
            Returns:
                A causal mask of shape (seq_len, seq_len).
            """
            mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
            mask = tf.expand_dims(tf.expand_dims(mask, 0), 0)
            # 三维情况下只复制到batch
            mask = tf.tile(mask, [batch_size, num_heads, 1, 1])
            return mask

        with tf.variable_scope(f"{name}", reuse=tf.AUTO_REUSE):
            # 判断输入维度
            input_shape = tf.shape(queries)
            batch_size = input_shape[0]
            seq_len = input_shape[1]
            input_dim = queries.get_shape().as_list()[-1]

            Q = tf.layers.dense(queries, dk, use_bias=False)
            K = tf.layers.dense(keys, dk, use_bias=False)
            V = tf.layers.dense(values, dk, use_bias=False)

            Q = split_heads(Q, num_heads, dk) # (?, num_heads, seq_len, dim_in/num_heads)
            K = split_heads(K, num_heads, dk)
            V = split_heads(V, num_heads, dk)

            mask = None
            if causal_mask:
                # mask = create_causal_mask(batch_size, num_heads, seq_len)
                mask = create_causal_mask(batch_size, seq_len, num_heads)

            scaled_attention, attention_weights = scaled_dot_product_attention(Q, K, V, mask=mask) # (?, num_heads, seq_len, dim_in/num_heads)
            scaled_attention = tf.transpose(scaled_attention, [0, 2, 1, 3])

            concat_attention = tf.reshape(scaled_attention, [batch_size, seq_len, dk])

            output = tf.layers.dense(concat_attention, input_dim)
            output = tf.layers.dropout(output, rate=dropout_rate, seed=2025, training=training)
            # output = tf.nn.dropout(output, rate=dropout_rate) if training else output
        return output
    
def feed_forward_network(name, x, dim, training=True, dropout_rate=0.0):
    with tf.variable_scope(f"{name}", reuse=tf.AUTO_REUSE):
        input_dim = x.get_shape().as_list()[-1]
        x = tf.layers.dense(x, dim, activation=tf.nn.relu)
        x = tf.layers.dense(x, input_dim)
        x = tf.layers.dropout(x, rate=dropout_rate, seed=2025, training=training)
        # x = tf.cond(training, lambda: tf.nn.dropout(x, rate=dropout_rate), lambda: x)
        return x

class EncoderLayer:
    def __init__(self, name, dim, num_heads, dk, dropout_rate):
        super(EncoderLayer, self).__init__()
        self.name = name
        self.dim = dim
        self.num_heads = num_heads
        self.dk = dk
        self.dropout_rate = dropout_rate

    def forward(self, x, training, causal_mask=False):
        with tf.variable_scope(f"{self.name}", reuse=tf.AUTO_REUSE):
            attn_output = multi_head_attention("encoder_mha", x, x, x, self.dk, self.num_heads, self.dropout_rate, training=training, causal_mask=causal_mask) # MHA
            out1 = layer_norm("encoder_ln1", x + attn_output) # Add & Norm  # (?, seq_len, dim)

            ffn_output = feed_forward_network("encoder_ffn", out1, self.dim, training=training) # (?, seq_len, dim)
            out2 = layer_norm("encoder_ln2", out1 + ffn_output)
        
        return out2

class DecoderLayer:
    def __init__(self, name, dim, num_heads, dk, dropout_rate, training=False):
        super(DecoderLayer, self).__init__()
        self.name = name
        self.dim = dim
        self.num_heads = num_heads
        self.dk = dk
        self.dropout_rate = dropout_rate

    def forward(self, x, enc_output, training, causal_mask=False):
        with tf.variable_scope(f"{self.name}", reuse=tf.AUTO_REUSE):
            # 1. Self Attention
            self_attn_output = multi_head_attention("decoder_mha", x, x, x, self.dk, self.num_heads, self.dropout_rate, training=training, causal_mask=True)
            out1 = layer_norm("decoder_ln1", x + self_attn_output)
            # 2. Cross Attention
            cross_attn_output = multi_head_attention("decoder_mha_cross", out1, enc_output, enc_output, self.dk, self.num_heads, self.dropout_rate, training=training, causal_mask=causal_mask)
            out2 = layer_norm("decoder_ln2", out1 + cross_attn_output)
            # 3. Feed Forward
            ffn_output = feed_forward_network("decoder_ffn", out2, self.dim, training=training)
            out3 = layer_norm("decoder_ln3", out2 + ffn_output)
            return out3

class PositionLayer:
    def __init__(self, name, dim, num_heads, dk, dropout_rate, training=False):
        super(PositionLayer, self).__init__()
        self.name = name
        self.dim = dim
        self.num_heads = num_heads
        self.dk = dk
        self.dropout_rate = dropout_rate
        
    def forward(self, x, enc_output, training, causal_mask=False):
        with tf.variable_scope(f"{self.name}", reuse=tf.AUTO_REUSE):
            cross_attn_output = multi_head_attention("position_mha_cross", x, enc_output, enc_output, self.dk, self.num_heads, self.dropout_rate, training=training, causal_mask=False)
            out1 = layer_norm("ln1", x + cross_attn_output)
            attn_output = multi_head_attention("position_mha", out1, out1, out1, self.dk, self.num_heads, self.dropout_rate, training=training, causal_mask=causal_mask)
            out2 = layer_norm("ln2", x + attn_output)
            ffn_output = feed_forward_network("ffn1", out2, self.dim, training=training, dropout_rate=self.dropout_rate)
            out3 = layer_norm("ln3", out2 + ffn_output)
        
        return out3

"""*************************************************** Set Transformer ******************************************************"""
"""
    Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks
    from https://github.com/juho-lee/set_transformer/blob/master/modules.py
"""

class SetTransformerModel():
    def __init__(self, dim, dim_in, num_inds, num_seeds, num_heads, dropout_rate=0.0, training=False):
        super(SetTransformerModel, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dim_in = dim_in
        self.dropout_rate = dropout_rate
        self.I = tf.get_variable(
            "Inducing_Points", shape=[num_inds, dim_in], initializer=tf.random_uniform_initializer()
        )
        self.S = tf.get_variable(
            "Seed_vectors", shape=[num_seeds, dim_in], initializer=tf.random_uniform_initializer()
        )

    def MAB(self, name, x, y, training):
        with tf.variable_scope(name):
            mha_output = multi_head_attention("mha_0", x, y, y, self.dim, self.num_heads, self.dropout_rate, training=training, causal_mask=False)
            out1 = layer_norm(x + mha_output)
            ffn = feed_forward_network("mab_ffn_0", self.dim, out1, dropout_rate=self.dropout_rate, training=training)
            ffn_output = ffn(out1, training=training)
            out2 = layer_norm(out1 + ffn_output)
            return out2

    def ISAB(self, name, x, training):
        # 两层 MHA
        with tf.variable_scope(f"{name}", reuse=tf.AUTO_REUSE):
            self.I = tf.tile(tf.expand_dims(self.I, 0), [tf.shape(x)[0], 1, 1])
            h = self.MAB("mab_0", self.I, x, training)
            output = self.MAB("mab_1", x, h, training)

        return output

    def PMA(self, name, x, training):
        with tf.variable_scope(f"{name}", reuse=tf.AUTO_REUSE):
            self.S = tf.tile(tf.expand_dims(self.S, 0), [tf.shape(x)[0], 1, 1])
            output = self.MAB("mab_0", self.S, x, training)

        return output


def simple_lhuc_network(inputs: Tensor, unit1: int, unit2: int, name_scope="lhuc"):
    with tf.variable_scope(name_scope, reuse=tf.AUTO_REUSE):
        output = inputs
        with tf.variable_scope("lhuc_layer_0"):
            output = tf.layers.dense(output, unit1, activation=tf.nn.relu)
        with tf.variable_scope("lhuc_layer_1"):
            output = 2.0 * tf.layers.dense(output, unit2, activation=tf.nn.sigmoid)
        return output


@tf.custom_gradient
def swish(x):
  sigx = tf.nn.sigmoid(x)
  y = x * sigx
  def grad(dy):
    return dy * (y + (1. - y) * sigx)
  return y, grad

def output_attention(loss, query, data, dim, values=None, need_initialize_values=False):
    #  (?,candidates_size,n,64)
    #  n: output num or seq_len，nh: num_head
    scope = loss + "_self_attention"
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
      batch_size = tf.shape(data)[0]
      cand_size = tf.shape(data)[0]
      data_size = data.get_shape().as_list()
      if need_initialize_values:
        values = tf.get_variable(scope+"_values", shape=[1, 1, data_size[2], data_size[3]], dtype=tf.float32, initializer=tf.zeros_initializer())
        values = tf.tile(values, [batch_size, cand_size, 1, 1])
      querys = tf.layers.dense(query, dim)  # [batch_size, cand_size, query_length, hidden_dim]
      keys = tf.layers.dense(data, dim)  # [batch_size, cand_size, sequence_length, hidden_dim]
      values = tf.layers.dense(values, dim)  # [batch_size, cand_size, sequence_length, n_classes]

      #flat_dim = m * 128
      attention = tf.matmul(querys, keys, transpose_b=True)  # [batch_size, cand_size, sequence_length, sequence_length]
      d_k = tf.cast(tf.shape(keys)[-1], dtype=tf.float32)
      attention = tf.divide(attention, tf.sqrt(d_k))  # [batch_size, cand_size, sequence_length, sequence_length]
      attention = tf.nn.softmax(attention, axis=-1)  # [batch_size, cand_size, sequence_length, sequence_length]
      output = tf.matmul(attention, values)  # [batch_size, cand_size, sequence_length, hidden_dim]
      return output
class PLE:
    def __init__(self, tasks_names, shared_key="shared", cgc_layers=2, task_expert_num=1, shared_expert_num=3,
                 expert_tower_dim = [128,128], gate_tower_dim = [64,32], print_ops = []):
        super().__init__()
        self.tasks_names = tasks_names #["next", "wtd", "ctr"]
        self.shared_key = shared_key #"wtd"
        self.cgc_layers = cgc_layers #1
        self.task_expert_num = task_expert_num #1
        self.shared_expert_num = shared_expert_num #4
        self.expert_tower_dim = expert_tower_dim  # [128,128]
        self.gate_tower_dim = gate_tower_dim # [64,32]
        self.print_ops = print_ops

    # 单个专家塔
    def expert(self, input_feature):
        output = input_feature
        for i in range(0, len(self.expert_tower_dim)):
            with tf.variable_scope(f"expert_layer_{i}", reuse=tf.AUTO_REUSE):
                dim = self.expert_tower_dim[i]
                output = tf.layers.dense(output, units=dim, activation=swish)
        return output

    # 共享专家塔
    def shared_experts(self, input_feature):
        with tf.variable_scope("shared_experts", reuse=tf.AUTO_REUSE):
            shared_expert_out_list = []
            for i in range(0, self.shared_expert_num):
                expert_out = self.expert(input_feature)  # [-1, cand_size,128]
                shared_expert_out_list.append(expert_out)
            shared_expert_out = tf.stack(shared_expert_out_list, axis=-1)  # [-1,cand_size,128, expert_num]
            return shared_expert_out

    # 任务独享专家塔
    def task_experts(self, task_name, input_feature):
        with tf.variable_scope("{}_expert".format(task_name), reuse=tf.AUTO_REUSE):
            task_expert_out_list = []
            for i in range(0, self.task_expert_num):
                expert_out = self.expert(input_feature)  # [-1,cand_size,128]
                task_expert_out_list.append(expert_out)
            task_expert_out = tf.stack(task_expert_out_list, axis=-1)  # [-1,cand_size,128, expert_num]
            return task_expert_out

    def gate(self, input_feature, task_name, unit_num):
        with tf.variable_scope("{}_gate".format(task_name), reuse=tf.AUTO_REUSE):
            output = input_feature
            for dim in self.gate_tower_dim:
                output = tf.layers.dense(output, units=dim, activation=swish)
            output = tf.layers.dense(output, units=unit_num, activation=tf.nn.sigmoid)
            gate = tf.nn.softmax(logits=output, axis=-1)  # [-1,cand_size,expert_num]
            return gate

    def CGC(self, layer_index, input_feature_dict):
        # 输入：(?,candidates_size,32)
        with tf.variable_scope("layer{}".format(layer_index), reuse=tf.AUTO_REUSE):
            task_outputs = []  # 各个塔直接的输出结果，供共享塔使用
            task_weighted_outputs = {}  # 加权求和之后的结果，供下一层使用
            shared_expert_outs = self.shared_experts(input_feature_dict.get(self.shared_key))  # [-1,cand_size,128, expert_num]
            is_4d = len(shared_expert_outs.get_shape().as_list()) == 4
            task_outputs.append(shared_expert_outs)
            for task_name in self.tasks_names:
                task_expert_outs = self.task_experts(task_name, input_feature_dict.get(task_name))
                task_outputs.append(task_expert_outs)
                task_shared_expert_outs = tf.concat([shared_expert_outs, task_expert_outs], axis=-1)
                gate = self.gate(input_feature_dict.get(task_name), task_name=task_name,
                                 unit_num=self.task_expert_num + self.shared_expert_num)  # [-1,cand_size,expert_num]
                # self.print_ops.append(tf.print(f"ple gate: ", gate[2][:8], summarize = 8, output_stream=sys.stdout))
                if is_4d:
                    gate = tf.tile(tf.expand_dims(gate, dim=2), multiples=[1, 1, self.expert_tower_dim[-1], 1])  # [-1,cand_size,128,total_expert_num]
                else:
                    gate = tf.tile(tf.expand_dims(gate, dim=1), multiples=[1, self.expert_tower_dim[-1], 1])
                output = gate * task_shared_expert_outs  # [-1,cand_size,128,total_expert_num]
                output = tf.reduce_sum(output, axis=-1)  # [-1,cand_size,128]
                task_weighted_outputs[task_name] = output

            if layer_index != self.cgc_layers - 1:  # 中间层进行共享塔的计算
                all_expert_outs = tf.concat(task_outputs, axis=2)
                shared_gate = self.gate(input_feature_dict.get(self.shared_key), task_name=self.shared_key,
                                        unit_num=(len(self.tasks_names) * self.task_expert_num + self.shared_expert_num))  # [-1,cand_size,total_expert_num]
                if is_4d:
                    shared_gate = tf.tile(tf.expand_dims(shared_gate, dim=2), multiples=[1, 1, self.expert_tower_dim[-1], 1])  # [-1,cand_size,128,total_expert_num]
                else:
                    shared_gate = tf.tile(tf.expand_dims(shared_gate, dim=1), multiples=[1, self.expert_tower_dim[-1], 1])
                shared_expert_output = shared_gate * all_expert_outs  # [-1,cand_size,128,total_expert_num]
                shared_expert_output = tf.reduce_sum(shared_expert_output, axis=-1)  # [-1,cand_size,128]
                task_weighted_outputs[self.shared_key] = shared_expert_output
            return task_weighted_outputs

    def __call__(self, input_feature_dict):  # task -> 输入
        with tf.variable_scope("ple", reuse=tf.AUTO_REUSE):
            for layer_index in range(self.cgc_layers):
                input_feature_dict = self.CGC(layer_index, input_feature_dict)
            return input_feature_dict
