import os
import sys
import json
import logging
import argparse
import functools
import tensorflow as tf
import numpy


def gen_autodis_net(feas, feas_num, head=3, units=[8], scope="default"):
    # 默认feas中特征都是numric，feas.shape[-1] 和feas_num一样
    def gen_meta_emb(feas, feas_num, head, units):
        emb_output = []
        for k in range(head):
            emb_net = feas
            for idx, unit in enumerate(units):
                emb_net = tf.layers.dense(
                    emb_net,
                    unit*feas_num,
                    activation=tf.nn.leaky_relu,
                    kernel_initializer=tf.keras.initializers.he_normal(seed=1),
                    name='autodis_meta_emb{}_layer{}'.format(k, idx))
            emb_output.append(tf.reshape(emb_net, [-1,feas_num,units[-1]]))
        emb_output = tf.stack(emb_output, axis=-2)
        print("cying_emb_output：", emb_output)
        return emb_output #b * fea_num * head * dim

    def gen_auto_dis(feas, feas_num, head, t=1.0): 
        feas = tf.expand_dims(feas, -1) #batch * feas_num * 1
        auto_kernels = tf.get_variable(name="auto_kernel", shape=[feas_num, head], initializer=tf.keras.initializers.he_normal(seed=1)) 
        auto_kernels = tf.expand_dims(auto_kernels,0)# 1 * feas_num * head
        feas = tf.transpose(feas, (1,0,2)) 
        auto_kernels = tf.transpose(auto_kernels, (1,0,2))
        auto_output = tf.matmul(feas, auto_kernels) # feas_num * batch * head
        auto_output = tf.nn.softmax(auto_output/t)
        auto_output = tf.transpose(auto_output, (1,0,2))
        print("cying_auto_emb:", auto_output)
        return auto_output  #batch * feas_num * head

    emb = gen_meta_emb(feas, feas_num, head, units)
    wt = gen_auto_dis(feas, feas_num, head)
    autodis_emb = tf.multiply(tf.reshape(wt,[-1,feas_num,head,1]), emb) #feas_num * batch * embdim ,其中autodis_embedding维度为bacth * feas_num * head * embdim
    autodis_emb = tf.reduce_sum(autodis_emb, axis=-2)
    return autodis_emb
   
def multi_head_attention(query, key, head_num=1, mask=True, scope = "name"):
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    dim = key.get_shape().as_list()[-1]
    q = tf.layers.dense(query, head_num * dim, activation=tf.nn.leaky_relu)
    k = tf.layers.dense(key, head_num * dim, activation=tf.nn.leaky_relu)
    v = tf.layers.dense(key, head_num * dim, activation=tf.nn.leaky_relu)
    
    q = tf.concat(tf.split(q,head_num,axis=-1),axis=0)
    k = tf.concat(tf.split(k,head_num,axis=-1),axis=0)
    v = tf.concat(tf.split(v,head_num,axis=-1),axis=0)

    output = tf.matmul(q, tf.transpose(k,[0,2,1]))
    output = output/(dim**0.5)

    #去掉padding值
    key_mask = tf.sign(tf.abs(tf.reduce_sum(key, axis=-1)))
    key_mask = tf.tile(key_mask, [head_num,1])
    key_mask = tf.tile(tf.expand_dims(key_mask, axis=1),[1,query.get_shape().as_list()[1],1])
    padding = tf.ones_like(output)*(-2**32+1)
    output = tf.where(tf.equal(key_mask,0),padding,output)
    if mask :
      diag_val = tf.ones_like(output[0::])
      mask = tf.linalg.LinearOperatorLowerTriangular(diag_val).to_dense() # 下三角，队首为最近看的，队尾是很久以前看的
      padding = tf.ones_like(output)*(-2**32+1)
      output = tf.where(tf.equal(mask,0),padding,output)

    output = tf.nn.softmax(output)
    query_mask = tf.sign(tf.abs(tf.reduce_sum(query,axis=-1)))
    query_mask = tf.tile(query_mask,[head_num,1])
    query_mask = tf.tile(tf.expand_dims(query_mask,-1),[1,1,key.get_shape().as_list()[1]])

    output = output * query_mask
    output = tf.matmul(output,v)
    output = tf.concat(tf.split(output,head_num,axis=0),axis=-1)
  return output

@tf.custom_gradient
def swish(x):
  sigx = tf.nn.sigmoid(x)
  y = x * sigx
  def grad(dy):
    return dy * (y + (1. - y) * sigx)
  return y, grad

def simple_lhuc_network(inputs, unit1, unit2, name, weight_name, extra_inputs=[]):
  with tf.name_scope('{}_lhuc'.format(name)):
    output = inputs
    final_outputs = []
    with tf.name_scope('{}_lhuc_layer_{}'.format(name, 0)):
      output = tf.layers.dense(output, unit1, tf.nn.relu, name='dense_{}_{}'.format(name, 0), weight_name='{}_layer1_param'.format(weight_name))
    with tf.name_scope('{}_lhuc_layer_{}'.format(name, 1)):
      origin_output = 2.0 * tf.layers.dense(output, unit2, tf.nn.sigmoid, name='dense_{}_{}'.format(name, 1), weight_name='{}_layer2_param'.format(weight_name))
      final_outputs.append(origin_output)
      for i, extra_input in enumerate(extra_inputs):
        extra_output = 2.0 * tf.layers.dense(output, extra_input, tf.nn.sigmoid, name='extra_{}_{}'.format(name, i), weight_name='kernel_extra_{}_{}_layer2_param'.format(weight_name, i))
        final_outputs.append(extra_output)
    return tf.concat(final_outputs, 1)

def fc_layer(loss_name, net, hidden_units, activation = tf.nn.relu, last_unit = None, norm = False): 
  with tf.variable_scope("{}_fc_layer".format(loss_name), reuse=tf.AUTO_REUSE): 
      for i, hidden_unit in enumerate(hidden_units) : 
          net = tf.layers.dense(net, hidden_unit, activation=activation) 
          #net = tf.clip_by_value(net, eps, 1 - eps) 
          #net = tf.layers.batch_normalization(net) 
      if norm : 
        net = tf.layers.batch_normalization(net) 
      if last_unit == 1: 
        net = tf.layers.dense(net, last_unit, activation=tf.nn.sigmoid) 
      elif last_unit != None: 
        net = tf.layers.dense(net, last_unit, activation=None)
  return net 

def get_weight(weight_name, shape):
    weight = tf.get_variable(weight_name, shape)
    return weight

def dense_layer(inputs, size, weight_name, act=None, summary=False):
    assert len(inputs.shape) >= 2
    input_shape = tf.shape(inputs)
    weight = get_weight(weight_name, (inputs.shape[-1] + 1, size))

    bias_shape = tf.concat([input_shape[:-1], [1]], 0)
    bias_input = tf.fill(bias_shape, 1.0, name="bias_input")
    if len(inputs.shape) == 2:
      o = tf.matmul(tf.concat([inputs, bias_input], len(inputs.shape)-1), weight, name=weight_name + "_mul")
    else:
      o = tf.tensordot(tf.concat([inputs, bias_input], len(inputs.shape)-1), weight, name=weight_name + "_mul", axes=(-1, 0))
    if act is not None:
        o = act(o, name=weight_name+'_act')
    if summary:
        tf.summary.histogram(weight_name, weight)
        tf.summary.histogram(weight_name + "_out", o)
    return o

def simple_dense_network(inputs, units, name, weight_name_template, act=tf.nn.relu):
    output = inputs
    for i, unit in enumerate(units):
        # output = tf.layers.Dense(unit, act, name='dense_{}_{}'.format(name, i))(output)
        output = dense_layer(
            output,
            unit,
            weight_name_template.format(i + 1),
            act,
        )
    return output


def gate_tower(loss_name, net, hidden_units, net2):
    with tf.variable_scope("{}_gate_tower".format(loss_name), reuse=tf.AUTO_REUSE):
        dim = net.get_shape().as_list()[-1]
        net2 = tf.concat([net, net2], -1)
        for i, hidden_unit in enumerate(hidden_units):
            weights = gate_layer(loss_name + str(i) + "_layer", net2, dim)
            net = tf.multiply(weights, net)
            net = dense_layer(net, hidden_unit, loss_name + '_' + str(i), act = tf.nn.relu)
            dim = hidden_unit
    return net

def gate_layer(loss_name, net, dim):
    with tf.variable_scope("{}_gate_layer".format(loss_name), reuse=tf.AUTO_REUSE):
        #net = tf.stop_gradient(net) # 不回传
        net = dense_layer(net, 128, loss_name + '_fc', act = tf.nn.relu)
        weights = dense_layer(net, dim, loss_name + '_weight', act = tf.nn.sigmoid)
        weights = weights * 2
    return weights

def shape_list(x):
    """
    deal with dynamic shape in tensorflow cleanly
    """
    ps = x.get_shape().as_list()
    ts = tf.shape(x)
    return [ts[i] if ps[i] is None else ps[i] for i in range(len(ps))]

def gelu(x):
    return 0.5*x*(1+tf.tanh(tf.math.sqrt(2/tf.math.pi)*(x+0.044715*tf.pow(x, 3))))

def _norm(x, g=None, b=None, e=1e-5, axis=[1]):
    u = tf.reduce_mean(x, axis=axis, keepdims=True)
    s = tf.reduce_mean(tf.square(x-u), axis=axis, keepdims=True)
    x = (x - u) * tf.rsqrt(s + e)
    if g is not None and b is not None:
        x = x*g + b
    return x

def norm(x, scope, axis=[-1]):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        n_state = shape_list(x)[-1]
        g = tf.get_variable("g", [n_state], initializer=tf.constant_initializer(1))
        b = tf.get_variable("b", [n_state], initializer=tf.constant_initializer(0))
        return _norm(x, g, b, axis=axis)

def dropout(x, pdrop, train):
    if train and pdrop > 0:
        x = tf.nn.dropout(x, 1-pdrop)
    return x

def mask_attn_weights(w, lower = -1, upper = 0):
    n = shape_list(w)[-1]
    b = tf.matrix_band_part(tf.ones([n, n]), lower, upper)
    b = tf.reshape(b, [1, 1, n, n])
    w = w*b + -1e9*(1-b)
    return w


def attention_fun(Q, K, loss, mask=None):
    with tf.variable_scope(loss + "_attention", reuse=tf.AUTO_REUSE):
      attention = tf.matmul(Q, K, transpose_b=True)  # [batch_size, sequence_length, sequence_length]
      d_k = tf.cast(tf.shape(K)[-1], dtype=tf.float32)
      attention = tf.divide(attention, tf.sqrt(d_k))  # [batch_size, sequence_length, sequence_length]

      if mask :
        attention = mask_attn_weights(attention, mask[0], mask[1])
      attention = tf.nn.softmax(attention, axis=-1)  # [batch_size, sequence_length, sequence_length]
      return attention

def self_attention(scope, query, data, n, nh, dim, mask):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
      batch_size = tf.shape(data)[0]
      #query = tf.reshape(query, [batch_size, n, query_dim])
      #data = tf.reshape(data, [batch_size, m, 64])
      querys = tf.layers.dense(query, nh * dim)  # [batch_size, query_length, hidden_dim]
      keys = tf.layers.dense(data, nh * dim)  # [batch_size, sequence_length, hidden_dim]
      values = tf.layers.dense(data, nh * dim)  # [batch_size, sequence_length, n_classes]

      querys = tf.stack(tf.split(querys, nh, axis=2))
      keys = tf.stack(tf.split(keys, nh, axis=2))
      values = tf.stack(tf.split(values, nh, axis=2)) #(head_num, batch_size, sequence_length, att_embedding_size)

      #flat_dim = m * 128
      attention = attention_fun(querys, keys, scope, mask)  # [head_num, batch_size, query_length, sequence_length]
      output = tf.matmul(attention, values)  # [head_num, batch_size, query_length, n_classes]
      output = tf.transpose(output,  perm=[1, 2, 0, 3]) # (batch_size, query_length ,hn, att_embedding_sizev)
      output = tf.reshape(output, [batch_size, n, nh * dim])
      return output

def self_attention_lora(scope, query, data, n, nh, dim, mask=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
      batch_size = tf.shape(data)[0]
      data_size = data.get_shape().as_list()
      values = tf.get_variable(scope+"_values", shape=[1, data_size[1], data_size[2]], dtype=tf.float32, initializer=tf.zeros_initializer())
      values = tf.tile(values, [batch_size, 1, 1])
      querys = tf.layers.dense(query, nh * dim)  # [batch_size, query_length, hidden_dim]
      keys = tf.layers.dense(data, nh * dim)  # [batch_size, sequence_length, hidden_dim]
      values = tf.layers.dense(values, nh * dim)  # [batch_size, sequence_length, n_classes]

      querys = tf.stack(tf.split(querys, nh, axis=2))
      keys = tf.stack(tf.split(keys, nh, axis=2))
      values = tf.stack(tf.split(values, nh, axis=2)) #(head_num, batch_size, sequence_length, att_embedding_size)

      #flat_dim = m * 128
      attention = attention_fun(querys, keys, scope, mask)  # [head_num, batch_size, query_length, sequence_length]
      output = tf.matmul(attention, values)  # [head_num, batch_size, query_length, n_classes]
      output = tf.transpose(output,  perm=[1, 2, 0, 3]) # (batch_size, query_length ,hn, att_embedding_sizev)
      output = tf.reshape(output, [batch_size, n, nh * dim])
      return output

def mlp(scope, x, n_state):
    with tf.variable_scope(scope):
        nx = shape_list(x)[-1]
        h = dense_layer(x, n_state, scope + "_c_fc", act = tf.nn.relu)
        h2 = dense_layer(h, nx, scope + "_c_fc2")
        #h2 = dropout(h2, resid_pdrop, kuiba_utils.train_mode())
        return h2

def bl_matmul(A, B):
  return tf.einsum('mij,jk->mik', A, B)
 
def neuralsort(s, tau = 4):
  """
  s: input elements to be sorted. Shape: batch_size x n x 1 tau: temperature for relaxation. Scalar.
  """
  n = tf.shape(s)[1]
  one = tf.ones((n, 1), dtype = tf.float32)
  A_s = tf.abs(s - tf.transpose(s, perm=[0, 2, 1]))
  #print("A_s.shape = {}".format(A_s.get_shape().as_list()))
  B = bl_matmul(A_s, tf.matmul(one, tf.transpose(one)))
  #print("B_s.shape = {}".format(B.get_shape().as_list()))
  scaling = tf.cast(n + 1 - 2 * (tf.range(n) + 1), dtype = tf.float32)
  scaling = tf.expand_dims(scaling, 0)
  #scaling = tf.tile(scaling, [rown, 1, 1])
  #print("scale.shape = {}".format(scaling.get_shape().as_list()))
  C = bl_matmul(s, scaling)
  P_max = tf.transpose(C-B, perm=[0, 2, 1])
  P_hat = tf.nn.softmax(P_max / tau, -1)
  return P_hat

def short_net(loss, query, data_list, list_len, list_dim, query_dim, nh = 1,
              dim = 128, stop_gradient_list = [], action_item_size=64, att_emb_size=32):
    scope = loss + "_short_net"
    no_stop_gradient = False
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        batch_size = tf.shape(query)[0]
        query_len = query.shape[1]
        new_data_list = []
        for i in range(len(data_list)):
            input_data = data_list[i]
            if (not no_stop_gradient and stop_gradient_list[i]):
              input_data = tf.stop_gradient(data_list[i])
            new_data_list.append(tf.reshape(input_data, [batch_size, list_len, list_dim[i]]))
        data = tf.concat(new_data_list, 2)
        data = tf.layers.dense(data, action_item_size, activation=tf.nn.relu)
        query = tf.reshape(query, [batch_size, query_len, query_dim])
        #output = self_attention(scope, query, data, 1, nh, dim) # [b,  nh * dim]
        output = short_transformer(loss, query, data, query_len, nh = 1, dim = query_dim) # [b,  nh * dim]
        #output = tf.layers.dense(output, dim, activation=tf.nn.relu)
    return output

def short_transformer(loss, query, data, n, nh = 1, dim = 256, mask = None): 
    scope = loss + '_transformer' 
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE): 
        batch_size = tf.shape(data)[0] 
        a = self_attention(scope, query, data, n, nh, dim, mask = mask) 
        a = tf.layers.dense(a, dim, activation=None, name=scope + "_proj_fc_one_layer")
        data = tf.tile(tf.reshape(tf.reduce_mean(data, axis=1), [batch_size, 1, dim]), [1, n, 1]) 
        output = norm(data + a, scope + '_ln_1') 
        m = mlp(scope + "_mlp", output, dim * 4) 
        output = norm(output + m, scope + '_ln_2') 
    return output

def gate_layer2(loss_name, net, dim): 
    with tf.variable_scope("{}_gate_layer2".format(loss_name), reuse=tf.AUTO_REUSE): 
        #net = tf.stop_gradient(net) # 不回传 
        net = tf.layers.dense(net, 128, activation = tf.nn.relu)  
        weights = tf.layers.dense(net, dim, activation = tf.nn.sigmoid)  
        weights = weights 
    return weights 

def SRGA_transformer(loss, query, data, query_gate, n, nh = 1, dim = 256, mask_GSA = None, mask_LSA = None): 
    scope = loss + '_srga_transformer' 
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE): 
        batch_size = tf.shape(data)[0] 
        GSA = self_attention(scope, query, data, n, nh, dim, mask = mask_GSA)
        LSA = self_attention(scope, query, data, n, nh, dim, mask = mask_LSA)
        gate = gate_layer2(scope, query_gate, dim)
        a = gate * GSA + (1 - gate) * LSA
        output = norm(data + a, scope + '_ln_1')
        m = mlp(scope + "_mlp", output, dim * 4)
        output = norm(output + m, scope + '_ln_2')
    return output

class PLE:
    def __init__(self, tasks_names, shared_key="shared", cgc_layers=2, task_expert_num=1, shared_expert_num=3,
                 expert_tower_dim = [128,128], gate_tower_dim = [64,32]):
        super().__init__()
        self.tasks_names = tasks_names #["next", "wtd", "play"]
        self.shared_key = shared_key #"wtd"
        self.cgc_layers = cgc_layers #1
        self.task_expert_num = task_expert_num #1
        self.shared_expert_num = shared_expert_num #4
        self.expert_tower_dim = expert_tower_dim  # [128,128]
        self.gate_tower_dim = gate_tower_dim # [64,32]

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
                expert_out = self.expert(input_feature)  # [-1,256]
                shared_expert_out_list.append(expert_out)
            shared_expert_out = tf.stack(shared_expert_out_list, axis=2)  # [-1,256, expert_num]
            return shared_expert_out

    # 任务独享专家塔
    def task_experts(self, task_name, input_feature):
        with tf.variable_scope("{}_expert".format(task_name), reuse=tf.AUTO_REUSE):
            task_expert_out_list = []
            for i in range(0, self.task_expert_num):
                expert_out = self.expert(input_feature)  # [-1,256]
                task_expert_out_list.append(expert_out)
            task_expert_out = tf.stack(task_expert_out_list, axis=2)  # [-1,256, expert_num]
            return task_expert_out

    def gate(self, input_feature, task_name, unit_num):
        with tf.variable_scope("{}_gate".format(task_name), reuse=tf.AUTO_REUSE):
            output = input_feature
            for dim in self.gate_tower_dim:
                output = tf.layers.dense(output, units=dim, activation=swish)
            output = tf.layers.dense(output, units=unit_num, activation=tf.nn.sigmoid)  # [-1,expert_num]
            gate = tf.nn.softmax(logits=output, axis=-1)
            return gate  # [-1,expert_num]

    def CGC(self, layer_index, input_feature_dict):
        with tf.variable_scope("layer{}".format(layer_index), reuse=tf.AUTO_REUSE):
            task_outputs = []  # 各个塔直接的输出结果，供共享塔使用
            task_weighted_outputs = {}  # 加权求和之后的结果，供下一层使用
            shared_expert_outs = self.shared_experts(input_feature_dict.get(self.shared_key))
            task_outputs.append(shared_expert_outs)
            for task_name in self.tasks_names:
                # if task_name == "play":
                #   task_expert_outs = self.task_experts(task_name,  tf.stop_gradient(input_feature_dict.get(task_name)))
                # else:
                task_expert_outs = self.task_experts(task_name, input_feature_dict.get(task_name))
                task_outputs.append(task_expert_outs)
                task_shared_expert_outs = tf.concat([shared_expert_outs, task_expert_outs], axis=2)
                gate = self.gate(input_feature_dict.get(task_name), task_name=task_name,
                                 unit_num=self.task_expert_num + self.shared_expert_num)
                gate = tf.tile(tf.expand_dims(gate, dim=1),
                               multiples=[1, self.expert_tower_dim[-1], 1])  # [-1,256,total_expert_num]
                output = gate * task_shared_expert_outs  # [-1,256,total_expert_num]
                output = tf.reduce_sum(output, axis=-1)  # [-1,256]
                task_weighted_outputs[task_name] = output

            if layer_index != self.cgc_layers - 1:  # 中间层进行共享塔的计算
                all_expert_outs = tf.concat(task_outputs, axis=2)
                shared_gate = self.gate(input_feature_dict.get(self.shared_key), task_name=self.shared_key,
                                        unit_num=(len(self.tasks_names) * self.task_expert_num + self.shared_expert_num))
                shared_gate = tf.tile(tf.expand_dims(shared_gate, dim=1),
                                      multiples=[1, self.expert_tower_dim[-1], 1])  # [-1,256,total_expert_num]
                shared_expert_output = shared_gate * all_expert_outs  # [-1,256,total_expert_num]
                shared_expert_output = tf.reduce_sum(shared_expert_output, axis=-1)  # [-1,256]
                task_weighted_outputs[self.shared_key] = shared_expert_output
            return task_weighted_outputs

    def __call__(self, input_feature_dict):  # task -> 输入
        with tf.variable_scope("ple", reuse=tf.AUTO_REUSE):
            for layer_index in range(self.cgc_layers):
                input_feature_dict = self.CGC(layer_index, input_feature_dict)
            return input_feature_dict

def scaled_dot_product_attention(Q, K, V, mask=None):
    matmul_qk = tf.matmul(Q, K, transpose_b=True)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    if mask is not None:
        scaled_attention_logits = scaled_attention_logits + (mask * -1e9)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, V)
    return output, attention_weights

def multi_head_attention(name, queries, keys, values, dim_in=512, num_heads=8, dropout_rate=0.0, training=False, causal_mask=False):
    def split_heads(x, num_heads, dim_in):
        # 判断输入维度
        input_shape = tf.shape(x)
        # 三维输入 [batch_size, seq_len, dim]
        batch_size = input_shape[0]
        assert dim_in % num_heads == 0
        depth = dim_in // num_heads
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

        Q = tf.layers.dense(queries, dim_in, use_bias=False)
        K = tf.layers.dense(keys, dim_in, use_bias=False)
        V = tf.layers.dense(values, dim_in, use_bias=False)

        Q = split_heads(Q, num_heads, dim_in) # (?, num_heads, seq_len, dim_in/num_heads)
        K = split_heads(K, num_heads, dim_in)
        V = split_heads(V, num_heads, dim_in)

        mask = None
        if causal_mask:
            # mask = create_causal_mask(batch_size, num_heads, seq_len)
            mask = create_causal_mask(batch_size, seq_len, num_heads)

        scaled_attention, attention_weights = scaled_dot_product_attention(Q, K, V, mask=mask) # (?, num_heads, seq_len, dim_in/num_heads)
        scaled_attention = tf.transpose(scaled_attention, [0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention, [batch_size, seq_len, dim_in])

        output = tf.layers.dense(concat_attention, dim_in)
        output = tf.layers.dropout(output, rate=dropout_rate, seed=2025, training=training)
        # output = tf.nn.dropout(output, rate=dropout_rate) if training else output
    return output