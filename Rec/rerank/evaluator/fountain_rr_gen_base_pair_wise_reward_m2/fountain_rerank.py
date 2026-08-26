import os
import sys
import json
import logging
import argparse
import functools
import tensorflow as tf
import numpy
from numpy.core.fromnumeric import transpose
from tensorflow.keras.backend import expand_dims,repeat_elements,sum
from mio_tensorflow.config import MioConfig
from mio_tensorflow.variable import MioVariable, MioEmbedding
from mio_tensorflow.collection import MioCollections
import model_utils as mu

parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=['train', 'predict'])
parser.add_argument('--dryrun', dest="dryrun", const=True, default=False, nargs='?')
parser.add_argument('--with_kai', action="store_true")
parser.add_argument('--text', action="store_true")
args = parser.parse_args()

if not args.dryrun and not args.with_kai:
    # monkey patch
    import mio_tensorflow.patch as mio_tensorflow_patch
    mio_tensorflow_patch.apply()

logging.basicConfig()

base_config = '/home/caoying03/complete_pro/model/kuiba_to_kai/rerank/lintao/base.yaml'
config = MioConfig.from_base_yaml(base_config, clear_embeddings=True, clear_params=True,
                                  dryrun=args.dryrun, label_with_kv=True, grad_no_scale=False,
                                  predict=(args.mode == 'predict'),
                                  with_kai=args.with_kai, kconf="McHotL2RTower")

config_from_kuiba = json.load(open("kai_kuiba_config.json"))
all_sparse_input_dict = {}
all_dense_input_dict = {}
use_dragonfly_io = True

def new_sized_embedding(name, dim, expand, slots, common):
  if args.mode == 'predict' or args.with_kai:
      return config.new_embedding(name, dim=dim, expand=expand, slots=slots, common=common, sized=True)
  x = config.new_embedding(name, dim=dim, expand=expand, common=common, slots=slots)
  offset = tf.cast(config.get_signs(slots[0])[1], tf.int32)
  size_var = offset[1:] - offset[0:-1]
  return x, size_var

def get_sparse_input(feature_name, dim, slot_id, expand, common):
  feature_name = "KAI_" + feature_name
  if args.mode == 'predict':
    pass
  else:
    common = False
  if feature_name in all_sparse_input_dict.keys():
    return all_sparse_input_dict[feature_name]

  sparse_input, sparse_size = new_sized_embedding(feature_name, dim, expand, [slot_id], common)
  all_sparse_input_dict[feature_name] = (sparse_input, sparse_size)
  print ("new_embedding" + feature_name + ", dim:" + str(dim) + ", slot:" + str(slot_id))
  return sparse_input, sparse_size

def kai_output_embedding(feature_name, output_layer):
  sign_feature_slot = config_from_kuiba["sign_feature_slot"]
  sign_feature_dim = config_from_kuiba["sign_feature_dim"]
  sign_feature_expand = config_from_kuiba["sign_feature_expand"]
  sign_feature_is_common = config_from_kuiba["sign_feature_is_common"]
  sparse_input_name = feature_name
  dim = sign_feature_dim[sparse_input_name]
  slot_id = sign_feature_slot[sparse_input_name]
  expand = sign_feature_expand[sparse_input_name]
  common = sign_feature_is_common[sparse_input_name]
  sparse_input, sparse_size = get_sparse_input(sparse_input_name, dim, slot_id, expand, common)
  if (args.with_kai):
    config.custom_opt[sparse_input] = {"opt_type" : "AssignAdd" }
    config.custom_gradients[sparse_input] = output_layer
  else:
    tf.assign(sparse_input, output_layer)

def get_dense_input(name, dim=1, default_value=0.0):
  if name in all_dense_input_dict.keys():
    return all_dense_input_dict[name]
  print ("get_label:" + name, dim)

  sign_feature_dim = config_from_kuiba["sign_feature_dim"]
  if name in sign_feature_dim.keys():
    dim = sign_feature_dim[name]
  if use_dragonfly_io and name.startswith("KAI_"):
    if name[4:] in sign_feature_dim.keys():
      dim = sign_feature_dim[name[4:]]

  dense_input = config.get_extra_param(name, size=dim, default_value=default_value)
  all_dense_input_dict[name] = dense_input
  return dense_input

def get_kuiba_parameter_dim(name):
  sign_feature_dim = config_from_kuiba["sign_feature_dim"]
  return sign_feature_dim[name]

def get_kuiba_loss_relative(loss, with_label_dict, with_label_value, with_sample_weight):
  parameters_dict = {}
  parameters_size_dict = {}
  sign_feature_slot = config_from_kuiba["sign_feature_slot"]
  sign_feature_dim = config_from_kuiba["sign_feature_dim"]
  sign_feature_expand = config_from_kuiba["sign_feature_expand"]
  sign_feature_is_common = config_from_kuiba["sign_feature_is_common"]
  sign_feature_pooling = config_from_kuiba["sign_feature_pooling"]
  loss_config = config_from_kuiba["loss_functions"][loss]

  for input_name in loss_config["all_inputs"]:
    if input_name in loss_config["sparse_inputs"]:
      sparse_input_name = input_name
      dim = sign_feature_dim[sparse_input_name]
      slot_id = sign_feature_slot[sparse_input_name]
      expand = sign_feature_expand[sparse_input_name]
      common = sign_feature_is_common[sparse_input_name]
      sparse_input, sparse_size = get_sparse_input(sparse_input_name, dim, slot_id, expand, common)
      pooling_type = sign_feature_pooling[sparse_input_name]
      parameters_dict[sparse_input_name] = sparse_input
      parameters_size_dict[sparse_input_name] = sparse_size
      if pooling_type == 2:
        one = tf.fill(tf.shape(sparse_size), 1.0)
        float_sparse_size = tf.cast(sparse_size, dtype=tf.float32)
        div_tensor = tf.math.maximum(float_sparse_size, one)
        parameters_dict[sparse_input_name] = sparse_input / div_tensor;
        print ("average pooling:", sparse_input_name, parameters_dict[sparse_input_name])
    else:
      assert(input_name in loss_config["dense_inputs"])
      dense_input_name = input_name
      if use_dragonfly_io:
        parameters_dict[dense_input_name] = get_dense_input("KAI_" + dense_input_name)
      else:
        parameters_dict[dense_input_name] = get_dense_input(dense_input_name)

  label_dict = {}
  label_value_dict = {}
  for loss_name,config in config_from_kuiba["loss_functions"].items():
    output_dim = 1
    if "output_dim" in config.keys():
      output_dim = config["output_dim"]
    if (with_label_dict):
      label_dict[loss_name] = get_dense_input(loss_name + "_label", dim=output_dim)
    if (with_label_value):
      label_value_dict[loss_name] = get_dense_input(loss_name + "_label_value")
  sample_weight = None
  sample_flag = None
  if (with_sample_weight):
    sample_flag = get_dense_input(loss + "_sample_flag")
    weight_attr = loss_config["weight_attr"]
    if weight_attr == "":
      sample_weight = sample_flag
    else:
      sample_weight = get_dense_input(weight_attr, default_value=1.0) * sample_flag

  return parameters_dict, parameters_size_dict, label_dict, label_value_dict, sample_weight, sample_flag

def sum_loss_tensor_dict(loss_dict):
  sum_loss = None
  for key,loss in loss_dict.items():
    if (sum_loss == None):
      sum_loss = loss
    else:
      sum_loss += loss
  return sum_loss

def get_parameter_names_by_loss_name(loss_name):
  loss_config = config_from_kuiba["loss_functions"][loss_name];
  input_set = {}
  all_inputs = []
  for input_name in loss_config["all_inputs"]:
    if input_name in input_set.keys():
      continue
    input_set[input_name] = 1
    all_inputs.append(input_name)
  return all_inputs

def get_sparse_bp_var_list(sparse_name_list):
  sparse_list = tf.get_collection_ref(MioCollections.MIO_EMBEDDINGS)
  # name.[4:] 是为了去掉 "KAI_" 前缀
  return [x for x in sparse_list if x.name[4:] in sparse_name_list]

def KaiBatchNorm(inputs, training=True, name="batch_normalization"):
  # 这里加一个前缀是因为可能以"_" 开头非法命名
  if (args.with_kai == True):
    return config.batch_norm(inputs, "KAIBNPREFIX_" + name)
  else:
    # 异步不会自动添加 scope
    return config.batch_norm(inputs, tf.get_variable_scope().name + "/KAIBNPREFIX_" + name)

# NOTE(linta): 照搬https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/contrib/layers/python/layers/layers.py
# layer_norm 的实现, 将 model_variable 换成 tf.get_variable
def KaiLayerNorm(inputs,
                 center=True,
                 scale=True,
                 activation_fn=None,
                 reuse=None,
                 variables_collections=None,
                 outputs_collections=None,
                 trainable=True,
                 begin_norm_axis=1,
                 begin_params_axis=-1,
                 scope=None):
  from tensorflow.contrib.layers.python.layers import utils
  from tensorflow.python.framework import ops
  from tensorflow.python.ops import init_ops
  from tensorflow.python.ops import variable_scope
  from tensorflow.python.ops import nn
  with variable_scope.variable_scope(
      scope, 'LayerNorm', [inputs], reuse=reuse) as sc:
    inputs = ops.convert_to_tensor(inputs)
    inputs_shape = inputs.shape
    inputs_rank = inputs_shape.ndims
    if inputs_rank is None:
      raise ValueError('Inputs %s has undefined rank.' % inputs.name)
    dtype = inputs.dtype.base_dtype
    if begin_norm_axis < 0:
      begin_norm_axis = inputs_rank + begin_norm_axis
    if begin_params_axis >= inputs_rank or begin_norm_axis >= inputs_rank:
      raise ValueError('begin_params_axis (%d) and begin_norm_axis (%d) '
                       'must be < rank(inputs) (%d)' %
                       (begin_params_axis, begin_norm_axis, inputs_rank))
    params_shape = inputs_shape[begin_params_axis:]
    if not params_shape.is_fully_defined():
      raise ValueError(
          'Inputs %s: shape(inputs)[%s:] is not fully defined: %s' %
          (inputs.name, begin_params_axis, inputs_shape))
    # Allocate parameters for the beta and gamma of the normalization.
    beta, gamma = None, None
    if center:
      beta_collections = utils.get_variable_collections(variables_collections,
                                                        'beta')
      beta = tf.get_variable(
          'beta',
          shape=params_shape,
          dtype=dtype,
          initializer=init_ops.zeros_initializer(),
          collections=beta_collections,
          trainable=trainable)
    if scale:
      gamma_collections = utils.get_variable_collections(
          variables_collections, 'gamma')
      gamma = tf.get_variable(
          'gamma',
          shape=params_shape,
          dtype=dtype,
          initializer=init_ops.ones_initializer(),
          collections=gamma_collections,
          trainable=trainable)
    # Calculate the moments on the last axis (layer activations).
    norm_axes = list(range(begin_norm_axis, inputs_rank))
    mean, variance = nn.moments(inputs, norm_axes, keep_dims=True)
    # Compute layer normalization using the batch_normalization function.
    variance_epsilon = 1e-12
    outputs = nn.batch_normalization(
        inputs,
        mean,
        variance,
        offset=beta,
        scale=gamma,
        variance_epsilon=variance_epsilon)
    outputs.set_shape(inputs_shape)
    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return utils.collect_named_outputs(outputs_collections, sc.name, outputs)


SEQUENCE_LEN = 6
POSITION_EMBEDDING_SIZE = 64
print_ops = []

@tf.custom_gradient
def swish(x):
  sigx = tf.nn.sigmoid(x)
  y = x * sigx
  def grad(dy):
    return dy * (y + (1. - y) * sigx)
  return y, grad

def _log(name, tensor):
  print("log:{},{}".format(name, str(tensor)))

def simple_lhuc_network(inputs, unit1, unit2, name, weight_name, extra_inputs=[]):
  with tf.name_scope('{}_lhuc'.format(name)):
    output = inputs
    final_outputs = []
    with tf.name_scope('{}_lhuc_layer_{}'.format(name, 0)):
      output = mio_dense_layer(output, unit1, tf.nn.relu, name='dense_{}_{}'.format(name, 0), weight_name='{}_layer1_param'.format(weight_name))
    with tf.name_scope('{}_lhuc_layer_{}'.format(name, 1)):
      origin_output = 2.0 * mio_dense_layer(output, unit2, tf.nn.sigmoid, name='dense_{}_{}'.format(name, 1), weight_name='{}_layer2_param'.format(weight_name))
      final_outputs.append(origin_output)
      for i, extra_input in enumerate(extra_inputs):
        extra_output = 2.0 * mio_dense_layer(output, extra_input, tf.nn.sigmoid, name='extra_{}_{}'.format(name, i), weight_name='kernel_extra_{}_{}_layer2_param'.format(weight_name, i))
        final_outputs.append(extra_output)
    return tf.concat(final_outputs, 1)

def mio_dense_layer(i, units, activation, name, weight_name,
                    use_bias=True, extra_inputs=[]):
  # tf.Dense-like layer similar to that of mio-dnn
  with tf.name_scope(name):
    rows = tf.shape(i)[0]
    if use_bias:
      weight = tf.get_variable(weight_name, (i.get_shape()[1]+1, units))
      bias_input = tf.fill([rows, 1], 1.0, name='bias_input')
      o = tf.matmul(tf.concat([i, bias_input], 1), weight, name=name + '_mul')

      with tf.variable_scope(name):
        for i, extra_input in enumerate(extra_inputs):
          extra_kernel = tf.get_variable(extra_name.format(i), (extra_input.get_shape()[1], units))
          o += tf.matmul(extra_input, extra_kernel)
    else:
      weight = tf.get_variable(weight_name, (i.get_shape()[1], units))
      o = tf.matmul(i, weight, name=name + '_mul')
    if activation is not None:
      o = activation(o)
    return o


def fc_layer(loss_name, net, hidden_units, last_unit = None, norm = False):
  with tf.variable_scope("{}_fc_layer".format(loss_name), reuse=tf.AUTO_REUSE):
      for i, hidden_unit in enumerate(hidden_units) :
          net = tf.layers.dense(net, hidden_unit, activation=tf.nn.relu)
          #net = tf.clip_by_value(net, eps, 1 - eps)
          #net = tf.layers.batch_normalization(net)
      if norm :
        net = tf.layers.batch_normalization(net)
        #net = KaiLayerNorm(inputs=net, center=True, scale=True)
      if last_unit == 1:
        net = tf.layers.dense(net, last_unit, activation=None)
      elif last_unit != None:
        net = tf.layers.dense(net, last_unit, activation=None)
  return net

def get_weight(weight_name, shape):
    weight = tf.get_variable(weight_name, shape)
    return weight

#[input + bias]*weight
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
        #net2 = tf.concat([tf.stop_gradient(net), net2], 1)
        net2 = tf.concat([net, net2], 1)
        for i, hidden_unit in enumerate(hidden_units):
            #if i == 0 or i == len(hidden_units) - 1:
            weights = gate_layer(loss_name + str(i) + "_layer", net2, dim) # 1遍dnn后获取gate的weight权重
            net = tf.multiply(weights, net)
            net = dense_layer(net, hidden_unit, loss_name + '_' + str(i), act = tf.nn.relu)
            #net = tf.layers.batch_normalization(net)
            dim = hidden_unit
        #net = tf.layers.dense(net, last_dim, activation=None)
        #if kuiba_utils.train_mode():
        #    tf.summary.histogram(loss_name + "_output", net)
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
    return 0.5*x*(1+tf.tanh(math.sqrt(2/math.pi)*(x+0.044715*tf.pow(x, 3))))

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

def _attn(q, k, v, train=False, scale=False):
    w = tf.matmul(q, k)

    if scale:
        n_state = shape_list(v)[-1]
        w = w*tf.rsqrt(tf.cast(n_state, tf.float32))

    w = mask_attn_weights(w)
    w = tf.nn.softmax(w)

    w = dropout(w, attn_pdrop, train)

    a = tf.matmul(w, v)
    return a


def attention_fun(Q, K, loss, mask=None):
    with tf.variable_scope(loss + "_attention", reuse=tf.AUTO_REUSE):
      attention = tf.matmul(Q, K, transpose_b=True)  # [batch_size, sequence_length, sequence_length]
      d_k = tf.cast(tf.shape(K)[-1], dtype=tf.float32)
      attention = tf.divide(attention, tf.sqrt(d_k))  # [batch_size, sequence_length, sequence_length]

      if mask :
        attention = mask_attn_weights(attention, mask[0], mask[1])
      attention = tf.nn.softmax(attention, axis=-1)  # [batch_size, sequence_length, sequence_length]
      return attention

def self_attention(loss, query, data, n, nh, dim, mask):
    scope = loss + "_self_attention"
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
      batch_size = tf.shape(data)[0]
      #query = tf.reshape(query, [batch_size, n, query_dim])
      #data = tf.reshape(data, [batch_size, m, 64])
      querys = tf.layers.dense(query, nh * dim)  # [batch_size, query_length, hidden_dim]
      keys = tf.layers.dense(data, nh * dim)  # [batch_size, sequence_length, hidden_dim]
      values = tf.layers.dense(data, nh * dim)  # [batch_size, sequence_length, n_classes]

      #Q = tf.get_variable(scope + '_q_trans_matrix', (dim, dim * nh))  # [emb, att_emb * hn]
      #K = tf.get_variable(scope + '_k_trans_matrix', (dim, dim * nh))
      #V = tf.get_variable(scope + '_v_trans_matrix', (dim, dim * nh))
      #querys = tf.tensordot(query, Q, axes=(-1, 0))  # (batch_size,sq_q,att_embedding_size*head_num)
      #keys = tf.tensordot(data, K, axes=(-1, 0))
      #values = tf.tensordot(data, V, axes=(-1, 0)) # (batch_size,sq_v,att_embedding_size*head_num)

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

def transformer(loss, query, data, n, nh = 1, dim = 256, mask = None):
    scope = loss + '_transformer'
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        batch_size = tf.shape(data)[0]
        a = self_attention(scope, query, data, n, nh, dim, mask = mask)
        a = dense_layer(a, dim, scope + "_proj")
        #a = dropout(a, resid_pdrop, kuiba_utils.train_mode())
        output = norm(data + a, scope + '_ln_1')
        m = mlp(scope + "_mlp", output, dim * 4)
        output = norm(output + m, scope + '_ln_2')
        #output = tf.reshape(output, [batch_size, n, nh * dim]) # [b, 8, 30 * 16]
        #output = tf.reduce_sum(output, 1)
        #output = tf.layers.dense(output, 64, activation=None)
    #if kuiba_utils.predict_mode():
    #    output = tf.identity(output, loss + '_embedding')
    return output

def user_tower(loss, query, data, query_dim, nh = 1, dim = 128):
    scope = loss + "_user_tower"
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        batch_size = tf.shape(data)[0]
        query = tf.reshape(query, [batch_size, 1, query_dim])
        output = self_attention(scope, query, data, 1, nh, dim, mask = False) # [b,  nh * dim]
        output = tf.reshape(output, [batch_size, nh * dim]) # [b, 8, 30 * 16]
        output = tf.layers.dense(output, dim, activation=tf.nn.relu)
        #output = fc_layer(scope, output, [256, 128], activation = tf.nn.relu)
        #output = tf.layers.dense(output, 256, activation=tf.nn.relu)
        #output = tf.layers.dense(output, 128, activation=tf.nn.relu)
        #output = tf.layers.dense(output, 128, activation=tf.nn.relu)
    return output

def attention_layer(loss, query, data, query_dim, nh = 1, dim = 128):
    scope = loss + "_user_tower"
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        batch_size = tf.shape(data)[0]
        query = tf.reshape(query, [batch_size, 1, query_dim])
        query = tf.tile(query, [1,30,1])
        output = self_attention(scope, query, data, 30, nh, dim, mask = False) # [b,  nh * dim]
        output = tf.layers.dense(output, dim, activation=tf.nn.relu)
    return output

def LSTM(loss, input_x, n_hidden, init_state = None):
    '''
    返回静态单层GRU单元的输出，以及cell状态
    args:
        input_x:输入张量 形状为[batch_size,n_steps,n_input]
        n_steps:时序总数
        n_hidden：gru单元输出的节点个数 即隐藏层节点数
    '''
    scope = loss + "_lstm"
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
      #把输入input_x按列拆分，并返回一个有n_steps个张量组成的list 如batch_sizex28x28的输入拆成[(batch_size,28),((batch_size,28))....] 
      #如果是调用的是静态rnn函数，需要这一步处理   即相当于把序列作为第一维度 
      #gru_cell = tf.contrib.rnn.GRUCell(num_units=n_hidden)
      #gru_cells = [tf.nn.rnn_cell.GRUCell(n) for n in [n_hidden, n_hidden]]
      #stacked_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(gru_cells)
      #静态rnn函数传入的是一个张量list  每一个元素都是一个(batch_size,n_input)大小的张量 
      #hiddens,states = tf.contrib.rnn.static_rnn(cell=stacked_rnn_cell,inputs=input_x, initial_state = init_state, dtype=tf.float32)
      gru_cells = [tf.nn.rnn_cell.GRUCell(n) for n in [n_hidden]]
      stacked_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(gru_cells)
      #静态rnn函数传入的是一个张量list  每一个元素都是一个(batch_size,n_input)大小的张量
      if init_state != None:
        hiddens,states = tf.nn.dynamic_rnn(cell=stacked_rnn_cell,inputs=input_x, initial_state = init_state, dtype=tf.float32)
      else:
        hiddens,states = tf.nn.dynamic_rnn(cell=stacked_rnn_cell,inputs=input_x, dtype=tf.float32)
          
    return hiddens

def cross_layer(name, x0):
  with tf.variable_scope("{}_cross_layer".format(name), reuse=tf.AUTO_REUSE):
    input_dim = x0.get_shape().as_list()[-1]
    w = tf.get_variable("weight", [input_dim], initializer=tf.truncated_normal_initializer(stddev=0.01))
    b = tf.get_variable("bias", [input_dim], initializer=tf.truncated_normal_initializer(stddev=0.01))
    xb = tf.tensordot(tf.reshape(x0, [-1, 1, input_dim]), w, 1)
    return x0 * xb + b + x0

# action list和候选做个co-attention
def Set2ListInteraction(name, candidates, data, historys, atten, dx = 64, dh = 80, n=6, m=30):
  with tf.variable_scope("{}_set2list_layer".format(name), reuse=tf.AUTO_REUSE):
    S = tf.concat([candidates, data], axis = 2)
    L = tf.concat([historys, atten], axis = 2)
    WIA = tf.get_variable("WIA", [2*dx, 2*dh], initializer=tf.truncated_normal_initializer(stddev=0.01))
    CIA = tf.matmul(tf.tensordot(S, WIA, 1), L, transpose_b = True)
    CIA = tf.nn.tanh(CIA)
    WS = tf.get_variable("WS", [2*dx, n], initializer=tf.truncated_normal_initializer(stddev=0.01))
    WL = tf.get_variable("WL", [2*dh, n], initializer=tf.truncated_normal_initializer(stddev=0.01))
    QS = tf.nn.tanh(tf.tensordot(S, WS, 1) + tf.matmul(CIA, tf.tensordot(L, WL, 1)))
    AS = tf.nn.softmax(QS)
    QL = tf.nn.tanh(tf.matmul(tf.tensordot(S, WS, 1), CIA))
    AL = tf.nn.softmax(QL)
    Sp = tf.matmul(AS, S)
    Lp = tf.matmul(AL, L)
    return Sp, Lp

def ctr_model(parameters_dict, loss_name):
  parameters_names = get_parameter_names_by_loss_name(loss_name)
  parameters = [parameters_dict[name] for name in parameters_names]
  user_fea_num = 2
  photo_fea_num = 2

  user = tf.concat(parameters[ : user_fea_num], axis = 1)
  data = tf.concat(parameters[user_fea_num : ], 1)
  #
  with tf.variable_scope("ctr_model", reuse=tf.AUTO_REUSE):
    output = fc_layer('ctr_dnn', tf.concat([user, data], 1), [128, 64, 32], 1)
    output = tf.sigmoid(output)

  if args.mode=="predict":
    output = tf.identity(output, "{}".format(loss_name))
  return {loss_name : output} 

def sample_gumbel(shape, eps = 1e-20): 
  U = tf.random_uniform(shape, minval=0, maxval=1)
  return -tf.log(-tf.log(U + eps) + eps)

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

def slide_ctr_model(parameters_dict, loss_name):
  parameters_names = get_parameter_names_by_loss_name(loss_name)
  parameters = [parameters_dict[name] for name in parameters_names]
  user_fea_num = 17 
  photo_fea_num = 21 

  user = tf.concat(parameters[0 : user_fea_num], 1)
  parameters = parameters[user_fea_num : ]
  rown = tf.shape(parameters[0])[0]

  names = loss_name.split('_')
  name = 'hot_ctr'
  n = (int) (names[-1])
  if n == 2:
    name = 'splash_ctr'

  output_dict = {}
  labels = []
  data = []
  ctr_labels = []
  preds = []
  weights = []
  gates = []
  one = tf.ones([rown, 1])
    
  batch = photo_fea_num + 2
  assert( batch * n == len(parameters))
  n_s = 5
  with tf.variable_scope("slide_model", reuse=tf.AUTO_REUSE):
    for i in range(n):
      p = parameters[i * batch : (i+1) * batch]
      ctr_labels.append(p[-2])
      data0 = tf.concat(p[0:-2], 1)
      weights.append(p[-1])
      
      #bias = gate_tower(name + '_vtr_dnn' + str(i), data0, [128], user)
      input3 = simple_dense_network(tf.concat([data0, user], -1), [256, 128, 64], name + '_item_dnn', name + '_item_dnn{}_param')
      data.append(input3)
    
    data = tf.concat(data, 1)
    #data = tf.reshape(data, (rown, n, 64))
    #scores = transformer(name, data, data, n, nh = 1, dim = 64, mask = [-1, 0])
    #scores = LSTM(name, data, 64)
    #scores = simple_dense_network(tf.concat((user, atten_output, [32], name + '_item_proj', name + '_item_proj{}_param')

    scores = tf.reshape(data, (rown, n, 64))
    scores = simple_dense_network(scores, [32], name + '_item_proj', name + '_item_proj{}_param')
    scores = dense_layer(scores, 1, name + '_output_logit', act = None) 


    scores = tf.reshape(scores, (rown, n, 1))
    #scores_sample = tf.tile(scores, [n_s, 1, 1])
    #scores_sample += sample_gumbel([rown * n_s, n, 1])
    #scores_sample = neuralsort(scores_sample, tau = 1)
    #scores_sample = neuralsort(scores, tau = 1)
    #labels = tf.reshape(tf.concat(labels, -1), (rown, n, 1))
    #labels_sample = neuralsort(labels, tau = 1e-10)
    
    #labels_sample = tf.one_hot(tf.concat(labels, -1), n) 
    #labels_sample = tf.tile(labels, [n_s, 1, 1])

    losses = []
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    #        labels=labels_sample, logits=tf.log(scores_sample + 1e-20), dim=2), -1)
    #losses.append(tf.reduce_mean(loss))

    scores = tf.reshape(scores, (rown, n))
    scores = tf.split(scores, [1] * n, 1)

    for i in range(n):
      scores[i] = tf.sigmoid(scores[i])
      losses.append(tf.losses.log_loss(labels=ctr_labels[i], predictions=scores[i], reduction=tf.losses.Reduction.SUM))
        
  output_dict['loss'] = [tf.reduce_sum(tf.stack(losses))]
  output_dict['preds'] = [scores[0]]
  output_dict['labels'] = [ctr_labels[0]]
  output_dict['q_names'] = [loss_name]
  #if (args.mode == 'predict'):
  #  for i in range(n):
  #    output = scores[i]
  #    if name == 'splash_ctr':
  #      out_name = 'splash_pos'
  #    else:
  #      out_name = 'pos'
  #    output = tf.identity(output, "{}".format(out_name + str(i)))
  #    output_dict[out_name + str(i)] = output
  return output_dict

def get_pos_embedding(name, pos_index, batch_size, dim=64):
  with tf.variable_scope(name+"pos_embedding", reuse=tf.AUTO_REUSE):
    embedding_param = tf.get_variable(name="pos_embedding", shape=[6, dim])
    ids = tf.ones([batch_size, 1], name="id_{}".format(pos_index), dtype=tf.int32) * pos_index
    embedding = tf.nn.embedding_lookup(ids=ids, params=embedding_param)
    embedding = tf.reshape(embedding, shape=[-1, dim])
    return embedding

def pxtr_trans_layer(feature_name, pxtr_dense):
    # 记录个性化修正前后的分布情况
    import math
    with tf.variable_scope("function_" + feature_name, reuse=tf.AUTO_REUSE):
      pxtr_clip = tf.clip_by_value(pxtr_dense, clip_value_min=0.0, clip_value_max=300.0)  # filter ilegal
      pxtr_bn = tf.math.subtract(pxtr_clip, tf.tile(tf.math.reduce_min(pxtr_clip, axis=0, keepdims=True),
                                                    [tf.shape(pxtr_clip)[0], 1])) #减最小值
      pxtr_bn_bn = tf.math.divide(pxtr_bn,
                                  tf.tile(tf.math.reduce_max(pxtr_bn, axis=0, keepdims=True) + 0.000000001,
                                          [tf.shape(pxtr_bn)[0], 1])) #除以最大值

      log_bn_pxtr = 1.0 + pxtr_bn
      log_v2 = tf.log(log_bn_pxtr) #log

      tanh_bn_scale_pxtr = pxtr_bn_bn * 4.0
      tanh_v3 = tf.nn.tanh(tanh_bn_scale_pxtr) #tanh

      reverse_pxtr = -1.0 * tf.log(1 / (pxtr_clip + 1.0))
      sigmoid_v1 = tf.sigmoid(reverse_pxtr * 1.0) #sigmoid

      sin_bn_pxtr = pxtr_bn_bn * math.pi / 2
      sin_v2 = tf.math.sin(sin_bn_pxtr) #sin

      power_v1 = tf.pow(pxtr_clip, 0.5)
      power_v2 = tf.pow(pxtr_clip, 2.0)
      exp_v1 = tf.pow(2.0, -1.0 * pxtr_clip) #pow

      pxtr_gated_list = [log_v2, tanh_v3, sigmoid_v1, sin_v2, power_v1, power_v2, exp_v1, pxtr_dense]
      pxtr_dense_gated = tf.stack(pxtr_gated_list, axis=2)

      pxtr_dense_gated = tf.layers.dense(pxtr_dense_gated, 1, activation=None, name="score_{}".format(feature_name))
      pxtr_dense_gated = tf.squeeze(pxtr_dense_gated, axis=2)

      return pxtr_dense_gated

def slide_model(parameters_dict, loss_name):
  parameters_names = get_parameter_names_by_loss_name(loss_name)
  parameters = [parameters_dict[name] for name in parameters_names]
  user_fea_num = 17 
  photo_fea_num = 39 

  n = 30 
  #real_show_list，点击aid_list，tag_list，play_list
  click = tf.concat(parameters[0 : n], 1)
  click_aid = tf.concat(parameters[n : n * 2], 1)
  click_tag = tf.concat(parameters[n * 2 : n * 3], 1)
  click_play = tf.concat(parameters[n * 3: n * 4], 1)
  #用户侧基本特征
  user = tf.concat(parameters[n * 4: n * 4 + user_fea_num], axis = 1)
  
  parameters = parameters[n * 4 + user_fea_num : ]

  rown = tf.shape(user)[0]
  click = tf.reshape(click,  (rown, n, 32))
  click_aid = tf.reshape(click_aid,  (rown, n, 32))
  click_tag = tf.reshape(click_tag,  (rown, n, 8))
  click_play = tf.reshape(click_play,  (rown, n, 8))
  atten = tf.concat([click, click_aid, click_tag, click_play], 2) #[b,30,80]
  
  #rown = tf.shape(parameters[0])[0]

  names = loss_name.split('_')
  name = loss_name
  n = (int) (names[-1])
  if n == 2:
    name = 'splash_l2r'


  output_dict = {}
  labels = []
  data = []
  ctr_labels = []
  preds = []
  weights = []
  gates = []
  play_weights = []
  one = tf.ones([rown, 1])
  label_num = 3
  if name == 'slide_next_6':
    label_num = 2
  batch = photo_fea_num + label_num #item特征+weight+label
  assert( batch * n == len(parameters)) # n是item的个数
  n_s = 5
  with tf.variable_scope("slide_model", reuse=tf.AUTO_REUSE):
    if name == 'slide_next_6':
      user = tf.stop_gradient(user)
      atten = tf.stop_gradient(atten)
    # atten = user_tower(name+"_ctr_user", user, atten, query_dim = 320, nh = 1, dim = 80)
    # query = tf.concat([user, atten], 1)
    for i in range(n):
      p = parameters[i * batch : (i+1) * batch]
      ctr_label = p[-(label_num-1)]
      ctr_labels.append(ctr_label)
      interact_pxtr = tf.concat(p[0:12], 1) #item的互动xtr特征
      watchtime_pxtr = tf.concat(p[12:17], 1) #item的时长特征
      data0 = tf.concat(p[17:-label_num], 1)  #item基础特征
      pos_emb = p[-(label_num+1)] # 位置特征
      if name == 'slide_next_6':
        interact_pxtr = tf.stop_gradient(interact_pxtr)
        watchtime_pxtr = tf.stop_gradient(watchtime_pxtr)
        data0 = tf.stop_gradient(data0)
        pos_emb = tf.stop_gradient(pos_emb)
      #labels.append(tf.cast(p[-1], tf.int32))
      play_wt = 1.0
      if name == 'slide_l2r_6' :
         play_weights.append(p[-1])
      #   play_wt = 1/(1 + tf.pow(1.2,-tf.maximum(play_wt,1))) + 0.2
      #   play_wt = tf.where(tf.greater(ctr_label,0), play_wt,1/play_wt)
      weights.append(p[-label_num])
      
      query = tf.concat([user, pos_emb, data0], 1) #用户侧基本特征+位置特征+item侧基本特征
      query_gate = simple_dense_network(query, [256, 128, 64], name + '_item_dnn', name + '_item_dnn{}_param') # 3层dnn [b,64]
      data1 = gate_tower(name + '_vtr_dnn1', query_gate, [64], interact_pxtr) #[b,64] 
      data2 = gate_tower(name + '_vtr_dnn2', query_gate, [64], watchtime_pxtr)#[b,64]
      gate = gate_layer(name + '_vtr_gate', query, 64)#[b,64]
      input2 = data1 * gate + data2 * (1 - gate)#[b,64] #门控单元控制互动和时长的贡献
      pxtr_dense = tf.concat([interact_pxtr, watchtime_pxtr], axis =1)
      pxtr_cross = cross_layer(name+"_cross", pxtr_dense) #[时长，互动]进行2阶交叉
      pxtr_trans = pxtr_trans_layer(name + "_trans", pxtr_dense)  #[时长，互动]各种转化
      data3 = gate_tower(name + '_vtr_dnn3', query_gate, [64], tf.concat([pxtr_cross, pxtr_trans], axis=1))
      gate2 = gate_layer(name + '_vtr_gate2', query, 64)
      input3 = input2 * gate2 + data3 * (1 - gate2)
      #input3 = simple_dense_network(input2, [64], name + str(i) + '_item_dnn', name + str(i) + '_item_dnn{}_param')
      data.append(input3)
    
    data = tf.concat(data, 1)
    data = tf.reshape(data, (rown, n, 64))
    candidates = transformer(name, data, data, n, nh = 1, dim = 64, mask = None)
    historys = attention_layer(name+"_attention_layer", user, atten, query_dim = 320, nh = 1, dim = 80)
    S = tf.concat([candidates, data], axis = 2)
    L = tf.reshape(tf.concat([historys, atten], axis = 2),(rown, n, 800))
    scores = simple_dense_network(tf.concat([S, L],axis = 2), [32], name + '_item_proj', name + '_item_proj{}_param')
   
    # Sp, Lp = Set2ListInteraction(name, candidates, data, historys, atten)
    # scores = tf.concat([Sp, Lp], axis=2)
    #scores = LSTM(name, data, 64)
    #scores = simple_dense_network(tf.concat((user, atten_output, [32], name + '_item_proj', name + '_item_proj{}_param')
    # scores = simple_dense_network(scores, [64, 32], name + '_item_proj', name + '_item_proj{}_param')
    
    scores = dense_layer(scores, 1, name + '_output_logit', act = None) 

    scores = tf.reshape(scores, (rown, n, 1))
    #scores_sample = tf.tile(scores, [n_s, 1, 1])
    #scores_sample += sample_gumbel([rown * n_s, n, 1])
    #scores_sample = neuralsort(scores_sample, tau = 1)
    #scores_sample = neuralsort(scores, tau = 1)
    
    #labels = tf.reshape(tf.concat(labels, -1), (rown, n, 1))
    #labels_sample = neuralsort(labels, tau = 1e-10)
    
    #labels_sample = tf.one_hot(tf.concat(labels, -1), n) 
    #labels_sample = tf.tile(labels, [n_s, 1, 1])

    losses = []
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    #        labels=labels_sample, logits=tf.log(scores_sample + 1e-20), dim=2), -1)
    #losses.append(tf.reduce_mean(loss) * 0.1)

    scores = tf.reshape(scores, (rown, n))
    scores = tf.split(scores, [1] * n, 1)
    logits = scores
    if name == 'slide_next_6':
      for i in range(n):
        scores[i] = tf.sigmoid(scores[i])
        # losses.append(-tf.reduce_sum(tf.log(tf.clip_by_value(scores[i], 1e-10, 0.9999))*weights[i]-100*scores[i]))
        losses.append(tf.losses.log_loss(labels=ctr_labels[i], predictions=scores[i], weights = weights[i], reduction=tf.losses.Reduction.SUM))
        # tp1 = -tf.reduce_sum(tf.log(tf.clip_by_value(scores[i], 1e-10, 0.9999))*weights[i]-100*scores[i])
        # tp2 = tf.losses.log_loss(labels=ctr_labels[i], predictions=scores[i], weights = weights[i], reduction=tf.losses.Reduction.SUM)
        # tp1 = tf.Print(tp1, ["cying_tp1:"+name, tp1], summarize=-1)
        # tp2 = tf.Print(tp2, ["cying_tp2:"+name, tp2], summarize=-1)
        # tf.summary.histogram('tp1', tp1, family="weight")
        # tf.summary.histogram('tp2', tp2, family="weight")
    else:
      for i in range(n):
        scores[i] = tf.sigmoid(scores[i])
        losses.append(-tf.reduce_sum(tf.log(tf.clip_by_value(scores[i], 1e-10, 0.9999))*play_weights[i]-100*scores[i]))


  output_dict['loss'] = [tf.reduce_sum(tf.stack(losses))]
  output_dict['preds'] = [scores[0]]
  output_dict['labels'] = [ctr_labels[0]]
  output_dict['q_names'] = [loss_name]
  output_dict['weights'] = [weights[0]]
  if name == 'slide_l2r_6':
    output_dict['play_weights'] = tf.concat(play_weights, 0)
  if (args.mode == 'predict'):
    for i in range(n):
      output = scores[i]
      if name == 'splash_l2r':
        out_name = 'splash_pos'
      elif name == 'slide_l2r_6':
        out_name = 'pos'
      else:
        out_name = 'next'
      output = tf.identity(output, "{}".format(out_name + str(i)))
      output_dict[out_name + str(i)] = output
  return output_dict

def slide_point_model(parameters_dict, name, loss_name):
  parameters_names = get_parameter_names_by_loss_name(name)
  parameters = [parameters_dict[name] for name in parameters_names]
  user_fea_num, photo_fea_num = 31, 43
  rown, name = tf.shape(parameters[0])[0], loss_name
   
  # 用户基础特征
  user = tf.concat(parameters[0: 2], axis = 1)
  user_basic =  tf.concat(parameters[0: 8], axis = 1)
  user_exp_action_count =  tf.concat(parameters[8: 13], axis = 1)
  user_realtime_pids_list = tf.concat(parameters[13: 19], axis = 1) 
  user_fountain_pids_list = tf.concat(parameters[19: 24], axis = 1)
  user_all_pids_list = tf.concat(parameters[24: 31], axis = 1)
  user_hetu_list = tf.concat(parameters[29: 31], axis = 1)
  
  if name == 'slide_point_next':
      user = tf.stop_gradient(user)
      user_basic = tf.stop_gradient(user_basic)
      user_exp_action_count  =  tf.stop_gradient(user_exp_action_count)
      user_realtime_pids_list  =  tf.stop_gradient(user_realtime_pids_list)
      user_fountain_pids_list  =  tf.stop_gradient(user_fountain_pids_list)
      user_all_pids_list  =  tf.stop_gradient(user_all_pids_list)
      user_hetu_list  =  tf.stop_gradient(user_hetu_list)

  user_hetu_dnn_list = simple_dense_network(user_hetu_list, [64], name+'_user_hetu_list_dnn', name+'_user_hetu_list__dnn{}_param') 

  #user_all_pids_list ,user_realtime_pids_list 执行self attention；user_fountain_pids_list进行target attention
  # realtime_pids_reshape = tf.reshape(user_realtime_pids_list,[rown, 120, 32])
  # realtime_pids = mu.multi_head_attention(realtime_pids_reshape, realtime_pids_reshape, mask=False,scope="mulhead_realtime_pids_list")
  # realtime_pids = tf.reduce_mean(realtime_pids,1)
  # user_all_pids_reshape = tf.reshape(user_all_pids_list,[rown, 350, 32])
  # all_pids = mu.multi_head_attention(user_all_pids_reshape, user_all_pids_reshape, mask=False,scope="mulhead_all_pids_list")
  # all_pids = tf.reduce_mean(all_pids,1)
  
  n = 6 # 训练每条样本6条pid信息，预测每条样本1条pid
  if (args.mode == 'predict'):
    n = 1
  label_num = 3
  if name == 'slide_point_next':
    label_num = 2
  batch = photo_fea_num + label_num
  output_dict, labels, data, querys, ctr_labels, weights, play_weights, pvtrs, pwtds, preds, gates, losses = {}, [], [], [],[], [], [],[], [], [], [], []

  # pid基础特征
  parameters = parameters[user_fea_num : ]
  with tf.variable_scope("slide_point_model", reuse=tf.AUTO_REUSE):
    for i in range(n):
      p = parameters[i * batch : (i+1) * batch]
      weights.append(p[-label_num])
      ctr_labels.append(p[-(label_num-1)])
      if name == 'slide_point_l2r' :
        play_weights.append(p[-1])
      interact_pxtr = tf.concat(p[0:12], 1) #item的互动xtr特征
      pvtrs.append(p[12])
      watchtime_pxtr = tf.concat(p[12:18], 1) #item的时长特征
      # interact_buck_pxtr = tf.concat(p[18:29], 1)
      # watchtime_buck_pxtr = tf.concat(p[29:34], 1) 
      watchtime_buck_pxtr = p[18]
      exp_pxtr = tf.concat(p[19:27], 1) 
      pid = tf.concat(p[27:29], 1) 
      pid_basic = tf.concat(p[27:-(label_num+1)], 1)  #item基础特征
      pid_hetu_info = tf.concat(p[36:42], 1)  #hetu特征
      pos_emb = p[-(label_num+1)] # 位置特征
      if name == 'slide_point_next':
        interact_pxtr = tf.stop_gradient(interact_pxtr)
        watchtime_pxtr = tf.stop_gradient(watchtime_pxtr)
        # interact_buck_pxtr = tf.stop_gradient(interact_buck_pxtr)
        watchtime_buck_pxtr = tf.stop_gradient(watchtime_buck_pxtr)
        exp_pxtr = tf.stop_gradient(exp_pxtr)
        pid = tf.stop_gradient(pid)
        pid_basic = tf.stop_gradient(pid_basic)
        pos_emb = tf.stop_gradient(pos_emb)
      
      play_wt = 1.0
      # if name == 'slide_point_l2r' and args.mode == 'train':
      #   play_wt = p[-1] 
      #   play_wt = 1/(1 + tf.pow(1.2,-tf.maximum(play_wt,1))) + 0.2
      #   play_wt = tf.where(tf.greater(ctr_label,0), play_wt, tf.ones_like(play_wt))
      

      # pxtr numric parse --细化解析方式
      pxtr_dense = tf.concat([interact_pxtr, watchtime_pxtr], axis =1)
      pxtr_cross = cross_layer(name+"_cross", pxtr_dense) #[时长，互动]进行2阶交叉
      pxtr_trans = pxtr_trans_layer(name + "_trans", pxtr_dense)  #[时长，互动]各种转化
      
      pxtr_autodis = mu.gen_autodis_net(tf.concat([interact_pxtr,watchtime_pxtr],axis=-1), 18, 3) #autodis
      pxtr_autodis = tf.reduce_mean(pxtr_autodis,1)

      # pxtr emb parse
      # interact_buck_pxtr = tf.concat(p[18:29], 1)
      # watchtime_buck_pxtr = tf.concat(p[29:34], 1) 
      # exp_pxtr = tf.concat(p[34:42], 1) 
      pid_buck_pxtr = tf.concat([watchtime_buck_pxtr],1)

      # 交叉特征
      basic_cross = cross_layer(name+"_basic_cross", tf.concat([user,pid_basic],axis=1)) #[pid基础特征，用户基础特征]进行2阶交叉
      # buck_pxtr_cross = tf.reshape(tf.matmul(tf.reshape(user_exp_action_count, [rown,5,8]) , tf.transpose(tf.reshape(pid_buck_pxtr, [rown,1,8]),(0,2,1))),[rown,80])
      exp_pxtr_cross = tf.reshape(tf.matmul(tf.reshape(user_exp_action_count, [rown,5,8]) , tf.transpose(tf.reshape(exp_pxtr, [rown,8,8]),(0,2,1))),[rown,40])
      hetu_cross = tf.reshape(tf.matmul(tf.reshape(user_hetu_dnn_list, [rown,8,8]) , tf.transpose(tf.reshape(pid_hetu_info, [rown,6,8]),(0,2,1))) ,[rown,48])

      #target attention
      user_fountain_pids_lists = tf.split(user_fountain_pids_list,5,axis=1)
      user_fountain_pids_multi_rel = []
      for i,fountain_pids in enumerate(user_fountain_pids_lists):
        fountain_pids = tf.reshape(fountain_pids, (rown, 50, 32))
        target_fountain_his = mu.multi_head_attention(tf.expand_dims(pid_basic,1), fountain_pids, mask=False, scope="user_fountain_pids_lists_{}".format(i))
        user_fountain_pids_multi_rel.append(tf.reduce_mean(target_fountain_his,1))

      #所有特征
      all_features = tf.concat([user_basic,user_exp_action_count,user_realtime_pids_list,user_all_pids_list, 
              pid_basic, pxtr_dense, pxtr_cross, pxtr_trans, pxtr_autodis, watchtime_buck_pxtr,exp_pxtr,
              basic_cross,exp_pxtr_cross]+user_fountain_pids_multi_rel, 1) #用户侧基本特征+位置特征+item侧基本特征
      #transformer 使用pid和user当成slot
      all_features_gate = gate_tower(name + '_all_features_dnn1', all_features,  [512, 256, 128, 64], tf.concat([user,pid], 1)) #[b,64] 

      data1 = gate_tower(name + '_vtr_dnn1', tf.concat([interact_pxtr,watchtime_pxtr],axis=1), [64], all_features_gate) #[b,64] 
      data3 = gate_tower(name + '_vtr_dnn3', tf.concat([pxtr_cross, pxtr_trans],axis=1), [64], all_features_gate)
      data4 = gate_tower(name + '_vtr_dnn4', pid_buck_pxtr, [64], all_features_gate) #[b,64] 
      data5 = gate_tower(name + '_vtr_dnn5', exp_pxtr, [64], all_features_gate) #[b,64] 

      output = tf.concat([all_features_gate, data1, data3, data4, data5], axis = -1)
      data.append(output)
    input = tf.concat(data, 0)
    query_gate = simple_dense_network(input, [256, 128, 64], name + '_dnn', name + '_dnn{}_param') #[b,64]
    logit = dense_layer(query_gate, 1, name + '_output_logit', act = None) 
    pred = tf.sigmoid(logit) 
    
  output_dict['pred'] = pred
  output_dict['logit'] = logit
  output_dict['labels'] = tf.concat(ctr_labels, 0)
  output_dict['weights'] = tf.concat(weights, 0)
  if name == 'slide_point_l2r' :
    output_dict['play_weights'] = tf.concat(play_weights, 0)
  
  if (args.mode == 'predict'):
    if name == "slide_point_next" :
      out_name = "point_next"
    else :
      out_name = "point_pos"
    output = tf.identity(pred, out_name)
    output_dict[out_name] = output
  return output_dict


loss_model_dict = {
  # "slide_l2r_6" : slide_model,
  # "slide_next_6" : slide_model,
}

loss_point_dict = {
  "slide_point_l2r" : slide_point_model,
  "slide_point_next"  : slide_point_model
}

if (args.mode == 'train'):
  targets, loss_tensor_dict, dense_loss_tensor_dict, point_loss, sum_loss = [], {}, {}, {}, 0.0
  
  # point model
  for loss_name ,model in loss_point_dict.items():
    if "next" in loss_name:
      parameters_dict, join_parameters_size_dict, label_dict, label_value_dict, weight, sample_flag = get_kuiba_loss_relative("slide_next_6", True, True, True)
      point_output = model(parameters_dict, "slide_next_6", loss_name)
      preds_point = point_output['pred']
      labels_point = point_output['labels']
      weights_point = point_output['weights']
      loss_point = tf.losses.log_loss(labels=labels_point, predictions=preds_point, weights = weights_point, reduction=tf.losses.Reduction.SUM)
      sum_loss += loss_point
      batch_preds_point,_,_,_,_,_ = tf.split(preds_point, 6, 0)
      batch_labels_point,_,_,_,_,_ = tf.split(labels_point, 6, 0)
      weights_point,_,_,_,_,_ = tf.split(weights_point, 6, 0)
      targets.append((loss_name+"_auc", batch_preds_point, batch_labels_point, sample_flag, "auc"))
      targets.append((loss_name+"_lr", batch_preds_point, weights_point, sample_flag, "linear_regression"))
      targets.append((loss_name+"_pos_lr", batch_preds_point, batch_labels_point * weights_point, sample_flag, "linear_regression"))
    else:
      parameters_dict, join_parameters_size_dict, label_dict, label_value_dict, weight, sample_flag = get_kuiba_loss_relative("slide_l2r_6", True, True, True)
      point_output = model(parameters_dict, "slide_l2r_6", loss_name)
      preds_point = point_output['pred']
      labels_point = point_output['labels']
      weights_point = point_output['weights']
      play_weights = point_output['play_weights']
      logit_point = point_output['logit']

      wt_shape = tf.reshape(weights_point, [-1, 6])
      logit_shape =tf.reshape(logit_point, [-1, 6])
      label_shape = tf.reshape(labels_point,[-1,6])
      weights_points, logit_points = [], []
      label_pair, weight_pair, pred_pair = {}, {}, {}
      weights_points=tf.split(wt_shape,[1,1,1,1,1,1],axis = 1)
      logit_points=tf.split(logit_shape,[1,1,1,1,1,1],axis = 1)
      ones_val = tf.ones_like(weights_points[0])
      zero_val = tf.zeros_like(weights_points[0])
      
      for i in range(6):
        targets.append(("pos" + str(i), tf.reshape(tf.nn.sigmoid(logit_shape[:,i]),[-1,1]), tf.reshape(label_shape[:,i],[-1,1]), sample_flag, "auc"))
        targets.append(("wpos" + str(i), tf.reshape(tf.nn.sigmoid(logit_shape[:,i]),[-1,1]), tf.reshape(wt_shape[:,i],[-1,1]), sample_flag, "linear_regression"))
        for j in range(i+1,6):
          label = tf.where(tf.greater(weights_points[i], weights_points[j]), ones_val, zero_val) 
          weight = tf.abs(weights_points[i]/(i+1) + weights_points[j]/(j+1) - weights_points[j]/(i+1) - weights_points[i]/(j+1))
          weight = tf.where(tf.greater_equal(weights_points[i], weights_points[j]) , weight+0.5, tf.sigmoid(weight))
          pred = tf.sigmoid(logit_points[i] - logit_points[j])
          loss_tensor = -tf.reduce_sum(tf.log(tf.clip_by_value(pred, 1e-10, 0.9999))*weight-12*pred)
          # loss_tensor = tf.losses.log_loss(labels = label, predictions = pred, weights = weight, reduction=tf.losses.Reduction.SUM )
          loss_tensor_dict[f"pos_{i}_{j}"] = loss_tensor
          targets.append((loss_name+f"_{i}_{j}_auc", pred, label, sample_flag, "auc"))
          targets.append((loss_name+f"_{i}_{j}_lr",  pred, weight, sample_flag, "linear_regression"))

          label = tf.Print(label, ["cying_label: ", label], summarize=-1)
          weight = tf.Print(weight, ["cying_weight: ", weight], summarize=-1)
          pred = tf.Print(pred, ["cying_pred: ", pred], summarize=-1)
          loss_tensor = tf.Print(loss_tensor, ["cying_loss_tensor: ", loss_tensor], summarize=-1)
          tf.summary.histogram('label', label, family="weight")
          tf.summary.histogram('weight', weight, family="weight")
          tf.summary.histogram('pred', pred, family="weight")
          tf.summary.histogram('loss_tensor', loss_tensor, family="weight")


  for loss_name, model in loss_model_dict.items():
    parameters_dict, join_parameters_size_dict, label_dict, label_value_dict, weight, sample_flag = get_kuiba_loss_relative(loss_name, True, True, True)
    
    xtr_output= model(parameters_dict, loss_name)
    preds = xtr_output['preds']
    labels = xtr_output['labels']
    q_names = xtr_output['q_names']
    loss_tensor = xtr_output['loss']
    weights = xtr_output["weights"]
    if name == 'slide_l2r_6':
      play_weights = xtr_output['play_weights']
    for i, pred in enumerate(preds):
      auc = 'auc'
      #if q_names[i].find('l2r') != -1:
      #  auc = 'linear_regression'
      targets.append((q_names[i]+"_auc", pred, labels[i], sample_flag, "auc"))
      targets.append((q_names[i]+"_lr", pred, weights_point, sample_flag, "linear_regression"))

      if q_names[i] == "slide_next_6":
        sum_loss += loss_tensor[i]
      else:
        targets.append((loss_name+"_play_lr", pred, play_weights, sample_flag, "linear_regression"))
        sum_loss += loss_tensor[i]
  loss_tensor_dict["slide_next_model"] = sum_loss
  sum_loss = tf.Print(sum_loss, ["cying_sum_loss: ", sum_loss], summarize=-1)
  tf.summary.histogram('sum_loss', sum_loss, family="weight")


  if args.with_kai:
      config.dump_kai_training_config('./training/conf', targets, loss=None, text=args.text,
                                      dense_loss=sum_loss_tensor_dict(loss_tensor_dict),
                                      sparse_loss=sum_loss_tensor_dict(loss_tensor_dict),
                                      extra_outputs={'sum_loss': sum_loss,'loss_point':sum_loss})
  else:
      dense_loss=sum_loss_tensor_dict(dense_loss_tensor_dict)
      sparse_loss=sum_loss_tensor_dict(loss_tensor_dict)
      optimizer = tf.train.GradientDescentOptimizer(1, name="opt")
      sparse_grads_and_vars = optimizer.compute_gradients(sparse_loss, tf.get_collection_ref(MioCollections.MIO_EMBEDDINGS))
      print(tf.get_collection_ref(MioCollections.MIO_VARIABLES))
      dense_grads_and_vars = optimizer.compute_gradients(dense_loss, [x for x in tf.get_collection_ref(MioCollections.MIO_VARIABLES) if x.trainable])
      opt1 = optimizer.apply_gradients(sparse_grads_and_vars)
      opt2 = optimizer.apply_gradients(dense_grads_and_vars)
      config.dump_training_config('./training/conf', targets, opts=[opt1, opt2], text=args.text)

if args.mode == 'predict':
    targets = []
    for loss_name, model in loss_point_dict.items():
      if "next" in loss_name:
        name = "slide_next_6"
      else:
        name = "slide_l2r_6"
      parameters_dict, join_parameters_size_dict, label_dict, label_value_dict, weight, sample_flag = get_kuiba_loss_relative(name, False, False, False)
      all_outputs  = slide_point_model(parameters_dict, name,loss_name)
      for (k, v) in all_outputs.items():
        if k.find('next') != -1:
          targets.append((k, v))
        if k.find('pos') != -1:
          targets.append((k, v))

    for loss_name, model in loss_model_dict.items():
      parameters_dict, join_parameters_size_dict, label_dict, label_value_dict, weight, sample_flag = get_kuiba_loss_relative(loss_name, False, False, False)

      xtr_output= model(parameters_dict, loss_name)
      # for (k, v) in xtr_output.items():
      #   if k.find('pos') != -1:
      #     targets.append((k, v))
      #   if k.find('next') != -1:
      #     targets.append((k, v))

    q_names, preds = zip(*targets)
    
    config.dump_predict_config('./predict', targets, input_type=3, extra_preds=q_names)
# if args.mode == 'predict':
#   targets = []
#   for loss_name, model in loss_model_dict.items():
#     parameters_dict, join_parameters_size_dict, label_dict, label_value_dict, weight, sample_flag = get_kuiba_loss_relative(loss_name, False, False, False)

#     xtr_output= model(parameters_dict, loss_name)
#     for (k, v) in xtr_output.items():
#       if k.find('pos') != -1:
#         targets.append((k, v))
#       if k.find('next') != -1:
#         targets.append((k, v))

#   q_names, preds = zip(*targets)
#   config.dump_predict_config('./predict', targets, input_type=3, extra_preds=q_names)


