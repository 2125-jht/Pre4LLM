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

base_config = '/home/mpi/kap_kuiba_to_kai/lintao/base.yaml'
config = MioConfig.from_base_yaml(base_config, clear_embeddings=True, clear_params=True,
                                  dryrun=args.dryrun, label_with_kv=True, grad_no_scale=False,
                                  predict=(args.mode == 'predict'),
                                  with_kai=args.with_kai, kconf="McHotL2RTower")

config_from_kuiba = json.load(open("./kai_kuiba_config.json"))
all_sparse_input_dict = {}
all_dense_input_dict = {}
use_dragonfly_io = True

view_list_dim = [32, 32, 8, 8, 4, 4, 4]
ltr_view_sg_list = [False, False, False, False, False, False, False]
view_list_length = 50

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
  loss_config = config_from_kuiba["loss_functions"][loss];

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
  #if kuiba_utils.predict_mode(): 
  #    net = tf.identity(net, "{}_embedding".format(loss_name)) 
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
        #net2 = tf.concat([tf.stop_gradient(net), net2], 1)
        net2 = tf.concat([net, net2], -1)
        for i, hidden_unit in enumerate(hidden_units):
            #if i == 0 or i == len(hidden_units) - 1:
            weights = gate_layer(loss_name + str(i) + "_layer", net2, dim)
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

def get_pos_embedding(name, pos_index, batch_size, dim=64):
  with tf.variable_scope(name+"pos_embedding", reuse=tf.AUTO_REUSE):
    embedding_param = tf.get_variable(name="pos_embedding", shape=[10, dim])
    ids = tf.ones([batch_size, 1], name="id_{}".format(pos_index), dtype=tf.int32) * pos_index
    embedding = tf.nn.embedding_lookup(ids=ids, params=embedding_param)
    embedding = tf.reshape(embedding, shape=[-1, dim])
    return embedding

def short_net(loss, query, data_list, list_len, list_dim, query_dim, nh = 1,
              dim = 128, stop_gradient_list = [], action_item_size=64, att_emb_size=32):
    scope = loss + "_short_net"
    no_stop_gradient = False
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        batch_size = tf.shape(query)[0]
        new_data_list = []
        for i in range(len(data_list)):
            input_data = data_list[i]
            if (not no_stop_gradient and stop_gradient_list[i]):
              input_data = tf.stop_gradient(data_list[i])
            new_data_list.append(tf.reshape(input_data, [batch_size, list_len, list_dim[i]]))
        data = tf.concat(new_data_list, 2)
        data = tf.layers.dense(data, action_item_size, activation=tf.nn.relu)
        query = tf.reshape(query, [batch_size, 10, query_dim])
        #output = self_attention(scope, query, data, 1, nh, dim) # [b,  nh * dim]
        output = short_transformer(loss, query, data, 10, nh = 1, dim = query_dim) # [b,  nh * dim]
        #output = tf.layers.dense(output, dim, activation=tf.nn.relu)
    return output
 
def fc_one_layer(loss_name, net, hidden_unit, act = None): 
  with tf.variable_scope("{}_fc_one_layer".format(loss_name), reuse=tf.AUTO_REUSE): 
      net = tf.layers.dense(net, hidden_unit, activation=act) 
  return net 

def short_transformer(loss, query, data, n, nh = 1, dim = 256, mask = None): 
    scope = loss + '_transformer' 
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE): 
        batch_size = tf.shape(data)[0] 
        a = self_attention(scope, query, data, n, nh, dim, mask = mask) 
        a = fc_one_layer(scope + "_proj", a, dim) 
        #a = dropout(a, resid_pdrop, kuiba_utils.train_mode())
        data = tf.tile(tf.reshape(tf.reduce_mean(data, axis=1), [batch_size, 1, dim]), [1, n, 1]) 
        output = norm(data + a, scope + '_ln_1') 
        m = mlp(scope + "_mlp", output, dim * 4) 
        output = norm(output + m, scope + '_ln_2') 
        #output = tf.reshape(output, [batch_size, n, nh * dim]) # [b, 8, 30 * 16] 
        #output = tf.reduce_sum(output, 1) 
        #output = tf.layers.dense(output, 64, activation=None) 
    #if kuiba_utils.predict_mode(): 
    #    output = tf.identity(output, loss + '_embedding') 
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

def ctr_model(parameters_dict, loss_name): 
  parameters_name = get_parameter_names_by_loss_name(loss_name) 
  parameters = [parameters_dict[name] for name in parameters_name] 
  user_fea_num = 22
  view_list_num = 7
  photo_fea_num = 47
 
  user = tf.concat(parameters[:user_fea_num], axis = 1) 
  parameters = parameters[user_fea_num + view_list_num : ]
  
  rown = tf.shape(user)[0] 
  n=50
  cl_net_pid = tf.reshape(parameters_dict["l2r_user_view_list_pids"], (rown, n, 32))
  cl_net_aid = tf.reshape(parameters_dict["l2r_user_view_list_aids"], (rown, n, 32))
  cl_net_hetu1 =  tf.reshape(parameters_dict["l2r_user_view_list_hetu1"], (rown, n, 8))
  cl_net_hetu2 = tf.reshape(parameters_dict["l2r_user_view_list_hetu2"], (rown, n, 8))
  cl_input = tf.concat([cl_net_pid, cl_net_aid, cl_net_hetu1, cl_net_hetu2], 2)
    
  cl_input = tf.reduce_mean(cl_input, 1)
 
  weights = [] 
  labels = [] 
  data = [] 
  output_dict = {}
  one = tf.ones([rown, 1]) 
   
  batch = photo_fea_num + 2
  n = 10 
  for i in range(n): 
    p = parameters[i * batch : (i+1) * batch] 
    labels.append(p[-2]) 
    data.append(tf.concat(p[0:-2], 1)) 
    weights.append(one) 
    pos_emb = get_pos_embedding('pos_' + str(i), i, rown, dim=64)
    data[i] = fc_layer(loss_name + 'ctr_dnn', tf.concat([user, cl_input, data[i], pos_emb], 1), [512, 256, 128, 64]) 
   
  scores = []
  losses = []
  for i in range(n) : 
    output = fc_layer(loss_name + '_proj', data[i], [64, 32], last_unit = 1)
    scores.append(output)
    losses.append(tf.losses.log_loss(labels=labels[i], predictions=scores[i], weights = weights[i], reduction=tf.losses.Reduction.SUM))
   
  output_dict['loss'] = losses
  output_dict['preds'] = [scores[i] for i in range(n)]
  output_dict['labels'] = [labels[i] for i in range(n)]
  output_dict['q_names'] = [loss_name + '_idx' + str(i) for i in range(n)]
  return output_dict

def ctr_label_model(parameters_dict, loss_name):
  scores, labels, _, _, _, _, _, _ = poslabel_model(parameters_dict, loss_name)
  losses = []
  output_dict = {}
  n = 10
  for i in range(n) : 
    losses.append(tf.losses.log_loss(labels=labels[i], predictions=scores[i], reduction=tf.losses.Reduction.SUM))
   
  output_dict['loss'] = losses
  output_dict['preds'] = [scores[i] for i in range(n)]
  output_dict['labels'] = [labels[i] for i in range(n)]
  output_dict['q_names'] = [loss_name + '_idx' + str(i) for i in range(n)]

  return output_dict

class PLE:
    def __init__(self, tasks_names, shared_key="shared", cgc_layers=2, task_expert_num=1, shared_expert_num=3,
                 expert_tower_dim = [128,128], gate_tower_dim = [64,32]):
        super().__init__()
        self.tasks_names = tasks_names
        self.shared_key = shared_key
        self.cgc_layers = cgc_layers
        self.task_expert_num = task_expert_num
        self.shared_expert_num = shared_expert_num
        self.expert_tower_dim = expert_tower_dim  # [128,128]
        self.gate_tower_dim = gate_tower_dim # [64]

    # 单个专家塔
    def expert(self, input_feature):
        output = input_feature
        for i in range(0, len(self.expert_tower_dim)):
            dim = self.expert_tower_dim[i]
            output = tf.layers.dense(output, units=dim, activation=tf.nn.relu)
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
                output = tf.layers.dense(output, units=dim, activation=tf.nn.relu)
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

def poslabel_model(parameters_dict, loss_name):
  parameters_name = get_parameter_names_by_loss_name(loss_name) 
  parameters = [parameters_dict[name] for name in parameters_name] 
  user_fea_num = 22
  view_list_num = 7
  photo_fea_num = 47 
  context_fea_num = 40

  user = tf.concat(parameters[:user_fea_num], axis = 1)
  context_other_emb = tf.concat(parameters[-(context_fea_num+2):-2], axis = 1)
  labels_list = [parameters[-2]]
  weights_list = [parameters[-1]]

  parameters = parameters[user_fea_num + view_list_num : ] 
  
  rown = tf.shape(user)[0] 
  n = 50
  cl_net_pid = tf.reshape(parameters_dict["l2r_user_view_list_pids"], (rown, n, 32))
  cl_net_aid = tf.reshape(parameters_dict["l2r_user_view_list_aids"], (rown, n, 32))
  cl_net_hetu1 =  tf.reshape(parameters_dict["l2r_user_view_list_hetu1"], (rown, n, 8))
  cl_net_hetu2 = tf.reshape(parameters_dict["l2r_user_view_list_hetu2"], (rown, n, 8))
  cl_net_ev = tf.reshape(parameters_dict["l2r_user_view_list_ev"], (rown, n, 4))
  cl_net_sv = tf.reshape(parameters_dict["l2r_user_view_list_sv"], (rown, n, 4))
  cl_net_lv = tf.reshape(parameters_dict["l2r_user_view_list_lv"], (rown, n, 4))
  cl_input = [cl_net_pid, cl_net_aid, cl_net_hetu1, cl_net_hetu2, cl_net_ev, cl_net_sv, cl_net_lv]
  
  # user = tf.stop_gradient(user) 
  weights = [] 
  labels = [] 
  data0 = [] 
   
  batch = photo_fea_num + 2
  ltr_pids = []
  pos_embeddings = []
  interact_pxtrs = []
  watchtime_pxtrs = []
  hetus = []
  n = 10
  for i in range(n): 
    p = parameters[i * batch : (i+1) * batch] 
    ltr_pid = tf.concat(p[20:22], -1)
    ltr_pids.append(ltr_pid)
    labels.append(p[-2])
    interact_pxtr = tf.concat(p[0:13], 1)
    watchtime_pxtr = tf.concat(p[13:20], 1)
    interact_pxtrs.append(interact_pxtr)
    watchtime_pxtrs.append(watchtime_pxtr)
    data0.append(tf.concat(p[20:-2], 1))
    hetus.append(tf.concat(p[44:46], 1))
    weights.append(p[-1]) 
    pos_emb = get_pos_embedding(loss_name + 'pos_' + str(i), i, rown, dim=64)
    pos_emb = pos_emb
    pos_embeddings.append(pos_emb)
    #data[i] = gate_tower(loss_name + '_dnn' + str(i), data[i], [512, 256, 128], gate) 
    #bias = fc_layer(prefix + 'ctr_dnn', tf.concat([user, atten, data[i]], 1), [256], norm = True)
  # 升维操作
  ltr_pid_emb = tf.stack(ltr_pids, axis = 1)
  datas = tf.stack(data0, axis = 1)
  interact_pxtrs = tf.stack(interact_pxtrs, axis = 1)
  watchtime_pxtrs = tf.stack(watchtime_pxtrs, axis = 1)
  hetu_emb = tf.stack(hetus, axis = 1)
  pos_embeddings = tf.stack(pos_embeddings, axis = 1)
  #bias2 = fc_layer('l2r_dnn', tf.concat([user, atten, bias, pid, uid], 1), [512])
  user_ems = tf.tile(tf.expand_dims(user, 1), [1, n, 1])
  short_net_query = tf.concat([user_ems, ltr_pid_emb, hetu_emb], axis=2)
  query_dim = 64
  short_net_query = fc_layer(loss_name + 'ltr_user_short', short_net_query, [query_dim])
  view_net_output = short_net(loss_name +"ltr_view_list",
                              short_net_query,
                              cl_input,
                              view_list_length,
                              view_list_dim,
                              query_dim = query_dim,
                              nh = 1,
                              dim = 64,
                              stop_gradient_list = ltr_view_sg_list,
                              action_item_size = 64,
                              att_emb_size = 64)
  gate_fea =  tf.concat([user_ems, view_net_output, datas, interact_pxtrs, watchtime_pxtrs], axis=-1)
  # query_fea = fc_layer('l2r_dnn', gate_fea, [512, 256, 128, 64]) 
  # data1 = gate_tower('vtr_dnn1', query_fea, [64], interact_pxtrs)
  # data2 = gate_tower('vtr_dnn2', query_fea, [64], watchtime_pxtrs)
  # gate = gate_layer2('vtr_gate', query_fea, 64)
  # data = data1 * gate + data2 * (1 - gate)
  data = fc_layer(loss_name + 'l2r_dnn', gate_fea, [512, 256, 128, 64]) 
  
  data = tf.add(data, pos_embeddings)
  global_output = transformer(loss_name + "ltr_trf_1", data, data, n, nh=1, dim=64, mask=[-1,0])
  local_output = transformer(loss_name + "ltr_trf_2", data, data, n, nh=1, dim=64, mask=[2,2])
  atten_output = tf.concat([global_output, local_output], axis=2)
  # atten_output = SRGA_transformer(loss_name + "ltr_srga", data, data, data, n, nh = 1, dim = 64, mask_GSA = [-1, 0], mask_LSA = [2, 2]) # [batch, 4, 128]
  atten_output = tf.reshape(atten_output, (rown, n * 128)) 
  atten_outputs = tf.split(atten_output, [128] * n, 1)
  
  # 接一个 PLE 网络输出 ctr 和 l2r 目标
  contexts = []
  context_query_fea = tf.concat([user, tf.concat(ltr_pids, axis=1), tf.concat(hetus, axis=1), tf.reshape(view_net_output, (-1, n*64))], axis=1)
  context_query_fea = tf.stop_gradient(context_query_fea)
  photo_ple_layer = PLE(["ctr", "l2r"], shared_key="l2r", cgc_layers = 1, task_expert_num=1, shared_expert_num=4,
                    expert_tower_dim = [128,128], gate_tower_dim = [64,32])
  output_ctr_scores = []
  output_l2r_scores = []
  for i in range(n) : 
    input_feature_dict = {"ctr": atten_outputs[i], "l2r": atten_outputs[i]}
    output_fea_dict = photo_ple_layer(input_feature_dict)
    for key in output_fea_dict.keys():
      output_hidden = fc_layer(key + 'ltr_proj', output_fea_dict[key], [64, 32])
      output_hidden = fc_layer(key + 'ltr_out_none', output_hidden, [32], activation = None)
      output_pos = fc_layer(key + 'ltr_out_pos', output_hidden, [1], activation = tf.nn.sigmoid)
      if key == "l2r":
        contexts.append(tf.stop_gradient(output_hidden))
        output_l2r_scores.append(output_pos)
      elif key == "ctr":
        output_ctr_scores.append(output_pos)
  return output_ctr_scores, output_l2r_scores, labels, weights, contexts, context_query_fea, context_other_emb, labels_list, weights_list

def poslabel_ltr_model(parameters_dict, loss_name):
  ctr_scores, ltr_scores, labels, weights, contexts, context_query_fea, context_other_emb, labels_list, weights_list = poslabel_model(parameters_dict, loss_name)
  losses = []
  output_dict = {}
  n = 10
  # ctr 单点模型
  if (args.mode == 'train'):
    for i in range(n):
      losses.append(tf.losses.log_loss(labels=labels[i], predictions=ctr_scores[i], reduction=tf.losses.Reduction.SUM))

  # l2r 单点模型
  
  for i in range(n):
      losses.append(tf.losses.log_loss(labels=labels[i], predictions=ltr_scores[i], weights = weights[i], reduction=tf.losses.Reduction.SUM))
  # list 模型
  context_input_emb = tf.concat(contexts, axis = 1)
  # context_other_emb = tf.concat(parameters[-(context_fea_num+2):-2], axis = 1)
  context_input = tf.concat([context_input_emb, context_other_emb], 1)
  context_weight = gate_layer("context_gate", context_query_fea, context_input.shape[1])
  context_output = context_input * context_weight
  output = fc_layer(loss_name+"context_ltr_out", context_output, [256, 128, 64], last_unit=1)
  scores_list = [output]
  for i in range(1):
    losses.append(tf.losses.log_loss(labels=labels_list[i], predictions=scores_list[i], weights = weights_list[i], reduction=tf.losses.Reduction.SUM))
  output_dict['loss'] = losses
  output_dict['preds'] = [ctr_scores[i] for i in range(n)] + [ltr_scores[i] for i in range(n)] + scores_list
  output_dict['labels'] = [labels[i] for i in range(n)] + [labels[i] for i in range(n)] + labels_list
  output_dict['q_names'] = ["ctr" + '_idx' + str(i) for i in range(n)] + ["l2r" + '_idx' + str(i) for i in range(n)] + ["list_ltr0"]
  if (args.mode == 'predict'):
    for i in range(n+1):
      if i < 10:
        output = ltr_scores[i]
        out_name = 'l2r_pos'
        output = tf.identity(output, "{}".format(out_name + str(i)))
        output_dict[out_name + str(i)] = output
      else:
        output = scores_list[0]
        out_name = 'list_ltr'
        output = tf.identity(output, "{}".format(out_name + str(0)))
        output_dict[out_name + str(0)] = output
  return output_dict

def listlabel_ltr_model(parameters_dict, loss_name):
  parameters_name = get_parameter_names_by_loss_name(loss_name) 
  parameters = [parameters_dict[name] for name in parameters_name] 
  context_fea_num = 40
  _, _, _, contexts, context_query_fea = poslabel_model(parameters_dict, loss_name)
  context_input_emb = tf.concat(contexts, axis = 1)
  context_other_emb = tf.concat(parameters[-(context_fea_num+2):-2], axis = 1)
  context_input = tf.concat([context_input_emb, context_other_emb], 1)
  context_weight = gate_layer("context_gate", context_query_fea, context_input.shape[1])
  context_output = context_input * context_weight
  labels = [parameters[-2]]
  weights = [parameters[-1]]
  output = fc_layer(loss_name+"context_ltr_out", context_output, [256, 128, 64], last_unit=1)
  # print_op = tf.print("ljy_tower:context_input_emb=",context_input_emb,
  #                             "context_other_emb=",context_other_emb,
  #                             "output=", output,
  #                             summarize=1024*5,output_stream=sys.stdout)
  # print_ops.append(print_op)
  scores = [output]
  losses = []
  output_dict = {}
  n = 1
  for i in range(n):
      losses.append(tf.losses.log_loss(labels=labels[i], predictions=scores[i], weights = weights[i], reduction=tf.losses.Reduction.SUM))
  output_dict['loss'] = losses
  output_dict['preds'] = [scores[i] for i in range(n)]
  output_dict['labels'] = [labels[i] for i in range(n)]
  output_dict['q_names'] = [loss_name + '_idx' + str(i) for i in range(n)]
  if (args.mode == 'predict'):
    for i in range(n):
      output = scores[i]
      if loss_name == 'l2r_10':
        out_name = 'l2r_pos'
      else:
        out_name = 'list_ltr'
      output = tf.identity(output, "{}".format(out_name + str(i)))
      output_dict[out_name + str(i)] = output
  # return output_dict, print_op
  return output_dict

loss_model_dict = { 
  # "ctr_10" : ctr_label_model, 
  "l2r_10" : poslabel_ltr_model, 
  # "context_ltr_1": listlabel_ltr_model
} 

if (args.mode == 'train'):
  targets = []
  loss_tensor_dict = {}
  dense_loss_tensor_dict = {}

  for loss_name, model in loss_model_dict.items():
    parameters_dict, join_parameters_size_dict, label_dict, label_value_dict, weight, sample_flag = get_kuiba_loss_relative(loss_name, True, True, True)

    # if loss_name == 'context_ltr_1':
    #   xtr_output, print_op = model(parameters_dict, loss_name)
    # else:
    xtr_output = model(parameters_dict, loss_name)
    preds = xtr_output['preds']
    labels = xtr_output['labels']
    q_names = xtr_output['q_names']
    loss_tensor = xtr_output['loss']
    for i, pred in enumerate(preds):
      auc = 'auc'
      #if q_names[i].find('l2r') != -1:
      #  auc = 'linear_regression'
      targets.append((q_names[i], pred, labels[i], sample_flag, auc))
      loss_tensor_dict[q_names[i]] = loss_tensor[i]
      dense_loss_tensor_dict[q_names[i]] = loss_tensor[i]
 
  if args.with_kai:
      config.dump_kai_training_config('./training/conf', targets, loss=None, text=args.text,
                                      dense_loss=sum_loss_tensor_dict(dense_loss_tensor_dict),
                                      sparse_loss=sum_loss_tensor_dict(loss_tensor_dict));
  else:
      dense_loss=sum_loss_tensor_dict(dense_loss_tensor_dict)
      sparse_loss=sum_loss_tensor_dict(loss_tensor_dict)
      optimizer = tf.train.GradientDescentOptimizer(1, name="opt")
      sparse_grads_and_vars = optimizer.compute_gradients(sparse_loss, tf.get_collection_ref(MioCollections.MIO_EMBEDDINGS))
      print(tf.get_collection_ref(MioCollections.MIO_VARIABLES))
      dense_grads_and_vars = optimizer.compute_gradients(dense_loss, [x for x in tf.get_collection_ref(MioCollections.MIO_VARIABLES) if x.trainable])
      opt1 = optimizer.apply_gradients(sparse_grads_and_vars)
      opt2 = optimizer.apply_gradients(dense_grads_and_vars)
      # if loss_name == 'context_ltr_1':
      #   config.dump_training_config('./training/conf', targets, opts=[opt1, opt2, print_op], text=args.text)
      # else:
      config.dump_training_config('./training/conf', targets, opts=[opt1, opt2], text=args.text)

if args.mode == 'predict':
  targets = []
  for loss_name, model in loss_model_dict.items():
    parameters_dict, join_parameters_size_dict, label_dict, label_value_dict, weight, sample_flag = get_kuiba_loss_relative(loss_name, False, False, False)

    xtr_output= model(parameters_dict, loss_name)
    for (k, v) in xtr_output.items():
      if k.find('l2r_pos') != -1:
        targets.append((k, v))
      if k.find('list_ltr') != -1:
        targets.append((k, v))

  q_names, preds = zip(*targets)
  config.dump_predict_config('./predict/conf', targets, input_type=3, extra_preds=q_names)
