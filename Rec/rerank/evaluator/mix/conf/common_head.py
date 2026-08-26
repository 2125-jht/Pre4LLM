import os
import sys
import json
import logging
import argparse
import functools
import tensorflow as tf
import numpy
from numpy.core.fromnumeric import transpose
from tensorflow.keras.backend import expand_dims,repeat_elements
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

base_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'lintao/base.yaml')

config = MioConfig.from_base_yaml(base_config, clear_embeddings=True, clear_params=True,
                                  dryrun=args.dryrun, label_with_kv=True, grad_no_scale=False,
                                  predict=(args.mode == 'predict'),
                                  with_kai=args.with_kai)

config_from_kuiba = json.load(open("./kai_kuiba_config.json"))
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


