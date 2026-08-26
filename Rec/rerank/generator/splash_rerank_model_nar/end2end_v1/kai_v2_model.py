# -*- coding: utf-8 -*-
'''
------------------------------------------------------------------------
@Description :  
@Author :  邓英杰
@Time :  2025/01/17 17:39:15
------------------------------------------------------------------------
'''

from __future__ import print_function
MODEL_TRANS_ORIGIN='cpp'

import json
import yaml
import logging
import os
import sys
import time

import argparse
import tensorflow as tf

from model import FountainDeepLtrMultiTaskModel
from feature_attr_extract import *

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'predict'], dest='mode', default='train')
parser.add_argument('--dryrun', dest='dryrun', const=True, default=False, nargs='?')
parser.add_argument('--with_kai', default=False)
# parser.add_argument('--with_kai', default=True)
parser.add_argument('--text', default=False)
parser.add_argument('--tower', choices=None, dest='tower', default='False')
parser.add_argument('--with_kai_v2', default=True) #False True 
# parser.add_argument('--with_kai_v2', default=False) #False True 
args = parser.parse_known_args()[0]
is_training = args.mode == "train"
LIST_SIZE = 6
CANDIDATES_SIZE = 60

print_ops = []
# 目前这段逻辑功能未知
if args.with_kai_v2:
    import kai.tensorflow as config
    import tensorflow.compat.v1 as tf
    default_param_attr = config.nn.ParamAttr(initializer=config.nn.UniformInitializer(0.0001),
                                             access_method=config.nn.ProbabilityAccess(100.0),
                                             recycle_method=config.nn.UnseendaysRecycle(delete_after_unseen_days=30, delete_threshold=1.0, allow_dynamic_delete=True))
    config.nn.set_default_param_attr(default_param_attr)

    def filter_mask_wrapper(dataset):
        # 1. 声明字段
        #  sample_type为字段名，特征类型dataset.DENSE表示稠密，tf.int64为数据类型，dim为1
        # dataset.add_feature('fountain_fulllink_rerank_index_list', dataset.DENSE, tf.int64, max_length=60)
        # dataset.add_feature('fountain_fulllink_rerank_index_weight_list', dataset.DENSE, tf.int64, max_length=60)
        dataset.add_feature('context_info__real_show_index_list', dataset.DENSE, tf.int64, max_length=30)
        dataset.add_feature('context_info__real_show_list', dataset.DENSE, tf.int64, max_length=30)
        dataset.add_feature('fountain_fulllink_rerank_realshow_label_weight_list', dataset.DENSE, tf.int64, max_length=30)
        dataset.add_feature('context_info__first_screen', dataset.DENSE, tf.int64, max_length=30)


        # 2.声明mask，batch是一个dict，key为声明的字段名，value根据特征类型分为2种情况：
        # dataset.DENSE: 值为tf.Tensor
        # dataset.SPARSE: 值为元组: (tf.Tensor, tf.Tensor)，
        #   其中第一个tensor表示feasign，第二个tensor表示cumsum
        #   可以使用tf.RaggedTensor.from_row_splits转成RaggedTensor
        def mask_fn(batch):

            realshow = batch["context_info__real_show_list"]
            realshow = tf.RaggedTensor.from_row_splits(realshow[0], row_splits= realshow[1])
            realshow = realshow.to_tensor()

            realshow_weight = batch["fountain_fulllink_rerank_realshow_label_weight_list"]
            realshow_weight = tf.RaggedTensor.from_row_splits(realshow_weight[0], row_splits= realshow_weight[1])
            realshow_weight = realshow_weight.to_tensor()

            total_play_time = tf.reduce_sum(realshow_weight, axis=-1) - CANDIDATES_SIZE
            realshow = tf.reduce_sum(realshow, axis=-1)

            context_page = batch['context_info__first_screen']

            # print_ops.append(tf.print("total_play_time=", total_play_time, output_stream=sys.stdout))
            # print_ops.append(tf.print("context_page=", context_page, output_stream=sys.stdout))
            # print_ops.append(tf.print("context_page len=", len(context_page), output_stream=sys.stdout))
            # print_ops.append(tf.print("realshow_weight", realshow_weight, output_stream=sys.stdout))
            print(f"realshow shape: {realshow.shape}")
            print(f"realshow_weight shape: {realshow_weight.shape}")
            
            is_short_request = tf.math.less(total_play_time, 20) # 60%分位数
            is_unvalid_request = tf.math.less(total_play_time, 5)
            
            fountain_click_label = batch["fountain_click_label_list"]
            fountain_click_label = tf.RaggedTensor.from_row_splits(fountain_click_label[0], row_splits=fountain_click_label[1])
            fountain_click_label = fountain_click_label.to_tensor()
            fountain_click_sum = tf.reduce_sum(fountain_click_label, axis=-1)

            # 长视频需要满足点击数>=2
            # long_video_quality = tf.math.greater_equal(fountain_click_sum, 2)
            
            # 过滤规则:
            # 0. 只保留首屏样本
            # 1. 曝光数要>=2
            # 2. 至少要有一条有效点击
            # mask = tf.math.logical_or(
            #     tf.math.logical_or(
            #         tf.math.not_equal(context_page, 1), # 过滤非首屏
            #         tf.math.less(fountain_click_sum, 1), # 过滤无有效点击
            #     ),
            #     is_short_request # 过滤时长在中位数以下的序列
            # )
            mask = tf.math.logical_or(
                    tf.math.not_equal(context_page, 1), # 过滤非首屏
                    tf.math.less(realshow, 1),
                ),
            # return mask
            return tf.math.less(realshow, 2)
        # 3.返回mask_fn
        return mask_fn
    # 注册过滤条件
    config.declare_sample_filter(filter_mask_wrapper, data_source_name='train')
else:
    import tensorflow as tf
    from mio_tensorflow.config import MioConfig
    if not args.dryrun and not args.with_kai:
        import mio_tensorflow.patch as mio_tensorflow_patch
    
        mio_tensorflow_patch.apply()
    
    logging.basicConfig()
    base_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), './base.yaml')
    config = MioConfig.from_base_yaml(base_config,
                                      clear_embeddings=True,
                                      clear_params=True, #False,
                                      dryrun=args.dryrun,
                                      label_with_kv=True,
                                      grad_no_scale=False,
                                      with_kai=args.with_kai,
                                  predict=(args.mode != "train"))
# label name和cofea_reader.py中各label前缀保持一致
all_model_labels = [
    "fountain_fulllink_rerank_index_list",
    "fountain_fulllink_rerank_index_weight_list",
    "fountain_fulllink_rerank_realshow_label_weight_list",
    "fountain_click_label_list",
    "fountain_wtd_label_list",
    "fountain_ltr_label_list",
    "fountain_ltr_weight_list",
    "fountain_finish_label_list",
]

realshow_labels = [
    "context_info__real_show_index_list",
    "context_info__real_show_list",
]

print("common_attr_names: ", [attr.attr_name for attr in all_features if attr.is_common])
print("all_feature_name: ", [attr.attr_name for attr in all_features])

def mark_common_attr():
    common_embeddings = []
    for attr in all_features:
        if attr.is_common:
            common_embeddings.append(attr.attr_name)
    with open('./infer/dnn_model.yaml', "r+") as f:
        yaml_config = yaml.load(f.read(), Loader=yaml.FullLoader)
        print(yaml_config['embedding']['slots_config'][0])
        for idx, slot_config in enumerate(yaml_config['embedding']['slots_config']):
            if slot_config['input_name'] in common_embeddings:
                yaml_config['embedding']['slots_config'][idx]['is_common'] = True
        f.seek(0)
        yaml.dump(yaml_config, f)
        f.truncate()

def get_dense_fea(name, list_dim):
    assert name in realshow_labels, name
    return config.get_dense_fea(name, dim=list_dim, dtype=tf.int64)

def get_label(name, list_dim):
    assert name in all_model_labels, name
    return config.get_label(name, dim=list_dim)

def get_param_dict(print_ops):
    """
    train and dnn infer：不需要区分common or no_common,(infer配置中对应的tensorflow_use_batching=true)
    tower infer : 需要区分attr是common or no_common
    :return:
    """
    if args.with_kai_v2:
        # share embedding
        config.declare_reallocate_slots(share_input_slots,
                             share_output_slots,
                             remap=True,
                             inplace=True)
        # 需要额外copy的特征
        config.declare_reallocate_slots(copy_input_slots,
                             copy_output_slots,
                             remap=True,
                             inplace=False)
    feature_emb_dict = {}
    feature_emb_size_dict = {}
    for attr in all_features:
        print("--->>> feature %s start" % attr.attr_name)
        if not is_training:
            if not attr.expand:
                attr.expand = 1
            embed = None
            if attr.is_common:
                embed = config.new_embedding(attr.attr_name, dim=attr.dim, slots=attr.slots, expand=attr.expand, compress_group='USER')
            else:
                embed = config.new_embedding(attr.attr_name, dim=attr.dim, slots=attr.slots, expand=attr.expand)
            feature_emb_dict[attr.attr_name] = embed
            feature_emb_size_dict[attr.attr_name] = 1
        else:
            # if not attr.expand:
            #     attr.expand = 1
            if attr.attr_name in photo_fea_names + ["photo_id_v2"]:
                if not attr.expand:
                    attr.expand = CANDIDATES_SIZE
                else:
                    attr.expand *= CANDIDATES_SIZE
            print("attr ",attr,"attr.dim ", attr.dim, "attr.slots ",attr.slots, "attr.expand", attr.expand)
            feature_emb_dict[attr.attr_name] = config.new_embedding(attr.attr_name, dim=attr.dim, slots=attr.slots, expand=attr.expand)
        # 这里会再把expand进行reduce sum
        if attr.expand is not None and attr.expand > 1:
            feature_emb_dict[attr.attr_name] = tf.reshape(feature_emb_dict[attr.attr_name], [-1, attr.expand, attr.dim])
            # feature_emb_dict[attr.attr_name] = tf.reduce_sum(feature_emb_dict[attr.attr_name], axis=1)

        if args.with_kai_v2: 
            sparse_feature = config.get_sparse_fea(name=str(attr.slots[0])) 
            offset = sparse_feature[1]
            size_var = offset[1:] - offset[0:-1]
            feature_emb_size_dict[attr.attr_name] = size_var
            if attr.slots[0] in ["3122", "3123", "3124", "3125", "3126", "3127", "3128", "3129"]:
                tt = tf.RaggedTensor.from_row_splits(values=sparse_feature[0], row_splits=sparse_feature[1]).to_tensor()
                print_ops.append(tf.print(f"[Test test] {attr.attr_name} slot " + str(attr.slots[0]), tt[10], output_stream=sys.stdout))
        elif args.with_kai:
            offset = tf.cast(config.get_signs(attr.slots[0])[1], tf.int32)
            size_var = offset[1:] - offset[0:-1]
            feature_emb_size_dict[attr.attr_name] = size_var        
        print("--->>> feature {} = {}".format(attr.attr_name, feature_emb_dict[attr.attr_name]))
        print("--->>> feature %s normal" % attr.attr_name)

    return feature_emb_dict, feature_emb_size_dict

# all_model_labels里的名字和cofea_reader.py中名字对应
def get_label_dict(list_dim):
    label_value_dict = {}
    for label_name in all_model_labels:
        print(f"====> get_label, name : {label_name}")
        label_value = get_label(label_name, list_dim)
        label_value_dict[label_name] = label_value

    for label_name in realshow_labels:
        print(f"====> get_label, name : {label_name}")
        label_value = get_dense_fea(label_name, list_dim)
        label_value_dict[label_name] = label_value

    return label_value_dict

def get_dense_dict(dense_feas, list_dim):
    dense_value_dict = {}
    for name in dense_feas:
        dense_value_dict[name] = config.get_extra_param(name, size=list_dim, default_value=0.0)

    return dense_value_dict

def sum_loss_tensor_dict(loss_dict):
    sum_loss = None
    for key, loss in loss_dict.items():
        if (sum_loss == None):
            sum_loss = loss
        else:
            sum_loss += loss
    return sum_loss

def list_precision(predict, label_value_dict):
    predict = tf.identity(predict) # (?, list_size+1, candidates_size+3)
    
    gen_model_label = label_value_dict['context_info__real_show_list'] # 形如 [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    print('gen_model_label', gen_model_label.get_shape().as_list())
    gen_model_label = tf.reshape(gen_model_label, [-1, CANDIDATES_SIZE])
    true_label = tf.cast(tf.greater(gen_model_label, 0), tf.float32) # [None, candidates_size]
    true_label = true_label[:,:LIST_SIZE]
    print("true_label shape", true_label.get_shape().as_list())
    indices_matrix = tf.tile(tf.expand_dims(tf.range(0, CANDIDATES_SIZE), 0), [tf.shape(gen_model_label)[0], 1])
    print("indices_matrix shape ", indices_matrix.shape)
    true_index = tf.where(tf.greater(gen_model_label, 0), indices_matrix, tf.fill(tf.shape(gen_model_label), 0))
    true_index = true_index[:,:LIST_SIZE]
    print("true_index shape ", true_index.shape)

    # predict = predict[:,:-1,2:-1] # (?, list_size, candidates_size)
    print('list_precision ', predict.shape)

    _, rank_index = tf.math.top_k(predict, 1, sorted=True) # 返回最后一维最大值index, (?, candidates_size, 1)
    rank_index = tf.squeeze(rank_index, -1)

    print_ops.append(tf.print('[train] true_label ', true_label[2], summarize = -1, output_stream=sys.stdout))
    # print_ops.append(tf.print('[train] true_index ', true_index[2], summarize = -1, output_stream=sys.stdout))
    print_ops.append(tf.print('[train] select_index ', rank_index[2], summarize = -1, output_stream=sys.stdout))
    print_ops.append(tf.print('[train] predict ', predict[2][:8], summarize = -1, output_stream=sys.stdout))

    hit_cnt = tf.reduce_sum(tf.cast(tf.equal(true_index, rank_index), tf.float32) * true_label, axis=-1, keep_dims=True)
    avg_precision = hit_cnt / (tf.reduce_sum(true_label, -1, keep_dims=True)+1e-9)
    tf.summary.scalar('avg_precision', tf.reduce_mean(avg_precision))
    print_ops.append(tf.print('[train] avg_precision ', tf.reduce_mean(avg_precision), summarize = 8, output_stream=sys.stdout))

def set_zero_topk(pred, indices):
    batch_size, seq_len, vocab_len = tf.shape(pred)[0],tf.shape(pred)[1],tf.shape(pred)[2]
    index = tf.expand_dims(tf.range(0,batch_size),axis=1)*seq_len*vocab_len+ tf.expand_dims(tf.range(0, seq_len),axis=0)*vocab_len
    index = tf.expand_dims(index, axis=2)
    # print("index shape ",index.shape)
    selected_token = tf.expand_dims(tf.expand_dims(indices,axis=1),axis=2)
    selected_token = tf.cast(selected_token, tf.int32)
    # print("tensor shape ",tensor.shape)
    selected_token = tf.tile(selected_token,[1,seq_len,1])+index
    # print("selected token shape ", selected_token.shape)
    pred = tf.reshape(pred,(batch_size*seq_len*vocab_len, -1))
    selected_token = tf.reshape(selected_token, (batch_size*seq_len*1,-1))

    output_tensor = tf.tensor_scatter_nd_update(pred, selected_token, tf.expand_dims(tf.ones(batch_size*seq_len)*float("-inf"),axis=1))
    output_tensor = tf.reshape(output_tensor, [batch_size, seq_len, vocab_len])
    return output_tensor

def set_zero_by_idx(pred, indices):
  batch_size, beam_len, vocab_len = tf.shape(pred)[0],tf.shape(pred)[1],tf.shape(pred)[2]
  batch_size, beam_len, seq_len = tf.shape(indices)[0],tf.shape(indices)[1],tf.shape(indices)[2]
  pred = tf.reshape(pred,(batch_size*beam_len, -1))
  indices = tf.reshape(indices,(batch_size*beam_len, -1))
  flat_indices = indices + tf.reshape(tf.range(0, batch_size*beam_len*vocab_len, vocab_len), (batch_size*beam_len, 1))
  flat_indices = tf.reshape(flat_indices, (-1,1))
  pred = tf.reshape(pred, (batch_size*beam_len*vocab_len,))
  # pred = tf.tensor_scatter_nd_update(pred, flat_indices, tf.zeros(batch_size*beam_len*seq_len))
  pred = tf.tensor_scatter_nd_update(pred, flat_indices, tf.ones(batch_size*beam_len*seq_len)*float("-inf"))
  pred = tf.reshape(pred,(batch_size, beam_len, vocab_len))

  return pred

def beam_search(prediction, k=10, list_size=LIST_SIZE):
    batch_size, seq_length, vocab_size = tf.shape(prediction)[0],tf.shape(prediction)[1],tf.shape(prediction)[2]

    log_prob, indices = tf.nn.top_k(prediction[:,0,:], k, sorted=True) #sort k largest logits
    indices = tf.expand_dims(indices, -1) #bs,k,1
    for n1 in range(1, list_size):
        predict_temp = tf.expand_dims(prediction[:,n1,:], 1) #bs,1,vocab_size
        predict_temp = tf.tile(predict_temp, [1, k, 1])       #bs,k,vocab_size
        if n1 >= 2:
            predict_temp = set_zero_by_idx(predict_temp, new_indices) #set k largest logits to -inf
        else:
            predict_temp = set_zero_by_idx(predict_temp, indices) #set k largest logits to -inf
        log_prob_temp = tf.expand_dims(log_prob, -1) + predict_temp #bs,k,vocab_size

        log_prob, index_temp = tf.nn.top_k(tf.reshape(log_prob_temp, [batch_size, -1]), k, sorted=True) #sort k largest logits
        idx_begin = index_temp // vocab_size #bs,k
        idx_concat = index_temp % vocab_size #bs,k

        new_indices = tf.zeros([batch_size, k, n1+1], dtype=tf.int32) #bs,k,n1+1
        idx_expand = tf.expand_dims(idx_begin, -1) #bs,k,1
        new_indices = tf.concat([tf.batch_gather(indices, idx_begin), tf.expand_dims(idx_concat, -1)], axis=-1) #bs,k,n1+1
        indices = new_indices

    return indices, log_prob

def greedy_search(prediction):
    seq_len = 6
    generated_tokens = []
    for i in range(seq_len):
        logits = prediction[:, i, :]
        _, sampled_token = tf.nn.top_k(logits, k=1)
        # print("prediction shape ",prediction.shape)
        # print("sampled_token shape ",sampled_token.shape)
        sampled_token = tf.squeeze(sampled_token, axis=-1)
        prediction = set_zero_topk(prediction, sampled_token)
        generated_tokens.append(sampled_token)
    generated_tokens = tf.stack(generated_tokens, axis=-1)

    return generated_tokens


# print info
worker_global_step = config.get_step()
ops = [tf.print("====> step", worker_global_step, summarize=-1, output_stream=sys.stdout)]

############################################################

# 获取模型output
all_param_dict, feature_emb_size_dict = get_param_dict(print_ops=print_ops)
print("feature_emb_size_dict ", feature_emb_size_dict)
label_value_dict = get_label_dict(list_dim=CANDIDATES_SIZE)
dense_value_dict = {}
dense_value_dict = get_dense_dict(["context_info__pctr", "context_info__pltr", "context_info__plvtr", "context_info__pwtr", "context_info__pvtr", "context_info__psvtr"], CANDIDATES_SIZE)

model_class = FountainDeepLtrMultiTaskModel(all_param_dict, label_value_dict, dense_value_dict, print_ops, list_size=LIST_SIZE, candidates_size=CANDIDATES_SIZE, training=True)


if is_training:
    print(f"====> train, gen...")
    # rerank_list_score_list = tf.reshape(config.get_dense_fea("rerank_list_score_list", dim=LIST_SIZE * 15, dtype=tf.int64), [-1, CANDIDATES_SIZE])
    # rerank_list_item_idx_flat_list = tf.reshape(config.get_dense_fea("rerank_list_item_idx_flat_list", dim=15, dtype=tf.int64), [-1, CANDIDATES_SIZE])

    logits, loss, item_weight, generator_loss, generator_logits, cl_loss_states, cl_loss, predict, item_embeddings, gen_loss, gt_label = model_class.model()
    loss_sum = (loss+cl_loss_states) #eval loss sum
    tf.summary.scalar('gen_loss', gen_loss)
    tf.summary.scalar('eval_loss', loss)
    tf.summary.scalar('eval_cl_loss_hidden_states', cl_loss_states)
    tf.summary.scalar('eval_loss_sum', loss_sum)

    tf.summary.scalar('gen_eval_loss', generator_loss)
    tf.summary.scalar('gen_cl_loss_item_position',cl_loss)
    print("logits ",logits)
    print_ops = model_class.print_ops

    targets = []
    sum_loss = 0.0

    list_precision(predict, label_value_dict)
    # generator_loss_sum = 1000*(generator_loss+cl_loss)
    alpha = 0.5
    generator_loss_sum = (generator_loss+cl_loss+alpha*gen_loss)
    # generator_loss_sum = (generator_loss+cl_loss)
    tf.summary.scalar('gen_loss_sum', generator_loss_sum)

    gen_model_label = label_value_dict['context_info__real_show_list']
    gen_model_label = tf.reshape(gen_model_label, [-1, CANDIDATES_SIZE])
    gen_model_label = tf.cast(gen_model_label, dtype=tf.int32)
    distill_label = tf.cast(tf.math.greater(gen_model_label,0), tf.float32)
    
     # recall v1
    item_score = tf.reduce_sum(predict, axis=1) #bs,cand_size
    predict = tf.identity(predict)
    _, ranked_item_index = tf.math.top_k(item_score, LIST_SIZE, sorted=True)
    item_hit = tf.batch_gather(distill_label,ranked_item_index) 
    # print_ops.append(tf.print("ranked_item_index ", ranked_item_index[:5], summarize=10, output_stream=sys.stdout))
    recall_6_th = tf.reduce_sum(item_hit[:,:6], -1, keep_dims=True) / (tf.reduce_sum(distill_label, -1, keep_dims=True)+1e-9)
    recall_4_th = tf.reduce_sum(item_hit[:,:4], -1, keep_dims=True) / (tf.reduce_sum(distill_label, -1, keep_dims=True)+1e-9)
    tf.summary.scalar('recall 6 v1', tf.reduce_mean(recall_6_th))
    tf.summary.scalar('recall 4 v1', tf.reduce_mean(recall_4_th))
    print_ops.append(tf.print('[train] recall_6 ', tf.reduce_mean(recall_6_th), summarize = 8, output_stream=sys.stdout))

    #recall v2
    greedy_indices = greedy_search(predict)
    greedy_hit = tf.batch_gather(distill_label, greedy_indices)
    recall_4_th_greedy = tf.reduce_sum(greedy_hit[:,:4], -1, keep_dims=True) / (tf.reduce_sum(distill_label, -1, keep_dims=True) + 1e-9)
    tf.summary.scalar('recall 4 greedy', tf.reduce_mean(recall_4_th_greedy))

    # 打印、监控
    # for attr_name, emb in all_param_dict.items():
    #     tf.summary.scalar(f"{attr_name}/mean", tf.reduce_mean(emb))
    #     # tf.summary.scalar(f"{attr_name}/abs_mean", tf.reduce_mean(tf.abs(emb)))
    #     tf.summary.scalar(f"{attr_name}/norm2", tf.reduce_mean(tf.norm(emb, axis=1)))
    #     embedding_gradients = tf.gradients(loss, emb)
    #     print(f"{attr_name}/gradient, {embedding_gradients}")
    #     tf.summary.scalar(f"{attr_name}/gradient/mean", tf.reduce_mean(embedding_gradients))
    #     # tf.summary.scalar(f"{attr_name}/gradient/abs_mean", tf.reduce_mean(tf.abs(embedding_gradients)))
    #     tf.summary.scalar(f"{attr_name}/gradient/norm2", tf.reduce_mean(tf.norm(embedding_gradients, axis=1)))

    gen_model_label = label_value_dict['context_info__real_show_list'] # 形如 [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    print('gen_model_label', gen_model_label.get_shape().as_list())
    gen_model_label = tf.reshape(gen_model_label, [-1, CANDIDATES_SIZE])
    true_label = tf.cast(tf.greater(gen_model_label, 0), tf.float32) # [None, candidates_size]
    print("true_label shape", true_label.get_shape().as_list())
    
    with tf.control_dependencies(print_ops):
        logits = tf.reduce_sum(logits, axis=-1)
        logits = tf.expand_dims(logits, axis=-1)
        zero = tf.zeros_like(logits)
        one = tf.ones_like(logits)
        print("zero shape", zero.shape)
        print("one shape", one.shape)
        targets.append(('recall', logits, zero, one, 'linear_regression'))

    if args.with_kai_v2:
        sparse_optimizer = config.optimizer.Adam(0.001)
        dense_optimizer = config.optimizer.Adam(0.00005)
        total_sparse_var = config.get_sparse_trainable_variables()
        total_dense_var = config.get_dense_trainable_variables()
        dense_optimizer_gen = config.optimizer.Adam(0.00005)

        dense_gen_var_list = []
        dense_eval_var_list = []
        for var in total_dense_var:
            if "generator" in var.name:
                dense_gen_var_list.append(var)
            else:
                dense_eval_var_list.append(var)

        sparse_optimizer.minimize(generator_loss_sum+loss_sum, var_list=total_sparse_var) #gen-eval loss + eval loss
        dense_optimizer.minimize(loss_sum, var_list=dense_eval_var_list) #eval loss
        dense_optimizer_gen.minimize(generator_loss_sum, var_list=dense_gen_var_list) #gen-eval loss
        opts = [sparse_optimizer, dense_optimizer, dense_optimizer_gen]
        # sparse_optimizer.minimize(loss, var_list=config.get_collection(config.GraphKeys.EMBEDDING_INPUT))
        # dense_optimizer.minimize(loss, var_list=config.get_collection(config.GraphKeys.TRAINABLE_VARIABLES))
        # opts = [sparse_optimizer, dense_optimizer]
    else:
        optimizer = tf.train.GradientDescentOptimizer(1, name="opt")
        grad_var = optimizer.compute_gradients(loss)
        opt = optimizer.apply_gradients(grad_var)
        opts = [opt]

    if args.dryrun:
        pass  # config.mock_and_profile(opt, './training_log/', batch_sizes=[128, 288])
    elif args.with_kai:
        print(f"====> train, with kai")
        # print(f"====> dump btq, user_top: {user_top}, photo_top: {photo_top}")
        config.dump_kai_training_config('./training', targets, loss=sum_loss, text=args.text, init_params_in_tf=True)
    elif args.with_kai_v2:
        config.build_model(optimizer=opts, metrics=targets)
    else:
        config.dump_training_config('./training', targets, opts=opts, text=args.text)
else:
    model_class.training = False
    # logits, generated_sequence = model_class.model()
    predict, item_embedding = model_class.model()
    targets = []
    #beam
    predict_=tf.identity(predict)
    indices, log_prob = beam_search(predict_,k=10,list_size=4) #(?,10,6)
    selected_score, selected_index = tf.nn.top_k(log_prob[0, :], 5, sorted=True) #(5,)
    indices=indices[0] #(10,6)
    selected_indices_beam = tf.gather(indices,selected_index) #(5,6)
    for i in range(3):#0-2
        selected_indices_i = selected_indices_beam[i,:] #(6,)
        tensor_zeros = tf.zeros(tf.shape(predict_)[2], dtype=tf.int32)
        output_tensor = tf.tensor_scatter_nd_update(tensor_zeros, tf.expand_dims(selected_indices_i, 1), tf.range(selected_indices_i.shape[0], 0, -1))
        output_tensor = tf.reshape(output_tensor, [-1, 1])
        output_tensor = tf.cast(output_tensor, dtype=tf.float32)
        # print("output_tensor shape", output_tensor.shape)
        pred_output = tf.identity(output_tensor) #(?,1)
        # print("pred_output shape",pred_output.shape)
        # targets.append((f"rerank_gen_score_{i+5}",pred_output))
        targets.append((f"rerank_gen_score_{i}",pred_output))

    q_names, preds = zip(*targets)
    config.dump_predict_config(
        "./infer/",
        targets,
        input_type=3,
        extra_preds=q_names,
    )
    print("====> q_name: ", q_names)
    mark_common_attr()

print(f"====> is_training: {is_training}, tower: {args.tower}, dryrun: {args.dryrun}")
