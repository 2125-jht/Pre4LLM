# -*- coding: utf-8 -*-
'''
------------------------------------------------------------------------
@Description :  
@Author :  邓英杰
@Time :  2025/01/17 17:39:15
------------------------------------------------------------------------
'''

MODEL_TRANS_ORIGIN='cpp'

import yaml
import logging
import os
import sys

import argparse
import tensorflow as tf

from feature_attr_extract import * 
# from model import FountainDeepLtrMultiTaskModel
# from nce_model import FountainDeepLtrMultiTaskModel
from ppo_model import GenModel

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
LIST_NUM = 15
CANDIDATES_SIZE = 60

print_ops = []
# 目前这段逻辑功能未知
if args.with_kai_v2:
    import kai.tensorflow as config
    import tensorflow.compat.v1 as tf
    default_param_attr = config.nn.ParamAttr(initializer=config.nn.UniformInitializer(0.0001),
                                             access_method=config.nn.ProbabilityAccess(100.0),
                                             recycle_method=config.nn.UnseendaysRecycle(delete_after_unseen_days=30, delete_threshold=0.1, allow_dynamic_delete=True))
    config.nn.set_default_param_attr(default_param_attr)

    def filter_mask_wrapper(dataset):
        # 1. 声明字段
        #  sample_type为字段名，特征类型dataset.DENSE表示稠密，tf.int64为数据类型，dim为1
        # dataset.add_feature('fountain_fulllink_rerank_index_list', dataset.DENSE, tf.int64, max_length=60)
        # dataset.add_feature('fountain_fulllink_rerank_index_weight_list', dataset.DENSE, tf.int64, max_length=60)
        dataset.add_feature('context_info__real_show_index_list', dataset.DENSE, tf.int64, max_length=60)
        dataset.add_feature('context_info__real_show_list', dataset.DENSE, tf.int64, max_length=60)
        dataset.add_feature('fountain_fulllink_rerank_realshow_label_weight_list', dataset.DENSE, tf.int64, max_length=60)
        dataset.add_feature('context_info__first_screen', dataset.DENSE, tf.int64, max_length=60)
        dataset.add_feature('tab', dataset.DENSE, tf.int64, max_length=60)
        dataset.add_feature('3017', dataset.DENSE, tf.int64, max_length=60)
        dataset.add_feature('3019', dataset.DENSE, tf.int64, max_length=60)
        dataset.add_feature('3030', dataset.DENSE, tf.int64, max_length=60)
        dataset.add_feature('1031', dataset.DENSE, tf.int64, max_length=60)


        # 2.声明mask，batch是一个dict，key为声明的字段名，value根据特征类型分为2种情况：
        # dataset.DENSE: 值为tf.Tensor
        # dataset.SPARSE: 值为元组: (tf.Tensor, tf.Tensor)，
        #   其中第一个tensor表示feasign，第二个tensor表示cumsum
        #   可以使用tf.RaggedTensor.from_row_splits转成RaggedTensor
        def mask_fn(batch):
            realshow = batch["context_info__real_show_list"]
            print_ops.append(tf.print("context_info__real_show_list0=", realshow[0], summarize = 10, output_stream=sys.stdout))
            print_ops.append(tf.print("context_info__real_show_list1=", realshow[1], summarize = 10, output_stream=sys.stdout))
            realshow = tf.RaggedTensor.from_row_splits(realshow[0], row_splits= realshow[1])
            realshow = realshow.to_tensor()
            #print_ops.append(tf.print("context_info__real_show_list=", realshow[0], summarize = 10, output_stream=sys.stdout))
            
            realshow_weight = batch["fountain_fulllink_rerank_realshow_label_weight_list"]
            realshow_weight = tf.RaggedTensor.from_row_splits(realshow_weight[0], row_splits= realshow_weight[1])
            realshow_weight = realshow_weight.to_tensor()
            
            # photo_hetu_tag_level5_list = tf.RaggedTensor.from_row_splits(batch["1031"][0], row_splits= batch["1031"][1]).to_tensor()
            # print_ops.append(tf.print("photo_hetu_tag_level5_list=", photo_hetu_tag_level5_list[0], summarize = 10, output_stream=sys.stdout))

            total_play_time = tf.reduce_sum(realshow_weight, axis=-1) - CANDIDATES_SIZE
            realshow = tf.reduce_sum(realshow, axis=-1)
            
            context_page = batch['context_info__first_screen']
            tab = batch['tab']
        
            # print_ops.append(tf.print("tab=", tab, output_stream=sys.stdout))
            # print_ops.append(tf.print("context_page=", context_page, output_stream=sys.stdout))
            # print_ops.append(tf.print("context_page len=", len(context_page), output_stream=sys.stdout))
            # print_ops.append(tf.print("realshow_weight", realshow_weight, output_stream=sys.stdout))
            print(f"realshow shape: {realshow.shape}")
            print(f"realshow_weight shape: {realshow_weight.shape}")
            
            is_short_request = tf.math.less(total_play_time, 20) # 60%分位数
            
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
            #     tf.math.not_equal(context_page, 1),
            #     tf.math.less(realshow, 1),
            # )
            mask = tf.math.logical_or(
                tf.math.equal(context_page, 1),
                tf.math.less(realshow, 1),
            )
            return mask
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
]

realshow_labels = [
    "context_info__real_show_index_list",
    "context_info__real_show_list",
    "context_info__playing_time_list",
    "context_info__click_list",
    "context_info__like_list",
    "context_info__follow_list",
    "context_info__comment_list",
    "context_info__forward_list",
    "context_info__fountain_slide_to_next_list",
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

def get_param_dict():
    """
    train and dnn infer：不需要区分common or no_common,(infer配置中对应的tensorflow_use_batching=true)
    tower infer : 需要区分attr是common or no_common
    :return:
    """
    # if args.with_kai_v2:
    #     # share embedding
    #     config.declare_reallocate_slots(share_input_slots,
    #                          share_output_slots,
    #                          remap=True,
    #                          inplace=True)
    #     # 需要额外copy的特征
    #     config.declare_reallocate_slots(copy_input_slots,
    #                          copy_output_slots,
    #                          remap=True,
    #                          inplace=False)
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
            if attr.attr_name in photo_fea_names:
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
            if attr.slots[0] == 16:
                tt = tf.RaggedTensor.from_row_splits(values=sparse_feature[0], row_splits=sparse_feature[1]).to_tensor()
                #print_ops.append(tf.print("[Test test] slot " + str(attr.slots[0]), tt, output_stream=sys.stdout))
        elif args.with_kai:
            offset = tf.cast(config.get_signs(attr.slots[0])[1], tf.int32)
            size_var = offset[1:] - offset[0:-1]
            feature_emb_size_dict[attr.attr_name] = size_var        
        print("--->>> feature {} = {}, shape={}".format(attr.attr_name, feature_emb_dict[attr.attr_name], feature_emb_dict[attr.attr_name].shape))
        # print("--->>> feature %s normal" % attr.attr_name)

    return feature_emb_dict, feature_emb_size_dict

# all_model_labels里的名字和cofea_reader.py中名字对应
def get_label_dict(list_dim=CANDIDATES_SIZE):
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
        # dense_value_dict[name] = config.get_extra_param(name, size=list_dim, default_value=0.0)
        dense_value_dict[name] = config.get_extra_param(name, default_value=0.0)

    return dense_value_dict


def sum_loss_tensor_dict(loss_dict):
    sum_loss = None
    for key, loss in loss_dict.items():
        if (sum_loss == None):
            sum_loss = loss
        else:
            sum_loss += loss
    return sum_loss

def set_zero_topk(pred, indices):
    batch_size, seq_len, vocab_len = tf.shape(pred)[0],pred.shape[1],pred.shape[2]
    # 计算展平后的 batch 和 seq 偏移位置
    index = tf.expand_dims(tf.range(0,batch_size),axis=1) * seq_len * vocab_len + tf.expand_dims(tf.range(0, seq_len),axis=0) * vocab_len # (?, seq_len)
    index = tf.expand_dims(index, axis=2) # (?, seq_len, 1)
    # print("index shape ",index.shape)
    selected_token = tf.expand_dims(tf.expand_dims(indices, axis=1), axis=2) # (?, 1, 1)
    selected_token = tf.cast(selected_token, tf.int32)
    selected_token = tf.tile(selected_token, [1, seq_len, 1]) + index # (?, seq_len, 1)
    # print("selected token shape ", selected_token.shape)
    pred = tf.reshape(pred, (batch_size * seq_len * vocab_len, 1))
    selected_token = tf.reshape(selected_token, (batch_size * seq_len * 1, 1))

    output_tensor = tf.tensor_scatter_nd_update(pred, selected_token, tf.expand_dims(tf.ones(batch_size * seq_len) * float("-inf"), axis=1))
    output_tensor = tf.reshape(output_tensor, [batch_size, seq_len, vocab_len])
    return output_tensor

def greedy_search(prediction):
    generated_tokens = []
    for i in range(LIST_SIZE):
        logits = prediction[:, i, :] # (?, candidates_size)
        _, sampled_token = tf.nn.top_k(logits, k=1) # (?, k=1)
        sampled_token = tf.squeeze(sampled_token, axis=-1) # (?,)
        prediction = set_zero_topk(prediction, sampled_token)
        generated_tokens.append(sampled_token)
    generated_tokens = tf.stack(generated_tokens, axis=-1) # (?, list_size)
    return generated_tokens

def list_recall(predict, label_value_dict):
    predict = tf.identity(predict) # (?, list_size+1, candidates_size+3)
    predict = predict[:,:-1,2:-1] # (?, list_size, candidates_size)，下标从0开始
    print('list_recall predict', predict.shape)
    
    gen_model_label = label_value_dict['context_info__real_show_list'] # 形如 [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    print('gen_model_label', gen_model_label.get_shape().as_list())
    gen_model_label = tf.reshape(gen_model_label, [-1, CANDIDATES_SIZE])
    show_label = tf.cast(tf.greater(gen_model_label, 0), tf.float32) # [None, candidates_size]
    true_label = show_label[:,:LIST_SIZE]
    print("true_label shape", true_label.get_shape().as_list())
    indices_matrix = tf.tile(tf.expand_dims(tf.range(0, CANDIDATES_SIZE), 0), [tf.shape(gen_model_label)[0], 1])
    print("indices_matrix shape ", indices_matrix.shape)
    true_index = tf.where(tf.greater(gen_model_label, 0), indices_matrix, tf.fill(tf.shape(gen_model_label), 0))
    true_index = true_index[:,:LIST_SIZE]
    print("true_index shape ", true_index.shape)

    _, rank_index = tf.math.top_k(predict, 1, sorted=True) # 返回最后一维最大值index, (?, list_size, 1)
    rank_index = tf.squeeze(rank_index, -1)

    # print_ops.append(tf.print('[train] true_label ', true_label[2], summarize = -1, output_stream=sys.stdout))
    # print_ops.append(tf.print('[train] true_index ', true_index[2], summarize = -1, output_stream=sys.stdout))
    # print_ops.append(tf.print('[train] select_index ', rank_index[2], summarize = -1, output_stream=sys.stdout))
    # print_ops.append(tf.print('[train] predict ', predict[2][:8], summarize = -1, output_stream=sys.stdout))

    hit_cnt = tf.reduce_sum(tf.cast(tf.equal(true_index, rank_index), tf.float32) * true_label, axis=-1, keep_dims=True)
    avg_precision = hit_cnt / (tf.reduce_sum(true_label, -1, keep_dims=True)+1e-9)
    tf.summary.scalar('avg_precision', tf.reduce_mean(avg_precision))
    # print_ops.append(tf.print('[train] avg_precision ', tf.reduce_mean(avg_precision), summarize = 8, output_stream=sys.stdout))

    # 不重复选取
    greedy_indices = greedy_search(predict) #bs,6
    greedy_hit = tf.batch_gather(show_label, greedy_indices)
    recall_6_th_greedy = tf.reduce_sum(greedy_hit, -1, keep_dims=True) / (tf.reduce_sum(true_label, -1, keep_dims=True)+1e-9)
    tf.summary.scalar('recall_6_th_greedy', tf.reduce_mean(recall_6_th_greedy))
    # print_ops.append(tf.print('[train] recall_6_th_greedy ', tf.reduce_mean(recall_6_th_greedy), summarize = 8, output_stream=sys.stdout))

def cal_batch_advantage(reward, mask):
    mask = tf.cast(mask, reward.dtype)
    valid_cnt = tf.reduce_sum(mask)
    mean = tf.reduce_sum(reward * mask) / (valid_cnt + 1e-8)
    variance = (reward - mean) ** 2 * mask
    std = tf.sqrt(tf.reduce_sum(variance) / (valid_cnt + 1e-8))
    advantages = (reward - mean) / (std + 1e-8)
    return advantages

def get_watch_time_from_vtr(pvtr, duration_s):
    buckets = [126.143,37.273,37.273,37.273,49.909,73.636,108.556,116.71,115.661,112.282,117.694,120.773,113.152,113.58,116.71,120.994,117.205,114.166,114.916,110.194,104.811,102.394,100.992,105.644,107.073,110.415,110.693,105.249,108.215,106.411,110.046,103.66,107.075,107.948,102.366,106.835,104.614,106.755,107.392,103.63,98.364,98.318,101.976,97.505,99.748,99.906,101.857,100.387,102.698,103.719,104.998,103.746,106.468,108.6,106.418,107.294,110.825,112.583,113.497,113.473,114.885,110.998,113.476,114.182,110.493,112.166,112.849,115.205,113.069,116.622,115.864,116.927,112.597,116.769,114.353,115.245,115.381,114.476,113.123,118.325,120.576,117.788,115.617,119.428,119.337,121.104,121.076,121.622,123.891,122.986,119.524,121.759,124.767,126.54,122.851,123.598,123.747,121.141,126.368,122.234,124.698,123.941,122.459,125.179,128.054,124.017,123.927,127.821,126.8,125.761,129.136,126.184,128.474,130.522,132.295,131.511,130.809,129.382,132.497,131.264,134.051,134.566,132.249,135.828,135.531,131.979,137.039,136.273,138.381,138.364,139.18,139.395,139.402,142.823,141.631,142.814,141.64,141.355,140.215,141.915,140.216,142.513,143.464,146.272,146.592,145.636,147.262,144.395,149.201,146.603,146.636,146.351,147.59,151.337,147.944,149.681,149.202,149.958,146.294,154.688,150.646,153.921,153.576,153.557,149.261,148.648,152.067,150.784,150.381,155.05,155.099,155.092,149.341,149.552,156.568,158.64,155.796,157.338,153.212,155.447,153.174,151.656,155.98,155.608,149.921,157.445,158.027,159.689,156.586,155.805,149.556,156.661,161.279,156.972,160.079,158.68,156.277,157.08,156.773,154.777,200.0]
    buckets = tf.constant(buckets, dtype=tf.float32) # (200,)
    vtr_max = tf.ones_like(duration_s, dtype=tf.int32) * 200
    # vtr_max = tf.constant(200, shape=[1, 1], dtype=tf.int32)
    # vtr_max = tf.tile(vtr_max, [tf.shape(duration_s)[0], tf.shape(duration_s)[1]]) # (cand_size, 1)
    print("vtr_max ", vtr_max, " duration_s ", duration_s)
    vtr_indices = tf.where(duration_s > 200, vtr_max, duration_s)
    print("vtr_indices ", vtr_indices)
    max_time = tf.gather(buckets, vtr_indices) # (cand_size, 1)
    print("max_time ", max_time)
    wt = pvtr * max_time
    return wt

def vtr_encode(playtime_s, duration_s):
    buckets = [178, 90, 90, 90, 90, 90, 90, 90, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 77, 79, 78, 79, 81, 83, 84, 86, 86, 86, 88, 89, 90, 90, 88, 88, 91, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 95, 96, 97, 98, 98, 98, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 103, 103, 103, 103, 103, 103, 103, 103, 103, 106, 106, 106, 106, 106, 104, 105, 106, 107, 107, 108, 110, 111, 111, 112, 113, 111, 111, 111, 111, 111, 111, 111, 111, 111, 113, 113, 113, 113, 113, 113, 113, 113, 116, 116, 116, 116, 116, 116, 116, 116, 116, 116, 116, 116, 123, 124, 125, 126, 127, 128, 129, 130, 132, 130, 130, 131, 133, 132, 131, 133, 134, 135, 137, 138, 135, 139, 139, 131, 136, 137, 131, 134, 139, 135, 130, 131, 139, 132, 136, 134, 131, 138, 131, 136, 133, 130, 139, 141, 140, 136, 137, 134, 136, 130, 136, 132, 141, 135, 130, 143, 135, 141, 142, 130, 135, 141, 130, 139, 138, 145, 135, 145, 134, 136, 143, 143, 143, 144, 137, 140, 134, 132, 135, 136, 131, 140, 145, 130, 149, 145, 132, 149, 147, 130, 136, 146, 133, 133, 139, 138, 146, 143, 130, 130, 133, 130, 130, 130, 147, 156, 149, 164, 145, 154, 129, 150, 151, 147, 152, 152, 147, 144, 144, 149, 152, 143, 148, 151, 147, 151, 149, 155, 148, 138, 140, 145, 142, 137, 139, 144, 146, 146, 154, 133, 143, 137, 141, 154, 140, 145, 151, 135, 143, 140, 142, 134, 141, 132, 135, 119, 135, 135, 128, 102]
    buckets = tf.constant(buckets, dtype=tf.float32) # (300,)
    vtr_max = tf.ones_like(duration_s, dtype=tf.int32) * 300
    duration_s = tf.cast(duration_s, dtype=tf.int32)
    print("vtr_max ", vtr_max, " duration_s ", duration_s)
    vtr_indices = tf.where(duration_s > 300, vtr_max, duration_s)
    print("vtr_indices ", vtr_indices)
    max_time = tf.cast(tf.gather(buckets, vtr_indices), dtype=tf.float32) # (cand_size, 1)
    print("max_time ", max_time)
    pvtr = tf.cast(playtime_s, dtype=tf.float32) * 1.0 / max_time
    return pvtr
def vtr_decode(pvtr, duration_s):
    buckets = [178, 90, 90, 90, 90, 90, 90, 90, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 77, 79, 78, 79, 81, 83, 84, 86, 86, 86, 88, 89, 90, 90, 88, 88, 91, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 95, 96, 97, 98, 98, 98, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 103, 103, 103, 103, 103, 103, 103, 103, 103, 106, 106, 106, 106, 106, 104, 105, 106, 107, 107, 108, 110, 111, 111, 112, 113, 111, 111, 111, 111, 111, 111, 111, 111, 111, 113, 113, 113, 113, 113, 113, 113, 113, 116, 116, 116, 116, 116, 116, 116, 116, 116, 116, 116, 116, 123, 124, 125, 126, 127, 128, 129, 130, 132, 130, 130, 131, 133, 132, 131, 133, 134, 135, 137, 138, 135, 139, 139, 131, 136, 137, 131, 134, 139, 135, 130, 131, 139, 132, 136, 134, 131, 138, 131, 136, 133, 130, 139, 141, 140, 136, 137, 134, 136, 130, 136, 132, 141, 135, 130, 143, 135, 141, 142, 130, 135, 141, 130, 139, 138, 145, 135, 145, 134, 136, 143, 143, 143, 144, 137, 140, 134, 132, 135, 136, 131, 140, 145, 130, 149, 145, 132, 149, 147, 130, 136, 146, 133, 133, 139, 138, 146, 143, 130, 130, 133, 130, 130, 130, 147, 156, 149, 164, 145, 154, 129, 150, 151, 147, 152, 152, 147, 144, 144, 149, 152, 143, 148, 151, 147, 151, 149, 155, 148, 138, 140, 145, 142, 137, 139, 144, 146, 146, 154, 133, 143, 137, 141, 154, 140, 145, 151, 135, 143, 140, 142, 134, 141, 132, 135, 119, 135, 135, 128, 102]
    buckets = tf.constant(buckets, dtype=tf.float32) # (200,)
    vtr_max = tf.ones_like(duration_s, dtype=tf.int32) * 200
    duration_s = tf.cast(duration_s, dtype=tf.int32)
    print("vtr_max ", vtr_max, " duration_s ", duration_s)
    vtr_indices = tf.where(duration_s > 300, vtr_max, duration_s)
    print("vtr_indices ", vtr_indices)
    max_time = tf.cast(tf.gather(buckets, vtr_indices), dtype=tf.float32) # (cand_size, 1)
    print("max_time ", max_time)
    wt = pvtr * max_time
    return wt

def wtd_encode(duration, play_time, duration_bucket, play_time_buckets_ragged):
    '''
      duration: (?,cand_size)
      play_time: (?,cand_size)
      2-D wtd
    '''
    duration_bucket = tf.tile(tf.expand_dims(duration_bucket, axis=0), [tf.shape(duration)[0], 1])
    bucket_idx = tf.searchsorted(sorted_sequence=duration_bucket, values=tf.cast(duration, tf.float32), out_type=tf.int32) # (?,cand_size)
    print("bucket_idx ", bucket_idx)
    selected_play_buckets = tf.gather(play_time_buckets_ragged, bucket_idx) # (?, cand_size, ?)
    play_time_bucket_lengths = play_time_buckets_ragged.row_lengths() # (?,cand_size,2)
    selected_play_buckets_dense = selected_play_buckets.to_tensor(default_value=1e19)  # (?, cand_size, max_len)
    bucket_idx2 = tf.searchsorted(sorted_sequence=selected_play_buckets_dense,
                                  values=tf.expand_dims(tf.cast(play_time, tf.float32), axis=-1), out_type=tf.int32) # (?,cand_size,1)
    bucket_idx2 = tf.squeeze(bucket_idx2, axis=-1)
    print("bucket_idx2 ", bucket_idx2)
    selected_bucket_lengths = tf.gather(play_time_bucket_lengths, bucket_idx) # (?, cand_size)
    ratio = tf.cast(bucket_idx2, tf.float32) / tf.cast(selected_bucket_lengths + 1, tf.float32)
    ratio = tf.clip_by_value(ratio, 0, 1)

    return ratio

def wtd_decode(ratio, duration, duration_bucket, play_time_buckets_ragged):
    '''
      duration: (?,cand_size)
      ratio: (?,cand_size)
      2-D wtd
    '''
    batch_size = tf.shape(duration)[0]
    cand_size = duration.shape[1]
    duration_bucket = tf.tile(tf.expand_dims(duration_bucket, axis=0), [batch_size, 1])
    play_time_bucket_lengths = play_time_buckets_ragged.row_lengths() # (?,cand_size,2)
    bucket_idx = tf.searchsorted(sorted_sequence=duration_bucket, values=tf.cast(duration, tf.float32), out_type=tf.int32) # (?,cand_size)
    selected_play_buckets = tf.gather(play_time_buckets_ragged, bucket_idx) # (?, cand_size, ?)
    selected_bucket_lengths = tf.gather(play_time_bucket_lengths, bucket_idx) # (?, cand_size)
    print("selected_bucket_lengths ", selected_bucket_lengths)
    original_bucket_idx = ratio * tf.cast(selected_bucket_lengths + 1, tf.float32) # (?, cand_size)
    lower_bucket_id = tf.floor(original_bucket_idx) # (?, cand_size)
    lower_bucket_id = tf.where(lower_bucket_id < 0.0, tf.zeros_like(lower_bucket_id), lower_bucket_id)
    print("original_bucket_value ", original_bucket_idx)
    lower_bucket_id = tf.where(tf.cast(lower_bucket_id, tf.int64) > selected_bucket_lengths, selected_bucket_lengths, tf.cast(lower_bucket_id, tf.int64))
    print("lower_bucket_id ", lower_bucket_id)
    # 需要补充边界值 最小值
    selected_play_buckets = tf.concat([tf.zeros([batch_size, cand_size, 1], dtype=selected_play_buckets.dtype), selected_play_buckets], axis=-1)
    print("selected_bucket_lengths ", selected_bucket_lengths)
    # 最后一维不同因此不支持直接gather 转为 tensor
    selected_play_buckets = selected_play_buckets.to_tensor(default_value=-1) # (?, cand_size, max_size)
    print("selected_play_buckets ", selected_play_buckets)
    lower_bucket_value = tf.gather(selected_play_buckets, tf.expand_dims(lower_bucket_id, axis=-1), batch_dims=2)
    lower_bucket_value = tf.squeeze(lower_bucket_value, axis=-1) # (?, cand_size)
    print("lower_bucket_value ", lower_bucket_value)
    higher_bucket_id = tf.where(lower_bucket_id >= selected_bucket_lengths, selected_bucket_lengths, lower_bucket_id + 1)
    print("higher_bucket_id ", higher_bucket_id)
    higher_bucket_value = tf.gather(selected_play_buckets, tf.expand_dims(higher_bucket_id, axis=-1), batch_dims=2)
    higher_bucket_value = tf.squeeze(higher_bucket_value, axis=-1) # (?, cand_size)
    print("higher_bucket_value ", higher_bucket_value)
    play_time = tf.where(lower_bucket_id >= selected_bucket_lengths, lower_bucket_value,
                         lower_bucket_value + (higher_bucket_value - lower_bucket_value) \
                         * (original_bucket_idx - tf.cast(lower_bucket_id, tf.float32)))
    return play_time

def wilson_smoothing(clicks, impressions, z=1.96):
    clicks = tf.cast(clicks, tf.float32)
    impressions = tf.cast(impressions, tf.float32) + 1e-7
    # 计算原始CTR (防止除0)
    p = tf.where(impressions > 0, clicks / impressions, tf.zeros_like(impressions))
    # Wilson平滑计算
    z2 = tf.square(z)
    n = impressions
    term1 = (p + z2/(2*n)) / (1 + z2/n)
    term2 = z * tf.sqrt((p*(1-p) + z2/(4*n)) / (n + 1e-7)) / (1 + z2/n)
    smoothed_ctr = term1 - term2
    return tf.clip_by_value(smoothed_ctr, 0.0, 1.0)

def get_base_label(duration_s, playtime_s):
    # ltr
    base_ltr_label = tf.where(tf.less(duration_s, 7.0), tf.greater(playtime_s, 7.0),
                      tf.where(tf.logical_and(tf.greater_equal(duration_s, 7.0), tf.less(duration_s, 18.0)),
                               tf.greater(playtime_s, duration_s), tf.greater(playtime_s, 18.0)))
    base_ltr_label = tf.cast(base_ltr_label, tf.float32)
    # lvtr
    threshold = (duration_s * 28.0 + 180.0) / 33.0
    base_lvtr_label = tf.where(tf.less(duration_s, 3.0), tf.greater_equal(playtime_s, 18.0),
        tf.where(tf.less(duration_s, 36.0), tf.greater_equal(playtime_s, threshold), tf.greater_equal(playtime_s, 36.0))
    )
    base_lvtr_label = tf.cast(base_lvtr_label, tf.float32)
    # finish
    finish_threshold = tf.where(tf.less(duration_s, 18.0), tf.ones_like(duration_s) * 18.0, duration_s)
    base_finish_label = tf.greater_equal(playtime_s, finish_threshold)
    base_finish_label = tf.cast(base_finish_label, tf.float32)
    return base_ltr_label, base_lvtr_label, base_finish_label, None

def get_play_labels(duration, play_time):
    # boundaries = [0.0, 3.0, 9.0, 14.0, 22.0, 40.0, 67.0, 165.0]
    # thresholds = [[12.0, 18.0], [7.0, 18.0], [7.0, 18.0], [7.0, 18.0], [7.0, 20.0], [7.0, 28.0], [7.0, 27.0], [7.0, 23.0]]
    boundaries = [0.0, 9.0, 14.0, 22.0, 40.0, 67.0, 165.0]
    thresholds = [[14.0, 18.0], [10.0, 18.0], [11.0, 18.0], [12.0, 20.0], [14.0, 28.0], [13.0, 27.0], [9.0, 23.0]]
    boundaries_tensor = tf.constant(boundaries, dtype=tf.float32)
    boundaries_tensor = tf.tile(tf.expand_dims(boundaries_tensor, axis=0), [tf.shape(duration)[0], 1])
    bucket_idx = tf.searchsorted(
        sorted_sequence=boundaries_tensor,
        values=tf.cast(duration, tf.float32),
        side="right"
    ) - 1
    max_idx = tf.constant(len(thresholds) - 1, dtype=tf.int32)
    bucket_idx = tf.clip_by_value(bucket_idx, 0, max_idx)
    evtr_threshold = tf.gather(tf.constant([t[0] for t in thresholds]), bucket_idx)
    lvtr_threshold = tf.gather(tf.constant([t[1] for t in thresholds]), bucket_idx)
    
    # 判断播放时长是否达到阈值
    evtr_label = tf.cast(tf.greater_equal(play_time, evtr_threshold), dtype=tf.float32)
    lvtr_label = tf.cast(tf.greater_equal(play_time, lvtr_threshold), dtype=tf.float32)
    
    return evtr_label, lvtr_label

# print info
worker_global_step = config.get_step()
ops = [tf.print("====> step", worker_global_step, summarize=-1, output_stream=sys.stdout)]

############################################################
############################################################
# wtd 配置
wtd_buckets = tf.constant(wtd_config["buckets"], dtype=tf.float32)
wtd_configs = tf.ragged.constant(wtd_config["configs"], dtype=tf.float32)
# wtd_buckets_fountain = tf.constant(wtd_config_fountain["buckets"], dtype=tf.float32)
# wtd_configs_fountain = tf.ragged.constant(wtd_config_fountain["configs"], dtype=tf.float32)
list_wtd_buckets_fountain = tf.constant(list_wtd_config_fountain["buckets"], dtype=tf.float32)
list_wtd_configs_fountain = tf.ragged.constant(list_wtd_config_fountain["configs"], dtype=tf.float32)
# 获取模型output
all_param_dict, feature_emb_size_dict, _ = get_param_dict()
label_value_dict = {}
# label
label_value_dict["context_info__first_screen"] = tf.cast(tf.reshape(config.get_dense_fea("context_info__first_screen", dim=1, dtype=tf.int64), [-1, 1]), dtype=tf.float32)
label_value_dict["show_label"] = tf.cast(tf.reshape(config.get_dense_fea("context_info__real_show_list", dim=CANDIDATES_SIZE, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["play_time_s"] = tf.cast(tf.reshape(config.get_dense_fea("context_info__playing_time_list", dim=CANDIDATES_SIZE, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32) / 1000.0
label_value_dict["play_time_s"] = tf.clip_by_value(label_value_dict["play_time_s"], 0, 3600)
label_value_dict["like_label"] = tf.cast(tf.reshape(config.get_dense_fea("context_info__like_list", dim=CANDIDATES_SIZE, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["follow_label"] = tf.cast(tf.reshape(config.get_dense_fea("context_info__follow_list", dim=CANDIDATES_SIZE, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["comment_label"] = tf.cast(tf.reshape(config.get_dense_fea("context_info__comment_list", dim=CANDIDATES_SIZE, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["forward_label"] = tf.cast(tf.reshape(config.get_dense_fea("context_info__forward_list", dim=CANDIDATES_SIZE, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["slide_label"] = tf.cast(tf.reshape(config.get_dense_fea("context_info__fountain_slide_to_next_list", dim=CANDIDATES_SIZE, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["click_label"] = tf.cast(tf.reshape(config.get_label("fountain_click_label_list", dim=CANDIDATES_SIZE), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["wtd_label"] = tf.cast(tf.reshape(config.get_label("fountain_wtd_label_list", dim=CANDIDATES_SIZE), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["finish_label"] = tf.cast(tf.reshape(config.get_label("fountain_finish_label_list", dim=CANDIDATES_SIZE), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["ltr_label"] = tf.cast(tf.reshape(config.get_label("fountain_ltr_label_list", dim=CANDIDATES_SIZE), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["evtr_label"], label_value_dict["lvtr_label"] = get_play_labels(label_value_dict["photo_info__duration_ms_list"] / 1000, label_value_dict["play_time_s"])
# dense feature
dense_dim = CANDIDATES_SIZE if is_training else 1
label_value_dict["pctr"] = tf.cast(tf.reshape(config.get_label("context_info__pctr_list", dim=dense_dim), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["plvtr"] = tf.cast(tf.reshape(config.get_label("context_info__plvtr_list", dim=dense_dim), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["pvtr"] = tf.cast(tf.reshape(config.get_label("context_info__pvtr_list", dim=dense_dim), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["pwtd"] = tf.cast(tf.reshape(config.get_label("context_info__pwtd_list", dim=dense_dim), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["pltr"] = tf.cast(tf.reshape(config.get_label("context_info__pltr_list", dim=dense_dim), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["pcmtr"] = tf.cast(tf.reshape(config.get_label("context_info__pcmtr_list", dim=dense_dim), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["pwtr"] = tf.cast(tf.reshape(config.get_label("context_info__pwtr_list", dim=dense_dim), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["photo_info__duration_ms_list"] = tf.cast(tf.reshape(config.get_dense_fea("photo_info__duration_ms_list", dim=dense_dim, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["photo_info__duration_ms_list"] = tf.clip_by_value(label_value_dict["photo_info__duration_ms_list"], 0, 36000 * 1000)
label_value_dict["photo_info__explore_stat__click_count_list"] = tf.cast(tf.reshape(config.get_dense_fea("photo_info__explore_stat__click_count_list", dim=dense_dim, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["photo_info__explore_stat__real_show_count_list"] = tf.cast(tf.reshape(config.get_dense_fea("photo_info__explore_stat__real_show_count_list", dim=dense_dim, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["photo_info__explore_stat__like_count_list"] = tf.cast(tf.reshape(config.get_dense_fea("photo_info__explore_stat__like_count_list", dim=dense_dim, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["photo_info__explore_stat__long_play_count_list"] = tf.cast(tf.reshape(config.get_dense_fea("photo_info__explore_stat__long_play_count_list", dim=dense_dim, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["photo_info__explore_stat__short_play_count_list"] = tf.cast(tf.reshape(config.get_dense_fea("photo_info__explore_stat__short_play_count_list", dim=dense_dim, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["photo_info__explore_stat__follow_count_list"] = tf.cast(tf.reshape(config.get_dense_fea("photo_info__explore_stat__follow_count_list", dim=dense_dim, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["photo_info__explore_stat__view_length_sum_list"] = tf.cast(tf.reshape(config.get_dense_fea("photo_info__explore_stat__view_length_sum_list", dim=dense_dim, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["photo_info__fountain_stats__real_show_count_list"] = tf.cast(tf.reshape(config.get_dense_fea("photo_info__fountain_stats__real_show_count_list", dim=dense_dim, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["photo_info__fountain_stats__like_count_list"] = tf.cast(tf.reshape(config.get_dense_fea("photo_info__fountain_stats__like_count_list", dim=dense_dim, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["photo_info__fountain_stats__follow_count_list"] = tf.cast(tf.reshape(config.get_dense_fea("photo_info__fountain_stats__follow_count_list", dim=dense_dim, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["photo_info__fountain_stats__long_play_count_list"] = tf.cast(tf.reshape(config.get_dense_fea("photo_info__fountain_stats__long_play_count_list", dim=dense_dim, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["photo_info__fountain_stats__short_play_count_list"] = tf.cast(tf.reshape(config.get_dense_fea("photo_info__fountain_stats__short_play_count_list", dim=dense_dim, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["photo_info__fountain_stats__view_length_sum_list"] = tf.cast(tf.reshape(config.get_dense_fea("photo_info__fountain_stats__view_length_sum_list", dim=dense_dim, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32)

label_value_dict["photo_emp_explore_ctr"] = wilson_smoothing(label_value_dict["photo_info__explore_stat__click_count_list"],
                                                             label_value_dict["photo_info__explore_stat__real_show_count_list"])
label_value_dict["photo_emp_explore_ltr"] = wilson_smoothing(label_value_dict["photo_info__explore_stat__like_count_list"],
                                                             label_value_dict["photo_info__explore_stat__click_count_list"])
label_value_dict["photo_emp_explore_avg_time"] = tf.cast(label_value_dict["photo_info__explore_stat__view_length_sum_list"], tf.float32) / 1000.0 \
                                                / (tf.cast(label_value_dict["photo_info__explore_stat__click_count_list"], tf.float32) + 1e-7)
label_value_dict["photo_emp_fountain_lvtr"] = wilson_smoothing(label_value_dict["photo_info__fountain_stats__long_play_count_list"],
                                                               label_value_dict["photo_info__fountain_stats__real_show_count_list"])
label_value_dict["photo_emp_fountain_svtr"] = wilson_smoothing(label_value_dict["photo_info__fountain_stats__short_play_count_list"],
                                                               label_value_dict["photo_info__fountain_stats__real_show_count_list"])
label_value_dict["photo_emp_fountain_ltr"] = wilson_smoothing(label_value_dict["photo_info__fountain_stats__like_count_list"],
                                                              label_value_dict["photo_info__fountain_stats__real_show_count_list"])
label_value_dict["photo_emp_fountain_wtr"] = wilson_smoothing(label_value_dict["photo_info__fountain_stats__follow_count_list"],
                                                              label_value_dict["photo_info__fountain_stats__real_show_count_list"])
label_value_dict["photo_emp_fountain_avg_time"] = tf.cast(label_value_dict["photo_info__fountain_stats__view_length_sum_list"], dtype=tf.float32) / 1000.0 \
                                                / (tf.cast(label_value_dict["photo_info__fountain_stats__real_show_count_list"], dtype=tf.float32) + 1e-7)
label_value_dict["photo_emp_fountain_avg_fintr"] = tf.cast(label_value_dict["photo_emp_fountain_avg_time"], dtype=tf.float32) \
                                                / (tf.cast(label_value_dict["photo_info__duration_ms_list"], dtype=tf.float32) / 1000.0 + 1e-7) # 确认下是否为 ms


point_wise_tasks = ["ltr", "vtr", "click", "slide", "wtd"]
# list_wise_tasks = ["listwise_wtd"]
list_wise_tasks = []
model_class = GenModel(all_param_dict, label_value_dict, print_ops, list_size=LIST_SIZE, candidates_size=CANDIDATES_SIZE, list_num=LIST_NUM,
                             point_wise_tasks=point_wise_tasks, list_wise_tasks=list_wise_tasks)


if is_training:
    print(f"====> train, gen...")

    predict, nce_loss, gen_loss, bpr_loss, reward_loss, sub_seq_loss = model_class.model(training=True)
    print("return predict ",predict.shape)
    print_ops = model_class.print_ops

    targets = []
    # loss = gen_loss
    loss = gen_loss + sub_seq_loss / 2.0 + nce_loss / 1000.0 + bpr_loss
    # loss = gen_loss + reward_loss / 50.0 + bpr_loss / 20.0
    tf.summary.scalar('gen_loss', gen_loss)
    tf.summary.scalar('nce_loss', nce_loss)
    tf.summary.scalar('sub_seq_loss', sub_seq_loss)
    tf.summary.scalar('reward_loss', reward_loss)
    tf.summary.scalar('bpr_loss', bpr_loss)
    
    list_recall(predict, label_value_dict)

    with tf.control_dependencies(print_ops):
        logits = tf.reduce_sum(predict, axis=-1)
        logits = tf.expand_dims(logits, axis=-1)
        zero = tf.zeros_like(logits)
        one = tf.ones_like(logits)
        print("zero shape", zero.shape)
        print("one shape", one.shape)
        targets.append(('recall', logits, zero, one, 'linear_regression'))

    if args.with_kai_v2:
        sparse_optimizer = config.optimizer.Adam(0.001)
        # dense_optimizer = config.optimizer.Adam(0.00005)
        dense_optimizer = config.optimizer.Adam(0.0001)
        sparse_optimizer.minimize(loss, var_list=config.get_collection(config.GraphKeys.EMBEDDING_INPUT))
        dense_optimizer.minimize(loss, var_list=config.get_collection(config.GraphKeys.TRAINABLE_VARIABLES))
        opts = [sparse_optimizer, dense_optimizer]
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
        config.dump_kai_training_config('./training', targets, loss=loss, text=args.text, init_params_in_tf=True)
    elif args.with_kai_v2:
        config.build_model(optimizer=opts, metrics=targets)
    else:
        config.dump_training_config('./training', targets, opts=opts, text=args.text)
else:
    model_class._training = False
    # logits: [batch_size, beam_size, seq_length, vocab_size], 
    # generated_sequence: [batch_size, beam_size, seq_length]
    # probs[0]: [batch_size, beam_size, vocab_size]
    beam_size = 2
    max_length = 12
    logits, generated_sequence, preward, best_sequences, probs = model_class.model(training=False, beam_size=beam_size, max_length=max_length)
    photo_id_emb = all_param_dict["photo_id"]
    # context_cascade_pctr_emb = all_param_dict["context_cascade_pctr"]

    targets = []

    # best_sequence = best_sequences[0] - 2 # (4,)
    # size = tf.shape(best_sequence)[0]
    # values = tf.range(size, 0, -1) # 递减score [4, 3, 2, 1]
    # vocab_size = tf.shape(logits)[2]-3
    # tensor_zeros = tf.zeros(vocab_size, dtype=tf.int32)
    # scores = tf.tensor_scatter_nd_update(tensor_zeros, tf.expand_dims(best_sequence, axis=1), values)
    # scores = tf.cast(tf.reshape(scores, [-1, 1]), dtype=tf.float32)
    # print(f"scores shape: {scores.shape}")
    # targets.append((f"rerank_gen_score_0",scores))
    selected_indices = generated_sequence[0] - 2 # [beam_size, seq_length]
    vocab_size = tf.shape(logits)[-1] - 3
    for i in range(beam_size):
        selected_indices_i = selected_indices[i, :] # [seq_length]
        tensor_zeros = tf.zeros(vocab_size, dtype=tf.int32)
        output_tensor = tf.tensor_scatter_nd_update(tensor_zeros, tf.expand_dims(selected_indices_i, 1), tf.range(selected_indices_i.shape[0], 0, -1))
        output_tensor = tf.reshape(output_tensor, [-1, 1])
        output_tensor = tf.cast(output_tensor, dtype=tf.float32)
        # print("output_tensor shape", output_tensor.shape)
        pred_output = tf.identity(output_tensor) #(?,1)
        print("pred_output shape",pred_output.shape)
        targets.append((f"rerank_gen_score_{i}",pred_output))

    probs = [tf.transpose(x[0][:, 2:-1], perm=[1, 0]) for x in probs] # (30, beam_size)
    targets.append(("photo_id_emb", tf.identity(photo_id_emb)))
    # targets.append(("context_cascade_pctr_emb", tf.identity(context_cascade_pctr_emb)))
    # targets.append(("preward", tf.identity(tf.reshape(preward[:,2:-1], [-1, 1]))))
    targets.append(("logits_0", tf.identity(probs[0])))
    targets.append(("logits_1", tf.identity(probs[1])))
    targets.append(("logits_2", tf.identity(probs[2])))
    targets.append(("logits_3", tf.identity(probs[3])))
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
