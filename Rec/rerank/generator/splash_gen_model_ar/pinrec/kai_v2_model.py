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
from model_v1 import FountainDeepLtrMultiTaskModel

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'predict'], dest='mode', default='train')
parser.add_argument('--dryrun', dest='dryrun', const=True, default=False, nargs='?')
parser.add_argument('--with_kai', action='store_true', default=False)
parser.add_argument('--text', default=False)
parser.add_argument('--tower', choices=None, dest='tower', default='False')
parser.add_argument('--with_kai_v2', action='store_false', default=True)
args = parser.parse_known_args()[0]
is_training = args.mode == "train"
print("args: ", args)
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
                                             recycle_method=config.nn.UnseendaysRecycle(delete_after_unseen_days=30, delete_threshold=0.1, allow_dynamic_delete=True))
    config.nn.set_default_param_attr(default_param_attr)

    def filter_mask_wrapper(dataset):
        # 1. 声明字段
        #  sample_type为字段名，特征类型dataset.DENSE表示稠密，tf.int64为数据类型，dim为1
        # dataset.add_feature('fountain_fulllink_rerank_index_list', dataset.DENSE, tf.int64, max_length=60)
        # dataset.add_feature('fountain_fulllink_rerank_index_weight_list', dataset.DENSE, tf.int64, max_length=60)
        dataset.add_feature('context_info__real_show_index_list', dataset.DENSE, tf.int64, max_length=60)
        dataset.add_feature('context_info__real_show_list', dataset.DENSE, tf.int64, max_length=60)
        dataset.add_feature('context_info__first_screen', dataset.DENSE, tf.int64, max_length=60)
        dataset.add_feature('tab', dataset.DENSE, tf.int64, max_length=60)


        # 2.声明mask，batch是一个dict，key为声明的字段名，value根据特征类型分为2种情况：
        # dataset.DENSE: 值为tf.Tensor
        # dataset.SPARSE: 值为元组: (tf.Tensor, tf.Tensor)，
        #   其中第一个tensor表示feasign，第二个tensor表示cumsum
        #   可以使用tf.RaggedTensor.from_row_splits转成RaggedTensor
        def mask_fn(batch):
            realshow = batch["context_info__real_show_list"]
            realshow = tf.RaggedTensor.from_row_splits(realshow[0], row_splits= realshow[1])
            realshow = realshow.to_tensor()
            
            realshow = tf.reduce_sum(realshow, axis=-1)
            
            context_page = batch['context_info__first_screen']
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
            if attr.attr_name in photo_fea_names:
                if not attr.expand:
                    attr.expand = CANDIDATES_SIZE
                else:
                    attr.expand *= CANDIDATES_SIZE
            else:
                if args.with_kai:
                    if not attr.expand:
                        attr.expand = 1
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
    
    gen_model_label = label_value_dict['show_label'] # 形如 [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
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
    has_pos = tf.greater(tf.reduce_sum(true_label, -1, keep_dims=True), 0)
    safe_recall = tf.where(has_pos, recall_6_th_greedy, tf.zeros_like(recall_6_th_greedy))
    tf.summary.scalar('recall_6_th_greedy', tf.reduce_mean(safe_recall))
    # print_ops.append(tf.print('[train] recall_6_th_greedy ', tf.reduce_mean(recall_6_th_greedy), summarize = 8, output_stream=sys.stdout))

def get_play_labels(duration, play_time):
    # boundaries = [0.0, 3.0, 7.0, 9.0, 12.0, 17.0, 20.0, 58.0, 90.0, 120.0, 180, 300, 420, 600]
    # thresholds = [[5.6, 7.7], [5.5, 7.0], [8.6, 12.1], [9.5, 12.2], [11.4, 13.4], [14.3, 16.7], [17.9, 20.2], [20.5, 28.8], [22.1, 48.2], [19.9, 46.3], [18.9, 45.1], [14.3, 36.9], [11.0, 28.4], [10.6, 27.0], [8.5, 21.9]]
    # >3s的25分位数、50分位数
    boundaries = [0.0, 8.366, 10.3, 12.433, 15.066, 17.3, 20.431, 24.833, 29.333, 33.916, 39.033, 46.566, 54.7, 62.933, 76.366, 99.166, 132.566, 178.266, 235, 360.433, 1108.266,]
    thresholds = [[6.055, 10.156], [7.911, 10.059], [9.32, 11.301], [9.86, 12.584], [9.925, 14.407], [10.756, 16.37], [10.744, 18.545], [10.559, 20.73], [11.39, 24.744], [11.658, 27.298], [12.064, 29.611], [12.791, 32.704], [13.126, 35.619], [12.963, 37.619], [13.138, 40.433], [12.474, 40.972], [12.375, 43.145], [11.915, 42.533], [10.931, 38.012], [9.436, 28.892], [8.499, 20.747], [8.738, 19.911]]

    # 0506版本，有效播放50%分位数，长播65分位数
    # boundaries = [0, 8.366, 10.3, 12.433, 15.066, 17.3, 20.431, 24.833, 29.333, 33.916, 39.033, 46.566, 54.7, 62.933, 76.366, 99.166, 178.266, 235, 360.433, 1108.266]
    # thresholds = [[4.529, 8.124], [8.56, 10.781], [10.154, 11.684], [11.228, 12.894], [12.009, 14.768], [13.51, 16.661], [13.406, 18.912], [13.038, 21.209], [14.57, 25.462], [15.108, 28.599], [16.205, 31.515], [17.891, 35.461], [18.748, 38.867], [18.451, 41.105], [19.012, 44.266], [17.148, 43.874], [15.472, 44.873], [13.181, 38.552], [10.074, 27.184], [8.925, 19.689], [9.554, 19.591]]
    boundaries_tensor = tf.constant(boundaries, dtype=tf.float32)
    boundaries_tensor = tf.tile(tf.expand_dims(boundaries_tensor, axis=0), [tf.shape(duration)[0], 1])
    bucket_idx = tf.searchsorted(sorted_sequence=boundaries_tensor, values=tf.cast(duration, tf.float32), side="left") # 左开右闭
    max_idx = tf.constant(len(thresholds) - 1, dtype=tf.int32)
    bucket_idx = tf.clip_by_value(bucket_idx, 0, max_idx)
    evtr_threshold = tf.gather(tf.constant([t[0] for t in thresholds]), bucket_idx)
    lvtr_threshold = tf.gather(tf.constant([t[1] for t in thresholds]), bucket_idx)
    
    # 判断播放时长是否达到阈值
    evtr_label = tf.cast(tf.greater_equal(play_time, evtr_threshold), dtype=tf.float32)
    # lvtr_label = tf.cast(tf.greater_equal(play_time, lvtr_threshold), dtype=tf.float32)
    thres_mid = (duration * 28.0 + 180.0) / 33.0  # [B, 1]，单位 s
    lvtr_threshold = tf.where(
        tf.less(duration, 3.0),
        tf.fill(tf.shape(duration), 18.0),
        tf.where(
            tf.greater(duration, 36.0),
            tf.fill(tf.shape(duration), 36.0),
            thres_mid
        )
    )  # [B, 1]
    lvtr_label = tf.cast(tf.greater_equal(play_time, lvtr_threshold), dtype=tf.float32)
    svtr_label = tf.where(play_time < tf.minimum(duration, 3.0), tf.ones_like(play_time), tf.zeros_like(play_time))
    
    return evtr_label, lvtr_label, svtr_label


# print info
worker_global_step = config.get_step()
ops = [tf.print("====> step", worker_global_step, summarize=-1, output_stream=sys.stdout)]

############################################################
############################################################
# wtd 配置
wtd_buckets = tf.constant(wtd_config["buckets"], dtype=tf.float32)
wtd_configs = tf.ragged.constant(wtd_config["configs"], dtype=tf.float32)
# 获取模型output
all_param_dict, feature_emb_size_dict = get_param_dict()
label_value_dict = {}
# label
label_value_dict["show_label"] = tf.cast(tf.reshape(config.get_dense_fea("context_info__real_show_list", dim=CANDIDATES_SIZE, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["play_time_s"] = tf.cast(tf.reshape(config.get_dense_fea("context_info__playing_time_list", dim=CANDIDATES_SIZE, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32) / 1000.0
label_value_dict["like_label"] = tf.cast(tf.reshape(config.get_dense_fea("context_info__like_list", dim=CANDIDATES_SIZE, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["follow_label"] = tf.cast(tf.reshape(config.get_dense_fea("context_info__follow_list", dim=CANDIDATES_SIZE, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["comment_label"] = tf.cast(tf.reshape(config.get_dense_fea("context_info__comment_list", dim=CANDIDATES_SIZE, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["forward_label"] = tf.cast(tf.reshape(config.get_dense_fea("context_info__forward_list", dim=CANDIDATES_SIZE, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["slide_label"] = tf.cast(tf.reshape(config.get_dense_fea("context_info__fountain_slide_to_next_list", dim=CANDIDATES_SIZE, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["click_label"] = tf.cast(tf.reshape(config.get_dense_fea("context_info__click_list", dim=CANDIDATES_SIZE, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["wtd_label"] = tf.cast(tf.reshape(config.get_label("fountain_wtd_label_list", dim=CANDIDATES_SIZE), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["finish_label"] = tf.cast(tf.reshape(config.get_label("fountain_finish_label_list", dim=CANDIDATES_SIZE), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["ltr_label"] = tf.cast(tf.reshape(config.get_label("fountain_ltr_label_list", dim=CANDIDATES_SIZE), [-1, CANDIDATES_SIZE]), dtype=tf.float32)

# dense feature
dense_dim = CANDIDATES_SIZE if is_training else 1
label_value_dict["pctr"] = tf.cast(tf.reshape(config.get_label("context_info__pctr_list", dim=dense_dim), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["plvtr"] = tf.cast(tf.reshape(config.get_label("context_info__plvtr_list", dim=dense_dim), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["pvtr"] = tf.cast(tf.reshape(config.get_label("context_info__pvtr_list", dim=dense_dim), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["pwtd"] = tf.cast(tf.reshape(config.get_label("context_info__pwtd_list", dim=dense_dim), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["pltr"] = tf.cast(tf.reshape(config.get_label("context_info__pltr_list", dim=dense_dim), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["pcmtr"] = tf.cast(tf.reshape(config.get_label("context_info__pcmtr_list", dim=dense_dim), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["pwtr"] = tf.cast(tf.reshape(config.get_label("context_info__pwtr_list", dim=dense_dim), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
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
label_value_dict["photo_info__duration_ms_list"] = tf.cast(tf.reshape(config.get_dense_fea("photo_info__duration_ms_list" if is_training else "duration_ms_list", dim=dense_dim, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32)

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

play_time_s = label_value_dict["play_time_s"]
duration_s = label_value_dict["photo_info__duration_ms_list"] / 1000
evtr_label, lvtr_label, svtr_label = get_play_labels(duration_s, play_time_s)
label_value_dict["evtr_label"] = evtr_label
label_value_dict["lvtr_label"] = lvtr_label
label_value_dict["svtr_label"] = svtr_label

model_class = FountainDeepLtrMultiTaskModel(all_param_dict, label_value_dict, print_ops, list_size=LIST_SIZE, candidates_size=CANDIDATES_SIZE)


if is_training:
    print(f"====> train, gen...")

    predict, nce_loss, gen_loss, bpr_loss = model_class.model(training=True)
    # predict (?, list_size, cand_size)
    # gen_loss = gen_loss # 对齐量级
    nce_loss = nce_loss / 10000.0 # 对齐量级
    bpr_loss = bpr_loss / 100.0 # 容易主导训练
    print("return predict ",predict.shape)
    print_ops = model_class.print_ops

    targets = []
    loss = gen_loss + nce_loss
    # loss = gen_loss
    tf.summary.scalar('gen_loss', gen_loss)
    tf.summary.scalar('nce_loss', nce_loss)
    # tf.summary.scalar('bpr_loss', bpr_loss)
    
    list_recall(predict, label_value_dict)

    with tf.control_dependencies(print_ops):
        # pos0_pred = predict[:, 0, :] # (?, cand_size)
        show_label = label_value_dict["show_label"] # (?, cand_size)
        show_label = tf.tile(tf.expand_dims(show_label, axis=1), [1, LIST_SIZE, 1])
        predict = predict[:,:-1,2:-1] # (?, list_size, candidates_size)，下标从0开始
        targets.append(('show_avg', predict, show_label, tf.ones_like(predict, dtype=tf.float32), 'auc'))

    if args.with_kai_v2:
        sparse_optimizer = config.optimizer.Adam(0.0005)
        dense_optimizer = config.optimizer.Adam(0.00005)
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
    beam_size = 5
    max_length = 6
    # max_length = 20
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
    # for i in range(10):
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
    # targets.append(("context_cascade_pctr_emb", tf.identity(context_cascade_pctr_emb)))
    targets.append(("preward", tf.identity(tf.reshape(preward[:,2:-1], [-1, 1]))))
    targets.append(("logits_0", tf.identity(probs[0])))
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
