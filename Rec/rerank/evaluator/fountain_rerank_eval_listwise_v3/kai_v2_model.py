from __future__ import print_function

MODEL_TRANS_ORIGIN='cpp'

import yaml
import logging
import os
import sys

import argparse
import tensorflow as tf

from feature_attr_extract import * 
from model import EvaluatorModel

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

CANDIDATES_SIZE = 60
LIST_NUM = 30
LIST_SIZE = 6

# v2 List Value loss 权重。原分支保留为兼容对照，便于确认 v3 没有破坏
# 已经跑通的 P(K) × PrefixValue 结果。
LENGTH_LOSS_WEIGHT = 1.0
PREFIX_WT_LOSS_WEIGHT = 1.0
PREFIX_EVV_LOSS_WEIGHT = 0.5
LIST_WT_LOSS_WEIGHT = 0.5
LIST_EVV_LOSS_WEIGHT = 0.2
PREFIX_MONOTONIC_LOSS_WEIGHT = 0.1

# v3 新增 loss 权重。逐位置增量 WT 是主任务；EVV 提供较轻的行为辅助，
# List 总 WT、请求 baseline 和相对 delta 负责约束最终使用尺度。
INCREMENTAL_WT_LOSS_WEIGHT = 1.0
INCREMENTAL_EVV_LOSS_WEIGHT = 0.2
REACH_INCREMENTAL_LIST_WT_LOSS_WEIGHT = 0.5
REQUEST_BASELINE_LOSS_WEIGHT = 0.5
RELATIVE_DELTA_LOSS_WEIGHT = 0.5

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
        dataset.add_feature('context_info__real_show_index_list', dataset.DENSE, tf.int64, max_length=60)
        dataset.add_feature('context_info__real_show_list', dataset.DENSE, tf.int64, max_length=60)
        dataset.add_feature('context_info__first_screen', dataset.DENSE, tf.int64, max_length=60)
        dataset.add_feature('tab', dataset.DENSE, tf.int64, max_length=60)
        dataset.add_feature('3017', dataset.DENSE, tf.int64, max_length=60)
        dataset.add_feature('3019', dataset.DENSE, tf.int64, max_length=60)
        dataset.add_feature('3030', dataset.DENSE, tf.int64, max_length=60)
        dataset.add_feature('1031', dataset.DENSE, tf.int64, max_length=60)

        dataset.add_feature('rerank_list_score_list', dataset.DENSE, tf.float32, max_length=15)
        dataset.add_feature('rerank_list_item_idx_flat_list', dataset.DENSE, tf.int64, max_length=90)
        dataset.add_feature('fountain_fulllink_rerank_index_list', dataset.DENSE, tf.float32, max_length=60)

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
            
            # rerank_list_score_list = tf.RaggedTensor.from_row_splits(batch["rerank_list_score_list"][0], row_splits= batch["rerank_list_score_list"][1])
            # rerank_list_score_list = rerank_list_score_list.to_tensor()

            context_page = batch['context_info__first_screen']
            print(f"realshow shape: {realshow.shape}")
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
print_ops = []

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
    uid_dict = {}
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
            # feature_emb_size_dict[attr.attr_name] = 1
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
            offset = tf.cast(sparse_feature[1], tf.int32)
            size_var = offset[1:] - offset[0:-1]
            feature_emb_size_dict[attr.attr_name] = size_var
            if attr.slots[0] == 16:
                uid = tf.RaggedTensor.from_row_splits(values=sparse_feature[0], row_splits=sparse_feature[1]).to_tensor()
                uid_dict["uid"] = uid
                #tt = tf.RaggedTensor.from_row_splits(values=uid, row_splits=offset).to_tensor()
                #print_ops.append(tf.print("[Test test] slot " + str(attr.slots[0]), tt, output_stream=sys.stdout))
        else:
            # 实际 infer 不可用，待研究
            offset = tf.cast(config.get_signs(attr.slots[0])[1], tf.int32)
            size_var = offset[1:] - offset[0:-1]
            feature_emb_size_dict[attr.attr_name] = size_var        
        print("--->>> feature {} = {}".format(attr.attr_name, feature_emb_dict[attr.attr_name]))
        print("--->>> feature %s normal" % attr.attr_name)

    return feature_emb_dict, feature_emb_size_dict, uid_dict


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

def get_play_labels(duration, play_time):
    # 沿用基线的分视频时长阈值，将单个 item 是否达到有效播放定义为 EVV 标签。
    boundaries = [0, 8.366, 10.3, 12.433, 15.066, 17.3, 20.431, 24.833, 29.333, 33.916, 39.033, 46.566, 54.7, 62.933, 76.366, 99.166, 178.266, 235, 360.433, 1108.266]
    evtr_thresholds = [4.529, 8.56, 10.154, 11.228, 12.009, 13.51, 13.406, 13.038, 14.57, 15.108, 16.205, 17.891, 18.748, 18.451, 19.012, 17.148, 15.472, 13.181, 10.074, 8.925, 9.554]
    boundaries_tensor = tf.constant(boundaries, dtype=tf.float32)
    boundaries_tensor = tf.tile(tf.expand_dims(boundaries_tensor, axis=0), [tf.shape(duration)[0], 1])
    bucket_idx = tf.searchsorted(sorted_sequence=boundaries_tensor, values=tf.cast(duration, tf.float32), side="left") # 左开右闭
    max_idx = tf.constant(len(evtr_thresholds) - 1, dtype=tf.int32)
    bucket_idx = tf.clip_by_value(bucket_idx, 0, max_idx)
    evtr_threshold = tf.gather(tf.constant(evtr_thresholds), bucket_idx)
    evtr_label = tf.cast(tf.greater_equal(play_time, evtr_threshold), dtype=tf.float32)
    svtr_label = tf.where(play_time < tf.minimum(duration, 3.0), tf.ones_like(play_time), tf.zeros_like(play_time))
    return evtr_label, svtr_label

# 以下 WTD 分桶配置完全沿用 point-wise 基线，不属于本次 List Value 改动。
wtd_buckets = tf.constant(wtd_config["buckets"], dtype=tf.float32)
wtd_configs = tf.ragged.constant(wtd_config["configs"], dtype=tf.float32)

all_param_dict, _, _ = get_param_dict()
label_value_dict = {}
label_value_dict["show_label"] = tf.cast(tf.reshape(config.get_dense_fea("context_info__real_show_list", dim=CANDIDATES_SIZE, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["play_time_s"] = tf.cast(tf.reshape(config.get_dense_fea("context_info__playing_time_list", dim=CANDIDATES_SIZE, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32) / 1000.0
label_value_dict["play_time_s"] = tf.clip_by_value(label_value_dict["play_time_s"], 0, 36000)
label_value_dict["like_label"] = tf.cast(tf.reshape(config.get_dense_fea("context_info__like_list", dim=CANDIDATES_SIZE, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["follow_label"] = tf.cast(tf.reshape(config.get_dense_fea("context_info__follow_list", dim=CANDIDATES_SIZE, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["comment_label"] = tf.cast(tf.reshape(config.get_dense_fea("context_info__comment_list", dim=CANDIDATES_SIZE, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["forward_label"] = tf.cast(tf.reshape(config.get_dense_fea("context_info__forward_list", dim=CANDIDATES_SIZE, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["click_label"] = tf.cast(tf.reshape(config.get_label("fountain_click_label_list", dim=CANDIDATES_SIZE), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["wtd_label"] = tf.cast(tf.reshape(config.get_label("fountain_wtd_label_list", dim=CANDIDATES_SIZE), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["finish_label"] = tf.cast(tf.reshape(config.get_label("fountain_finish_label_list", dim=CANDIDATES_SIZE), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
dense_dim = CANDIDATES_SIZE if is_training else 1
label_value_dict["pwtd"] = tf.cast(tf.reshape(config.get_label("context_info__pwtd_list", dim=dense_dim), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["photo_info__duration_ms_list"] = tf.cast(tf.reshape(config.get_dense_fea("photo_info__duration_ms_list", dim=dense_dim, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["photo_info__duration_ms_list"] = tf.clip_by_value(label_value_dict["photo_info__duration_ms_list"], 0, 36000 * 1000)
point_wise_tasks = ["ltr", "vtr", "click", "wtd"]
model_class = EvaluatorModel(
    all_param_dict,
    print_ops,
    list_size=LIST_SIZE,
    candidates_size=CANDIDATES_SIZE,
    list_num=LIST_NUM,
    point_wise_tasks=point_wise_tasks,
)

if is_training:
    batch_size = tf.shape(label_value_dict["show_label"])[0]
    zeros = tf.zeros([batch_size, 1], dtype=tf.float32)
    raw_show_label = label_value_dict["show_label"]
    realshow_num_raw = tf.reduce_sum(
        tf.cast(tf.greater(raw_show_label, 0.0), tf.int32),
        axis=-1,
    )
    realshow_num = tf.clip_by_value(realshow_num_raw, 1, LIST_SIZE)
    has_observed_prefix = tf.greater(realshow_num_raw, 0)
    label_value_dict["fountain_fulllink_rerank_index_list"] = config.get_extra_param("fountain_fulllink_rerank_index_list", size=CANDIDATES_SIZE) # 从1计数
    real_show_rerank_indices = tf.expand_dims(label_value_dict["fountain_fulllink_rerank_index_list"][:, :LIST_SIZE], axis=1) # (?, 1, LIST_SIZE) 曝光对应的index
    real_show_rerank_indices = tf.cast(real_show_rerank_indices, dtype=tf.int32)
    index_indices = tf.argsort((tf.reshape(label_value_dict['fountain_fulllink_rerank_index_list'], [-1, CANDIDATES_SIZE])), axis=-1)
    index_indices = tf.reshape(index_indices, [batch_size, CANDIDATES_SIZE])
    print("index_indices ", index_indices)
    for k, v in label_value_dict.items():
        label_value_dict[k] = tf.reshape(label_value_dict[k], [-1, CANDIDATES_SIZE])
        label_value_dict[k] = tf.gather(label_value_dict[k], index_indices, axis=1, batch_dims=1)
    for k, v in all_param_dict.items():
        if k in photo_fea_names:
            all_param_dict[k] = tf.gather(all_param_dict[k], index_indices, axis=1, batch_dims=1)
    rerank_list_score_list = config.get_extra_param("rerank_list_score_list", size=LIST_NUM)
    # 实际 kai2 中 get_dense_fea default_value 不起作用；get_extra_param 也不支持 default_value ; kai oncall说没办法解决 =，=。
    rerank_list_item_idx_flat_list = config.get_dense_fea("rerank_list_item_idx_flat_list", dim=LIST_NUM * LIST_SIZE, dtype=tf.int64, default_value=-1) + 1
    label_value_dict['rerank_list_score_list'] = tf.reshape(rerank_list_score_list, [-1, LIST_NUM])
    rerank_list_item_idx_flat_list = tf.cast(tf.reshape(rerank_list_item_idx_flat_list, [-1, LIST_NUM, LIST_SIZE]), tf.int32)

    # -------- 为 List Value 寻找可监督的候选 List --------
    # 真实日志只告诉我们已经曝光的 Prefix[1:K]，因此只要求候选 List 的前 K 位
    # 与真实曝光序列一致；K 之后是反事实位置，既不参与匹配，也不构造价值标签。
    # 若多个候选共享同一个真实前缀，使用旧分最高的候选，保证选择规则稳定、可复现。
    observed_position_mask = tf.expand_dims(
        tf.sequence_mask(realshow_num, maxlen=LIST_SIZE, dtype=tf.bool),
        axis=1,
    )  # (?, 1, list_size)
    prefix_item_match = tf.logical_or(
        tf.equal(rerank_list_item_idx_flat_list, real_show_rerank_indices),
        tf.logical_not(observed_position_mask),
    )
    prefix_list_match = tf.reduce_all(prefix_item_match, axis=-1) # (?, list_num)
    prefix_list_match = tf.logical_and(
        prefix_list_match,
        tf.expand_dims(has_observed_prefix, axis=-1),
    )
    has_prefix_match = tf.reduce_any(prefix_list_match, axis=-1) # (?,)
    masked_match_score = tf.where(
        prefix_list_match,
        tf.reshape(rerank_list_score_list, [-1, LIST_NUM]),
        tf.fill([batch_size, LIST_NUM], tf.constant(-1e9, dtype=tf.float32)),
    )
    matched_list_index = tf.argmax(masked_match_score, axis=-1, output_type=tf.int32)
    max_score_list_index = tf.argmax(rerank_list_score_list, axis=-1, output_type=tf.int32)

    # 两套分支使用不同的样本选择规则：
    # 1. point-wise 完全沿用基线的“旧分最高 List”，避免改变基础任务样本分布；
    # 2. List Value 只使用真实前缀匹配的 List，没有匹配项的样本不计算新增 loss。
    pointwise_list_index = max_score_list_index
    pointwise_list_mask = tf.one_hot(
        pointwise_list_index,
        depth=LIST_NUM,
        dtype=tf.float32,
    )
    listwise_match_mask = tf.one_hot(
        matched_list_index,
        depth=LIST_NUM,
        dtype=tf.float32,
    ) * tf.expand_dims(tf.cast(has_prefix_match, tf.float32), axis=-1)

    full_observed_mask = tf.equal(realshow_num, LIST_SIZE)
    full_observed_count = tf.reduce_sum(tf.cast(full_observed_mask, tf.float32))
    full_list_match_count = tf.reduce_sum(
        tf.cast(
            tf.logical_and(full_observed_mask, has_prefix_match),
            tf.float32,
        )
    )
    full_list_match_rate = full_list_match_count / (full_observed_count + 1e-8)

    # prefix_match_rate 决定新增任务实际能使用多少训练样本；
    # full_list_match_rate 仅观察 K=LIST_SIZE 时的严格全 List 匹配情况。
    tf.summary.scalar(
        "list_value/match/prefix_match_rate",
        tf.reduce_mean(tf.cast(has_prefix_match, tf.float32)),
    )
    tf.summary.scalar(
        "list_value/match/full_list_match_rate",
        full_list_match_rate,
    )

    model_class._training=True
    point_wise_output_dict, list_value_output_dict = model_class.model(list_index = rerank_list_item_idx_flat_list)
    print(f"====> train, gen...")
    show_label = label_value_dict["show_label"]
    context_pwtd = label_value_dict["pwtd"]
    vtr_label = label_value_dict["wtd_label"]
    play_time_s = label_value_dict["play_time_s"]
    duration_s = label_value_dict["photo_info__duration_ms_list"] / 1000
    wtd_label = wtd_encode(duration=duration_s, play_time=play_time_s, duration_bucket=wtd_buckets, play_time_buckets_ragged=wtd_configs)
    click_label = label_value_dict["click_label"]
    like_label = label_value_dict["like_label"]
    follow_label = label_value_dict["follow_label"]
    comment_label = label_value_dict["comment_label"]
    forward_label = label_value_dict["forward_label"]
    finish_label = label_value_dict["finish_label"]
    evtr_label, svtr_label = get_play_labels(duration_s, play_time_s)

    # 原有 point-wise LTR 权重及 advantage 逻辑保持不变。
    evtr_weight = 1.0 + finish_label + (1 - svtr_label) * 2.0 + like_label * 20.0 + follow_label * 200.0 + comment_label * 200.0 + forward_label * 50.0
    advantage_reward = tf.clip_by_value(play_time_s, 0, 400) + finish_label * 3.0
    advantage_reward = tf.clip_by_value(advantage_reward, 0.0, 200.0)
    advantage = cal_batch_advantage(advantage_reward, mask=show_label) # (?, list_size)
    advantage = tf.nn.relu(tf.clip_by_value(advantage, 0.0, 40.0)) * (1 - svtr_label) + 1.0

    # -------- 原有 point-wise 基线标签 --------
    list_show_label = tf.gather(tf.concat([zeros, show_label], axis=-1), rerank_list_item_idx_flat_list, axis=1, batch_dims=1) # (?, list_num, list_size)
    list_context_pwtd = tf.gather(tf.concat([zeros, context_pwtd], axis=-1), rerank_list_item_idx_flat_list, axis=1, batch_dims=1) # (?, list_num, list_size)
    list_play_time_s = tf.gather(tf.concat([zeros, play_time_s], axis=-1), rerank_list_item_idx_flat_list, axis=1, batch_dims=1) # (?, list_num, list_size)
    list_duration_s = tf.gather(tf.concat([zeros, duration_s], axis=-1), rerank_list_item_idx_flat_list, axis=1, batch_dims=1) # (?, list_num, list_size)
    list_vtr_label = tf.gather(tf.concat([zeros, vtr_label], axis=-1), rerank_list_item_idx_flat_list, axis=1, batch_dims=1) # (?, list_num, list_size)
    list_wtd_label = tf.gather(tf.concat([zeros, wtd_label], axis=-1), rerank_list_item_idx_flat_list, axis=1, batch_dims=1) # (?, list_num, list_size)
    list_click_label = tf.gather(tf.concat([zeros, click_label], axis=-1), rerank_list_item_idx_flat_list, axis=1, batch_dims=1) # (?, list_num, list_size)
    list_evtr_label = tf.gather(tf.concat([zeros, evtr_label], axis=-1), rerank_list_item_idx_flat_list, axis=1, batch_dims=1) # (?, list_num, list_size)
    list_evtr_weight = tf.gather(tf.concat([zeros, evtr_weight], axis=-1), rerank_list_item_idx_flat_list, axis=1, batch_dims=1) # (?, list_num, list_size)
    list_advantage = tf.gather(tf.concat([zeros, advantage], axis=-1), rerank_list_item_idx_flat_list, axis=1, batch_dims=1) # (?, list_num, list_size)

    # -------- 新增 List Value 标签 --------
    # 对每个真实曝光位置做累加，构造 Prefix[1:k] 的累计 WT/EVV。
    # prefix_label_mask 同时限制“候选 List 匹配”和“位置已经真实曝光”两个条件，
    # 因而不会拿 K 之后的反事实 item 做监督。
    prefix_watch_time_label = tf.cumsum(list_play_time_s, axis=-1)
    prefix_effective_vv_label = tf.cumsum(list_evtr_label, axis=-1)
    prefix_label_mask = tf.cast(observed_position_mask, tf.float32) \
        * tf.expand_dims(listwise_match_mask, axis=-1)

    list_play_time_s_reduced = tf.reduce_sum(list_play_time_s, axis=-1)  # (?, list_num)
    list_effective_vv_reduced = tf.reduce_sum(list_evtr_label, axis=-1)  # (?, list_num)

    # 将原有 point-wise VTR/WTD 都解码成秒，后面在完全相同的匹配 List 上
    # 构造“直接累加”和“到达概率加权”两组离线基线。这里只增加评估量，
    # 不进入 loss，也不会改变 point-wise 或 List Value 分支的训练。
    flat_duration_s_int = tf.reshape(
        tf.cast(list_duration_s, dtype=tf.int32),
        [batch_size, LIST_NUM * LIST_SIZE],
    )
    list_vtr_wt = get_watch_time_from_vtr(
        tf.reshape(
            point_wise_output_dict["vtr"],
            [batch_size, LIST_NUM * LIST_SIZE],
        ),
        flat_duration_s_int,
    )  # (?, list_num * list_size)
    list_wtd_wt = wtd_decode(
        tf.reshape(
            point_wise_output_dict["wtd"],
            [batch_size, LIST_NUM * LIST_SIZE],
        ),
        flat_duration_s_int,
        duration_bucket=wtd_buckets,
        play_time_buckets_ragged=wtd_configs,
    )  # (?, list_num * list_size)

    mask = tf.reshape(list_show_label * tf.expand_dims(pointwise_list_mask, axis=-1), [batch_size, LIST_NUM * LIST_SIZE])

    with tf.control_dependencies(print_ops):
        targets = []
        sum_loss = 0.0
        list_duration_s = tf.reshape(list_duration_s, [batch_size, LIST_NUM * LIST_SIZE])
        # -------- 原有 point-wise 基线损失 --------
        for loss_name in point_wise_output_dict:
            output = point_wise_output_dict[loss_name]
            output = tf.reshape(output, [batch_size, LIST_NUM * LIST_SIZE])
            print(loss_name, output)

            if loss_name == "ltr":
                list_evtr_weight = tf.reshape(list_evtr_weight, [batch_size, LIST_NUM * LIST_SIZE])
                label = tf.reshape(list_click_label, [batch_size, LIST_NUM * LIST_SIZE])
                loss = tf.losses.log_loss(labels=label, predictions=output, weights=mask * list_evtr_weight)
                targets.append((loss_name, output, label, mask, "auc"))
            elif loss_name == "vtr":
                list_vtr_label = tf.reshape(list_vtr_label, [batch_size, LIST_NUM * LIST_SIZE])
                loss = tf.losses.huber_loss(labels=list_vtr_label, predictions=output, weights=mask, delta=0.05)
                loss = loss * 150.0
                targets.append((loss_name, output, list_vtr_label, mask, "linear_regression"))
                targets.append(("list_vtr_wt", list_vtr_wt, tf.reshape(list_play_time_s, [batch_size, LIST_NUM * LIST_SIZE]), mask, "linear_regression"))
            elif loss_name == "wtd":
                list_wtd_label = tf.reshape(list_wtd_label, [batch_size, LIST_NUM * LIST_SIZE])
                loss = tf.losses.log_loss(labels=list_wtd_label, predictions=output, weights=mask)
                targets.append((loss_name, output, list_wtd_label, mask, "linear_regression"))
            elif loss_name == "click":
                weight = tf.reshape(list_advantage, [batch_size, LIST_NUM * LIST_SIZE])
                label = tf.reshape(list_click_label, [batch_size, LIST_NUM * LIST_SIZE])
                loss = tf.losses.log_loss(labels=label, predictions=output, weights=mask * weight)
                targets.append((loss_name, output, label, mask, "auc"))

            sum_loss += loss
            tf.summary.scalar('loss_' + loss_name, loss)

        # -------- 新增 List Value 损失 --------
        continue_logits = list_value_output_dict["continue_logits"]
        continue_probs = list_value_output_dict["continue_probs"]
        length_probs = list_value_output_dict["length_probs"]
        prefix_watch_time = list_value_output_dict["prefix_watch_time"]
        prefix_watch_time_log = list_value_output_dict["prefix_watch_time_log"]
        prefix_effective_vv = list_value_output_dict["prefix_effective_vv"]
        expected_watch_time = list_value_output_dict["expected_watch_time"]
        expected_effective_vv = list_value_output_dict["expected_effective_vv"]
        expected_consume_length = list_value_output_dict["expected_consume_length"]
        reach_probs = list_value_output_dict["reach_probs"]
        incremental_watch_time_log = list_value_output_dict["incremental_watch_time_log"]
        incremental_watch_time = list_value_output_dict["incremental_watch_time"]
        incremental_effective_vv = list_value_output_dict["incremental_effective_vv"]
        reach_incremental_watch_time = list_value_output_dict["reach_incremental_watch_time"]
        reach_incremental_effective_vv = list_value_output_dict["reach_incremental_effective_vv"]
        request_baseline_log = list_value_output_dict["request_baseline_log"]
        relative_list_delta = list_value_output_dict["relative_list_delta"]
        relative_list_watch_time = list_value_output_dict["relative_list_watch_time"]

        # reach_prob[i] 表示用户能够到达第 i 个位置的概率：
        # 第一个位置必达，后续位置由前面所有 continue 概率连乘得到。
        # 该张量现在由 model.py 统一生成，训练、推理和三个 v3 输出共用同一口径。
        # 候选 List 中索引 0 表示 padding。显式屏蔽 padding，避免 point-wise
        # 输出层的 bias 给空位置产生非零观看时长。
        candidate_item_mask = tf.cast(
            tf.greater(rerank_list_item_idx_flat_list, 0),
            tf.float32,
        )
        list_vtr_wt_3d = tf.reshape(
            list_vtr_wt,
            [batch_size, LIST_NUM, LIST_SIZE],
        ) * candidate_item_mask
        list_wtd_wt_3d = tf.reshape(
            list_wtd_wt,
            [batch_size, LIST_NUM, LIST_SIZE],
        ) * candidate_item_mask

        # 四个分数使用相同的 item WT，只改变是否乘 reach_prob：
        # 1. baseline_add_*：假设 6 个位置都会被消费，直接累加；
        # 2. reach_weighted_*：按当前 continue 头给出的到达概率衰减。
        # VTR/WTD 各保留一组，避免结论只依赖某一种 point-wise 时长解码。
        baseline_add_vtr_watch_time = tf.reduce_sum(
            list_vtr_wt_3d,
            axis=-1,
        )
        reach_weighted_vtr_watch_time = tf.reduce_sum(
            reach_probs * list_vtr_wt_3d,
            axis=-1,
        )
        baseline_add_wtd_watch_time = tf.reduce_sum(
            list_wtd_wt_3d,
            axis=-1,
        )
        reach_weighted_wtd_watch_time = tf.reduce_sum(
            reach_probs * list_wtd_wt_3d,
            axis=-1,
        )

        # 长度类别仍从 0 开始：真实消费 K 个 item，对应类别 K-1。
        # Hazard 标签逐位置表示“消费完当前位置后是否继续”：
        # K=3 -> [继续, 继续, 停止, 未观测, 未观测]；
        # K=6 -> [继续, 继续, 继续, 继续, 继续]，末端按右截断处理。
        length_label = tf.tile(
            tf.expand_dims(realshow_num - 1, axis=-1),
            [1, LIST_NUM],
        )
        continue_label_per_request = tf.cast(
            tf.expand_dims(realshow_num, axis=-1)
            > tf.reshape(tf.range(1, LIST_SIZE), [1, LIST_SIZE - 1]),
            tf.float32,
        )
        continue_labels = tf.tile(
            tf.expand_dims(continue_label_per_request, axis=1),
            [1, LIST_NUM, 1],
        )
        hazard_observed_mask = tf.sequence_mask(
            tf.minimum(realshow_num, LIST_SIZE - 1),
            maxlen=LIST_SIZE - 1,
            dtype=tf.float32,
        )
        hazard_observed_mask = tf.expand_dims(
            hazard_observed_mask,
            axis=1,
        ) * tf.expand_dims(listwise_match_mask, axis=-1)

        # -------- continuation 的请求内相对分诊断 --------
        # 原始 continue logit 同时包含用户整体活跃度、位置难度和候选 List 内容信号。
        # 为了单独观察候选 List 之间的内容差异，这里在每个请求、每个位置上，
        # 减去“其他有效候选 List”的平均 logit（leave-one-out peer mean）。
        # 该相对分只用于离线 AUC 诊断，不参与训练 loss，也不改变线上打分。
        candidate_list_mask = tf.reduce_max(candidate_item_mask, axis=-1)
        candidate_list_mask_3d = tf.expand_dims(candidate_list_mask, axis=-1)
        valid_list_count = tf.reduce_sum(
            candidate_list_mask_3d,
            axis=1,
            keepdims=True,
        )
        continue_logit_sum = tf.reduce_sum(
            continue_logits * candidate_list_mask_3d,
            axis=1,
            keepdims=True,
        )
        peer_list_count = tf.maximum(valid_list_count - 1.0, 1.0)
        peer_continue_logit_mean = (
            continue_logit_sum - continue_logits * candidate_list_mask_3d
        ) / peer_list_count
        relative_continue_probs = tf.sigmoid(
            continue_logits - peer_continue_logit_mean,
        )
        # 至少存在两个有效候选 List 才有请求内对照；否则不计入相对分 AUC。
        relative_hazard_mask = hazard_observed_mask * tf.cast(
            tf.greater(valid_list_count, 1.0),
            tf.float32,
        )

        hazard_ce = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=continue_labels,
            logits=continue_logits,
        )
        # 对每条 List 累加所有已观测决策，得到右截断的负对数似然；
        # 再按匹配 List 数量求均值，使每个请求保持相同的样本权重。
        length_nll_per_list = tf.reduce_sum(
            hazard_ce * hazard_observed_mask,
            axis=-1,
        )
        length_loss = tf.reduce_sum(length_nll_per_list) \
            / (tf.reduce_sum(listwise_match_mask) + 1e-8)
        length_prediction = tf.argmax(length_probs, axis=-1, output_type=tf.int32)
        length_accuracy = tf.reduce_sum(
            tf.cast(tf.equal(length_prediction, length_label), tf.float32)
            * listwise_match_mask
        ) / (tf.reduce_sum(listwise_match_mask) + 1e-8)

        # 前缀价值损失直接约束每一个已观测 Prefix。
        # WT 使用 log1p 压缩长尾；EVV 数值较小，可直接在原空间拟合。
        prefix_watch_time_label_log = tf.math.log1p(prefix_watch_time_label)
        prefix_wt_loss = tf.losses.huber_loss(
            labels=prefix_watch_time_label_log,
            predictions=prefix_watch_time_log,
            weights=prefix_label_mask,
            delta=0.5,
        )
        prefix_evv_loss = tf.losses.huber_loss(
            labels=prefix_effective_vv_label,
            predictions=prefix_effective_vv,
            weights=prefix_label_mask,
            delta=0.5,
        )

        # List 总价值损失约束经过 P(K) 加权后的最终期望值，
        # 使训练目标与后续按 expected value 比较 List 的使用方式一致。
        list_wt_loss = tf.losses.huber_loss(
            labels=tf.math.log1p(list_play_time_s_reduced),
            predictions=tf.math.log1p(expected_watch_time),
            weights=listwise_match_mask,
            delta=0.5,
        )
        list_evv_loss = tf.losses.huber_loss(
            labels=list_effective_vv_reduced,
            predictions=expected_effective_vv,
            weights=listwise_match_mask,
            delta=0.5,
        )

        # 累计价值随前缀变长不应下降；只在相邻位置均有真实监督时施加软约束。
        prefix_pair_mask = prefix_label_mask[:, :, 1:]
        wt_monotonic_error = tf.nn.relu(
            prefix_watch_time[:, :, :-1] - prefix_watch_time[:, :, 1:]
        )
        evv_monotonic_error = tf.nn.relu(
            prefix_effective_vv[:, :, :-1] - prefix_effective_vv[:, :, 1:]
        )
        monotonic_loss = tf.reduce_sum(
            (wt_monotonic_error + evv_monotonic_error) * prefix_pair_mask
        ) / (tf.reduce_sum(prefix_pair_mask) + 1e-8)

        weighted_length_loss = LENGTH_LOSS_WEIGHT * length_loss
        weighted_prefix_wt_loss = PREFIX_WT_LOSS_WEIGHT * prefix_wt_loss
        weighted_prefix_evv_loss = PREFIX_EVV_LOSS_WEIGHT * prefix_evv_loss
        weighted_list_wt_loss = LIST_WT_LOSS_WEIGHT * list_wt_loss
        weighted_list_evv_loss = LIST_EVV_LOSS_WEIGHT * list_evv_loss
        weighted_monotonic_loss = PREFIX_MONOTONIC_LOSS_WEIGHT * monotonic_loss
        list_value_loss = weighted_length_loss \
            + weighted_prefix_wt_loss \
            + weighted_prefix_evv_loss \
            + weighted_list_wt_loss \
            + weighted_list_evv_loss \
            + weighted_monotonic_loss
        sum_loss += list_value_loss

        # -------- v3：Reach × Incremental Value --------
        # 增量头只在 matched List 的真实曝光位置训练。与累计 Prefix 标签相比，
        # 第 t 位直接拟合该 item 自身贡献，和 sum(reach_t * value_t) 条件一致。
        incremental_value_mask = prefix_label_mask * candidate_item_mask
        incremental_wt_loss = tf.losses.huber_loss(
            labels=tf.math.log1p(list_play_time_s),
            predictions=incremental_watch_time_log,
            weights=incremental_value_mask,
            delta=0.5,
        )
        incremental_evv_loss = tf.losses.huber_loss(
            labels=list_evtr_label,
            predictions=incremental_effective_vv,
            weights=incremental_value_mask,
            delta=0.5,
        )
        reach_incremental_list_wt_loss = tf.losses.huber_loss(
            labels=tf.math.log1p(list_play_time_s_reduced),
            predictions=tf.math.log1p(reach_incremental_watch_time),
            weights=listwise_match_mask,
            delta=0.5,
        )

        # -------- v3：请求公共 baseline + 相对 List 增量 --------
        # baseline 单独拟合请求整体消费水平；delta 只拟合扣除 baseline 后的残差。
        # stop_gradient 防止 delta loss 反向改变 baseline，从而保持两者职责清晰。
        list_watch_time_label_log = tf.math.log1p(list_play_time_s_reduced)
        request_baseline_loss = tf.losses.huber_loss(
            labels=list_watch_time_label_log,
            predictions=request_baseline_log,
            weights=listwise_match_mask,
            delta=0.5,
        )
        relative_delta_label = (
            list_watch_time_label_log - tf.stop_gradient(request_baseline_log)
        )
        relative_delta_loss = tf.losses.huber_loss(
            labels=relative_delta_label,
            predictions=relative_list_delta,
            weights=listwise_match_mask,
            delta=0.5,
        )

        weighted_incremental_wt_loss = (
            INCREMENTAL_WT_LOSS_WEIGHT * incremental_wt_loss
        )
        weighted_incremental_evv_loss = (
            INCREMENTAL_EVV_LOSS_WEIGHT * incremental_evv_loss
        )
        weighted_reach_incremental_list_wt_loss = (
            REACH_INCREMENTAL_LIST_WT_LOSS_WEIGHT
            * reach_incremental_list_wt_loss
        )
        weighted_request_baseline_loss = (
            REQUEST_BASELINE_LOSS_WEIGHT * request_baseline_loss
        )
        weighted_relative_delta_loss = (
            RELATIVE_DELTA_LOSS_WEIGHT * relative_delta_loss
        )
        v3_loss = weighted_incremental_wt_loss \
            + weighted_incremental_evv_loss \
            + weighted_reach_incremental_list_wt_loss \
            + weighted_request_baseline_loss \
            + weighted_relative_delta_loss
        sum_loss += v3_loss

        def masked_mean(values, weights):
            values = tf.cast(values, tf.float32)
            weights = tf.cast(weights, tf.float32)
            return tf.reduce_sum(values * weights) / (tf.reduce_sum(weights) + 1e-8)

        # -------- 精简后的 TensorBoard 监控：消费长度 --------
        # 同时观察分类准确率、预测/标签均值及期望长度 MAE，
        # 用于区分“整体偏长/偏短”和“单样本预测不准”。
        length_label_value = tf.cast(length_label + 1, tf.float32)
        expected_length_mae = masked_mean(
            tf.abs(expected_consume_length - length_label_value),
            listwise_match_mask,
        )
        # 用 K=LIST_SIZE 的真实/预测占比检查长度头是否退化为多数类预测。
        # 两个比例都只在真实前缀匹配成功的 List 上统计，口径与 length loss 一致。
        label_full_length_rate = masked_mean(
            tf.cast(tf.equal(length_label, LIST_SIZE - 1), tf.float32),
            listwise_match_mask,
        )
        predicted_full_length_rate = masked_mean(
            tf.cast(tf.equal(length_prediction, LIST_SIZE - 1), tf.float32),
            listwise_match_mask,
        )
        non_full_length_mask = listwise_match_mask * tf.cast(
            tf.less(length_label, LIST_SIZE - 1),
            tf.float32,
        )
        non_full_length_accuracy = masked_mean(
            tf.cast(tf.equal(length_prediction, length_label), tf.float32),
            non_full_length_mask,
        )
        tf.summary.scalar(
            "list_value/length/accuracy",
            length_accuracy,
        )
        tf.summary.scalar(
            "list_value/length/predicted_k_mean",
            masked_mean(expected_consume_length, listwise_match_mask),
        )
        tf.summary.scalar(
            "list_value/length/label_k_mean",
            masked_mean(length_label_value, listwise_match_mask),
        )
        tf.summary.scalar(
            "list_value/length/expected_k_mae",
            expected_length_mae,
        )
        tf.summary.scalar(
            "list_value/length/label_k6_rate",
            label_full_length_rate,
        )
        tf.summary.scalar(
            "list_value/length/predicted_argmax_k6_rate",
            predicted_full_length_rate,
        )
        tf.summary.scalar(
            "list_value/length/non_k6_accuracy",
            non_full_length_accuracy,
        )

        # -------- TensorBoard 监控：价值校准 --------
        # pred/label ratio 判断整体尺度偏差，MAE 观察单样本绝对误差。
        expected_wt_pred_mean = masked_mean(
            expected_watch_time,
            listwise_match_mask,
        )
        expected_wt_label_mean = masked_mean(
            list_play_time_s_reduced,
            listwise_match_mask,
        )
        expected_wt_mae = masked_mean(
            tf.abs(expected_watch_time - list_play_time_s_reduced),
            listwise_match_mask,
        )
        expected_evv_pred_mean = masked_mean(
            expected_effective_vv,
            listwise_match_mask,
        )
        expected_evv_label_mean = masked_mean(
            list_effective_vv_reduced,
            listwise_match_mask,
        )
        expected_evv_mae = masked_mean(
            tf.abs(expected_effective_vv - list_effective_vv_reduced),
            listwise_match_mask,
        )
        tf.summary.scalar(
            "list_value/calibration/expected_wt_pred_label_ratio",
            expected_wt_pred_mean / (expected_wt_label_mean + 1e-8),
        )
        tf.summary.scalar(
            "list_value/calibration/expected_wt_mae",
            expected_wt_mae,
        )
        tf.summary.scalar(
            "list_value/calibration/expected_evv_pred_label_ratio",
            expected_evv_pred_mean / (expected_evv_label_mean + 1e-8),
        )
        tf.summary.scalar(
            "list_value/calibration/expected_evv_mae",
            expected_evv_mae,
        )

        # -------- TensorBoard 监控：加权损失贡献 --------
        # 记录乘过超参权重后的真实贡献，并合并同类 loss，避免曲线过多。
        tf.summary.scalar("list_value/loss_contribution/length", weighted_length_loss)
        tf.summary.scalar(
            "list_value/loss_contribution/prefix_value",
            weighted_prefix_wt_loss + weighted_prefix_evv_loss,
        )
        tf.summary.scalar(
            "list_value/loss_contribution/list_total",
            weighted_list_wt_loss + weighted_list_evv_loss,
        )
        tf.summary.scalar("list_value/loss_contribution/monotonic", weighted_monotonic_loss)
        tf.summary.scalar("list_value/loss_contribution/total", list_value_loss)

        # -------- v3 TensorBoard：只保留能直接判断成败的六条曲线 --------
        # loss 按“逐位置价值”和“List/相对价值”分两组，避免再次堆出过多曲线。
        tf.summary.scalar(
            "v3/loss_contribution/incremental_value",
            weighted_incremental_wt_loss + weighted_incremental_evv_loss,
        )
        tf.summary.scalar(
            "v3/loss_contribution/list_and_relative",
            weighted_reach_incremental_list_wt_loss
            + weighted_request_baseline_loss
            + weighted_relative_delta_loss,
        )
        tf.summary.scalar("v3/loss_contribution/total", v3_loss)

        reach_incremental_wt_pred_mean = masked_mean(
            reach_incremental_watch_time,
            listwise_match_mask,
        )
        reach_incremental_wt_mae = masked_mean(
            tf.abs(
                reach_incremental_watch_time - list_play_time_s_reduced
            ),
            listwise_match_mask,
        )
        tf.summary.scalar(
            "v3/calibration/reach_incremental_wt_pred_label_ratio",
            reach_incremental_wt_pred_mean / (expected_wt_label_mean + 1e-8),
        )
        tf.summary.scalar(
            "v3/calibration/reach_incremental_wt_mae",
            reach_incremental_wt_mae,
        )
        relative_delta_mean = masked_mean(
            relative_list_delta,
            candidate_list_mask,
        )
        relative_delta_variance = masked_mean(
            tf.square(relative_list_delta - relative_delta_mean),
            candidate_list_mask,
        )
        tf.summary.scalar(
            "v3/relative/delta_std",
            tf.sqrt(relative_delta_variance + 1e-8),
        )

        # 聚合所有已观测 hazard 决策，检查模型能否区分“继续”和“停止”。
        # 这里只增加离线 AUC target，不参与 loss，也不新增 TensorBoard 曲线。
        flat_hazard_size = LIST_NUM * (LIST_SIZE - 1)
        targets.append((
            "continuation",
            tf.reshape(continue_probs, [batch_size, flat_hazard_size]),
            tf.reshape(continue_labels, [batch_size, flat_hazard_size]),
            tf.reshape(hazard_observed_mask, [batch_size, flat_hazard_size]),
            "auc",
        ))
        # 阶段性验证已经完成，暂时关闭以下 stdout target；需要复验时再恢复。
        # 1. continuation_pos*：固定位置，排除聚合 AUC 中的位置先验；
        # 2. continuation_relative_pos*：再排除同请求候选共享的公共先验。
        #
        # 21 个稳定 pass 的验证结果：
        # - 聚合 continuation AUC 均值约 0.7239；
        # - pos1~pos5 AUC 均值依次约为
        #   0.7169 / 0.7050 / 0.7013 / 0.7150 / 0.7358；
        # - 请求内相对 AUC 均值依次约为
        #   0.5317 / 0.5484 / 0.5478 / 0.5422 / 0.5434。
        #
        # 实验结论：
        # - 固定位置后 AUC 仍在 0.70 以上，聚合 AUC 并非主要来自位置先验；
        # - 去掉同请求公共倾向后仅剩约 0.53~0.55，说明原始区分能力主要来自
        #   用户/请求公共信息，候选 List 内容存在稳定但较弱的增量信号；
        # - relative 分数是请求内残差，不是绝对继续概率。
        # for position_idx in range(LIST_SIZE - 1):
        #     targets.append((
        #         "continuation_pos{}".format(position_idx + 1),
        #         continue_probs[:, :, position_idx],
        #         continue_labels[:, :, position_idx],
        #         hazard_observed_mask[:, :, position_idx],
        #         "auc",
        #     ))
        # for position_idx in range(LIST_SIZE - 1):
        #     targets.append((
        #         "continuation_relative_pos{}".format(position_idx + 1),
        #         relative_continue_probs[:, :, position_idx],
        #         continue_labels[:, :, position_idx],
        #         relative_hazard_mask[:, :, position_idx],
        #         "auc",
        #     ))

        flat_prefix_mask = tf.reshape(prefix_label_mask, [batch_size, LIST_NUM * LIST_SIZE])
        targets.append((
            "prefix_watch_time",
            tf.reshape(prefix_watch_time, [batch_size, LIST_NUM * LIST_SIZE]),
            tf.reshape(prefix_watch_time_label, [batch_size, LIST_NUM * LIST_SIZE]),
            flat_prefix_mask,
            "linear_regression",
        ))
        targets.append((
            "prefix_effective_vv",
            tf.reshape(prefix_effective_vv, [batch_size, LIST_NUM * LIST_SIZE]),
            tf.reshape(prefix_effective_vv_label, [batch_size, LIST_NUM * LIST_SIZE]),
            flat_prefix_mask,
            "linear_regression",
        ))
        targets.append((
            "expected_consume_length",
            expected_consume_length,
            tf.cast(length_label + 1, tf.float32),
            listwise_match_mask,
            "linear_regression",
        ))
        targets.append((
            "expected_list_watch_time",
            expected_watch_time,
            list_play_time_s_reduced,
            listwise_match_mask,
            "linear_regression",
        ))
        # v3 三个主要输出使用完全相同的 matched List 和真实 List WT：
        # 1. 现有 point-wise WT 乘 reach，作为最低风险基线；
        # 2. 新增逐位置增量 WT 乘 reach，验证更一致的价值分解；
        # 3. request baseline + relative delta 重建绝对 WT，验证请求内残差。
        targets.append((
            "reach_pointwise_list_watch_time",
            reach_weighted_vtr_watch_time,
            list_play_time_s_reduced,
            listwise_match_mask,
            "linear_regression",
        ))
        targets.append((
            "reach_incremental_list_watch_time",
            reach_incremental_watch_time,
            list_play_time_s_reduced,
            listwise_match_mask,
            "linear_regression",
        ))
        targets.append((
            "relative_list_watch_time",
            relative_list_watch_time,
            list_play_time_s_reduced,
            listwise_match_mask,
            "linear_regression",
        ))
        targets.append((
            "relative_list_delta",
            relative_list_delta,
            relative_delta_label,
            listwise_match_mask,
            "linear_regression",
        ))
        # 阶段性验证已经完成，暂时关闭以下四个 additive/reach 对照输出。
        # 它们只用于验证消费到达概率是否优于直接累加，不参与训练 loss。
        #
        # 21 个稳定 pass 的 List WT AUC 均值：
        # - VTR 直接累加 0.5122，乘 reach 后 0.5159（21 次中提升 16 次）；
        # - WTD 直接累加 0.5043，乘 reach 后 0.5067（21 次中提升 17 次）；
        # - 当前 P(K) × PrefixValue 的 expected_list_watch_time 均值约 0.5165，
        #   对 reach×VTR 只赢 12/21 次，对 reach×WTD 赢 19/21 次。
        #
        # 实验结论：
        # - 消费到达概率加权方向成立，但绝对收益只有约 0.0025~0.0037 AUC；
        # - 当前复杂 List Value 对简单 reach×VTR 没有形成稳定优势，主要瓶颈
        #   仍是候选 List 间的有效监督不足，而不是消费概率公式不够复杂。
        # targets.append((
        #     "baseline_add_vtr_watch_time",
        #     baseline_add_vtr_watch_time,
        #     list_play_time_s_reduced,
        #     listwise_match_mask,
        #     "linear_regression",
        # ))
        # targets.append((
        #     "reach_weighted_vtr_watch_time",
        #     reach_weighted_vtr_watch_time,
        #     list_play_time_s_reduced,
        #     listwise_match_mask,
        #     "linear_regression",
        # ))
        # targets.append((
        #     "baseline_add_wtd_watch_time",
        #     baseline_add_wtd_watch_time,
        #     list_play_time_s_reduced,
        #     listwise_match_mask,
        #     "linear_regression",
        # ))
        # targets.append((
        #     "reach_weighted_wtd_watch_time",
        #     reach_weighted_wtd_watch_time,
        #     list_play_time_s_reduced,
        #     listwise_match_mask,
        #     "linear_regression",
        # ))
        targets.append((
            "expected_list_effective_vv",
            expected_effective_vv,
            list_effective_vv_reduced,
            listwise_match_mask,
            "linear_regression",
        ))
        list_context_pwtd = tf.reshape(list_context_pwtd, [batch_size, LIST_NUM * LIST_SIZE])
        targets.append(("list_context_pwtd", list_context_pwtd, tf.reshape(list_play_time_s, [batch_size, LIST_NUM * LIST_SIZE]), mask, "linear_regression"))

    if args.with_kai_v2:
        # config.set_slot_param_attr([44, 45], config.nn.ParamAttr(access_method=config.nn.ProbabilityAccess(100.0),
        #                                                            recycle_method=config.nn.UnseendaysRecycle(
        #                                                                delete_after_unseen_days=15,
        #                                                                delete_threshold=2.0,
        #                                                                allow_dynamic_delete=True)))
        # config.set_slot_param_attr([1, 2], config.nn.ParamAttr(access_method=config.nn.ProbabilityAccess(100.0),
        #                                                            recycle_method=config.nn.UnseendaysRecycle(
        #                                                                delete_after_unseen_days=30,
        #                                                                delete_threshold=2.0,
        #                                                                allow_dynamic_delete=False)))
        sparse_optimizer = config.optimizer.Adam(0.0007)
        dense_optimizer = config.optimizer.Adam(0.00005)
        sparse_optimizer.minimize(sum_loss, var_list=config.get_collection(config.GraphKeys.EMBEDDING_INPUT))
        dense_optimizer.minimize(sum_loss, var_list=config.get_collection(config.GraphKeys.TRAINABLE_VARIABLES))
        opts = [sparse_optimizer, dense_optimizer]
    else:
        optimizer = tf.train.GradientDescentOptimizer(1, name="opt")
        grad_var = optimizer.compute_gradients(sum_loss)
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
        config.dump_training_config('./training/conf', targets, opts=opts, text=args.text)
else:
    duration_s = tf.reshape(config.get_extra_param("duration_ms_infer", size=1, dtype=tf.float32), [1, -1]) / 1000.0 # (?, CANDIDATES_SIZE)
    print("duration_s ", duration_s)
    zeros = tf.zeros(shape=[tf.shape(duration_s)[0], 1], dtype=tf.float32)
    print("zeros ", zeros)
    context_info__pctr_infer = tf.reshape(config.get_extra_param("context_info__pctr_infer", size=1, dtype=tf.float32), [1, -1]) # (?, CANDIDATES_SIZE)
    context_info__pwtd_infer = tf.reshape(config.get_extra_param("context_info__pwtd_infer", size=1, dtype=tf.float32), [1, -1]) # (?, CANDIDATES_SIZE)
    context_info__pltr_infer = tf.reshape(config.get_extra_param("context_info__pltr_infer", size=1, dtype=tf.float32), [1, -1]) # (?, CANDIDATES_SIZE)
    context_info__pcmtr_infer = tf.reshape(config.get_extra_param("context_info__pcmtr_infer", size=1, dtype=tf.float32), [1, -1]) # (?, CANDIDATES_SIZE)
    context_info__pwtr_infer = tf.reshape(config.get_extra_param("context_info__pwtr_infer", size=1, dtype=tf.float32), [1, -1]) # (?, CANDIDATES_SIZE)
    context_info__pftr_infer = tf.reshape(config.get_extra_param("context_info__pftr_infer", size=1, dtype=tf.float32), [1, -1]) # (?, CANDIDATES_SIZE)
    context_info__plvtr_infer = tf.reshape(config.get_extra_param("context_info__plvtr_infer", size=1, dtype=tf.float32), [1, -1]) # (?, CANDIDATES_SIZE)
    context_info__psvtr_infer = tf.reshape(config.get_extra_param("context_info__psvtr_infer", size=1, dtype=tf.float32), [1, -1]) # (?, CANDIDATES_SIZE)
    # infer 时从1开始
    rerank_list_item_idx_flat_list = config.get_extra_param("rerank_list_item_idx_flat_list_double", size=LIST_NUM * LIST_SIZE, default_value=-1.0, common=True) + 1.0
    rerank_list_item_idx_flat_list = tf.reshape(rerank_list_item_idx_flat_list, [-1, LIST_NUM, LIST_SIZE])
    print("rerank_list_item_idx_flat_list ", rerank_list_item_idx_flat_list)
    rerank_list_item_idx_flat_list = tf.cast(rerank_list_item_idx_flat_list, tf.int32)
    duration_s = tf.concat([zeros, duration_s], axis=-1) # (?, CANDIDATES_SIZE + 1)
    list_duration_s = tf.gather(duration_s, rerank_list_item_idx_flat_list, axis=1, batch_dims=1) # (?, list_num, list_size)
    print("list_duration_s ", list_duration_s)
    list_context_pctr = tf.gather(tf.concat([zeros, context_info__pctr_infer], axis=-1), rerank_list_item_idx_flat_list, axis=1, batch_dims=1) # (?, list_num, list_size)
    list_context_pwtd = tf.gather(tf.concat([zeros, context_info__pwtd_infer], axis=-1), rerank_list_item_idx_flat_list, axis=1, batch_dims=1) # (?, list_num, list_size)
    list_context_pltr = tf.gather(tf.concat([zeros, context_info__pltr_infer], axis=-1), rerank_list_item_idx_flat_list, axis=1, batch_dims=1) # (?, list_num, list_size)
    list_context_pcmtr = tf.gather(tf.concat([zeros, context_info__pcmtr_infer], axis=-1), rerank_list_item_idx_flat_list, axis=1, batch_dims=1) # (?, list_num, list_size)
    list_context_pwtr = tf.gather(tf.concat([zeros, context_info__pwtr_infer], axis=-1), rerank_list_item_idx_flat_list, axis=1, batch_dims=1) # (?, list_num, list_size)
    list_context_pftr = tf.gather(tf.concat([zeros, context_info__pftr_infer], axis=-1), rerank_list_item_idx_flat_list, axis=1, batch_dims=1) # (?, list_num, list_size)
    list_context_plvtr = tf.gather(tf.concat([zeros, context_info__plvtr_infer], axis=-1), rerank_list_item_idx_flat_list, axis=1, batch_dims=1) # (?, list_num, list_size)
    list_context_psvtr = tf.gather(tf.concat([zeros, context_info__psvtr_infer], axis=-1), rerank_list_item_idx_flat_list, axis=1, batch_dims=1) # (?, list_num, list_size)
    # fountain_stats__long_play_count = tf.gather(tf.concat([zeros, label_value_dict["photo_info__fountain_stats__long_play_count_list"]], axis=-1), rerank_list_item_idx_flat_list, axis=1, batch_dims=1) # (?, list_num, list_size)
    # fountain_stats__real_show_count = tf.gather(tf.concat([zeros, label_value_dict["photo_info__fountain_stats__real_show_count_list"]], axis=-1), rerank_list_item_idx_flat_list, axis=1, batch_dims=1) # (?, list_num, list_size)

    model_class._training = False
    point_wise_output_dict, list_value_output_dict = model_class.model(rerank_list_item_idx_flat_list)
    pctr = tf.reshape(point_wise_output_dict["click"], [-1, LIST_NUM * LIST_SIZE])
    # print("pctr ", pctr)
    pvtr = tf.reshape(point_wise_output_dict["vtr"], [-1, LIST_NUM * LIST_SIZE])
    # pslide = tf.reshape(point_wise_output_dict["slide"], [-1, LIST_NUM * LIST_SIZE])
    pltr = tf.reshape(point_wise_output_dict["ltr"], [-1, LIST_NUM * LIST_SIZE])
    pwtd = tf.reshape(point_wise_output_dict["wtd"], [-1, LIST_NUM * LIST_SIZE])
    context_pctr = tf.reshape(list_context_pctr, [-1, LIST_NUM * LIST_SIZE])
    context_pwtd = tf.reshape(list_context_pwtd, [-1, LIST_NUM * LIST_SIZE])
    context_pltr = tf.reshape(list_context_pltr, [-1, LIST_NUM * LIST_SIZE])
    context_pcmtr = tf.reshape(list_context_pcmtr, [-1, LIST_NUM * LIST_SIZE])
    context_pwtr = tf.reshape(list_context_pwtr, [-1, LIST_NUM * LIST_SIZE])
    context_pftr = tf.reshape(list_context_pftr, [-1, LIST_NUM * LIST_SIZE])
    context_plvtr = tf.reshape(list_context_plvtr, [-1, LIST_NUM * LIST_SIZE])
    context_psvtr = tf.reshape(list_context_psvtr, [-1, LIST_NUM * LIST_SIZE])
    # context_lv_count = tf.reshape(fountain_stats__long_play_count, [-1, LIST_NUM * LIST_SIZE])
    # context_sv_count = tf.reshape(fountain_stats__real_show_count, [-1, LIST_NUM * LIST_SIZE])
    duration_s_flat = tf.reshape(tf.cast(list_duration_s, tf.int32), [-1, LIST_NUM * LIST_SIZE])
    pvtr_wt = get_watch_time_from_vtr(pvtr, duration_s_flat) # (? , CANDIDATES_SIZE)
    pwtd_wt = wtd_decode(pwtd, duration_s_flat, duration_bucket=wtd_buckets, play_time_buckets_ragged=wtd_configs) # (? , CANDIDATES_SIZE)
    rerank_list_item_idx_flat_list_print = tf.reshape(rerank_list_item_idx_flat_list, [-1, LIST_NUM * LIST_SIZE])

    # v2 兼容输出：P(K=k) × PrefixValue[1:k]。
    expected_vtr_wt = list_value_output_dict["expected_watch_time"]
    expected_effective_vv = list_value_output_dict["expected_effective_vv"]
    expected_consume_length = list_value_output_dict["expected_consume_length"]

    # -------- v3 三个可独立上线的 List 级输出 --------
    # reach_pointwise_wt 使用现有 point-wise VTR 解码出的 item WT，不增加价值头；
    # 另外两个输出直接来自新增的增量价值/相对价值分支。
    reach_probs = list_value_output_dict["reach_probs"]
    infer_candidate_item_mask = tf.cast(
        tf.greater(rerank_list_item_idx_flat_list, 0),
        tf.float32,
    )
    pvtr_wt_3d = tf.reshape(
        pvtr_wt,
        [-1, LIST_NUM, LIST_SIZE],
    ) * infer_candidate_item_mask
    reach_pointwise_wt = tf.reduce_sum(
        reach_probs * pvtr_wt_3d,
        axis=-1,
        name="reach_pointwise_wt",
    )
    reach_incremental_wt = list_value_output_dict["reach_incremental_watch_time"]
    relative_list_delta = list_value_output_dict["relative_list_delta"]
    relative_list_watch_time = list_value_output_dict["relative_list_watch_time"]
    expected_ltr = tf.reduce_sum(tf.reshape(pltr, [-1, LIST_NUM, LIST_SIZE]), axis=-1)  # (1, list_num)
    expected_pctr = tf.reduce_sum(tf.reshape(pctr, [-1, LIST_NUM, LIST_SIZE]), axis=-1)  # (1, list_num)
    expected_like = tf.reduce_sum(tf.reshape(context_pltr, [-1, LIST_NUM, LIST_SIZE]), axis=-1)  # (1, list_num)
    expected_comment = tf.reduce_sum(tf.reshape(context_pcmtr, [-1, LIST_NUM, LIST_SIZE]), axis=-1)  # (1, list_num)
    expected_follow = tf.reduce_sum(tf.reshape(context_pwtr, [-1, LIST_NUM, LIST_SIZE]), axis=-1)  # (1, list_num)

    targets = []
    targets.append((f"pctr", pctr))
    # targets.append((f"pslide", pslide))
    targets.append((f"pwtd", pwtd_wt))
    targets.append((f"pvtr", pvtr_wt))
    targets.append((f"pltr", pltr))
    targets.append((f"context_pctr", context_pctr))
    targets.append((f"context_pwtd", context_pwtd))
    targets.append((f"context_pltr", context_pltr))
    targets.append((f"context_pcmtr", context_pcmtr))
    targets.append((f"context_pwtr", context_pwtr))
    targets.append((f"context_pftr", context_pftr))
    targets.append((f"context_plvtr", context_plvtr))
    targets.append((f"context_psvtr", context_psvtr))
    targets.append((f"duration_s", duration_s))
    targets.append((f"rerank_list_item_idx_flat_list_print", rerank_list_item_idx_flat_list_print))
    # list wise
    targets.append((f"expected_vtr_wt", expected_vtr_wt))
    targets.append((f"expected_effective_vv", expected_effective_vv))
    targets.append((f"expected_consume_length", expected_consume_length))
    targets.append((f"reach_pointwise_wt", reach_pointwise_wt))
    targets.append((f"reach_incremental_wt", reach_incremental_wt))
    targets.append((f"relative_list_delta", relative_list_delta))
    # 该重建值主要用于离线检查 request baseline + delta 的绝对校准，
    # 线上排序优先使用有正有负的 relative_list_delta。
    targets.append((f"relative_list_watch_time", relative_list_watch_time))
    targets.append((f"expected_ltr", expected_ltr))
    targets.append((f"expected_pctr", expected_pctr))
    targets.append((f"expected_like", expected_like))
    targets.append((f"expected_comment", expected_comment))
    targets.append((f"expected_follow", expected_follow))

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
