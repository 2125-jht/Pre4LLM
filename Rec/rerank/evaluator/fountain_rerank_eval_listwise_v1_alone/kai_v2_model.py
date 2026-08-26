from __future__ import print_function

MODEL_TRANS_ORIGIN='cpp'

import yaml
import logging
import os

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

# List Value 第一版的独立训练目标；不存在 point-wise loss 或共享参数。
LENGTH_LOSS_WEIGHT = 1.0
PREFIX_WT_LOSS_WEIGHT = 1.0
PREFIX_ENGAGEMENT_LOSS_WEIGHT = 0.5
LIST_WT_LOSS_WEIGHT = 0.5
LIST_ENGAGEMENT_LOSS_WEIGHT = 0.2
PREFIX_MONOTONIC_LOSS_WEIGHT = 0.1
ENGAGEMENT_INTERACTION_POSITIVE_SAMPLE_WEIGHT = 3.0
LONG_WATCH_TIME_THRESHOLD_S = 120.0

# Engagement 将有效播放作为稠密基础价值，并按原 LTR 价值比例合并互动。
INTERACTION_LIKE_VALUE = 1.0
INTERACTION_COMMENT_VALUE = 10.0
INTERACTION_FOLLOW_VALUE = 10.0
INTERACTION_FORWARD_VALUE = 2.5
MAX_ITEM_ENGAGEMENT_VALUE = (
    1.0
    + INTERACTION_LIKE_VALUE
    + INTERACTION_COMMENT_VALUE
    + INTERACTION_FOLLOW_VALUE
    + INTERACTION_FORWARD_VALUE
)
MAX_ENGAGEMENT_VALUE = LIST_SIZE * MAX_ITEM_ENGAGEMENT_VALUE

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

all_param_dict, _, _ = get_param_dict()
label_value_dict = {}
if is_training:
    label_value_dict["show_label"] = tf.cast(tf.reshape(config.get_dense_fea("context_info__real_show_list", dim=CANDIDATES_SIZE, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
    label_value_dict["play_time_s"] = tf.cast(tf.reshape(config.get_dense_fea("context_info__playing_time_list", dim=CANDIDATES_SIZE, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32) / 1000.0
    label_value_dict["play_time_s"] = tf.clip_by_value(label_value_dict["play_time_s"], 0, 36000)
    label_value_dict["like_label"] = tf.clip_by_value(tf.cast(tf.reshape(config.get_dense_fea("context_info__like_list", dim=CANDIDATES_SIZE, dtype=tf.int64), [-1, CANDIDATES_SIZE]), tf.float32), 0.0, 1.0)
    label_value_dict["comment_label"] = tf.clip_by_value(tf.cast(tf.reshape(config.get_dense_fea("context_info__comment_list", dim=CANDIDATES_SIZE, dtype=tf.int64), [-1, CANDIDATES_SIZE]), tf.float32), 0.0, 1.0)
    label_value_dict["follow_label"] = tf.clip_by_value(tf.cast(tf.reshape(config.get_dense_fea("context_info__follow_list", dim=CANDIDATES_SIZE, dtype=tf.int64), [-1, CANDIDATES_SIZE]), tf.float32), 0.0, 1.0)
    label_value_dict["forward_label"] = tf.clip_by_value(tf.cast(tf.reshape(config.get_dense_fea("context_info__forward_list", dim=CANDIDATES_SIZE, dtype=tf.int64), [-1, CANDIDATES_SIZE]), tf.float32), 0.0, 1.0)
    # 模型组 PWTD 仅作为离线 List 级 AUC 对照，不是本模型训练目标。
    label_value_dict["pwtd"] = tf.cast(tf.reshape(config.get_label("context_info__pwtd_list", dim=CANDIDATES_SIZE), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
    label_value_dict["photo_info__duration_ms_list"] = tf.cast(tf.reshape(config.get_dense_fea("photo_info__duration_ms_list", dim=CANDIDATES_SIZE, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
    label_value_dict["photo_info__duration_ms_list"] = tf.clip_by_value(label_value_dict["photo_info__duration_ms_list"], 0, 36000 * 1000)
model_class = EvaluatorModel(
    all_param_dict,
    print_ops,
    list_size=LIST_SIZE,
    candidates_size=CANDIDATES_SIZE,
    list_num=LIST_NUM,
    max_engagement_value=MAX_ENGAGEMENT_VALUE,
)

if is_training:
    batch_size = tf.shape(label_value_dict["show_label"])[0]
    zeros = tf.zeros([batch_size, 1], dtype=tf.float32)
    # K 使用请求中 real_show=1 的实际个数。该计数在候选重排前计算；后续
    # argsort 只改变候选位置，不会改变 real_show 的总数。
    raw_show_label = label_value_dict["show_label"]
    realshow_count_raw = tf.reduce_sum(
        tf.cast(tf.greater(raw_show_label, 0.0), tf.int32),
        axis=-1,
    )
    raw_fountain_rerank_index = tf.cast(
        tf.reshape(
            config.get_extra_param(
                "fountain_fulllink_rerank_index_list",
                size=CANDIDATES_SIZE,
            ),
            [-1, CANDIDATES_SIZE],
        ),
        tf.int32,
    )  # item 在统一候选池中的 1-based 坐标
    raw_real_show_index = tf.cast(
        tf.reshape(
            config.get_dense_fea(
                "context_info__real_show_index_list",
                dim=CANDIDATES_SIZE,
                dtype=tf.int64,
            ),
            [-1, CANDIDATES_SIZE],
        ),
        tf.int32,
    )

    # real_show_index 是与原始 60 个 item 对齐的最终曝光位次。
    # 先用 real_show 筛出真实曝光 item，再按该位次升序排列，最后 gather
    # 它们在统一候选池中的坐标，得到事实曝光 Prefix。训练、评估和标签
    # 对齐都统一使用该序列，不再假设原始数组前 K 个 item 就是曝光顺序。
    raw_show_mask = tf.greater(raw_show_label, 0.0)
    exposure_sort_key = tf.where(
        raw_show_mask,
        raw_real_show_index,
        tf.fill(
            tf.shape(raw_real_show_index),
            tf.constant(2147483647, dtype=tf.int32),
        ),
    )
    exposure_order = tf.argsort(exposure_sort_key, axis=-1)
    factual_exposure_rerank_indices = tf.gather(
        raw_fountain_rerank_index,
        exposure_order,
        axis=1,
        batch_dims=1,
    )[:, :LIST_SIZE]
    factual_real_show_indices = tf.gather(
        raw_real_show_index,
        exposure_order,
        axis=1,
        batch_dims=1,
    )[:, :LIST_SIZE]
    factual_position_mask_2d = tf.sequence_mask(
        realshow_count_raw,
        maxlen=LIST_SIZE,
        dtype=tf.bool,
    )
    factual_exposure_rerank_indices = tf.where(
        factual_position_mask_2d,
        factual_exposure_rerank_indices,
        tf.zeros_like(factual_exposure_rerank_indices),
    )
    factual_real_show_indices = tf.where(
        factual_position_mask_2d,
        factual_real_show_indices,
        tf.zeros_like(factual_real_show_indices),
    )

    label_value_dict["fountain_fulllink_rerank_index_list"] = tf.cast(
        raw_fountain_rerank_index,
        tf.float32,
    )
    # 候选 List Prefix 匹配所使用的唯一事实序列。K 之后补 0 仅用于张量
    # 对齐，实际比较和所有 Prefix loss 都会被 observed_position_mask 屏蔽。
    real_show_rerank_indices = tf.expand_dims(
        factual_exposure_rerank_indices,
        axis=1,
    )  # (?, 1, LIST_SIZE)
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
    rerank_list_score_matrix = tf.reshape(
        rerank_list_score_list,
        [-1, LIST_NUM],
    )
    # 实际 kai2 中 get_dense_fea default_value 不起作用；get_extra_param 也不支持 default_value ; kai oncall说没办法解决 =，=。
    rerank_list_item_idx_flat_list = config.get_dense_fea("rerank_list_item_idx_flat_list", dim=LIST_NUM * LIST_SIZE, dtype=tf.int64, default_value=-1) + 1
    label_value_dict['rerank_list_score_list'] = rerank_list_score_matrix
    rerank_list_item_idx_flat_list = tf.cast(tf.reshape(rerank_list_item_idx_flat_list, [-1, LIST_NUM, LIST_SIZE]), tf.int32)

    # K 是 real_show=1 的数量，不再取最后一个 real_show 的物理位置。
    # 中间的 real_show=0 不是快速划过；事实曝光 item 必须先按
    # real_show_index 恢复最终曝光顺序，再与候选 List 做严格 Prefix 匹配。
    max_score_list_index = tf.argmax(
        rerank_list_score_matrix,
        axis=-1,
        output_type=tf.int32,
    )
    consume_depth = tf.clip_by_value(realshow_count_raw, 1, LIST_SIZE)
    has_observed_prefix = tf.greater(realshow_count_raw, 0)

    # -------- List 与真实曝光 Prefix 的一致性筛选 --------
    # 只比较前 K 个位置；候选 List 的这 K 个 item 必须与真实曝光顺序逐项
    # 相同。K 之后没有事实反馈，不参与 Prefix 匹配，也不参与 Prefix loss。
    observed_position_mask = tf.expand_dims(
        tf.sequence_mask(consume_depth, maxlen=LIST_SIZE, dtype=tf.bool),
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

    # 训练现在依赖 real_show_index，保留一个低成本的数据质量监控：所有事实
    # rank 必须非负，且排序后严格递增（允许有间隔，但不允许重复/缺失）。
    factual_rank_non_negative = tf.reduce_all(
        tf.logical_or(
            tf.greater_equal(factual_real_show_indices, 0),
            tf.logical_not(factual_position_mask_2d),
        ),
        axis=-1,
    )
    adjacent_factual_position_mask = tf.sequence_mask(
        tf.maximum(realshow_count_raw - 1, 0),
        maxlen=LIST_SIZE - 1,
        dtype=tf.bool,
    )
    factual_rank_strictly_increasing = tf.reduce_all(
        tf.logical_or(
            tf.greater(
                factual_real_show_indices[:, 1:],
                factual_real_show_indices[:, :-1],
            ),
            tf.logical_not(adjacent_factual_position_mask),
        ),
        axis=-1,
    )
    factual_rank_valid = tf.logical_and(
        factual_rank_non_negative,
        factual_rank_strictly_increasing,
    )
    # rank 无效时无法可靠恢复事实顺序，即便数值碰巧命中候选也不得训练。
    prefix_list_match = tf.logical_and(
        prefix_list_match,
        tf.expand_dims(factual_rank_valid, axis=-1),
    )
    has_prefix_match = tf.reduce_any(prefix_list_match, axis=-1) # (?,)

    masked_match_score = tf.where(
        prefix_list_match,
        rerank_list_score_matrix,
        tf.fill([batch_size, LIST_NUM], tf.constant(-1e9, dtype=tf.float32)),
    )
    matched_list_index = tf.argmax(masked_match_score, axis=-1, output_type=tf.int32)

    observed_request_weight = tf.cast(has_observed_prefix, tf.float32)
    matched_request_weight = tf.cast(has_prefix_match, tf.float32)

    # 训练只使用事实 Prefix 能够对齐的请求。若多个候选共享同一个事实
    # Prefix，则选择其中旧分最高的 List；完全没有匹配候选的请求 mask 为 0，
    # 不进入长度、Prefix WT/Engagement、List 总价值 loss 或 matched 指标。
    listwise_match_mask = tf.one_hot(
        matched_list_index,
        depth=LIST_NUM,
        dtype=tf.float32,
    ) * tf.expand_dims(matched_request_weight, axis=-1)

    # 训练与评估 mask 显式拆分：
    # 1. loss 继续只使用事实 Prefix 匹配的 listwise_match_mask；
    # 2. matched_eval_list_mask 用于可信事实口径，保持现有指标名；
    # 3. legacy_max_score_eval_list_mask 复刻旧版“每请求旧分最高 List”口径，
    #    仅用于和历史曲线比较，不代表该 List 一定与真实曝光 Prefix 对齐。
    matched_eval_list_mask = tf.stop_gradient(
        tf.identity(listwise_match_mask, name="matched_eval_list_mask")
    )
    legacy_max_score_eval_list_mask = tf.stop_gradient(
        tf.one_hot(
            max_score_list_index,
            depth=LIST_NUM,
            dtype=tf.float32,
            name="legacy_max_score_eval_list_mask",
        )
    )

    # 以下指标用于观察事实 Prefix 筛选质量。prefix_match_rate 同时也是实际
    # 训练请求保留率；observed_prefix_rate 应接近 1，否则说明上游 sample
    # filter 有零曝光样本泄漏。
    tf.summary.scalar(
        "list_value/match/observed_prefix_rate",
        tf.reduce_mean(observed_request_weight),
    )
    tf.summary.scalar(
        "list_value/match/prefix_match_rate",
        tf.reduce_mean(tf.cast(has_prefix_match, tf.float32)),
    )
    tf.summary.scalar(
        "list_value/match/training_request_drop_rate",
        1.0 - tf.reduce_mean(tf.cast(has_prefix_match, tf.float32)),
    )
    # real_show_index 已成为正式训练依赖，因此只保留一个长期数据质量指标。
    tf.summary.scalar(
        "list_value/match/factual_rank_valid_rate",
        tf.reduce_sum(
            tf.cast(factual_rank_valid, tf.float32)
            * observed_request_weight
        ) / (tf.reduce_sum(observed_request_weight) + 1e-8),
    )
    # 旧分最高 List 相关监控已完成阶段性验证，暂停输出：
    # - list_value/match/max_score_prefix_match_rate
    # - list_value/match/max_score_vs_matched_agreement_rate

    # Prefix 越长，匹配条件越严格。保留按真实消费长度拆分的
    # any-prefix 匹配率，用于观察候选集对真实曝光 Prefix 的覆盖。
    for consume_k in range(1, LIST_SIZE + 1):
        k_request_weight = tf.cast(
            tf.equal(consume_depth, consume_k),
            tf.float32,
        ) * observed_request_weight
        k_request_count = tf.reduce_sum(k_request_weight)
        # k1～k6_request_rate 已完成阶段性验证，暂停输出。
        tf.summary.scalar(
            "list_value/match/by_k/k{}_any_prefix_match_rate".format(
                consume_k
            ),
            tf.reduce_sum(k_request_weight * matched_request_weight)
            / (k_request_count + 1e-8),
        )
        # k1～k6_max_score_prefix_match_rate 已完成阶段性验证，暂停输出。

    model_class._training = True
    list_value_output_dict = model_class.model(
        list_index=rerank_list_item_idx_flat_list,
    )
    print(f"====> train standalone v1 list model, gen...")
    context_pwtd = label_value_dict["pwtd"]
    play_time_s = label_value_dict["play_time_s"]
    duration_s = label_value_dict["photo_info__duration_ms_list"] / 1000
    evtr_label, _ = get_play_labels(duration_s, play_time_s)
    interaction_value = (
        INTERACTION_LIKE_VALUE * label_value_dict["like_label"]
        + INTERACTION_COMMENT_VALUE * label_value_dict["comment_label"]
        + INTERACTION_FOLLOW_VALUE * label_value_dict["follow_label"]
        + INTERACTION_FORWARD_VALUE * label_value_dict["forward_label"]
    )
    engagement_value = evtr_label + interaction_value

    # 外部模型组的 item PWTD 只作为 List 级离线对照，不参与本模型训练。
    list_context_pwtd = tf.gather(tf.concat([zeros, context_pwtd], axis=-1), rerank_list_item_idx_flat_list, axis=1, batch_dims=1) # (?, list_num, list_size)
    list_wt_from_context_pwtd_sum = tf.reduce_sum(
        list_context_pwtd * tf.cast(
            tf.greater(rerank_list_item_idx_flat_list, 0),
            tf.float32,
        ),
        axis=-1,
    )
    list_play_time_s = tf.gather(tf.concat([zeros, play_time_s], axis=-1), rerank_list_item_idx_flat_list, axis=1, batch_dims=1) # (?, list_num, list_size)
    list_evtr_label = tf.gather(tf.concat([zeros, evtr_label], axis=-1), rerank_list_item_idx_flat_list, axis=1, batch_dims=1) # (?, list_num, list_size)
    list_interaction_value = tf.gather(tf.concat([zeros, interaction_value], axis=-1), rerank_list_item_idx_flat_list, axis=1, batch_dims=1)
    list_engagement_value = list_evtr_label + list_interaction_value

    # -------- 新增 List Value 标签 --------
    # 对每个真实曝光位置做累加，构造 Prefix[1:k] 的累计 WT/Engagement。
    # prefix_label_mask 同时限制“旧分最高 List”和“位置已经真实曝光”两个条件，
    # 因而不会拿 K 之后的反事实 item 做监督。
    prefix_watch_time_label = tf.cumsum(list_play_time_s, axis=-1)
    prefix_engagement_label = tf.cumsum(list_engagement_value, axis=-1)
    prefix_label_mask = tf.cast(observed_position_mask, tf.float32) \
        * tf.expand_dims(listwise_match_mask, axis=-1)

    list_play_time_s_reduced = tf.reduce_sum(list_play_time_s, axis=-1)  # (?, list_num)
    list_effective_vv_reduced = tf.reduce_sum(list_evtr_label, axis=-1)  # (?, list_num)
    list_interaction_reduced = tf.reduce_sum(list_interaction_value, axis=-1)
    list_engagement_reduced = tf.reduce_sum(list_engagement_value, axis=-1)

    with tf.control_dependencies(print_ops):
        targets = []
        # -------- List Value 损失 --------
        length_logits = list_value_output_dict["length_logits"]
        length_probs = list_value_output_dict["length_probs"]
        prefix_watch_time = list_value_output_dict["prefix_watch_time"]
        prefix_watch_time_log = list_value_output_dict["prefix_watch_time_log"]
        prefix_engagement = list_value_output_dict["prefix_engagement"]
        prefix_engagement_log = list_value_output_dict["prefix_engagement_log"]
        expected_watch_time = list_value_output_dict["expected_list_watch_time"]
        expected_engagement = list_value_output_dict["expected_list_engagement"]
        expected_consume_length = list_value_output_dict["expected_consume_length"]

        # 长度类别从 0 开始：真实消费 K 个 item，对应类别 K-1。
        # 同一请求下所有候选 List 共用真实 K，但只有旧分最高 List 会被 mask 选中。
        length_label = tf.tile(
            tf.expand_dims(consume_depth - 1, axis=-1),
            [1, LIST_NUM],
        )
        length_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=length_label,
            logits=length_logits,
        )
        length_loss = tf.reduce_sum(length_ce * listwise_match_mask) \
            / (tf.reduce_sum(listwise_match_mask) + 1e-8)
        length_prediction = tf.argmax(length_probs, axis=-1, output_type=tf.int32)
        length_accuracy = tf.reduce_sum(
            tf.cast(tf.equal(length_prediction, length_label), tf.float32)
            * listwise_match_mask
        ) / (tf.reduce_sum(listwise_match_mask) + 1e-8)

        # 前缀价值损失直接约束每一个已观测 Prefix。
        # WT 和综合 Engagement 均使用 log1p 压缩长尾。
        prefix_watch_time_label_log = tf.math.log1p(prefix_watch_time_label)
        prefix_wt_loss = tf.losses.huber_loss(
            labels=prefix_watch_time_label_log,
            predictions=prefix_watch_time_log,
            weights=prefix_label_mask,
            delta=0.5,
        )
        prefix_interaction_positive = tf.greater(
            tf.cumsum(list_interaction_value, axis=-1),
            0.0,
        )
        prefix_engagement_sample_weight = prefix_label_mask * tf.where(
            prefix_interaction_positive,
            tf.fill(
                tf.shape(prefix_engagement_label),
                tf.constant(ENGAGEMENT_INTERACTION_POSITIVE_SAMPLE_WEIGHT, tf.float32),
            ),
            tf.ones_like(prefix_engagement_label),
        )
        prefix_engagement_loss = tf.losses.huber_loss(
            labels=tf.math.log1p(prefix_engagement_label),
            predictions=prefix_engagement_log,
            weights=prefix_engagement_sample_weight,
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
        list_engagement_sample_weight = listwise_match_mask * tf.where(
            tf.greater(list_interaction_reduced, 0.0),
            tf.fill(
                tf.shape(list_engagement_reduced),
                tf.constant(ENGAGEMENT_INTERACTION_POSITIVE_SAMPLE_WEIGHT, tf.float32),
            ),
            tf.ones_like(list_engagement_reduced),
        )
        list_engagement_loss = tf.losses.huber_loss(
            labels=tf.math.log1p(list_engagement_reduced),
            predictions=tf.math.log1p(expected_engagement),
            weights=list_engagement_sample_weight,
            delta=0.5,
        )

        # 累计价值随前缀变长不应下降；只在相邻位置均有真实监督时施加软约束。
        prefix_pair_mask = prefix_label_mask[:, :, 1:]
        wt_monotonic_error = tf.nn.relu(
            prefix_watch_time[:, :, :-1] - prefix_watch_time[:, :, 1:]
        )
        engagement_monotonic_error = tf.nn.relu(
            prefix_engagement_log[:, :, :-1] - prefix_engagement_log[:, :, 1:]
        )
        monotonic_loss = tf.reduce_sum(
            (wt_monotonic_error + engagement_monotonic_error) * prefix_pair_mask
        ) / (tf.reduce_sum(prefix_pair_mask) + 1e-8)

        weighted_length_loss = LENGTH_LOSS_WEIGHT * length_loss
        weighted_prefix_wt_loss = PREFIX_WT_LOSS_WEIGHT * prefix_wt_loss
        weighted_prefix_engagement_loss = PREFIX_ENGAGEMENT_LOSS_WEIGHT * prefix_engagement_loss
        weighted_list_wt_loss = LIST_WT_LOSS_WEIGHT * list_wt_loss
        weighted_list_engagement_loss = LIST_ENGAGEMENT_LOSS_WEIGHT * list_engagement_loss
        weighted_monotonic_loss = PREFIX_MONOTONIC_LOSS_WEIGHT * monotonic_loss
        list_value_loss = weighted_length_loss \
            + weighted_prefix_wt_loss \
            + weighted_prefix_engagement_loss \
            + weighted_list_wt_loss \
            + weighted_list_engagement_loss \
            + weighted_monotonic_loss
        sum_loss = list_value_loss

        def masked_mean(values, weights):
            values = tf.cast(values, tf.float32)
            weights = tf.cast(weights, tf.float32)
            return tf.reduce_sum(values * weights) / (tf.reduce_sum(weights) + 1e-8)

        def masked_pred_label_ratio(predictions, labels, weights):
            predictions = tf.cast(predictions, tf.float32)
            labels = tf.cast(labels, tf.float32)
            weights = tf.cast(weights, tf.float32)
            return tf.reduce_sum(predictions * weights) \
                / (tf.reduce_sum(labels * weights) + 1e-8)

        def masked_wmape(predictions, labels, weights):
            predictions = tf.cast(predictions, tf.float32)
            labels = tf.cast(labels, tf.float32)
            weights = tf.cast(weights, tf.float32)
            return tf.reduce_sum(tf.abs(predictions - labels) * weights) \
                / (tf.reduce_sum(tf.abs(labels) * weights) + 1e-8)

        # -------- 精简后的 TensorBoard 监控：消费长度 --------
        # 同时观察分类准确率、预测/标签均值及期望长度 MAE，
        # 用于区分“整体偏长/偏短”和“单样本预测不准”。
        length_label_value = tf.cast(length_label + 1, tf.float32)
        expected_length_mae = masked_mean(
            tf.abs(expected_consume_length - length_label_value),
            listwise_match_mask,
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
        # 完整监控 P(K=1..LIST_SIZE)，避免只观察 K=6 时漏掉其他长度类别的
        # 分布偏移或塌缩。所有统计均使用与 length loss 相同的 one-hot List mask。
        length_true_class_probability = tf.reduce_sum(
            length_probs * tf.one_hot(
                length_label,
                depth=LIST_SIZE,
                dtype=tf.float32,
            ),
            axis=-1,
        )
        length_probability_entropy = -tf.reduce_sum(
            length_probs * tf.math.log(tf.maximum(length_probs, 1e-8)),
            axis=-1,
        )
        tf.summary.scalar(
            "list_value/length/probability/true_class_probability_mean",
            masked_mean(length_true_class_probability, listwise_match_mask),
        )
        tf.summary.scalar(
            "list_value/length/probability/top1_probability_mean",
            masked_mean(tf.reduce_max(length_probs, axis=-1), listwise_match_mask),
        )
        tf.summary.scalar(
            "list_value/length/probability/entropy_mean",
            masked_mean(length_probability_entropy, listwise_match_mask),
        )

        for consume_k in range(1, LIST_SIZE + 1):
            class_index = consume_k - 1
            class_label = tf.cast(
                tf.equal(length_label, class_index),
                tf.float32,
            )
            class_label_mask = listwise_match_mask * class_label
            class_prediction = tf.cast(
                tf.equal(length_prediction, class_index),
                tf.float32,
            )
            tf.summary.scalar(
                "list_value/length/by_k/k{}_label_rate".format(consume_k),
                masked_mean(class_label, listwise_match_mask),
            )
            tf.summary.scalar(
                "list_value/length/by_k/k{}_predicted_probability_mean".format(
                    consume_k
                ),
                masked_mean(
                    length_probs[:, :, class_index],
                    listwise_match_mask,
                ),
            )
            tf.summary.scalar(
                "list_value/length/by_k/k{}_predicted_argmax_rate".format(
                    consume_k
                ),
                masked_mean(class_prediction, listwise_match_mask),
            )
            tf.summary.scalar(
                "list_value/length/by_k/k{}_recall".format(consume_k),
                masked_mean(class_prediction, class_label_mask),
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
        expected_engagement_pred_mean = masked_mean(
            expected_engagement,
            listwise_match_mask,
        )
        expected_engagement_label_mean = masked_mean(
            list_engagement_reduced,
            listwise_match_mask,
        )
        expected_engagement_mae = masked_mean(
            tf.abs(expected_engagement - list_engagement_reduced),
            listwise_match_mask,
        )
        evv_only_mask = listwise_match_mask * tf.cast(
            tf.logical_and(
                tf.greater(list_effective_vv_reduced, 0.0),
                tf.less_equal(list_interaction_reduced, 0.0),
            ),
            tf.float32,
        )
        interaction_positive_mask = listwise_match_mask * tf.cast(
            tf.greater(list_interaction_reduced, 0.0),
            tf.float32,
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
            "list_value/calibration/expected_engagement_pred_label_ratio",
            expected_engagement_pred_mean / (expected_engagement_label_mean + 1e-8),
        )
        tf.summary.scalar(
            "list_value/calibration/expected_engagement_mae",
            expected_engagement_mae,
        )
        tf.summary.scalar(
            "list_value/engagement/evv_value_share",
            tf.reduce_sum(list_effective_vv_reduced * listwise_match_mask)
            / (tf.reduce_sum(list_engagement_reduced * listwise_match_mask) + 1e-8),
        )
        tf.summary.scalar(
            "list_value/engagement/interaction_value_share",
            tf.reduce_sum(list_interaction_reduced * listwise_match_mask)
            / (tf.reduce_sum(list_engagement_reduced * listwise_match_mask) + 1e-8),
        )
        tf.summary.scalar(
            "list_value/engagement/evv_only_pred_label_ratio",
            masked_mean(expected_engagement, evv_only_mask)
            / (masked_mean(list_engagement_reduced, evv_only_mask) + 1e-8),
        )
        tf.summary.scalar(
            "list_value/engagement/interaction_positive_pred_label_ratio",
            masked_mean(expected_engagement, interaction_positive_mask)
            / (masked_mean(list_engagement_reduced, interaction_positive_mask) + 1e-8),
        )
        tf.summary.scalar(
            "list_value/engagement/interaction_positive_rate",
            tf.reduce_sum(interaction_positive_mask)
            / (tf.reduce_sum(listwise_match_mask) + 1e-8),
        )

        # -------- TensorBoard 监控：List 时长诊断 --------
        # 使用真实 K 选取模型预测的 Prefix WT，作为 Oracle-K 口径。
        # 它与最终 expected WT 的差距可用来区分长度概率误差和
        # Prefix WT 数值头误差；该值只用于训练期离线监控。
        oracle_k_watch_time = tf.reduce_sum(
            prefix_watch_time * tf.one_hot(
                length_label,
                depth=LIST_SIZE,
                dtype=tf.float32,
            ),
            axis=-1,
        )
        # Fixed-K6 对照始终取完整 List 的第 6 个 Prefix WT，
        # 不使用真实 K，也不乘长度概率，因此是推理可用的诊断基线。
        fixed_k6_watch_time = prefix_watch_time[:, :, LIST_SIZE - 1]
        tf.summary.scalar(
            "list_value/watch_time/global/wmape",
            masked_wmape(
                expected_watch_time,
                list_play_time_s_reduced,
                listwise_match_mask,
            ),
        )
        tf.summary.scalar(
            "list_value/watch_time/global/underprediction_rate",
            masked_mean(
                tf.cast(
                    tf.less(expected_watch_time, list_play_time_s_reduced),
                    tf.float32,
                ),
                listwise_match_mask,
            ),
        )
        tf.summary.scalar(
            "list_value/watch_time/oracle_k/pred_label_ratio",
            masked_pred_label_ratio(
                oracle_k_watch_time,
                list_play_time_s_reduced,
                listwise_match_mask,
            ),
        )
        tf.summary.scalar(
            "list_value/watch_time/oracle_k/wmape",
            masked_wmape(
                oracle_k_watch_time,
                list_play_time_s_reduced,
                listwise_match_mask,
            ),
        )

        # 按真实消费长度观察最终 expected WT 的校准与相对误差。
        # request rate 已由 length/by_k/kK_label_rate 提供，这里不重复输出。
        for consume_k in range(1, LIST_SIZE + 1):
            k_watch_time_mask = listwise_match_mask * tf.cast(
                tf.equal(length_label, consume_k - 1),
                tf.float32,
            )
            tf.summary.scalar(
                "list_value/watch_time/by_k/k{}_pred_label_ratio".format(
                    consume_k
                ),
                masked_pred_label_ratio(
                    expected_watch_time,
                    list_play_time_s_reduced,
                    k_watch_time_mask,
                ),
            )
            tf.summary.scalar(
                "list_value/watch_time/by_k/k{}_wmape".format(consume_k),
                masked_wmape(
                    expected_watch_time,
                    list_play_time_s_reduced,
                    k_watch_time_mask,
                ),
            )

        model_abs_error = tf.abs(
            expected_watch_time - list_play_time_s_reduced
        )
        pwtd_abs_error = tf.abs(
            list_wt_from_context_pwtd_sum - list_play_time_s_reduced
        )
        tf.summary.scalar(
            "list_value/watch_time/vs_pwtd/abs_error_win_rate",
            masked_mean(
                tf.cast(tf.less(model_abs_error, pwtd_abs_error), tf.float32),
                listwise_match_mask,
            ),
        )

        # log1p 训练容易压低原始秒数空间的长尾，因此单独观察
        # 真实 List WT >= LONG_WATCH_TIME_THRESHOLD_S 的样本。
        long_watch_mask = listwise_match_mask * tf.cast(
            tf.greater_equal(
                list_play_time_s_reduced,
                tf.constant(LONG_WATCH_TIME_THRESHOLD_S, tf.float32),
            ),
            tf.float32,
        )
        tf.summary.scalar(
            "list_value/watch_time/long_watch/request_rate",
            tf.reduce_sum(long_watch_mask)
            / (tf.reduce_sum(listwise_match_mask) + 1e-8),
        )
        tf.summary.scalar(
            "list_value/watch_time/long_watch/pred_label_ratio",
            masked_pred_label_ratio(
                expected_watch_time,
                list_play_time_s_reduced,
                long_watch_mask,
            ),
        )
        tf.summary.scalar(
            "list_value/watch_time/long_watch/wmape",
            masked_wmape(
                expected_watch_time,
                list_play_time_s_reduced,
                long_watch_mask,
            ),
        )

        # -------- TensorBoard 监控：加权损失贡献 --------
        # 记录乘过超参权重后的真实贡献，并合并同类 loss，避免曲线过多。
        tf.summary.scalar("list_value/loss_contribution/length", weighted_length_loss)
        tf.summary.scalar(
            "list_value/loss_contribution/prefix_value",
            weighted_prefix_wt_loss + weighted_prefix_engagement_loss,
        )
        tf.summary.scalar(
            "list_value/loss_contribution/list_total",
            weighted_list_wt_loss + weighted_list_engagement_loss,
        )
        tf.summary.scalar("list_value/loss_contribution/monotonic", weighted_monotonic_loss)
        tf.summary.scalar("list_value/loss_contribution/total", list_value_loss)

        flat_prefix_mask = tf.reshape(prefix_label_mask, [batch_size, LIST_NUM * LIST_SIZE])
        targets.append((
            "prefix_watch_time",
            tf.reshape(prefix_watch_time, [batch_size, LIST_NUM * LIST_SIZE]),
            tf.reshape(prefix_watch_time_label, [batch_size, LIST_NUM * LIST_SIZE]),
            flat_prefix_mask,
            "linear_regression",
        ))
        targets.append((
            "prefix_engagement",
            tf.reshape(prefix_engagement, [batch_size, LIST_NUM * LIST_SIZE]),
            tf.reshape(prefix_engagement_label, [batch_size, LIST_NUM * LIST_SIZE]),
            flat_prefix_mask,
            "linear_regression",
        ))
        targets.append((
            "expected_consume_length",
            expected_consume_length,
            tf.cast(length_label + 1, tf.float32),
            matched_eval_list_mask,
            "linear_regression",
        ))
        targets.append((
            "expected_consume_length_legacy_max_score",
            expected_consume_length,
            tf.cast(length_label + 1, tf.float32),
            legacy_max_score_eval_list_mask,
            "linear_regression",
        ))
        targets.append((
            "expected_list_watch_time",
            expected_watch_time,
            list_play_time_s_reduced,
            matched_eval_list_mask,
            "linear_regression",
        ))
        targets.append((
            "expected_list_watch_time_oracle_k",
            oracle_k_watch_time,
            list_play_time_s_reduced,
            matched_eval_list_mask,
            "linear_regression",
        ))
        targets.append((
            "expected_list_watch_time_fixed_k6",
            fixed_k6_watch_time,
            list_play_time_s_reduced,
            matched_eval_list_mask,
            "linear_regression",
        ))
        targets.append((
            "expected_list_watch_time_legacy_max_score",
            expected_watch_time,
            list_play_time_s_reduced,
            legacy_max_score_eval_list_mask,
            "linear_regression",
        ))
        targets.append((
            "list_wt_from_context_pwtd_sum",
            list_wt_from_context_pwtd_sum,
            list_play_time_s_reduced,
            matched_eval_list_mask,
            "linear_regression",
        ))
        targets.append((
            "list_wt_from_context_pwtd_sum_legacy_max_score",
            list_wt_from_context_pwtd_sum,
            list_play_time_s_reduced,
            legacy_max_score_eval_list_mask,
            "linear_regression",
        ))
        targets.append((
            "expected_list_engagement",
            expected_engagement,
            list_engagement_reduced,
            matched_eval_list_mask,
            "linear_regression",
        ))
        targets.append((
            "expected_list_engagement_legacy_max_score",
            expected_engagement,
            list_engagement_reduced,
            legacy_max_score_eval_list_mask,
            "linear_regression",
        ))
        targets.append((
            "expected_list_engagement_evv_only",
            expected_engagement,
            list_engagement_reduced,
            evv_only_mask,
            "linear_regression",
        ))
        targets.append((
            "expected_list_engagement_interaction_positive",
            expected_engagement,
            list_engagement_reduced,
            interaction_positive_mask,
            "linear_regression",
        ))
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
    # 外部 fullrank 分数不是本模型的单点预测头，只按 List 下标 gather 后
    # 原样导出，供 backbone 作为可选的线上基线分数使用。
    context_score_names = (
        "pctr", "pwtd", "pltr", "pcmtr",
        "pwtr", "pftr", "plvtr", "psvtr",
    )
    context_score_tensors = {
        name: tf.reshape(
            config.get_extra_param(
                "context_info__{}_infer".format(name),
                size=1,
                dtype=tf.float32,
            ),
            [1, -1],
        )
        for name in context_score_names
    }

    # infer 时索引从 1 开始，0 专门留给 padding。
    rerank_list_item_idx_flat_list = config.get_extra_param("rerank_list_item_idx_flat_list_double", size=LIST_NUM * LIST_SIZE, default_value=-1.0, common=True) + 1.0
    rerank_list_item_idx_flat_list = tf.reshape(rerank_list_item_idx_flat_list, [-1, LIST_NUM, LIST_SIZE])
    print("rerank_list_item_idx_flat_list ", rerank_list_item_idx_flat_list)
    rerank_list_item_idx_flat_list = tf.cast(rerank_list_item_idx_flat_list, tf.int32)

    zeros = tf.zeros(
        shape=[tf.shape(context_score_tensors["pctr"])[0], 1],
        dtype=tf.float32,
    )
    context_outputs = []
    for name in context_score_names:
        gathered_score = tf.gather(
            tf.concat([zeros, context_score_tensors[name]], axis=-1),
            rerank_list_item_idx_flat_list,
            axis=1,
            batch_dims=1,
        )
        context_outputs.append((
            "context_{}".format(name),
            tf.reshape(gathered_score, [-1, LIST_NUM * LIST_SIZE]),
        ))

    model_class._training = False
    list_value_output_dict = model_class.model(rerank_list_item_idx_flat_list)

    # 训练与推理统一使用 P(K=k) × PrefixValue[1:k]；不再导出任何单点头。
    expected_list_watch_time = list_value_output_dict["expected_list_watch_time"]
    expected_list_engagement = list_value_output_dict["expected_list_engagement"]
    expected_consume_length = list_value_output_dict["expected_consume_length"]

    targets = [
        ("expected_list_watch_time", expected_list_watch_time),
        ("expected_list_engagement", expected_list_engagement),
        ("expected_consume_length", expected_consume_length),
    ] + context_outputs

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
