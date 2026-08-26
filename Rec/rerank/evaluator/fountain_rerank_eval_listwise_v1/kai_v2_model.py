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

# List Value 第一版 loss 权重。保留 point-wise 主任务作为表征底座，
# 新任务先使用较保守的权重，便于单独观察长度分布和前缀价值的收益。
LENGTH_LOSS_WEIGHT = 1.0
PREFIX_WT_LOSS_WEIGHT = 1.0
PREFIX_EVV_LOSS_WEIGHT = 0.5
LIST_WT_LOSS_WEIGHT = 0.5
LIST_EVV_LOSS_WEIGHT = 0.2
PREFIX_MONOTONIC_LOSS_WEIGHT = 0.1

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
if is_training:
    # 仅用于 stdout 验证 real_show 内部 gap 是否对应继续下滑，不参与 loss。
    label_value_dict["slide_label"] = tf.cast(tf.reshape(config.get_dense_fea("context_info__fountain_slide_to_next_list", dim=CANDIDATES_SIZE, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
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
    realshow_count_raw = tf.reduce_sum(
        tf.cast(tf.greater(raw_show_label, 0.0), tf.int32),
        axis=-1,
    )
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
    rerank_list_score_matrix = tf.reshape(
        rerank_list_score_list,
        [-1, LIST_NUM],
    )
    # 实际 kai2 中 get_dense_fea default_value 不起作用；get_extra_param 也不支持 default_value ; kai oncall说没办法解决 =，=。
    rerank_list_item_idx_flat_list = config.get_dense_fea("rerank_list_item_idx_flat_list", dim=LIST_NUM * LIST_SIZE, dtype=tf.int64, default_value=-1) + 1
    label_value_dict['rerank_list_score_list'] = rerank_list_score_matrix
    rerank_list_item_idx_flat_list = tf.cast(tf.reshape(rerank_list_item_idx_flat_list, [-1, LIST_NUM, LIST_SIZE]), tf.int32)

    # K 表示用户在旧分选中 List 上最后到达的 real_show 物理位置，而不是
    # real_show 的数量。例如 [1, 1, 0, 1, 0, 1] 的 K 为 6，不是 4；
    # 中间的 0 保留为“到达但没有产生 real_show 价值”的快速划过位置。
    max_score_list_index = tf.argmax(
        rerank_list_score_matrix,
        axis=-1,
        output_type=tf.int32,
    )
    all_list_show_label = tf.gather(
        tf.concat([zeros, label_value_dict["show_label"]], axis=-1),
        rerank_list_item_idx_flat_list,
        axis=1,
        batch_dims=1,
    )  # (?, list_num, list_size)
    selected_list_coordinates = tf.stack(
        [tf.range(batch_size), max_score_list_index],
        axis=1,
    )
    selected_list_show_label = tf.gather_nd(
        all_list_show_label,
        selected_list_coordinates,
    )  # (?, list_size)
    physical_positions = tf.tile(
        tf.expand_dims(
            tf.range(1, LIST_SIZE + 1, dtype=tf.int32),
            axis=0,
        ),
        [batch_size, 1],
    )
    consume_depth_raw = tf.reduce_max(
        tf.where(
            tf.greater(selected_list_show_label, 0.0),
            physical_positions,
            tf.zeros_like(physical_positions),
        ),
        axis=-1,
    )
    consume_depth = tf.clip_by_value(consume_depth_raw, 1, LIST_SIZE)
    has_observed_prefix = tf.greater(consume_depth_raw, 0)

    # -------- List 与用户实际到达 Prefix 的一致性诊断 --------
    # K 取最后一个 real_show 的物理位置，Prefix[1:K] 允许存在 real_show=0 的
    # 快速划过位置；K 之后仍视为未到达的反事实位置。实际训练选择仍只看
    # 旧分最高 List，下面的 Prefix 匹配结果只用于监控上游顺序假设。
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
    has_prefix_match = tf.reduce_any(prefix_list_match, axis=-1) # (?,)
    masked_match_score = tf.where(
        prefix_list_match,
        rerank_list_score_matrix,
        tf.fill([batch_size, LIST_NUM], tf.constant(-1e9, dtype=tf.float32)),
    )
    matched_list_index = tf.argmax(masked_match_score, axis=-1, output_type=tf.int32)
    # 验证上游约束“旧分最高 List 就是实际曝光 List”。任意候选能够匹配
    # Prefix，并不代表最终选中的最高分候选也能匹配，因此两种口径都保留。
    max_score_prefix_match = tf.gather_nd(
        prefix_list_match,
        tf.stack(
            [tf.range(batch_size), max_score_list_index],
            axis=1,
        ),
    )
    observed_request_weight = tf.cast(has_observed_prefix, tf.float32)
    matched_request_weight = tf.cast(has_prefix_match, tf.float32)
    max_score_prefix_match_rate = tf.reduce_sum(
        tf.cast(max_score_prefix_match, tf.float32) * observed_request_weight
    ) / (tf.reduce_sum(observed_request_weight) + 1e-8)
    max_score_vs_matched_agreement_rate = tf.reduce_sum(
        tf.cast(
            tf.equal(max_score_list_index, matched_list_index),
            tf.float32,
        ) * matched_request_weight
    ) / (tf.reduce_sum(matched_request_weight) + 1e-8)

    # point-wise 与 List Value 统一使用旧分最高 List。Prefix 匹配只保留为
    # 数据诊断，不再过滤训练样本或改选 List。
    pointwise_list_index = max_score_list_index
    pointwise_list_mask = tf.one_hot(
        pointwise_list_index,
        depth=LIST_NUM,
        dtype=tf.float32,
    )
    listwise_match_mask = tf.one_hot(
        max_score_list_index,
        depth=LIST_NUM,
        dtype=tf.float32,
    )

    full_observed_mask = tf.equal(consume_depth, LIST_SIZE)
    full_observed_count = tf.reduce_sum(tf.cast(full_observed_mask, tf.float32))
    full_list_match_count = tf.reduce_sum(
        tf.cast(
            tf.logical_and(full_observed_mask, has_prefix_match),
            tf.float32,
        )
    )
    full_list_match_rate = full_list_match_count / (full_observed_count + 1e-8)

    # 以下指标仅用于数据诊断，不再决定 List Value 的训练样本。
    # observed_prefix_rate 应接近 1；否则说明上游 sample filter 有样本泄漏。
    tf.summary.scalar(
        "list_value/match/observed_prefix_rate",
        tf.reduce_mean(observed_request_weight),
    )
    tf.summary.scalar(
        "list_value/match/prefix_match_rate",
        tf.reduce_mean(tf.cast(has_prefix_match, tf.float32)),
    )
    tf.summary.scalar(
        "list_value/match/full_list_match_rate",
        full_list_match_rate,
    )
    tf.summary.scalar(
        "list_value/match/max_score_prefix_match_rate",
        max_score_prefix_match_rate,
    )
    tf.summary.scalar(
        "list_value/match/max_score_vs_matched_agreement_rate",
        max_score_vs_matched_agreement_rate,
    )

    # Prefix 越长，随机匹配越难。按真实消费长度拆分后，可以判断旧分最高
    # List 与曝光序列的偏差主要集中在哪一段消费深度。
    for consume_k in range(1, LIST_SIZE + 1):
        k_request_weight = tf.cast(
            tf.equal(consume_depth, consume_k),
            tf.float32,
        ) * observed_request_weight
        k_request_count = tf.reduce_sum(k_request_weight)
        tf.summary.scalar(
            "list_value/match/by_k/k{}_request_rate".format(consume_k),
            k_request_count / (tf.reduce_sum(observed_request_weight) + 1e-8),
        )
        tf.summary.scalar(
            "list_value/match/by_k/k{}_any_prefix_match_rate".format(
                consume_k
            ),
            tf.reduce_sum(k_request_weight * matched_request_weight)
            / (k_request_count + 1e-8),
        )
        tf.summary.scalar(
            "list_value/match/by_k/k{}_max_score_prefix_match_rate".format(
                consume_k
            ),
            tf.reduce_sum(
                k_request_weight
                * tf.cast(max_score_prefix_match, tf.float32)
            ) / (k_request_count + 1e-8),
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
    slide_label = label_value_dict["slide_label"]
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
    list_wt_from_context_pwtd_sum = tf.reduce_sum(
        list_context_pwtd * tf.cast(
            tf.greater(rerank_list_item_idx_flat_list, 0),
            tf.float32,
        ),
        axis=-1,
    )
    list_play_time_s = tf.gather(tf.concat([zeros, play_time_s], axis=-1), rerank_list_item_idx_flat_list, axis=1, batch_dims=1) # (?, list_num, list_size)
    list_duration_s = tf.gather(tf.concat([zeros, duration_s], axis=-1), rerank_list_item_idx_flat_list, axis=1, batch_dims=1) # (?, list_num, list_size)
    list_vtr_label = tf.gather(tf.concat([zeros, vtr_label], axis=-1), rerank_list_item_idx_flat_list, axis=1, batch_dims=1) # (?, list_num, list_size)
    list_wtd_label = tf.gather(tf.concat([zeros, wtd_label], axis=-1), rerank_list_item_idx_flat_list, axis=1, batch_dims=1) # (?, list_num, list_size)
    list_click_label = tf.gather(tf.concat([zeros, click_label], axis=-1), rerank_list_item_idx_flat_list, axis=1, batch_dims=1) # (?, list_num, list_size)
    list_evtr_label = tf.gather(tf.concat([zeros, evtr_label], axis=-1), rerank_list_item_idx_flat_list, axis=1, batch_dims=1) # (?, list_num, list_size)
    list_slide_label = tf.gather(tf.concat([zeros, slide_label], axis=-1), rerank_list_item_idx_flat_list, axis=1, batch_dims=1) # (?, list_num, list_size)
    list_evtr_weight = tf.gather(tf.concat([zeros, evtr_weight], axis=-1), rerank_list_item_idx_flat_list, axis=1, batch_dims=1) # (?, list_num, list_size)
    list_advantage = tf.gather(tf.concat([zeros, advantage], axis=-1), rerank_list_item_idx_flat_list, axis=1, batch_dims=1) # (?, list_num, list_size)

    # -------- 新增 List Value 标签 --------
    # 对每个真实曝光位置做累加，构造 Prefix[1:k] 的累计 WT/EVV。
    # prefix_label_mask 同时限制“旧分最高 List”和“位置已经真实曝光”两个条件，
    # 因而不会拿 K 之后的反事实 item 做监督。
    prefix_watch_time_label = tf.cumsum(list_play_time_s, axis=-1)
    prefix_effective_vv_label = tf.cumsum(list_evtr_label, axis=-1)
    prefix_label_mask = tf.cast(observed_position_mask, tf.float32) \
        * tf.expand_dims(listwise_match_mask, axis=-1)

    # 与老 base 版本保持一致：List 级 WT 标签直接累加全部位置的
    # play_time，不再额外乘 real_show mask。
    list_play_time_s_reduced = tf.reduce_sum(
        list_play_time_s,
        axis=-1,
    )  # (?, list_num)
    # 仅用于监控老版口径与 show-only 口径的标签差异。
    list_play_time_s_reduced_show_only = tf.reduce_sum(
        list_play_time_s * list_show_label,
        axis=-1,
    )
    list_effective_vv_reduced_raw = tf.reduce_sum(
        list_evtr_label,
        axis=-1,
    )
    list_effective_vv_reduced = tf.reduce_sum(
        list_evtr_label * list_show_label,
        axis=-1,
    )  # (?, list_num)

    # -------- P0/P1 数据口径验证（仅日志，不参与 loss）--------
    selected_list_mask_3d = tf.expand_dims(listwise_match_mask, axis=-1)
    binary_list_show_mask = tf.cast(
        tf.greater(list_show_label, 0.0),
        tf.float32,
    )
    binary_list_slide_mask = tf.cast(
        tf.greater(list_slide_label, 0.0),
        tf.float32,
    )

    # slide stdout 验证：判断 Prefix 内部 real_show=0 是否为继续下滑造成的
    # 快速划过，并检查 slide 标签与后续 real_show 的时序一致性。
    internal_real_show_gap_mask = (
        1.0 - binary_list_show_mask
    ) * prefix_label_mask
    internal_real_show_gap_count = tf.reduce_sum(
        internal_real_show_gap_mask
    )
    p1_internal_gap_slide_positive_rate = tf.reduce_sum(
        binary_list_slide_mask * internal_real_show_gap_mask
    ) / (internal_real_show_gap_count + 1e-8)

    later_realshow_exists = tf.cast(
        tf.greater(
            tf.cumsum(
                binary_list_show_mask,
                axis=-1,
                exclusive=True,
                reverse=True,
            ),
            0.0,
        ),
        tf.float32,
    )
    later_realshow_position_mask = (
        later_realshow_exists * selected_list_mask_3d
    )
    p1_later_realshow_after_no_slide_rate = tf.reduce_sum(
        (1.0 - binary_list_slide_mask) * later_realshow_position_mask
    ) / (tf.reduce_sum(later_realshow_position_mask) + 1e-8)

    last_realshow_position_mask = tf.expand_dims(
        tf.one_hot(
            consume_depth - 1,
            depth=LIST_SIZE,
            dtype=tf.float32,
        ) * tf.expand_dims(
            tf.cast(has_observed_prefix, tf.float32),
            axis=-1,
        ),
        axis=1,
    ) * selected_list_mask_3d
    p1_last_realshow_slide_positive_rate = tf.reduce_sum(
        binary_list_slide_mask * last_realshow_position_mask
    ) / (tf.reduce_sum(last_realshow_position_mask) + 1e-8)

    selected_unshown_mask = (
        1.0 - binary_list_show_mask
    ) * selected_list_mask_3d
    selected_unshown_count = tf.reduce_sum(selected_unshown_mask)
    p0_unshown_wt_nonzero_rate = tf.reduce_sum(
        tf.cast(tf.greater(tf.abs(list_play_time_s), 1e-6), tf.float32)
        * selected_unshown_mask
    ) / (selected_unshown_count + 1e-8)
    selected_shown_mask = binary_list_show_mask * selected_list_mask_3d
    selected_shown_count = tf.reduce_sum(selected_shown_mask)
    p0_shown_zero_wt_rate = tf.reduce_sum(
        tf.cast(tf.less_equal(tf.abs(list_play_time_s), 1e-6), tf.float32)
        * selected_shown_mask
    ) / (selected_shown_count + 1e-8)
    p0_real_show_evv_mismatch_rate = tf.reduce_sum(
        tf.cast(
            tf.not_equal(
                tf.greater(binary_list_show_mask, 0.0),
                tf.greater(list_evtr_label, 0.0),
            ),
            tf.float32,
        ) * selected_list_mask_3d
    ) / (
        tf.reduce_sum(selected_list_mask_3d) * float(LIST_SIZE) + 1e-8
    )
    p0_wt_label_abs_delta_mean = tf.reduce_sum(
        tf.abs(
            list_play_time_s_reduced - list_play_time_s_reduced_show_only
        )
        * listwise_match_mask
    ) / (tf.reduce_sum(listwise_match_mask) + 1e-8)
    p0_evv_label_abs_delta_mean = tf.reduce_sum(
        tf.abs(
            list_effective_vv_reduced_raw - list_effective_vv_reduced
        ) * listwise_match_mask
    ) / (tf.reduce_sum(listwise_match_mask) + 1e-8)

    # P1：sequence_mask(K_last) 表示用户到达的连续物理 Prefix；它与稀疏
    # real_show mask 的差异表示 Prefix 内部快速划过的 gap，不再视为标签错位。
    reference_prefix_mask = binary_list_show_mask * selected_list_mask_3d
    prefix_mask_mismatch = tf.cast(
        tf.not_equal(
            tf.greater(prefix_label_mask, 0.0),
            tf.greater(reference_prefix_mask, 0.0),
        ),
        tf.float32,
    )
    p1_internal_real_show_gap_request_rate = tf.reduce_sum(
        tf.cast(
            tf.reduce_any(tf.greater(prefix_mask_mismatch, 0.0), axis=-1),
            tf.float32,
        ) * listwise_match_mask
    ) / (tf.reduce_sum(listwise_match_mask) + 1e-8)
    selected_show_count = tf.reduce_sum(
        binary_list_show_mask * selected_list_mask_3d,
        axis=[1, 2],
    )
    selected_show_count_match = tf.equal(
        selected_show_count,
        tf.cast(realshow_count_raw, tf.float32),
    )
    p1_selected_show_count_mismatch_rate = tf.reduce_mean(
        tf.cast(tf.logical_not(selected_show_count_match), tf.float32)
    )
    internal_real_show_gap_request = tf.reduce_any(
        tf.reduce_any(tf.greater(prefix_mask_mismatch, 0.0), axis=-1),
        axis=-1,
    )
    count_match_weight = tf.cast(selected_show_count_match, tf.float32)
    p1_internal_real_show_gap_given_count_match_rate = tf.reduce_sum(
        tf.cast(internal_real_show_gap_request, tf.float32) * count_match_weight
    ) / (tf.reduce_sum(count_match_weight) + 1e-8)
    selected_position_count = (
        tf.reduce_sum(listwise_match_mask) * float(LIST_SIZE)
    )
    p0_show_label_nonbinary_rate = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.not_equal(list_show_label, 0.0),
                tf.not_equal(list_show_label, 1.0),
            ),
            tf.float32,
        ) * selected_list_mask_3d
    ) / (selected_position_count + 1e-8)
    selected_list_invalid = tf.reduce_any(
        tf.logical_or(
            tf.less_equal(rerank_list_item_idx_flat_list, 0),
            tf.greater(rerank_list_item_idx_flat_list, CANDIDATES_SIZE),
        ),
        axis=-1,
    )
    selected_list_invalid_request_rate = tf.reduce_sum(
        tf.cast(selected_list_invalid, tf.float32) * listwise_match_mask
    ) / (tf.reduce_sum(listwise_match_mask) + 1e-8)
    print_ops.append(tf.print(
        "[jht][list_value][P0_P1_validation]",
        "p0_unshown_wt_nonzero_rate=", p0_unshown_wt_nonzero_rate,
        "p0_shown_zero_wt_rate=", p0_shown_zero_wt_rate,
        "p0_real_show_evv_mismatch_rate=", p0_real_show_evv_mismatch_rate,
        "p0_wt_label_abs_delta_mean=", p0_wt_label_abs_delta_mean,
        "p0_evv_label_abs_delta_mean=", p0_evv_label_abs_delta_mean,
        "p1_internal_real_show_gap_request_rate=", p1_internal_real_show_gap_request_rate,
        "p1_selected_show_count_mismatch_rate=", p1_selected_show_count_mismatch_rate,
        "p1_internal_real_show_gap_given_count_match_rate=", p1_internal_real_show_gap_given_count_match_rate,
        "p1_internal_gap_slide_positive_rate=", p1_internal_gap_slide_positive_rate,
        "p1_later_realshow_after_no_slide_rate=", p1_later_realshow_after_no_slide_rate,
        "p1_last_realshow_slide_positive_rate=", p1_last_realshow_slide_positive_rate,
        "p0_show_label_nonbinary_rate=", p0_show_label_nonbinary_rate,
        "selected_list_invalid_request_rate=", selected_list_invalid_request_rate,
        output_stream=sys.stdout,
    ))

    # 基线 VTR 解码出的观看时长仅保留为离线对照指标，不进入 List Value loss。
    list_vtr_wt = get_watch_time_from_vtr(tf.reshape(point_wise_output_dict["vtr"], [batch_size, LIST_NUM * LIST_SIZE]),
                                          tf.reshape(tf.cast(list_duration_s, dtype=tf.int32), [batch_size, LIST_NUM * LIST_SIZE])) # (? , list_num * list_size)
    # 基线 WTD 输出是按视频时长桶归一化后的比例，先解码回 item WT 秒数。
    list_wtd_wt = wtd_decode(
        tf.reshape(
            point_wise_output_dict["wtd"],
            [batch_size, LIST_NUM * LIST_SIZE],
        ),
        tf.reshape(
            list_duration_s,
            [batch_size, LIST_NUM * LIST_SIZE],
        ),
        duration_bucket=wtd_buckets,
        play_time_buckets_ragged=wtd_configs,
    )
    # 与老 base 版本保持一致：将原单点 VTR/WTD 解码出的 item WT
    # 在 List 维直接求和，不对 padding 位置额外置零。
    list_wt_from_vtr_sum = tf.reduce_sum(
        tf.reshape(
            list_vtr_wt,
            [batch_size, LIST_NUM, LIST_SIZE],
        ),
        axis=-1,
    )
    list_wt_from_wtd_sum = tf.reduce_sum(
        tf.reshape(
            list_wtd_wt,
            [batch_size, LIST_NUM, LIST_SIZE],
        ),
        axis=-1,
    )

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
        length_logits = list_value_output_dict["length_logits"]
        length_probs = list_value_output_dict["length_probs"]
        prefix_watch_time = list_value_output_dict["prefix_watch_time"]
        prefix_watch_time_log = list_value_output_dict["prefix_watch_time_log"]
        prefix_effective_vv = list_value_output_dict["prefix_effective_vv"]
        expected_watch_time = list_value_output_dict["expected_watch_time"]
        expected_effective_vv = list_value_output_dict["expected_effective_vv"]
        expected_consume_length = list_value_output_dict["expected_consume_length"]

        # 长度类别从 0 开始：最后到达物理位置 K 对应类别 K-1。
        # K 取旧分选中 List 中最后一个 real_show 的位置，中间快速划过的
        # real_show=0 位置仍计入到达深度。
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
        # 两个比例都只在参与训练的旧分最高 List 上统计，口径与 length loss 一致。
        label_full_length_rate = masked_mean(
            tf.cast(tf.equal(length_label, LIST_SIZE - 1), tf.float32),
            listwise_match_mask,
        )
        predicted_full_length_rate = masked_mean(
            tf.cast(tf.equal(length_prediction, LIST_SIZE - 1), tf.float32),
            listwise_match_mask,
        )
        # argmax K=6 只能说明第六类是最大单类；平均 P(K=6) 可以进一步
        # 区分轻微偏向与高度确信。non_k6_accuracy 用于暴露多数类塌缩。
        predicted_full_length_probability_mean = masked_mean(
            length_probs[:, :, -1],
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
            "list_value/length/predicted_k6_probability_mean",
            predicted_full_length_probability_mean,
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
        targets.append((
            "list_wt_from_context_pwtd_sum",
            list_wt_from_context_pwtd_sum,
            list_play_time_s_reduced,
            listwise_match_mask,
            "linear_regression",
        ))
        targets.append((
            "list_wt_from_vtr_sum",
            list_wt_from_vtr_sum,
            list_play_time_s_reduced,
            listwise_match_mask,
            "linear_regression",
        ))
        targets.append((
            "list_wt_from_wtd_sum",
            list_wt_from_wtd_sum,
            list_play_time_s_reduced,
            listwise_match_mask,
            "linear_regression",
        ))
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

    # 训练与推理统一使用 P(K=k) × PrefixValue[1:k]。
    expected_list_watch_time = list_value_output_dict["expected_watch_time"]
    expected_list_effective_vv = list_value_output_dict["expected_effective_vv"]
    expected_consume_length = list_value_output_dict["expected_consume_length"]
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
    targets.append((f"expected_list_watch_time", expected_list_watch_time))
    targets.append((f"expected_list_effective_vv", expected_list_effective_vv))
    targets.append((f"expected_consume_length", expected_consume_length))
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
