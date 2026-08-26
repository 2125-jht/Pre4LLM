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
# 训练图同时前向两条 List：第 0 条是按真实曝光顺序恢复的
# factual Prefix，第 1 条是旧分最高 List，仅用于保留原评估口径。
# 所有 loss mask 只选中第 0 条；推理仍对 30 条候选逐条打分。
TRAIN_LIST_NUM = 2

# 每隔固定 step 打印 batch 中第一条请求的原始字段和恢复结果。
# 该日志专用于核对 slide / real_show / real_show_index 语义，
# 不改变任何训练标签或样本 mask。
EXPOSURE_DEBUG_PRINT_INTERVAL = 10

# Standalone 时长实验的全部优化目标；A/D 共享 backbone 与 hazard，
# 使用独立时长 head。EVV 恢复为低权重 item 二分类辅助任务，
# 互动任务仍停用。
LENGTH_LOSS_WEIGHT = 1.0
PREFIX_WT_LOSS_WEIGHT = 1.0
LIST_WT_LOSS_WEIGHT = 0.5
PREFIX_MONOTONIC_LOSS_WEIGHT = 0.1
WTD_D_LOSS_WEIGHT = 1.0
# EVV 只作为辅助监督，避免稀释 A/D 时长主目标。
EVV_ITEM_LOSS_WEIGHT = 0.2
LONG_WATCH_TIME_THRESHOLD_S = 120.0

# 合成换序 Pair 的 preference 实验暂时停用：不构造合成 List、不做额外
# 前向，也不把 preference loss 接入总训练目标。恢复时必须以重建后的事实
# Prefix 为监督边界，不能沿用旧 Top1、原数组前 K 项或反事实 suffix。

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
        dataset.add_feature('context_info__fountain_slide_to_next_list', dataset.DENSE, tf.int64, max_length=60)
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


def get_effective_vv_threshold(duration):
    """按视频时长桶返回 EVV 有效播放秒数阈值。"""
    boundaries = [
        0, 8.366, 10.3, 12.433, 15.066, 17.3, 20.431, 24.833,
        29.333, 33.916, 39.033, 46.566, 54.7, 62.933, 76.366,
        99.166, 178.266, 235, 360.433, 1108.266,
    ]
    evtr_thresholds = [
        4.529, 8.56, 10.154, 11.228, 12.009, 13.51, 13.406,
        13.038, 14.57, 15.108, 16.205, 17.891, 18.748, 18.451,
        19.012, 17.148, 15.472, 13.181, 10.074, 8.925, 9.554,
    ]
    tiled_boundaries = tf.tile(
        tf.expand_dims(tf.constant(boundaries, dtype=tf.float32), axis=0),
        [tf.shape(duration)[0], 1],
    )
    bucket_idx = tf.searchsorted(
        sorted_sequence=tiled_boundaries,
        values=tf.cast(duration, tf.float32),
        side="left",
        out_type=tf.int32,
    )
    bucket_idx = tf.clip_by_value(bucket_idx, 0, len(evtr_thresholds) - 1)
    return tf.gather(
        tf.constant(evtr_thresholds, dtype=tf.float32),
        bucket_idx,
    )


def get_effective_vv_label(duration, play_time):
    """由真实播放时长派生单 item EVV 二值标签。"""
    return tf.cast(
        tf.greater_equal(play_time, get_effective_vv_threshold(duration)),
        tf.float32,
    )


def wtd_encode(duration, play_time, duration_buckets, play_time_buckets):
    """按视频时长桶把播放秒数编码成 [0, 1] 的经验分位 ratio。"""
    tiled_duration_buckets = tf.tile(
        tf.expand_dims(duration_buckets, axis=0),
        [tf.shape(duration)[0], 1],
    )
    duration_bucket_idx = tf.searchsorted(
        sorted_sequence=tiled_duration_buckets,
        values=tf.cast(duration, tf.float32),
        out_type=tf.int32,
    )
    selected_play_buckets = tf.gather(
        play_time_buckets,
        duration_bucket_idx,
    )
    selected_play_buckets_dense = selected_play_buckets.to_tensor(
        default_value=1e19
    )
    play_bucket_idx = tf.searchsorted(
        sorted_sequence=selected_play_buckets_dense,
        values=tf.expand_dims(tf.cast(play_time, tf.float32), axis=-1),
        out_type=tf.int32,
    )
    play_bucket_idx = tf.squeeze(play_bucket_idx, axis=-1)
    selected_bucket_lengths = tf.gather(
        play_time_buckets.row_lengths(),
        duration_bucket_idx,
    )
    ratio = tf.cast(play_bucket_idx, tf.float32) \
        / tf.cast(selected_bucket_lengths + 1, tf.float32)
    return tf.clip_by_value(ratio, 0.0, 1.0)


def wtd_decode(ratio, duration, duration_buckets, play_time_buckets):
    """将 WTD ratio 按对应视频时长桶线性插值回播放秒数。"""
    batch_size = tf.shape(duration)[0]
    value_size = tf.shape(duration)[1]
    tiled_duration_buckets = tf.tile(
        tf.expand_dims(duration_buckets, axis=0),
        [batch_size, 1],
    )
    duration_bucket_idx = tf.searchsorted(
        sorted_sequence=tiled_duration_buckets,
        values=tf.cast(duration, tf.float32),
        out_type=tf.int32,
    )
    selected_play_buckets = tf.gather(
        play_time_buckets,
        duration_bucket_idx,
    )
    selected_bucket_lengths = tf.gather(
        play_time_buckets.row_lengths(),
        duration_bucket_idx,
    )
    original_bucket_idx = tf.clip_by_value(ratio, 0.0, 1.0) \
        * tf.cast(selected_bucket_lengths + 1, tf.float32)
    lower_bucket_id = tf.cast(tf.floor(original_bucket_idx), tf.int64)
    lower_bucket_id = tf.clip_by_value(
        lower_bucket_id,
        tf.zeros_like(lower_bucket_id),
        selected_bucket_lengths,
    )
    selected_play_buckets = tf.concat(
        [
            tf.zeros(
                [batch_size, value_size, 1],
                dtype=selected_play_buckets.dtype,
            ),
            selected_play_buckets,
        ],
        axis=-1,
    ).to_tensor(default_value=-1.0)
    lower_bucket_value = tf.squeeze(
        tf.gather(
            selected_play_buckets,
            tf.expand_dims(lower_bucket_id, axis=-1),
            batch_dims=2,
        ),
        axis=-1,
    )
    higher_bucket_id = tf.where(
        lower_bucket_id >= selected_bucket_lengths,
        selected_bucket_lengths,
        lower_bucket_id + 1,
    )
    higher_bucket_value = tf.squeeze(
        tf.gather(
            selected_play_buckets,
            tf.expand_dims(higher_bucket_id, axis=-1),
            batch_dims=2,
        ),
        axis=-1,
    )
    play_time = tf.where(
        lower_bucket_id >= selected_bucket_lengths,
        lower_bucket_value,
        lower_bucket_value
        + (higher_bucket_value - lower_bucket_value)
        * (original_bucket_idx - tf.cast(lower_bucket_id, tf.float32)),
    )
    return tf.clip_by_value(play_time, 0.0, 36000.0)


wtd_duration_buckets = tf.constant(wtd_config["buckets"], dtype=tf.float32)
wtd_play_time_buckets = tf.ragged.constant(
    wtd_config["configs"],
    dtype=tf.float32,
)


all_param_dict, _, _ = get_param_dict()
model_class = EvaluatorModel(
    all_param_dict,
    print_ops,
    list_size=LIST_SIZE,
    candidates_size=CANDIDATES_SIZE,
    list_num=LIST_NUM,
)

if is_training:
    label_value_dict = {}
    label_value_dict["show_label"] = tf.cast(tf.reshape(config.get_dense_fea("context_info__real_show_list", dim=CANDIDATES_SIZE, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
    raw_slide_to_next_label = tf.cast(
        tf.reshape(
            config.get_dense_fea(
                "context_info__fountain_slide_to_next_list",
                dim=CANDIDATES_SIZE,
                dtype=tf.int64,
            ),
            [-1, CANDIDATES_SIZE],
        ),
        tf.int32,
    )
    label_value_dict["slide_label"] = tf.cast(
        raw_slide_to_next_label,
        tf.float32,
    )
    label_value_dict["play_time_s"] = tf.cast(tf.reshape(config.get_dense_fea("context_info__playing_time_list", dim=CANDIDATES_SIZE, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32) / 1000.0
    label_value_dict["play_time_s"] = tf.clip_by_value(label_value_dict["play_time_s"], 0, 36000)
    label_value_dict["duration_s"] = tf.cast(
        tf.reshape(
            config.get_dense_fea(
                "photo_info__duration_ms_list",
                dim=CANDIDATES_SIZE,
                dtype=tf.int64,
            ),
            [-1, CANDIDATES_SIZE],
        ),
        tf.float32,
    ) / 1000.0
    label_value_dict["duration_s"] = tf.clip_by_value(
        label_value_dict["duration_s"],
        0.0,
        36000.0,
    )
    label_value_dict["pwtd"] = tf.cast(
        tf.reshape(
            config.get_label("context_info__pwtd_list", dim=CANDIDATES_SIZE),
            [-1, CANDIDATES_SIZE],
        ),
        dtype=tf.float32,
    )
    batch_size = tf.shape(label_value_dict["show_label"])[0]
    zeros = tf.zeros([batch_size, 1], dtype=tf.float32)
    raw_show_label = label_value_dict["show_label"]
    raw_play_time_s = label_value_dict["play_time_s"]
    realshow_num_raw = tf.reduce_sum(
        tf.cast(tf.greater(raw_show_label, 0.0), tf.int32),
        axis=-1,
    )
    realshow_num = tf.clip_by_value(realshow_num_raw, 1, LIST_SIZE)
    has_observed_prefix = tf.greater(realshow_num_raw, 0)
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

    # 与 v1_alone 保持一致：先按 real_show 筛出真实曝光 item，再按
    # real_show_index 恢复最终曝光顺序，最后映射到统一候选池坐标。
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
    factual_slide_to_next = tf.gather(
        raw_slide_to_next_label,
        exposure_order,
        axis=1,
        batch_dims=1,
    )[:, :LIST_SIZE]
    factual_play_time_s = tf.gather(
        raw_play_time_s,
        exposure_order,
        axis=1,
        batch_dims=1,
    )[:, :LIST_SIZE]
    factual_position_mask_2d = tf.sequence_mask(
        realshow_num_raw,
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
    # 调试张量用 -1 表示 PAD，避免与合法的 slide=0 / play_time=0
    # 混淆。这两个张量只打印和做监控，不进入 loss。
    factual_slide_to_next = tf.where(
        factual_position_mask_2d,
        factual_slide_to_next,
        tf.fill(tf.shape(factual_slide_to_next), tf.constant(-1, tf.int32)),
    )
    factual_play_time_s = tf.where(
        factual_position_mask_2d,
        factual_play_time_s,
        tf.fill(tf.shape(factual_play_time_s), tf.constant(-1.0, tf.float32)),
    )

    label_value_dict["fountain_fulllink_rerank_index_list"] = tf.cast(
        raw_fountain_rerank_index,
        tf.float32,
    )
    real_show_rerank_indices = tf.expand_dims(
        factual_exposure_rerank_indices,
        axis=1,
    )
    real_show_rerank_indices = tf.cast(real_show_rerank_indices, dtype=tf.int32)
    index_indices = tf.argsort((tf.reshape(label_value_dict['fountain_fulllink_rerank_index_list'], [-1, CANDIDATES_SIZE])), axis=-1)
    index_indices = tf.reshape(index_indices, [batch_size, CANDIDATES_SIZE])
    sorted_fountain_rerank_indices = tf.gather(
        raw_fountain_rerank_index,
        index_indices,
        axis=1,
        batch_dims=1,
    )
    expected_fountain_rerank_indices = tf.reshape(
        tf.range(1, CANDIDATES_SIZE + 1, dtype=tf.int32),
        [1, CANDIDATES_SIZE],
    )
    candidate_pool_index_valid = tf.reduce_all(
        tf.equal(
            sorted_fountain_rerank_indices,
            expected_fountain_rerank_indices,
        ),
        axis=-1,
    )
    print("index_indices ", index_indices)
    for k, v in label_value_dict.items():
        label_value_dict[k] = tf.reshape(label_value_dict[k], [-1, CANDIDATES_SIZE])
        label_value_dict[k] = tf.gather(label_value_dict[k], index_indices, axis=1, batch_dims=1)
    for k, v in all_param_dict.items():
        if k in photo_fea_names:
            all_param_dict[k] = tf.gather(all_param_dict[k], index_indices, axis=1, batch_dims=1)
    # 实际 kai2 中 get_dense_fea default_value 不起作用；get_extra_param 也不支持 default_value ; kai oncall说没办法解决 =，=。
    rerank_list_score_list = config.get_extra_param(
        "rerank_list_score_list",
        size=LIST_NUM,
    )
    rerank_list_score_matrix = tf.reshape(
        rerank_list_score_list,
        [-1, LIST_NUM],
    )
    candidate_list_item_idx_flat_list = config.get_dense_fea("rerank_list_item_idx_flat_list", dim=LIST_NUM * LIST_SIZE, dtype=tf.int64, default_value=-1) + 1
    candidate_list_item_idx_flat_list = tf.cast(tf.reshape(candidate_list_item_idx_flat_list, [-1, LIST_NUM, LIST_SIZE]), tf.int32)
    max_score_list_index = tf.argmax(
        rerank_list_score_matrix,
        axis=-1,
        output_type=tf.int32,
    )
    # max_score_list_index 的形状是 [B]。TensorFlow 1 要求
    # batch_dims < rank(indices)，因此不能对一维 indices 使用
    # tf.gather(..., batch_dims=1)；显式组成 [batch_id, list_id]
    # 坐标后用 gather_nd，得到 [B, LIST_SIZE]。
    legacy_max_score_list = tf.gather_nd(
        candidate_list_item_idx_flat_list,
        tf.stack(
            [tf.range(batch_size, dtype=tf.int32), max_score_list_index],
            axis=-1,
        ),
    )

    # -------- 30候选对事实 Prefix 的覆盖监控 --------
    # 真实日志只告诉我们已经曝光的 Prefix[1:K]，因此这里只比较候选 List
    # 的前 K 位。匹配结果不再筛选训练样本，也不再从候选中选择监督 List。
    observed_position_mask = tf.expand_dims(
        tf.sequence_mask(realshow_num, maxlen=LIST_SIZE, dtype=tf.bool),
        axis=1,
    )  # (?, 1, list_size)
    prefix_item_match = tf.logical_or(
        tf.equal(candidate_list_item_idx_flat_list, real_show_rerank_indices),
        tf.logical_not(observed_position_mask),
    )
    prefix_list_match = tf.reduce_all(prefix_item_match, axis=-1) # (?, list_num)
    prefix_list_match = tf.logical_and(
        prefix_list_match,
        tf.expand_dims(has_observed_prefix, axis=-1),
    )

    # rank 可以有间隔，但有效曝光 rank 必须为正且不能重复。0 表示上游
    # 缺失值，不能作为事实顺序；这也覆盖了仅曝光 1 个 item 且 rank=0 的情况。
    factual_rank_positive = tf.reduce_all(
        tf.logical_or(
            tf.greater(factual_real_show_indices, 0),
            tf.logical_not(factual_position_mask_2d),
        ),
        axis=-1,
    )
    adjacent_factual_position_mask = tf.sequence_mask(
        tf.maximum(realshow_num_raw - 1, 0),
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
        factual_rank_positive,
        factual_rank_strictly_increasing,
    )

    # 事实 Prefix 最终会直接作为模型 List 下标，因此还需保证候选坐标位于
    # [1, 60] 且互不重复。未曝光位置已经补 0，不参与有效性检查。
    factual_candidate_in_range = tf.reduce_all(
        tf.logical_or(
            tf.logical_and(
                tf.greater(factual_exposure_rerank_indices, 0),
                tf.less_equal(
                    factual_exposure_rerank_indices,
                    CANDIDATES_SIZE,
                ),
            ),
            tf.logical_not(factual_position_mask_2d),
        ),
        axis=-1,
    )
    factual_candidate_sort_key = tf.where(
        factual_position_mask_2d,
        factual_exposure_rerank_indices,
        tf.fill(
            tf.shape(factual_exposure_rerank_indices),
            tf.constant(2147483647, dtype=tf.int32),
        ),
    )
    factual_candidate_sort_order = tf.argsort(
        factual_candidate_sort_key,
        axis=-1,
    )
    factual_candidate_indices_sorted = tf.gather(
        factual_exposure_rerank_indices,
        factual_candidate_sort_order,
        axis=1,
        batch_dims=1,
    )
    factual_candidate_unique = tf.reduce_all(
        tf.logical_or(
            tf.greater(
                factual_candidate_indices_sorted[:, 1:],
                factual_candidate_indices_sorted[:, :-1],
            ),
            tf.logical_not(adjacent_factual_position_mask),
        ),
        axis=-1,
    )
    factual_candidate_valid = tf.logical_and(
        factual_candidate_in_range,
        factual_candidate_unique,
    )
    factual_sample_valid = tf.logical_and(
        has_observed_prefix,
        tf.logical_and(
            candidate_pool_index_valid,
            tf.logical_and(factual_rank_valid, factual_candidate_valid),
        ),
    )
    prefix_list_match = tf.logical_and(
        prefix_list_match,
        tf.expand_dims(factual_sample_valid, axis=-1),
    )
    has_prefix_match = tf.reduce_any(prefix_list_match, axis=-1) # (?,)
    observed_request_weight = tf.cast(has_observed_prefix, tf.float32)
    matched_request_weight = tf.cast(has_prefix_match, tf.float32)

    # 有真实曝光前缀（realshow_num_raw>0）的请求占比
    tf.summary.scalar(
        "list_value/match/observed_prefix_rate",
        tf.reduce_mean(observed_request_weight),
    )
    # 【已停用】候选 list 覆盖真实前缀的诊断（prefix_match_rate / candidate_prefix_miss_rate），
    # 来自旧的“候选匹配事实前缀”机制：匹配结果不再筛选训练样本、也不再选择监督
    # List，曲线无决策价值。上游 has_prefix_match / matched_request_weight 暂未清理。
    # tf.summary.scalar(
    #     "list_value/match/prefix_match_rate",
    #     tf.reduce_mean(tf.cast(has_prefix_match, tf.float32)),
    # )
    # tf.summary.scalar(
    #     "list_value/match/candidate_prefix_miss_rate",
    #     1.0 - tf.reduce_mean(tf.cast(has_prefix_match, tf.float32)),
    # )
    # 真实曝光 rank 为正且严格递增的请求占比（分母为有前缀的请求）
    tf.summary.scalar(
        "list_value/match/factual_rank_valid_rate",
        tf.reduce_sum(
            tf.cast(factual_rank_valid, tf.float32)
            * observed_request_weight
        ) / (tf.reduce_sum(observed_request_weight) + 1e-8),
    )
    # 事实候选下标在 [1,60] 且互不重复的请求占比
    tf.summary.scalar(
        "list_value/match/factual_candidate_valid_rate",
        tf.reduce_sum(
            tf.cast(factual_candidate_valid, tf.float32)
            * observed_request_weight
        ) / (tf.reduce_sum(observed_request_weight) + 1e-8),
    )
    # 候选池下标完整合法（排序后恰为 1..60）的请求占比
    tf.summary.scalar(
        "list_value/match/candidate_pool_index_valid_rate",
        tf.reduce_sum(
            tf.cast(candidate_pool_index_valid, tf.float32)
            * observed_request_weight
        ) / (tf.reduce_sum(observed_request_weight) + 1e-8),
    )
    factual_request_weight = tf.cast(factual_sample_valid, tf.float32)
    # 通过全部校验、进入事实训练的请求占比（样本保留率）
    tf.summary.scalar(
        "list_value/match/factual_training_request_rate",
        tf.reduce_mean(factual_request_weight),
    )
    # 被校验过滤掉的请求占比
    tf.summary.scalar(
        "list_value/match/factual_training_request_drop_rate",
        1.0 - tf.reduce_mean(factual_request_weight),
    )
    # 有效请求中原始曝光数 K>6 的截断样本占比
    tf.summary.scalar(
        "list_value/factual_training/raw_k_gt_list_size_rate",
        tf.reduce_sum(
            factual_request_weight
            * tf.cast(tf.greater(realshow_num_raw, LIST_SIZE), tf.float32)
        ) / (tf.reduce_sum(factual_request_weight) + 1e-8),
    )

    # 各 K 下的30候选 Prefix 覆盖率只作为数据诊断，不再决定训练保留率。
    for consume_k in range(1, LIST_SIZE + 1):
        k_request_weight = tf.cast(
            tf.equal(realshow_num, consume_k),
            tf.float32,
        ) * observed_request_weight
        k_request_count = tf.reduce_sum(k_request_weight)
        # 【已停用】候选覆盖真实前缀的分 K 诊断，旧机制遗留，无决策价值。
        # tf.summary.scalar(
        #     "list_value/match/by_k/k{}_any_prefix_match_rate".format(
        #         consume_k
        #     ),
        #     tf.reduce_sum(k_request_weight * matched_request_weight)
        #     / (k_request_count + 1e-8),
        # )
        k_factual_training_weight = k_request_weight * factual_request_weight
        # 分 K：消费长度=k 的请求中通过校验的比例
        tf.summary.scalar(
            "list_value/factual_training/by_k/k{}_valid_rate".format(
                consume_k
            ),
            tf.reduce_sum(k_factual_training_weight)
            / (k_request_count + 1e-8),
        )
        # 分 K：有效样本中各消费长度 K 的占比分布
        tf.summary.scalar(
            "list_value/factual_training/by_k/k{}_request_rate".format(
                consume_k
            ),
            tf.reduce_sum(k_factual_training_weight)
            / (tf.reduce_sum(factual_request_weight) + 1e-8),
        )

    play_time_s = label_value_dict["play_time_s"]
    duration_s = label_value_dict["duration_s"]
    evtr_label = get_effective_vv_label(duration_s, play_time_s)
    context_pwtd = label_value_dict["pwtd"]

    # 正式训练输入是按 real_show + real_show_index 恢复的事实
    # Prefix，K 后补 0(PAD)。模型是 causal 的，已曝光位置不会读取
    # 未来 PAD。额外前向旧分最高 List 只为保留历史评估，不进入任何 loss。
    # TensorFlow 1 的 tf.where(Select) 不会将 [B, 1] 条件自动
    # 广播到 [B, LIST_SIZE]，因此这里显式 tile，避免建图阶段形状错误。
    factual_sample_valid_2d = tf.tile(
        tf.expand_dims(factual_sample_valid, axis=-1),
        [1, LIST_SIZE],
    )
    safe_factual_exposure_rerank_indices = tf.where(
        factual_sample_valid_2d,
        factual_exposure_rerank_indices,
        tf.zeros_like(factual_exposure_rerank_indices),
    )
    rerank_list_item_idx_flat_list = tf.stack(
        [safe_factual_exposure_rerank_indices, legacy_max_score_list],
        axis=1,
    )  # (?, TRAIN_LIST_NUM=2, LIST_SIZE)
    factual_training_list_mask = tf.stack(
        [factual_request_weight, tf.zeros_like(factual_request_weight)],
        axis=-1,
    )
    factual_eval_list_mask = tf.stop_gradient(
        tf.identity(factual_training_list_mask, name="factual_eval_list_mask")
    )
    legacy_max_score_eval_list_mask = tf.stop_gradient(
        tf.stack(
            [tf.zeros_like(factual_request_weight), tf.ones_like(factual_request_weight)],
            axis=-1,
            name="legacy_max_score_eval_list_mask",
        )
    )
    complete_factual_list_mask = factual_training_list_mask * tf.cast(
        tf.expand_dims(tf.equal(realshow_num_raw, LIST_SIZE), axis=-1),
        tf.float32,
    )
    # K=6 完整事实样本占比（list 级训练/评估的样本量）
    tf.summary.scalar(
        "list_value/match/complete_factual_list_request_rate",
        tf.reduce_mean(tf.reduce_sum(complete_factual_list_mask, axis=-1)),
    )

    # -------- slide / 曝光序列字段联合排查日志 --------
    # 只在每 10 step 打印 batch 第 0 条请求，避免常规训练日志量
    # 失控。summarize=-1 保留完整 60 item 原始数组和30条候选 List。
    worker_global_step = config.get_step()
    debug_continue_from_k = tf.cast(
        tf.expand_dims(realshow_num_raw, axis=-1)
        > tf.reshape(tf.range(1, LIST_SIZE + 1), [1, LIST_SIZE]),
        tf.int32,
    )

    def print_exposure_debug_sample():
        # 每个字段单独一行，并通过 control dependency 串行化，
        # 避免多个 tf.print 在分布式运行时乱序。
        debug_print_ops = []

        def append_debug_line(section, *values):
            with tf.control_dependencies(debug_print_ops[-1:]):
                line_op = tf.print(
                    "[exposure_slide_debug]",
                    "step=", worker_global_step,
                    section,
                    *values,
                    summarize=-1,
                    output_stream=sys.stdout,
                )
            debug_print_ops.append(line_op)

        append_debug_line("BEGIN", "raw_k=", realshow_num_raw[0])
        append_debug_line(
            "RAW.real_show                 =",
            tf.cast(raw_show_label[0], tf.int32),
        )
        append_debug_line(
            "RAW.real_show_index           =",
            raw_real_show_index[0],
        )
        append_debug_line(
            "RAW.slide_to_next             =",
            raw_slide_to_next_label[0],
        )
        append_debug_line("RAW.play_time_s               =", raw_play_time_s[0])
        append_debug_line(
            "RAW.fountain_rerank_index     =",
            raw_fountain_rerank_index[0],
        )
        append_debug_line(
            "FACTUAL.source_array_index    =",
            exposure_order[0, :realshow_num[0]],
        )
        append_debug_line(
            "FACTUAL.real_show_index       =",
            factual_real_show_indices[0],
        )
        append_debug_line(
            "FACTUAL.rerank_index          =",
            factual_exposure_rerank_indices[0],
        )
        append_debug_line(
            "FACTUAL.slide_to_next         =",
            factual_slide_to_next[0],
        )
        append_debug_line(
            "FACTUAL.play_time_s           =",
            factual_play_time_s[0],
        )
        append_debug_line(
            "FACTUAL.continue_label_from_k =",
            debug_continue_from_k[0],
        )
        append_debug_line(
            "VALIDITY",
            "rank=", factual_rank_valid[0],
            "candidate=", factual_candidate_valid[0],
            "pool=", candidate_pool_index_valid[0],
            "sample=", factual_sample_valid[0],
            "any_prefix_match=", has_prefix_match[0],
        )
        append_debug_line(
            "MATCH.candidate_mask          =",
            prefix_list_match[0],
        )
        append_debug_line(
            "LEGACY.top_index=", max_score_list_index[0],
            "items=", legacy_max_score_list[0],
        )
        for candidate_idx in range(LIST_NUM):
            append_debug_line(
                "CANDIDATE[{:02d}]".format(candidate_idx),
                "items=", candidate_list_item_idx_flat_list[0, candidate_idx],
                "score=", rerank_list_score_matrix[0, candidate_idx],
                "prefix_match=", prefix_list_match[0, candidate_idx],
                "padding_like=", tf.logical_and(
                    tf.reduce_all(
                        tf.equal(
                            candidate_list_item_idx_flat_list[0, candidate_idx],
                            1,
                        )
                    ),
                    tf.equal(rerank_list_score_matrix[0, candidate_idx], 0.0),
                ),
                "is_legacy_top=", tf.equal(
                    max_score_list_index[0],
                    tf.constant(candidate_idx, dtype=tf.int32),
                ),
            )
        append_debug_line("END")

        with tf.control_dependencies(debug_print_ops[-1:]):
            return tf.constant(0, dtype=tf.int32)

    # 【已停用】曝光/slide 逐字段 debug 打印：每 EXPOSURE_DEBUG_PRINT_INTERVAL step
    # 把 batch 第 0 条请求的原始数组、事实前缀、30 条候选 list 全量打到 stdout，
    # 训练日志降噪。print_exposure_debug_sample 函数体保留未删，恢复时解除下方注释。
    # exposure_debug_token = tf.cond(
    #     tf.equal(
    #         tf.math.floormod(
    #             worker_global_step,
    #             tf.cast(EXPOSURE_DEBUG_PRINT_INTERVAL, worker_global_step.dtype),
    #         ),
    #         tf.cast(0, worker_global_step.dtype),
    #     ),
    #     print_exposure_debug_sample,
    #     lambda: tf.constant(0, dtype=tf.int32),
    # )
    # print_ops.append(exposure_debug_token)

    model_class._training = True
    list_value_output_dict = model_class.model(
        list_index=rerank_list_item_idx_flat_list,
        list_num=TRAIN_LIST_NUM,
    )
    print(f"====> train standalone list model, gen...")

    list_play_time_s = tf.gather(
        tf.concat([zeros, play_time_s], axis=-1),
        rerank_list_item_idx_flat_list,
        axis=1,
        batch_dims=1,
    )
    list_duration_s = tf.gather(
        tf.concat([zeros, duration_s], axis=-1),
        rerank_list_item_idx_flat_list,
        axis=1,
        batch_dims=1,
    )
    list_evtr_label = tf.gather(
        tf.concat([zeros, evtr_label], axis=-1),
        rerank_list_item_idx_flat_list,
        axis=1,
        batch_dims=1,
    )
    list_context_pwtd = tf.gather(
        tf.concat([zeros, context_pwtd], axis=-1),
        rerank_list_item_idx_flat_list,
        axis=1,
        batch_dims=1,
    )
    # -------- List Watch Time / EVV 标签 --------
    # 对每个真实曝光位置做累加，构造 Prefix[1:k] 的累计 WT/EVV。
    # prefix_label_mask 同时限制“事实样本有效”和“位置已经真实曝光”，
    # 因而不会拿 K 之后的 PAD 构造价值监督。
    prefix_watch_time_label = tf.cumsum(list_play_time_s, axis=-1)
    prefix_effective_vv_label = tf.cumsum(list_evtr_label, axis=-1)
    prefix_label_mask = tf.cast(observed_position_mask, tf.float32) \
        * tf.expand_dims(factual_training_list_mask, axis=-1)

    list_play_time_s_reduced = tf.reduce_sum(list_play_time_s, axis=-1)  # (?, list_num)
    list_effective_vv_reduced = tf.reduce_sum(list_evtr_label, axis=-1)  # (?, list_num)

    with tf.control_dependencies(print_ops):
        targets = []
        # -------- List Value 损失 --------
        continue_logits = list_value_output_dict["continue_logits"]
        continue_probs = list_value_output_dict["continue_probs"]
        length_probs = list_value_output_dict["length_probs"]
        prefix_watch_time = list_value_output_dict["prefix_watch_time"]
        prefix_watch_time_log = list_value_output_dict["prefix_watch_time_log"]
        wtd_ratio_d = list_value_output_dict["wtd_ratio_d"]
        evv_logits = list_value_output_dict["evv_logits"]
        evv_probs = list_value_output_dict["evv_probs"]
        prefix_effective_vv = list_value_output_dict["prefix_effective_vv"]

        # D 先按 duration-conditioned WTD 曲线把每个 item 的 ratio
        # 解码为秒，再构造 PrefixWT_D(k)。最终聚合与 A 完全一致：
        # sum_k P(K=k) * PrefixWT_D(k)。
        flat_list_item_size = TRAIN_LIST_NUM * LIST_SIZE
        wtd_ratio_d_flat = tf.reshape(
            wtd_ratio_d,
            [batch_size, flat_list_item_size],
        )
        list_duration_s_flat = tf.reshape(
            list_duration_s,
            [batch_size, flat_list_item_size],
        )
        list_play_time_s_flat = tf.reshape(
            list_play_time_s,
            [batch_size, flat_list_item_size],
        )
        wtd_label_d_flat = wtd_encode(
            list_duration_s_flat,
            list_play_time_s_flat,
            wtd_duration_buckets,
            wtd_play_time_buckets,
        )
        decoded_watch_time_d_flat = wtd_decode(
            wtd_ratio_d_flat,
            list_duration_s_flat,
            wtd_duration_buckets,
            wtd_play_time_buckets,
        )
        wtd_label_d = tf.reshape(
            wtd_label_d_flat,
            [batch_size, TRAIN_LIST_NUM, LIST_SIZE],
        )
        decoded_watch_time_d = tf.reshape(
            decoded_watch_time_d_flat,
            [batch_size, TRAIN_LIST_NUM, LIST_SIZE],
        )
        prefix_watch_time_d = tf.cumsum(decoded_watch_time_d, axis=-1)
        expected_list_watch_time_d = tf.reduce_sum(
            length_probs * prefix_watch_time_d,
            axis=-1,
            name="expected_list_watch_time_d",
        )

        # 四个核心输出：
        # - expected_consume_length：由 continue 概率得到的预期消费 item 数；
        # - A：sum_k P(K=k) * PrefixWT(k)；
        # - D：duration-conditioned WTD 单 item 解码 + A 的 P(K) 聚合。
        # - EVV：单 item 有效播放概率累加 + 同一 P(K) 聚合。
        expected_list_watch_time = list_value_output_dict["expected_list_watch_time"]
        expected_list_effective_vv = list_value_output_dict["expected_list_effective_vv"]
        expected_consume_length = list_value_output_dict["expected_consume_length"]

        # 与上面同一采样 step 输出模型概率，便于将 factual/legacy
        # 两条 List 的预测与 slide 字段、K 推导标签直接对齐。
        def print_exposure_prediction_debug_sample():
            prediction_print_ops = []

            def append_prediction_line(section, *values):
                with tf.control_dependencies(prediction_print_ops[-1:]):
                    line_op = tf.print(
                        "[exposure_slide_prediction_debug]",
                        "step=", worker_global_step,
                        section,
                        *values,
                        summarize=-1,
                        output_stream=sys.stdout,
                    )
                prediction_print_ops.append(line_op)

            append_prediction_line("BEGIN", "raw_k=", realshow_num_raw[0])
            append_prediction_line(
                "LABEL.factual_slide_to_next =",
                factual_slide_to_next[0],
            )
            append_prediction_line(
                "LABEL.continue_from_k       =",
                debug_continue_from_k[0],
            )
            append_prediction_line(
                "PRED.factual_continue       =",
                continue_probs[0, 0],
            )
            append_prediction_line(
                "PRED.legacy_continue        =",
                continue_probs[0, 1],
            )
            append_prediction_line(
                "PRED.factual_length_probs   =",
                length_probs[0, 0],
            )
            append_prediction_line(
                "PRED.legacy_length_probs    =",
                length_probs[0, 1],
            )
            append_prediction_line(
                "PRED.factual_expected_k     =",
                expected_consume_length[0, 0],
            )
            append_prediction_line(
                "PRED.legacy_expected_k      =",
                expected_consume_length[0, 1],
            )
            append_prediction_line("END")

            with tf.control_dependencies(prediction_print_ops[-1:]):
                return tf.constant(0, dtype=tf.int32)

        # 【已停用】预测值 debug 打印：与上面同一 step 输出 continue/length 概率
        # 做 label 对齐排查，训练日志降噪。print_exposure_prediction_debug_sample
        # 函数体保留未删；下方 identity 仅为挂载打印依赖，去掉后语义不变。
        # exposure_prediction_debug_token = tf.cond(
        #     tf.equal(
        #         tf.math.floormod(
        #             worker_global_step,
        #             tf.cast(
        #                 EXPOSURE_DEBUG_PRINT_INTERVAL,
        #                 worker_global_step.dtype,
        #             ),
        #         ),
        #         tf.cast(0, worker_global_step.dtype),
        #     ),
        #     print_exposure_prediction_debug_sample,
        #     lambda: tf.constant(0, dtype=tf.int32),
        # )
        # with tf.control_dependencies([exposure_prediction_debug_token]):
        #     continue_logits = tf.identity(continue_logits)
        #     continue_probs = tf.identity(continue_probs)
        #     length_probs = tf.identity(length_probs)

        # padding 位置在 gather list_context_pwtd 时已经取到前置零列，
        # 因此这里与原版线上聚合一致，直接对 List 内 item 分数求和。
        list_wt_from_context_pwtd_sum = tf.reduce_sum(
            list_context_pwtd,
            axis=-1,
        )
        # 复刻 backbone 线上 item 聚合的固定位置衰减。权重只依赖
        # 候选位置，不使用当前样本的真实曝光 K，因而可与 List
        # 分支作为可线上使用的公平对照。
        position_indices = tf.reshape(
            tf.range(1, LIST_SIZE + 1, dtype=tf.float32),
            [1, 1, LIST_SIZE],
        )
        context_pwtd_position_decay = 1.0 / (
            0.3 + tf.pow(position_indices, 0.6)
        )
        list_wt_from_context_pwtd_position_decay = tf.reduce_sum(
            list_context_pwtd
            * context_pwtd_position_decay,
            axis=-1,
        )

        # 长度类别仍从 0 开始：真实消费 K 个 item，对应类别 K-1。
        # Hazard 标签逐位置表示“消费完当前位置后是否继续”：
        # K=3 -> [继续, 继续, 停止, 未观测, 未观测]；
        # K=6 -> [继续, 继续, 继续, 继续, 继续]，末端按右截断处理。
        length_label = tf.tile(
            tf.expand_dims(realshow_num - 1, axis=-1),
            [1, TRAIN_LIST_NUM],
        )
        continue_label_per_request = tf.cast(
            tf.expand_dims(realshow_num, axis=-1)
            > tf.reshape(tf.range(1, LIST_SIZE), [1, LIST_SIZE - 1]),
            tf.float32,
        )
        continue_labels = tf.tile(
            tf.expand_dims(continue_label_per_request, axis=1),
            [1, TRAIN_LIST_NUM, 1],
        )
        hazard_position_mask = tf.sequence_mask(
            tf.minimum(realshow_num, LIST_SIZE - 1),
            maxlen=LIST_SIZE - 1,
            dtype=tf.float32,
        )
        slide_to_next_for_hazard = tf.cast(
            factual_slide_to_next[:, :LIST_SIZE - 1],
            tf.float32,
        )
        slide_binary_valid_mask = tf.cast(
            tf.logical_or(
                tf.equal(slide_to_next_for_hazard, 0.0),
                tf.equal(slide_to_next_for_hazard, 1.0),
            ),
            tf.float32,
        )
        slide_comparison_mask = hazard_position_mask \
            * tf.expand_dims(factual_request_weight, axis=-1) \
            * slide_binary_valid_mask
        slide_vs_k_disagreement = tf.cast(
            tf.not_equal(
                slide_to_next_for_hazard,
                continue_label_per_request,
            ),
            tf.float32,
        )
        # 【已停用】slide_debug 四个指标是 slide/曝光字段联查专用，排查结束后
        # 无日常监控价值；相关中间变量（slide_comparison_mask 等）暂未清理。
        # tf.summary.scalar(
        #     "list_value/slide_debug/slide_binary_valid_rate",
        #     tf.reduce_sum(
        #         hazard_position_mask
        #         * tf.expand_dims(factual_request_weight, axis=-1)
        #         * slide_binary_valid_mask
        #     ) / (
        #         tf.reduce_sum(
        #             hazard_position_mask
        #             * tf.expand_dims(factual_request_weight, axis=-1)
        #         ) + 1e-8
        #     ),
        # )
        # tf.summary.scalar(
        #     "list_value/slide_debug/slide_vs_k_disagreement_rate",
        #     tf.reduce_sum(slide_vs_k_disagreement * slide_comparison_mask)
        #     / (tf.reduce_sum(slide_comparison_mask) + 1e-8),
        # )
        nonterminal_position_mask = tf.sequence_mask(
            tf.maximum(tf.minimum(realshow_num_raw, LIST_SIZE) - 1, 0),
            maxlen=LIST_SIZE - 1,
            dtype=tf.float32,
        ) * tf.expand_dims(factual_request_weight, axis=-1)
        nonterminal_slide_mask = nonterminal_position_mask \
            * slide_binary_valid_mask
        # tf.summary.scalar(
        #     "list_value/slide_debug/nonterminal_slide_positive_rate",
        #     tf.reduce_sum(slide_to_next_for_hazard * nonterminal_slide_mask)
        #     / (tf.reduce_sum(nonterminal_slide_mask) + 1e-8),
        # )
        has_observed_terminal = tf.logical_and(
            tf.greater(realshow_num_raw, 0),
            tf.less(realshow_num_raw, LIST_SIZE),
        )
        terminal_position_mask = tf.one_hot(
            tf.maximum(realshow_num_raw - 1, 0),
            depth=LIST_SIZE - 1,
            dtype=tf.float32,
        ) * tf.expand_dims(
            factual_request_weight
            * tf.cast(has_observed_terminal, tf.float32),
            axis=-1,
        ) * slide_binary_valid_mask
        # tf.summary.scalar(
        #     "list_value/slide_debug/terminal_slide_positive_rate",
        #     tf.reduce_sum(slide_to_next_for_hazard * terminal_position_mask)
        #     / (tf.reduce_sum(terminal_position_mask) + 1e-8),
        # )
        hazard_position_mask = tf.expand_dims(
            hazard_position_mask,
            axis=1,
        )
        hazard_observed_mask = hazard_position_mask \
            * tf.expand_dims(factual_training_list_mask, axis=-1)
        # 保留原“旧分最高 List”的 continuation 评估口径。该 mask
        # 只选中第 1 条评估 List，不会进入 hazard loss。
        legacy_hazard_observed_mask = hazard_position_mask \
            * tf.expand_dims(legacy_max_score_eval_list_mask, axis=-1)

        hazard_ce = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=continue_labels,
            logits=continue_logits,
        )
        # 对每条 List 累加所有已观测决策，得到右截断的负对数似然；
        # 再按有效事实 Prefix 数量求均值，使每个请求保持相同样本权重。
        length_nll_per_list = tf.reduce_sum(
            hazard_ce * hazard_observed_mask,
            axis=-1,
        )
        length_loss = tf.reduce_sum(length_nll_per_list) \
            / (tf.reduce_sum(factual_training_list_mask) + 1e-8)
        length_prediction = tf.argmax(length_probs, axis=-1, output_type=tf.int32)
        length_accuracy = tf.reduce_sum(
            tf.cast(tf.equal(length_prediction, length_label), tf.float32)
            * complete_factual_list_mask
        ) / (tf.reduce_sum(complete_factual_list_mask) + 1e-8)

        # A：累计 Prefix WT，沿用原有 log1p Huber。
        prefix_watch_time_label_log = tf.math.log1p(prefix_watch_time_label)
        prefix_wt_loss = tf.losses.huber_loss(
            labels=prefix_watch_time_label_log,
            predictions=prefix_watch_time_log,
            weights=prefix_label_mask,
            delta=0.5,
        )

        # EVV 是单 item 二值事件，直接用 BCE 监督概率；不再回归
        # 累计 PrefixEVV，也不使用 reward/interaction 加权。
        evv_item_ce = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=list_evtr_label,
            logits=evv_logits,
        )
        evv_item_loss = tf.reduce_sum(evv_item_ce * prefix_label_mask) \
            / (tf.reduce_sum(prefix_label_mask) + 1e-8)

        # D 直接学习 duration-conditioned WTD ratio；解码后的秒数
        # 用于构造 PrefixWT_D(k) 并以与 A 相同的 P(K) 聚合。
        wtd_d_loss = tf.losses.log_loss(
            labels=wtd_label_d,
            predictions=wtd_ratio_d,
            weights=prefix_label_mask,
        )

        # 只有原始 K 恰好为 6 时，事实输入才是一条没有 PAD 的完整 List，
        # 可以安全地监督依赖全部位置的 expected List Value。K<6 的累计价值
        # 已经由 Prefix loss 监督；K>6 在本模型中属于截断样本，也不构造总值标签。
        list_total_supervision_mask = complete_factual_list_mask
        list_wt_loss = tf.losses.huber_loss(
            labels=tf.math.log1p(list_play_time_s_reduced),
            predictions=tf.math.log1p(expected_list_watch_time),
            weights=list_total_supervision_mask,
            delta=0.5,
        )
        # 合成换序 Pair 的 preference loss 暂停；标准 factual 监督不依赖它。

        # A 的累计 WT 随前缀变长不应下降。
        prefix_pair_mask = prefix_label_mask[:, :, 1:]
        wt_monotonic_error = tf.nn.relu(
            prefix_watch_time[:, :, :-1] - prefix_watch_time[:, :, 1:]
        )
        monotonic_loss = tf.reduce_sum(
            wt_monotonic_error * prefix_pair_mask
        ) / (tf.reduce_sum(prefix_pair_mask) + 1e-8)

        weighted_length_loss = LENGTH_LOSS_WEIGHT * length_loss
        weighted_prefix_wt_loss = PREFIX_WT_LOSS_WEIGHT * prefix_wt_loss
        weighted_list_wt_loss = LIST_WT_LOSS_WEIGHT * list_wt_loss
        weighted_wtd_d_loss = WTD_D_LOSS_WEIGHT * wtd_d_loss
        weighted_evv_item_loss = EVV_ITEM_LOSS_WEIGHT * evv_item_loss
        weighted_monotonic_loss = PREFIX_MONOTONIC_LOSS_WEIGHT * monotonic_loss
        list_value_loss = weighted_length_loss \
            + weighted_prefix_wt_loss \
            + weighted_list_wt_loss \
            + weighted_wtd_d_loss \
            + weighted_evv_item_loss \
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

        # -------- TensorBoard 监控：逐位置继续概率 --------
        # 所有校准指标严格使用事实 Prefix 的 hazard mask，不混入无效请求、
        # 未观测位置或 PAD。分位置统计可避免全局结果被位置先验掩盖。
        for hazard_position in range(LIST_SIZE - 1):
            position_hazard_mask = hazard_observed_mask[:, :, hazard_position]
            # 分位置：预测继续率/真实继续率，=1 表示该位置校准无偏
            tf.summary.scalar(
                "list_value/continuation/by_position/pos{}_pred_label_ratio".format(
                    hazard_position + 1
                ),
                masked_pred_label_ratio(
                    continue_probs[:, :, hazard_position],
                    continue_labels[:, :, hazard_position],
                    position_hazard_mask,
                ),
            )

        # 继续概率的 Brier 分数 mean((p-y)^2)：整体校准误差，越低越好；
        # 摆烂基线（预测常数 p̄）为 p̄(1-p̄)，主要看趋势和版本间对比。
        tf.summary.scalar(
            "list_value/continuation/global/brier_score",
            masked_mean(
                tf.square(continue_probs - continue_labels),
                hazard_observed_mask,
            ),
        )

        # -------- TensorBoard 监控：消费长度 --------
        # 完整 P(K) 和 expected length 需要完整六项输入。事实 Prefix 含 PAD
        # 时只监控逐位置 hazard；以下分布类指标仅使用无 PAD 的原始 K=6 样本。
        full_list_eval_mask = tf.stop_gradient(list_total_supervision_mask)
        length_label_value = tf.cast(length_label + 1, tf.float32)
        expected_length_mae = masked_mean(
            tf.abs(expected_consume_length - length_label_value),
            full_list_eval_mask,
        )
        # argmax P(K) 命中真实 K 的比例（仅 K=6 完整样本）
        tf.summary.scalar(
            "list_value/length/accuracy",
            length_accuracy,
        )
        # 期望消费长度 E[K] 的均值
        tf.summary.scalar(
            "list_value/length/predicted_k_mean",
            masked_mean(expected_consume_length, full_list_eval_mask),
        )
        # 真实 K 均值（完整样本恒为 6，作对照）
        tf.summary.scalar(
            "list_value/length/label_k_mean",
            masked_mean(length_label_value, full_list_eval_mask),
        )
        # |E[K]-真实K| 的平均误差
        tf.summary.scalar(
            "list_value/length/expected_k_mae",
            expected_length_mae,
        )

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
        # 真实 K 类别被赋予的预测概率均值，越高越好
        tf.summary.scalar(
            "list_value/length/probability/true_class_probability_mean",
            masked_mean(length_true_class_probability, full_list_eval_mask),
        )
        # P(K) 最高类别概率均值（预测置信度）
        tf.summary.scalar(
            "list_value/length/probability/top1_probability_mean",
            masked_mean(tf.reduce_max(length_probs, axis=-1), full_list_eval_mask),
        )
        # P(K) 分布熵（预测不确定度）
        tf.summary.scalar(
            "list_value/length/probability/entropy_mean",
            masked_mean(length_probability_entropy, full_list_eval_mask),
        )

        # 【已停用】分 K 消费长度指标：完整样本 mask 下真实 K 恒为 6，
        # label_rate/recall 退化为常数，argmax_rate 与 accuracy 重复，
        # 概率分布形状已由 probability/* 三个全局指标覆盖。
        # for consume_k in range(1, LIST_SIZE + 1):
        #     class_index = consume_k - 1
        #     class_label = tf.cast(
        #         tf.equal(length_label, class_index),
        #         tf.float32,
        #     )
        #     class_label_mask = full_list_eval_mask * class_label
        #     class_prediction = tf.cast(
        #         tf.equal(length_prediction, class_index),
        #         tf.float32,
        #     )
        #     tf.summary.scalar(
        #         "list_value/length/by_k/k{}_label_rate".format(consume_k),
        #         masked_mean(class_label, full_list_eval_mask),
        #     )
        #     tf.summary.scalar(
        #         "list_value/length/by_k/k{}_predicted_probability_mean".format(
        #             consume_k
        #         ),
        #         masked_mean(
        #             length_probs[:, :, class_index],
        #             full_list_eval_mask,
        #         ),
        #     )
        #     tf.summary.scalar(
        #         "list_value/length/by_k/k{}_predicted_argmax_rate".format(
        #             consume_k
        #         ),
        #         masked_mean(class_prediction, full_list_eval_mask),
        #     )
        #     tf.summary.scalar(
        #         "list_value/length/by_k/k{}_recall".format(consume_k),
        #         masked_mean(class_prediction, class_label_mask),
        #     )

        # -------- TensorBoard 监控：A/D 统一时长诊断 --------
        # 分位置观察 A 的累计 Prefix WT 均值校准；只统计真实到达该位置的样本。
        for prefix_position in range(LIST_SIZE):
            position_prefix_mask = prefix_label_mask[:, :, prefix_position]
            # 第 k 个 Prefix 的预估累计时长/真实累计时长，接近 1 表示均值校准。
            tf.summary.scalar(
                "list_value/watch_time/a/by_position/pos{}_pred_label_ratio".format(
                    prefix_position + 1
                ),
                masked_pred_label_ratio(
                    prefix_watch_time[:, :, prefix_position],
                    prefix_watch_time_label[:, :, prefix_position],
                    position_prefix_mask,
                ),
            )

        # Oracle-K 只用于拆分“长度头误差”和“Prefix WT 数值头误差”。
        oracle_k_watch_time = tf.reduce_sum(
            prefix_watch_time * tf.one_hot(
                length_label,
                depth=LIST_SIZE,
                dtype=tf.float32,
            ),
            axis=-1,
        )
        fixed_k6_watch_time = prefix_watch_time[:, :, LIST_SIZE - 1]
        # oracle-K（在真实K处取 PrefixWT(K)）的 pred/label，用于拆分长度头与数值头的误差来源
        tf.summary.scalar(
            "list_value/watch_time/oracle_k/pred_label_ratio",
            masked_pred_label_ratio(
                oracle_k_watch_time,
                list_play_time_s_reduced,
                factual_training_list_mask,
            ),
        )
        # oracle-K 的 WMAPE（误差下界参照）
        tf.summary.scalar(
            "list_value/watch_time/oracle_k/wmape",
            masked_wmape(
                oracle_k_watch_time,
                list_play_time_s_reduced,
                factual_training_list_mask,
            ),
        )

        pwtd_abs_error = tf.abs(
            list_wt_from_context_pwtd_sum - list_play_time_s_reduced
        )
        long_watch_mask = full_list_eval_mask * tf.cast(
            tf.greater_equal(
                list_play_time_s_reduced,
                tf.constant(LONG_WATCH_TIME_THRESHOLD_S, tf.float32),
            ),
            tf.float32,
        )
        short_watch_mask = full_list_eval_mask * tf.cast(
            tf.less(
                list_play_time_s_reduced,
                tf.constant(LONG_WATCH_TIME_THRESHOLD_S, tf.float32),
            ),
            tf.float32,
        )
        # 完整事实 List 中真实总时长 >=120s 的请求占比。
        tf.summary.scalar(
            "list_value/watch_time/ad/long_watch_request_rate",
            tf.reduce_sum(long_watch_mask)
            / (tf.reduce_sum(full_list_eval_mask) + 1e-8),
        )
        # 完整事实 List 中真实总时长 <120s 的请求占比。
        tf.summary.scalar(
            "list_value/watch_time/ad/short_watch_request_rate",
            tf.reduce_sum(short_watch_mask)
            / (tf.reduce_sum(full_list_eval_mask) + 1e-8),
        )
        ad_watch_time_predictions = (
            ("a", expected_list_watch_time),
            ("d", expected_list_watch_time_d),
            ("pwtd", list_wt_from_context_pwtd_sum),
        )
        for variant_name, variant_prediction in ad_watch_time_predictions:
            variant_abs_error = tf.abs(
                variant_prediction - list_play_time_s_reduced
            )
            summary_prefix = "list_value/watch_time/ad/{}".format(variant_name)
            # 预估总时长/真实总时长，接近 1 表示整体均值校准。
            tf.summary.scalar(
                summary_prefix + "/pred_label_ratio",
                masked_pred_label_ratio(
                    variant_prediction,
                    list_play_time_s_reduced,
                    full_list_eval_mask,
                ),
            )
            # 预估与真实 List 总时长的平均绝对误差，越低越好。
            tf.summary.scalar(
                summary_prefix + "/mae",
                masked_mean(variant_abs_error, full_list_eval_mask),
            )
            # List 总时长的加权平均绝对百分比误差，越低越好。
            tf.summary.scalar(
                summary_prefix + "/wmape",
                masked_wmape(
                    variant_prediction,
                    list_play_time_s_reduced,
                    full_list_eval_mask,
                ),
            )
            # 预估总时长低于真实值的请求占比，用于诊断系统性低估。
            tf.summary.scalar(
                summary_prefix + "/underprediction_rate",
                masked_mean(
                    tf.cast(
                        tf.less(variant_prediction, list_play_time_s_reduced),
                        tf.float32,
                    ),
                    full_list_eval_mask,
                ),
            )
            # 绝对误差小于 content_pwtd-sum 的请求占比，>0.5 表示逐请求胜多败少。
            tf.summary.scalar(
                summary_prefix + "/abs_error_win_rate_vs_pwtd",
                masked_mean(
                    tf.cast(tf.less(variant_abs_error, pwtd_abs_error), tf.float32),
                    full_list_eval_mask,
                ),
            )
            # 仅在真实总时长 >=120s 样本上的预估/真实均值比。
            tf.summary.scalar(
                summary_prefix + "/long_watch_pred_label_ratio",
                masked_pred_label_ratio(
                    variant_prediction,
                    list_play_time_s_reduced,
                    long_watch_mask,
                ),
            )
            # 仅在真实总时长 >=120s 样本上的 WMAPE，越低越好。
            tf.summary.scalar(
                summary_prefix + "/long_watch_wmape",
                masked_wmape(
                    variant_prediction,
                    list_play_time_s_reduced,
                    long_watch_mask,
                ),
            )
            # 仅在真实总时长 <120s 样本上的预估/真实均值比。
            tf.summary.scalar(
                summary_prefix + "/short_watch_pred_label_ratio",
                masked_pred_label_ratio(
                    variant_prediction,
                    list_play_time_s_reduced,
                    short_watch_mask,
                ),
            )
            # 仅在真实总时长 <120s 样本上的 WMAPE，越低越好。
            tf.summary.scalar(
                summary_prefix + "/short_watch_wmape",
                masked_wmape(
                    variant_prediction,
                    list_play_time_s_reduced,
                    short_watch_mask,
                ),
            )

        # D 的单 item WTD 解码精度，使用全部事实曝光位置。
        # WTD ratio 解码成秒后的单 item WMAPE，直接反映 D 的秒数还原误差。
        tf.summary.scalar(
            "list_value/watch_time/ad/d/decoded_item_wmape",
            masked_wmape(
                decoded_watch_time_d,
                list_play_time_s,
                prefix_label_mask,
            ),
        )
        # D 预测 WTD ratio 与 duration-conditioned 真实 ratio 的 MAE。
        tf.summary.scalar(
            "list_value/watch_time/ad/d/wtd_ratio_mae",
            masked_mean(
                tf.abs(wtd_ratio_d - wtd_label_d),
                prefix_label_mask,
            ),
        )

        # 位置衰减 content_pwtd 总分/真实 List 总时长，用于补充 sum 口径对照。
        tf.summary.scalar(
            "list_value/watch_time/ad/pwtd_position_decay/pred_label_ratio",
            masked_pred_label_ratio(
                list_wt_from_context_pwtd_position_decay,
                list_play_time_s_reduced,
                full_list_eval_mask,
            ),
        )

        # -------- TensorBoard 监控：EVV item 分类与 List 聚合 --------
        # 事实曝光 item 中达到有效播放阈值的真实正样本率。
        tf.summary.scalar(
            "list_value/evv/item/positive_rate",
            masked_mean(list_evtr_label, prefix_label_mask),
        )
        # item EVV 预测概率总和/正样本总和，接近 1 表示整体校准。
        tf.summary.scalar(
            "list_value/evv/item/pred_label_ratio",
            masked_pred_label_ratio(
                evv_probs,
                list_evtr_label,
                prefix_label_mask,
            ),
        )
        # item EVV 概率的 Brier 分数 mean((p-y)^2)，越低越好。
        tf.summary.scalar(
            "list_value/evv/item/brier_score",
            masked_mean(
                tf.square(evv_probs - list_evtr_label),
                prefix_label_mask,
            ),
        )
        # 聚合后 expected List EVV/真实 List EVV，接近 1 表示总量校准。
        tf.summary.scalar(
            "list_value/evv/list/pred_label_ratio",
            masked_pred_label_ratio(
                expected_list_effective_vv,
                list_effective_vv_reduced,
                full_list_eval_mask,
            ),
        )
        # 聚合后 expected List EVV 与真实 List EVV 的 MAE，越低越好。
        tf.summary.scalar(
            "list_value/evv/list/mae",
            masked_mean(
                tf.abs(expected_list_effective_vv - list_effective_vv_reduced),
                full_list_eval_mask,
            ),
        )

        # -------- TensorBoard 监控：A/D/EVV 加权 loss 贡献 --------
        # 长度 hazard NLL 经权重后对总 loss 的实际贡献。
        tf.summary.scalar("list_value/loss_contribution/length", weighted_length_loss)
        # A 的 Prefix 累计时长 log1p Huber 加权贡献。
        tf.summary.scalar(
            "list_value/loss_contribution/a_prefix_log",
            weighted_prefix_wt_loss,
        )
        # A 的完整 List expected WT log1p Huber 加权贡献。
        tf.summary.scalar(
            "list_value/loss_contribution/a_list_log",
            weighted_list_wt_loss,
        )
        # D 的 duration-conditioned WTD ratio soft-label BCE 加权贡献。
        tf.summary.scalar(
            "list_value/loss_contribution/d_wtd",
            weighted_wtd_d_loss,
        )
        # EVV 单 item 二分 BCE 加权贡献。
        tf.summary.scalar(
            "list_value/loss_contribution/evv_item_bce",
            weighted_evv_item_loss,
        )
        # A 的 Prefix WT 非单调惩罚加权贡献，越接近 0 越好。
        tf.summary.scalar(
            "list_value/loss_contribution/a_monotonic",
            weighted_monotonic_loss,
        )
        # 所有已启用 List 任务加权 loss 的总和。
        tf.summary.scalar("list_value/loss_contribution/total", list_value_loss)

        # 聚合所有已观测 hazard 决策，检查模型能否区分“继续”和“停止”。
        # 这里只增加离线 AUC target，不参与 loss，也不新增 TensorBoard 曲线。
        flat_hazard_size = TRAIN_LIST_NUM * (LIST_SIZE - 1)
        targets.append((
            "continuation",
            tf.reshape(continue_probs, [batch_size, flat_hazard_size]),
            tf.reshape(continue_labels, [batch_size, flat_hazard_size]),
            tf.reshape(hazard_observed_mask, [batch_size, flat_hazard_size]),
            "auc",
        ))
        targets.append((
            "continuation_legacy_max_score",
            tf.reshape(continue_probs, [batch_size, flat_hazard_size]),
            tf.reshape(continue_labels, [batch_size, flat_hazard_size]),
            tf.reshape(
                legacy_hazard_observed_mask,
                [batch_size, flat_hazard_size],
            ),
            "auc",
        ))
        # 如需复验固定位置 AUC，可恢复下面的 position target。
        # for position_idx in range(LIST_SIZE - 1):
        #     targets.append((
        #         "continuation_pos{}".format(position_idx + 1),
        #         continue_probs[:, :, position_idx],
        #         continue_labels[:, :, position_idx],
        #         hazard_observed_mask[:, :, position_idx],
        #         "auc",
        #     ))
        flat_prefix_mask = tf.reshape(
            prefix_label_mask,
            [batch_size, TRAIN_LIST_NUM * LIST_SIZE],
        )
        targets.append((
            "prefix_watch_time_a",
            tf.reshape(prefix_watch_time, [batch_size, TRAIN_LIST_NUM * LIST_SIZE]),
            tf.reshape(prefix_watch_time_label, [batch_size, TRAIN_LIST_NUM * LIST_SIZE]),
            flat_prefix_mask,
            "linear_regression",
        ))
        targets.append((
            "evv_item",
            tf.reshape(evv_probs, [batch_size, TRAIN_LIST_NUM * LIST_SIZE]),
            tf.reshape(list_evtr_label, [batch_size, TRAIN_LIST_NUM * LIST_SIZE]),
            flat_prefix_mask,
            "auc",
        ))
        targets.append((
            "prefix_effective_vv",
            tf.reshape(prefix_effective_vv, [batch_size, TRAIN_LIST_NUM * LIST_SIZE]),
            tf.reshape(prefix_effective_vv_label, [batch_size, TRAIN_LIST_NUM * LIST_SIZE]),
            flat_prefix_mask,
            "linear_regression",
        ))
        targets.append((
            "wtd_ratio_d",
            wtd_ratio_d_flat,
            wtd_label_d_flat,
            flat_prefix_mask,
            "linear_regression",
        ))
        targets.append((
            "decoded_watch_time_d",
            decoded_watch_time_d_flat,
            list_play_time_s_flat,
            flat_prefix_mask,
            "linear_regression",
        ))
        targets.append((
            "expected_consume_length",
            expected_consume_length,
            tf.cast(length_label + 1, tf.float32),
            full_list_eval_mask,
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
            "expected_list_watch_time_a",
            expected_list_watch_time,
            list_play_time_s_reduced,
            full_list_eval_mask,
            "linear_regression",
        ))
        targets.append((
            "expected_list_watch_time_d",
            expected_list_watch_time_d,
            list_play_time_s_reduced,
            full_list_eval_mask,
            "linear_regression",
        ))
        targets.append((
            "expected_list_effective_vv",
            expected_list_effective_vv,
            list_effective_vv_reduced,
            full_list_eval_mask,
            "linear_regression",
        ))
        targets.append((
            "expected_list_watch_time_oracle_k",
            oracle_k_watch_time,
            list_play_time_s_reduced,
            factual_eval_list_mask,
            "linear_regression",
        ))
        targets.append((
            "expected_list_watch_time_fixed_k6",
            fixed_k6_watch_time,
            list_play_time_s_reduced,
            full_list_eval_mask,
            "linear_regression",
        ))
        targets.append((
            "expected_list_watch_time_a_legacy_max_score",
            expected_list_watch_time,
            list_play_time_s_reduced,
            legacy_max_score_eval_list_mask,
            "linear_regression",
        ))
        targets.append((
            "expected_list_watch_time_d_legacy_max_score",
            expected_list_watch_time_d,
            list_play_time_s_reduced,
            legacy_max_score_eval_list_mask,
            "linear_regression",
        ))
        targets.append((
            "expected_list_effective_vv_legacy_max_score",
            expected_list_effective_vv,
            list_effective_vv_reduced,
            legacy_max_score_eval_list_mask,
            "linear_regression",
        ))
        targets.append((
            "list_wt_from_context_pwtd_sum",
            list_wt_from_context_pwtd_sum,
            list_play_time_s_reduced,
            full_list_eval_mask,
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
            "list_wt_from_context_pwtd_position_decay",
            list_wt_from_context_pwtd_position_decay,
            list_play_time_s_reduced,
            full_list_eval_mask,
            "linear_regression",
        ))
        targets.append((
            "list_wt_from_context_pwtd_position_decay_legacy_max_score",
            list_wt_from_context_pwtd_position_decay,
            list_play_time_s_reduced,
            legacy_max_score_eval_list_mask,
            "linear_regression",
        ))
        # 分段时长回归评估：与上面未分段的 watch-time target 使用完全
        # 相同的连续时长 label 和 linear_regression 评估，仅用 mask 分开
        # <120s 与 >=120s 样本。这组 target 用于判断模型在两个时长段内的效果。
        segmented_watch_time_predictions = (
            ("expected_list_watch_time_a", expected_list_watch_time),
            ("expected_list_watch_time_d", expected_list_watch_time_d),
            ("list_wt_from_context_pwtd_sum", list_wt_from_context_pwtd_sum),
        )
        for target_prefix, watch_time_prediction in segmented_watch_time_predictions:
            targets.append((
                target_prefix + "_lt120",
                watch_time_prediction,
                list_play_time_s_reduced,
                short_watch_mask,
                "linear_regression",
            ))
            targets.append((
                target_prefix + "_ge120",
                watch_time_prediction,
                list_play_time_s_reduced,
                long_watch_mask,
                "linear_regression",
            ))

        # 跨 120s 阈值的二分区分度，仅作次要参考。它回答的是模型能否把
        # >=120s List 排在 <120s List 前面，不表示各时长段内的回归/排序效果。
        # 解读时对照 ad/long_watch_request_rate（正样本基线比例）。
        # 注意：auc 类 target 的评估器按概率分处理（截断到 [0,1]），直接传
        # 秒级原始分会被全部截断为 1.0，AUC 退化为 0.5（PredictedCTR=1.0）。
        # 因此用 score/(score+threshold) 单调压缩到 (0,1)，阈值处恰为 0.5；
        # AUC 只依赖排序，单调变换不改变 AUC 值。
        long_watch_binary_label = tf.cast(
            tf.greater_equal(
                list_play_time_s_reduced,
                tf.constant(LONG_WATCH_TIME_THRESHOLD_S, tf.float32),
            ),
            tf.float32,
        )
        long_watch_score_scale = tf.constant(LONG_WATCH_TIME_THRESHOLD_S, tf.float32)
        targets.append((
            "expected_list_watch_time_a_threshold_120_auc",
            expected_list_watch_time / (expected_list_watch_time + long_watch_score_scale),
            long_watch_binary_label,
            full_list_eval_mask,
            "auc",
        ))
        targets.append((
            "expected_list_watch_time_d_threshold_120_auc",
            expected_list_watch_time_d / (expected_list_watch_time_d + long_watch_score_scale),
            long_watch_binary_label,
            full_list_eval_mask,
            "auc",
        ))
        targets.append((
            "list_wt_from_context_pwtd_sum_threshold_120_auc",
            list_wt_from_context_pwtd_sum / (list_wt_from_context_pwtd_sum + long_watch_score_scale),
            long_watch_binary_label,
            full_list_eval_mask,
            "auc",
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
    # 外部 fullrank 分数不属于 standalone 模型预测头；这里仅按候选 List
    # 的下标 gather 后原样导出，供 backbone 沿用 v2 的线上融合公式。
    duration_s_infer = tf.clip_by_value(
        tf.reshape(
            config.get_extra_param(
                "duration_ms_infer",
                size=1,
                dtype=tf.float32,
            ),
            [1, -1],
        ) / 1000.0,
        0.0,
        36000.0,
    )
    context_info__pctr_infer = tf.reshape(
        config.get_extra_param(
            "context_info__pctr_infer",
            size=1,
            dtype=tf.float32,
        ),
        [1, -1],
    )
    context_info__pwtd_infer = tf.reshape(
        config.get_extra_param(
            "context_info__pwtd_infer",
            size=1,
            dtype=tf.float32,
        ),
        [1, -1],
    )
    context_info__pltr_infer = tf.reshape(
        config.get_extra_param(
            "context_info__pltr_infer",
            size=1,
            dtype=tf.float32,
        ),
        [1, -1],
    )
    context_info__pcmtr_infer = tf.reshape(
        config.get_extra_param(
            "context_info__pcmtr_infer",
            size=1,
            dtype=tf.float32,
        ),
        [1, -1],
    )
    context_info__pwtr_infer = tf.reshape(
        config.get_extra_param(
            "context_info__pwtr_infer",
            size=1,
            dtype=tf.float32,
        ),
        [1, -1],
    )
    context_info__pftr_infer = tf.reshape(
        config.get_extra_param(
            "context_info__pftr_infer",
            size=1,
            dtype=tf.float32,
        ),
        [1, -1],
    )
    context_info__plvtr_infer = tf.reshape(
        config.get_extra_param(
            "context_info__plvtr_infer",
            size=1,
            dtype=tf.float32,
        ),
        [1, -1],
    )
    context_info__psvtr_infer = tf.reshape(
        config.get_extra_param(
            "context_info__psvtr_infer",
            size=1,
            dtype=tf.float32,
        ),
        [1, -1],
    )

    # 候选下标在 infer 输入中从 0 开始；额外预留的第 0 行是 padding。
    rerank_list_item_idx_flat_list = config.get_extra_param("rerank_list_item_idx_flat_list_double", size=LIST_NUM * LIST_SIZE, default_value=-1.0, common=True) + 1.0
    rerank_list_item_idx_flat_list = tf.reshape(rerank_list_item_idx_flat_list, [-1, LIST_NUM, LIST_SIZE])
    rerank_list_item_idx_flat_list = tf.cast(rerank_list_item_idx_flat_list, tf.int32)

    zeros = tf.zeros(
        shape=[tf.shape(context_info__pctr_infer)[0], 1],
        dtype=tf.float32,
    )

    def gather_context_score(raw_score):
        return tf.gather(
            tf.concat([zeros, raw_score], axis=-1),
            rerank_list_item_idx_flat_list,
            axis=1,
            batch_dims=1,
        )

    context_pctr = tf.reshape(
        gather_context_score(context_info__pctr_infer),
        [-1, LIST_NUM * LIST_SIZE],
    )
    context_pwtd = tf.reshape(
        gather_context_score(context_info__pwtd_infer),
        [-1, LIST_NUM * LIST_SIZE],
    )
    context_pltr = tf.reshape(
        gather_context_score(context_info__pltr_infer),
        [-1, LIST_NUM * LIST_SIZE],
    )
    context_pcmtr = tf.reshape(
        gather_context_score(context_info__pcmtr_infer),
        [-1, LIST_NUM * LIST_SIZE],
    )
    context_pwtr = tf.reshape(
        gather_context_score(context_info__pwtr_infer),
        [-1, LIST_NUM * LIST_SIZE],
    )
    context_pftr = tf.reshape(
        gather_context_score(context_info__pftr_infer),
        [-1, LIST_NUM * LIST_SIZE],
    )
    context_plvtr = tf.reshape(
        gather_context_score(context_info__plvtr_infer),
        [-1, LIST_NUM * LIST_SIZE],
    )
    context_psvtr = tf.reshape(
        gather_context_score(context_info__psvtr_infer),
        [-1, LIST_NUM * LIST_SIZE],
    )
    list_duration_s_infer = gather_context_score(duration_s_infer)

    model_class._training = False
    list_value_output_dict = model_class.model(rerank_list_item_idx_flat_list)
    wtd_ratio_d_infer_flat = tf.reshape(
        list_value_output_dict["wtd_ratio_d"],
        [-1, LIST_NUM * LIST_SIZE],
    )
    list_duration_s_infer_flat = tf.reshape(
        list_duration_s_infer,
        [-1, LIST_NUM * LIST_SIZE],
    )
    decoded_watch_time_d_infer = tf.reshape(
        wtd_decode(
            wtd_ratio_d_infer_flat,
            list_duration_s_infer_flat,
            wtd_duration_buckets,
            wtd_play_time_buckets,
        ),
        [-1, LIST_NUM, LIST_SIZE],
    )
    prefix_watch_time_d_infer = tf.cumsum(
        decoded_watch_time_d_infer,
        axis=-1,
    )
    expected_list_watch_time_d_infer = tf.reduce_sum(
        list_value_output_dict["length_probs"] * prefix_watch_time_d_infer,
        axis=-1,
        name="expected_list_watch_time_d_infer",
    )
    targets = [
        # 保留旧 key 作为 A 的兼容别名，同时显式导出 A/D 便于在线 shadow。
        ("expected_list_watch_time", list_value_output_dict["expected_list_watch_time"]),
        ("expected_list_watch_time_a", list_value_output_dict["expected_list_watch_time"]),
        ("expected_list_watch_time_d", expected_list_watch_time_d_infer),
        ("expected_list_effective_vv", list_value_output_dict["expected_list_effective_vv"]),
        ("expected_consume_length", list_value_output_dict["expected_consume_length"]),
        ("context_pctr", context_pctr),
        ("context_pwtd", context_pwtd),
        ("context_pltr", context_pltr),
        ("context_pcmtr", context_pcmtr),
        ("context_pwtr", context_pwtr),
        ("context_pftr", context_pftr),
        ("context_plvtr", context_plvtr),
        ("context_psvtr", context_psvtr),
    ]

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
