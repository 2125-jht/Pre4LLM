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

# Standalone List Value 的全部优化目标；不存在 point-wise loss 或共享参数。
LENGTH_LOSS_WEIGHT = 1.0
PREFIX_WT_LOSS_WEIGHT = 1.0
PREFIX_EVV_LOSS_WEIGHT = 0.5
LIST_WT_LOSS_WEIGHT = 0.5
LIST_EVV_LOSS_WEIGHT = 0.2
PREFIX_MONOTONIC_LOSS_WEIGHT = 0.1

# 单个 List 互动目标融合四类显式互动。相对比例沿用当前 evaluator 中
# 20/200/200/50 的业务价值关系，并统一除以 20 控制标签尺度。
INTERACTION_LIKE_VALUE = 1.0
INTERACTION_COMMENT_VALUE = 10.0
INTERACTION_FOLLOW_VALUE = 10.0
INTERACTION_FORWARD_VALUE = 2.5
MAX_ITEM_INTERACTION_VALUE = (
    INTERACTION_LIKE_VALUE
    + INTERACTION_COMMENT_VALUE
    + INTERACTION_FOLLOW_VALUE
    + INTERACTION_FORWARD_VALUE
)
MAX_INTERACTION_VALUE = LIST_SIZE * MAX_ITEM_INTERACTION_VALUE

# 互动比 WT 稀疏，正样本 Prefix/List 适度加权；任务总权重与 EVV 对齐，
# 低于当前主目标 WT，避免新增目标主导 standalone 底座。
INTERACTION_POSITIVE_SAMPLE_WEIGHT = 3.0
PREFIX_INTERACTION_LOSS_WEIGHT = 0.5
LIST_INTERACTION_LOSS_WEIGHT = 0.2
LONG_WATCH_TIME_THRESHOLD_S = 120.0

# 合成换序 Pair 的 preference 实验暂时停用：不构造合成 List、不做额外
# 前向，也不把 preference loss 接入总训练目标。恢复时需重新按 factual
# matched Prefix 设计样本，不能沿用旧 Top1/原数组前 K 项口径。

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


def get_effective_vv_threshold(duration):
    """按视频时长桶返回 EVV 的播放时长阈值。"""
    boundaries = [0, 8.366, 10.3, 12.433, 15.066, 17.3, 20.431, 24.833, 29.333, 33.916, 39.033, 46.566, 54.7, 62.933, 76.366, 99.166, 178.266, 235, 360.433, 1108.266]
    evtr_thresholds = [4.529, 8.56, 10.154, 11.228, 12.009, 13.51, 13.406, 13.038, 14.57, 15.108, 16.205, 17.891, 18.748, 18.451, 19.012, 17.148, 15.472, 13.181, 10.074, 8.925, 9.554]
    boundaries_tensor = tf.constant(boundaries, dtype=tf.float32)
    boundaries_tensor = tf.tile(tf.expand_dims(boundaries_tensor, axis=0), [tf.shape(duration)[0], 1])
    bucket_idx = tf.searchsorted(sorted_sequence=boundaries_tensor, values=tf.cast(duration, tf.float32), side="left") # 左开右闭
    max_idx = tf.constant(len(evtr_thresholds) - 1, dtype=tf.int32)
    bucket_idx = tf.clip_by_value(bucket_idx, 0, max_idx)
    return tf.gather(tf.constant(evtr_thresholds), bucket_idx)


def get_effective_vv_label(duration, play_time, evtr_threshold=None):
    """由真实播放时长派生 List Effective VV 标签。

    EVV 使用分视频时长桶阈值：

        d_bucket = searchsorted(duration_boundaries, duration, side="left")
        y_evv = 1[play_time >= evtr_thresholds[d_bucket]]

    y_evv 仅用于构造 Prefix/List Effective VV 监督。
    """
    if evtr_threshold is None:
        evtr_threshold = get_effective_vv_threshold(duration)
    evtr_label = tf.cast(tf.greater_equal(play_time, evtr_threshold), dtype=tf.float32)
    return evtr_label

all_param_dict, _, _ = get_param_dict()
model_class = EvaluatorModel(
    all_param_dict,
    print_ops,
    list_size=LIST_SIZE,
    candidates_size=CANDIDATES_SIZE,
    list_num=LIST_NUM,
    max_interaction_value=MAX_INTERACTION_VALUE,
)

if is_training:
    label_value_dict = {}
    label_value_dict["show_label"] = tf.cast(tf.reshape(config.get_dense_fea("context_info__real_show_list", dim=CANDIDATES_SIZE, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
    label_value_dict["play_time_s"] = tf.cast(tf.reshape(config.get_dense_fea("context_info__playing_time_list", dim=CANDIDATES_SIZE, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32) / 1000.0
    label_value_dict["play_time_s"] = tf.clip_by_value(label_value_dict["play_time_s"], 0, 36000)
    label_value_dict["like_label"] = tf.clip_by_value(
        tf.cast(tf.reshape(config.get_dense_fea("context_info__like_list", dim=CANDIDATES_SIZE, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32),
        0.0,
        1.0,
    )
    label_value_dict["comment_label"] = tf.clip_by_value(
        tf.cast(tf.reshape(config.get_dense_fea("context_info__comment_list", dim=CANDIDATES_SIZE, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32),
        0.0,
        1.0,
    )
    label_value_dict["follow_label"] = tf.clip_by_value(
        tf.cast(tf.reshape(config.get_dense_fea("context_info__follow_list", dim=CANDIDATES_SIZE, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32),
        0.0,
        1.0,
    )
    label_value_dict["forward_label"] = tf.clip_by_value(
        tf.cast(tf.reshape(config.get_dense_fea("context_info__forward_list", dim=CANDIDATES_SIZE, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32),
        0.0,
        1.0,
    )
    label_value_dict["pwtd"] = tf.cast(
        tf.reshape(
            config.get_label("context_info__pwtd_list", dim=CANDIDATES_SIZE),
            [-1, CANDIDATES_SIZE],
        ),
        dtype=tf.float32,
    )
    label_value_dict["photo_info__duration_ms_list"] = tf.cast(tf.reshape(config.get_dense_fea("photo_info__duration_ms_list", dim=CANDIDATES_SIZE, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
    label_value_dict["photo_info__duration_ms_list"] = tf.clip_by_value(label_value_dict["photo_info__duration_ms_list"], 0, 36000 * 1000)
    batch_size = tf.shape(label_value_dict["show_label"])[0]
    zeros = tf.zeros([batch_size, 1], dtype=tf.float32)
    raw_show_label = label_value_dict["show_label"]
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

    # rank 可以有间隔，但有效曝光 rank 必须非负且不能重复。无效请求即使
    # 数值碰巧命中候选，也不能作为事实监督样本。
    factual_rank_non_negative = tf.reduce_all(
        tf.logical_or(
            tf.greater_equal(factual_real_show_indices, 0),
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
        factual_rank_non_negative,
        factual_rank_strictly_increasing,
    )
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
    max_score_list_index = tf.argmax(
        rerank_list_score_list,
        axis=-1,
        output_type=tf.int32,
    )
    observed_request_weight = tf.cast(has_observed_prefix, tf.float32)
    matched_request_weight = tf.cast(has_prefix_match, tf.float32)
    # 所有 factual List 目标只监督重建 Prefix 能够匹配的请求；多个匹配候选
    # 中选择旧分最高者。未匹配请求的 mask 为0，不再用 Top1 强行承接标签。
    listwise_match_mask = tf.one_hot(
        matched_list_index,
        depth=LIST_NUM,
        dtype=tf.float32,
    ) * tf.expand_dims(matched_request_weight, axis=-1)
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
    tf.summary.scalar(
        "list_value/match/factual_rank_valid_rate",
        tf.reduce_sum(
            tf.cast(factual_rank_valid, tf.float32)
            * observed_request_weight
        ) / (tf.reduce_sum(observed_request_weight) + 1e-8),
    )

    # 与 v1_alone 一致，仅保留各 K 下实际候选覆盖/训练保留率。
    for consume_k in range(1, LIST_SIZE + 1):
        k_request_weight = tf.cast(
            tf.equal(realshow_num, consume_k),
            tf.float32,
        ) * observed_request_weight
        k_request_count = tf.reduce_sum(k_request_weight)
        tf.summary.scalar(
            "list_value/match/by_k/k{}_any_prefix_match_rate".format(
                consume_k
            ),
            tf.reduce_sum(k_request_weight * matched_request_weight)
            / (k_request_count + 1e-8),
        )

    play_time_s = label_value_dict["play_time_s"]
    context_pwtd = label_value_dict["pwtd"]
    duration_s = label_value_dict["photo_info__duration_ms_list"] / 1000
    evtr_threshold = get_effective_vv_threshold(duration_s)
    evtr_label = get_effective_vv_label(
        duration_s,
        play_time_s,
        evtr_threshold=evtr_threshold,
    )
    interaction_value = (
        INTERACTION_LIKE_VALUE * label_value_dict["like_label"]
        + INTERACTION_COMMENT_VALUE * label_value_dict["comment_label"]
        + INTERACTION_FOLLOW_VALUE * label_value_dict["follow_label"]
        + INTERACTION_FORWARD_VALUE * label_value_dict["forward_label"]
    )

    # 合成换序 Pair 的 preference 训练暂时停用：不构造 Y_w/Y_l，也不做
    # 额外两条 List 的前向。后续若恢复，必须基于重建后匹配到的事实 Prefix
    # 构造 Pair，不能再使用旧分 Top1 或原数组前 K 项作为事实曝光顺序。
    model_class._training = True
    list_value_output_dict = model_class.model(
        list_index=rerank_list_item_idx_flat_list,
    )
    print(f"====> train standalone list model, gen...")

    list_play_time_s = tf.gather(
        tf.concat([zeros, play_time_s], axis=-1),
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
    list_show_label = tf.gather(
        tf.concat([zeros, label_value_dict["show_label"]], axis=-1),
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
    list_interaction_value = tf.gather(
        tf.concat([zeros, interaction_value], axis=-1),
        rerank_list_item_idx_flat_list,
        axis=1,
        batch_dims=1,
    )

    # -------- List Value 标签 --------
    # 对每个真实曝光位置做累加，构造 Prefix[1:k] 的累计 WT/EVV/互动价值。
    # prefix_label_mask 同时限制“重建 Prefix 匹配的 factual List”和
    # “位置已经真实曝光”两个条件，因而不会拿 K 之后的反事实 item 做监督。
    prefix_watch_time_label = tf.cumsum(list_play_time_s, axis=-1)
    prefix_effective_vv_label = tf.cumsum(list_evtr_label, axis=-1)
    prefix_interaction_label = tf.cumsum(list_interaction_value, axis=-1)
    prefix_label_mask = tf.cast(observed_position_mask, tf.float32) \
        * tf.expand_dims(listwise_match_mask, axis=-1)

    list_play_time_s_reduced = tf.reduce_sum(list_play_time_s, axis=-1)  # (?, list_num)
    list_effective_vv_reduced = tf.reduce_sum(list_evtr_label, axis=-1)  # (?, list_num)
    list_interaction_reduced = tf.reduce_sum(list_interaction_value, axis=-1)  # (?, list_num)

    with tf.control_dependencies(print_ops):
        targets = []
        # -------- List Value 损失 --------
        continue_logits = list_value_output_dict["continue_logits"]
        continue_probs = list_value_output_dict["continue_probs"]
        length_probs = list_value_output_dict["length_probs"]
        prefix_watch_time = list_value_output_dict["prefix_watch_time"]
        prefix_watch_time_log = list_value_output_dict["prefix_watch_time_log"]
        prefix_effective_vv = list_value_output_dict["prefix_effective_vv"]
        prefix_interaction = list_value_output_dict["prefix_interaction"]
        prefix_interaction_log = list_value_output_dict["prefix_interaction_log"]
        # 四个核心 List 输出：
        # - expected_consume_length：由 continue 概率得到的预期消费 item 数；
        # - expected_list_watch_time：sum_k P(K=k) * PrefixWT(k)，单位为秒；
        # - expected_list_effective_vv：sum_k P(K=k) * PrefixEVV(k)，单位为
        #   有效播放次数。EVV 表示真实播放时长达到相应视频时长阈值。
        # - expected_list_interaction：sum_k P(K=k) * PrefixInteraction(k)，
        #   是 like/comment/follow/forward 融合后的综合互动价值。
        expected_list_watch_time = list_value_output_dict["expected_list_watch_time"]
        expected_list_effective_vv = list_value_output_dict["expected_list_effective_vv"]
        expected_list_interaction = list_value_output_dict["expected_list_interaction"]
        expected_consume_length = list_value_output_dict["expected_consume_length"]

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
        hazard_position_mask = tf.sequence_mask(
            tf.minimum(realshow_num, LIST_SIZE - 1),
            maxlen=LIST_SIZE - 1,
            dtype=tf.float32,
        )
        hazard_position_mask = tf.expand_dims(
            hazard_position_mask,
            axis=1,
        )
        hazard_observed_mask = hazard_position_mask \
            * tf.expand_dims(listwise_match_mask, axis=-1)
        legacy_hazard_observed_mask = hazard_position_mask \
            * tf.expand_dims(legacy_max_score_eval_list_mask, axis=-1)

        # -------- continuation 的请求内相对分诊断 --------
        # 原始 continue logit 同时包含用户整体活跃度、位置难度和候选 List 内容信号。
        # 为了单独观察候选 List 之间的内容差异，这里在每个请求、每个位置上，
        # 减去“其他有效候选 List”的平均 logit（leave-one-out peer mean）。
        # 该相对分只用于离线 AUC 诊断，不参与训练 loss，也不改变线上打分。
        candidate_list_mask = tf.cast(
            tf.reduce_any(
                tf.greater(rerank_list_item_idx_flat_list, 0),
                axis=-1,
            ),
            tf.float32,
        )
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
        # 再按参与训练的 matched factual List 数量求均值，使每个请求保持
        # 相同样本权重。
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
        prefix_interaction_label_log = tf.math.log1p(prefix_interaction_label)
        prefix_interaction_sample_weight = prefix_label_mask * tf.where(
            tf.greater(prefix_interaction_label, 0.0),
            tf.fill(
                tf.shape(prefix_interaction_label),
                tf.constant(INTERACTION_POSITIVE_SAMPLE_WEIGHT, tf.float32),
            ),
            tf.ones_like(prefix_interaction_label),
        )
        prefix_interaction_loss = tf.losses.huber_loss(
            labels=prefix_interaction_label_log,
            predictions=prefix_interaction_log,
            weights=prefix_interaction_sample_weight,
            delta=0.5,
        )

        # List 总价值损失约束经过 P(K) 加权后的最终期望值，
        # 使训练目标与后续按 expected value 比较 List 的使用方式一致。
        list_wt_loss = tf.losses.huber_loss(
            labels=tf.math.log1p(list_play_time_s_reduced),
            predictions=tf.math.log1p(expected_list_watch_time),
            weights=listwise_match_mask,
            delta=0.5,
        )
        list_evv_loss = tf.losses.huber_loss(
            labels=list_effective_vv_reduced,
            predictions=expected_list_effective_vv,
            weights=listwise_match_mask,
            delta=0.5,
        )
        list_interaction_sample_weight = listwise_match_mask * tf.where(
            tf.greater(list_interaction_reduced, 0.0),
            tf.fill(
                tf.shape(list_interaction_reduced),
                tf.constant(INTERACTION_POSITIVE_SAMPLE_WEIGHT, tf.float32),
            ),
            tf.ones_like(list_interaction_reduced),
        )
        list_interaction_loss = tf.losses.huber_loss(
            labels=tf.math.log1p(list_interaction_reduced),
            predictions=tf.math.log1p(expected_list_interaction),
            weights=list_interaction_sample_weight,
            delta=0.5,
        )

        # 合成换序 Pair 的 preference loss 暂停；标准 factual 监督不依赖它。

        # 累计价值随前缀变长不应下降；只在相邻位置均有真实监督时施加软约束。
        prefix_pair_mask = prefix_label_mask[:, :, 1:]
        wt_monotonic_error = tf.nn.relu(
            prefix_watch_time[:, :, :-1] - prefix_watch_time[:, :, 1:]
        )
        evv_monotonic_error = tf.nn.relu(
            prefix_effective_vv[:, :, :-1] - prefix_effective_vv[:, :, 1:]
        )
        interaction_monotonic_error = tf.nn.relu(
            prefix_interaction_log[:, :, :-1] - prefix_interaction_log[:, :, 1:]
        )
        monotonic_loss = tf.reduce_sum(
            (
                wt_monotonic_error
                + evv_monotonic_error
                + interaction_monotonic_error
            ) * prefix_pair_mask
        ) / (tf.reduce_sum(prefix_pair_mask) + 1e-8)

        weighted_length_loss = LENGTH_LOSS_WEIGHT * length_loss
        weighted_prefix_wt_loss = PREFIX_WT_LOSS_WEIGHT * prefix_wt_loss
        weighted_prefix_evv_loss = PREFIX_EVV_LOSS_WEIGHT * prefix_evv_loss
        weighted_prefix_interaction_loss = (
            PREFIX_INTERACTION_LOSS_WEIGHT * prefix_interaction_loss
        )
        weighted_list_wt_loss = LIST_WT_LOSS_WEIGHT * list_wt_loss
        weighted_list_evv_loss = LIST_EVV_LOSS_WEIGHT * list_evv_loss
        weighted_list_interaction_loss = (
            LIST_INTERACTION_LOSS_WEIGHT * list_interaction_loss
        )
        weighted_monotonic_loss = PREFIX_MONOTONIC_LOSS_WEIGHT * monotonic_loss
        list_value_loss = weighted_length_loss \
            + weighted_prefix_wt_loss \
            + weighted_prefix_evv_loss \
            + weighted_prefix_interaction_loss \
            + weighted_list_wt_loss \
            + weighted_list_evv_loss \
            + weighted_list_interaction_loss \
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
        # 所有校准指标严格使用 matched factual hazard mask，不混入未匹配请求、
        # 未观测位置或 padding。分位置统计可避免全局结果被位置先验掩盖。
        for hazard_position in range(LIST_SIZE - 1):
            position_hazard_mask = hazard_observed_mask[:, :, hazard_position]
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

        tf.summary.scalar(
            "list_value/continuation/global/brier_score",
            masked_mean(
                tf.square(continue_probs - continue_labels),
                hazard_observed_mask,
            ),
        )

        # 同一请求、同一位置下，观察模型对不同候选 List 的继续概率是否有响应。
        # 接近 0 只表示候选间分数趋同，不直接代表好坏；需结合 continuation
        # AUC 和校准指标判断模型是否主要依赖用户/位置公共先验。
        candidate_continue_prob_mean = tf.reduce_sum(
            continue_probs * candidate_list_mask_3d,
            axis=1,
            keepdims=True,
        ) / tf.maximum(valid_list_count, 1.0)
        candidate_continue_prob_variance = tf.reduce_sum(
            tf.square(continue_probs - candidate_continue_prob_mean)
            * candidate_list_mask_3d,
            axis=1,
            keepdims=True,
        ) / tf.maximum(valid_list_count, 1.0)
        candidate_continue_prob_std = tf.squeeze(
            tf.sqrt(candidate_continue_prob_variance + 1e-12),
            axis=1,
        )
        observed_hazard_request_mask = tf.reduce_sum(
            hazard_observed_mask,
            axis=1,
        )
        tf.summary.scalar(
            "list_value/continuation/global/candidate_std_mean",
            masked_mean(
                candidate_continue_prob_std,
                observed_hazard_request_mask,
            ),
        )

        # -------- TensorBoard 监控：消费长度 --------
        # 与 v1_alone 统一为全类别监控，避免只观察 K6 时漏掉其他长度类别
        # 的分布偏移或塌缩。统计口径与 factual length loss 完全一致。
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
            expected_list_watch_time,
            listwise_match_mask,
        )
        expected_wt_label_mean = masked_mean(
            list_play_time_s_reduced,
            listwise_match_mask,
        )
        expected_wt_mae = masked_mean(
            tf.abs(expected_list_watch_time - list_play_time_s_reduced),
            listwise_match_mask,
        )
        expected_evv_pred_mean = masked_mean(
            expected_list_effective_vv,
            listwise_match_mask,
        )
        expected_evv_label_mean = masked_mean(
            list_effective_vv_reduced,
            listwise_match_mask,
        )
        expected_evv_mae = masked_mean(
            tf.abs(expected_list_effective_vv - list_effective_vv_reduced),
            listwise_match_mask,
        )
        expected_interaction_pred_mean = masked_mean(
            expected_list_interaction,
            listwise_match_mask,
        )
        expected_interaction_label_mean = masked_mean(
            list_interaction_reduced,
            listwise_match_mask,
        )
        expected_interaction_mae = masked_mean(
            tf.abs(expected_list_interaction - list_interaction_reduced),
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
        tf.summary.scalar(
            "list_value/calibration/expected_interaction_pred_label_ratio",
            expected_interaction_pred_mean / (expected_interaction_label_mean + 1e-8),
        )
        tf.summary.scalar(
            "list_value/calibration/expected_interaction_mae",
            expected_interaction_mae,
        )
        tf.summary.scalar(
            "list_value/calibration/interaction_positive_rate",
            masked_mean(
                tf.cast(tf.greater(list_interaction_reduced, 0.0), tf.float32),
                listwise_match_mask,
            ),
        )

        # -------- TensorBoard 监控：List 时长诊断 --------
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
        tf.summary.scalar(
            "list_value/watch_time/global/wmape",
            masked_wmape(
                expected_list_watch_time,
                list_play_time_s_reduced,
                listwise_match_mask,
            ),
        )
        tf.summary.scalar(
            "list_value/watch_time/global/underprediction_rate",
            masked_mean(
                tf.cast(
                    tf.less(
                        expected_list_watch_time,
                        list_play_time_s_reduced,
                    ),
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
                    expected_list_watch_time,
                    list_play_time_s_reduced,
                    k_watch_time_mask,
                ),
            )
            tf.summary.scalar(
                "list_value/watch_time/by_k/k{}_wmape".format(consume_k),
                masked_wmape(
                    expected_list_watch_time,
                    list_play_time_s_reduced,
                    k_watch_time_mask,
                ),
            )

        model_abs_error = tf.abs(
            expected_list_watch_time - list_play_time_s_reduced
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
                expected_list_watch_time,
                list_play_time_s_reduced,
                long_watch_mask,
            ),
        )
        tf.summary.scalar(
            "list_value/watch_time/long_watch/wmape",
            masked_wmape(
                expected_list_watch_time,
                list_play_time_s_reduced,
                long_watch_mask,
            ),
        )

        # -------- TensorBoard 监控：加权损失贡献 --------
        # 记录乘过超参权重后的真实贡献，并合并同类 loss，避免曲线过多。
        tf.summary.scalar("list_value/loss_contribution/length", weighted_length_loss)
        tf.summary.scalar(
            "list_value/loss_contribution/prefix_value",
            weighted_prefix_wt_loss
            + weighted_prefix_evv_loss
            + weighted_prefix_interaction_loss,
        )
        tf.summary.scalar(
            "list_value/loss_contribution/list_total",
            weighted_list_wt_loss
            + weighted_list_evv_loss
            + weighted_list_interaction_loss,
        )
        tf.summary.scalar(
            "list_value/loss_contribution/interaction",
            weighted_prefix_interaction_loss + weighted_list_interaction_loss,
        )
        tf.summary.scalar("list_value/loss_contribution/monotonic", weighted_monotonic_loss)
        tf.summary.scalar("list_value/loss_contribution/total", list_value_loss)

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
        # 阶段性验证已经完成，暂时关闭以下 stdout target；需要复验时再恢复。
        # 1. continuation_pos*：固定位置，排除聚合 AUC 中的位置先验；
        # 2. continuation_relative_pos*：再排除同请求候选共享的公共先验。
        #
        # 旧实验是在 legacy Top1 mask 上得到的，切换 factual matched mask 后
        # 不可直接横向比较；如需复验应同时输出 matched 与 legacy 两套口径。
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
            "prefix_interaction",
            tf.reshape(prefix_interaction, [batch_size, LIST_NUM * LIST_SIZE]),
            tf.reshape(prefix_interaction_label, [batch_size, LIST_NUM * LIST_SIZE]),
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
            expected_list_watch_time,
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
            expected_list_watch_time,
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
            "list_wt_from_context_pwtd_position_decay",
            list_wt_from_context_pwtd_position_decay,
            list_play_time_s_reduced,
            matched_eval_list_mask,
            "linear_regression",
        ))
        targets.append((
            "list_wt_from_context_pwtd_position_decay_legacy_max_score",
            list_wt_from_context_pwtd_position_decay,
            list_play_time_s_reduced,
            legacy_max_score_eval_list_mask,
            "linear_regression",
        ))
        targets.append((
            "expected_list_effective_vv",
            expected_list_effective_vv,
            list_effective_vv_reduced,
            matched_eval_list_mask,
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
            "expected_list_interaction",
            expected_list_interaction,
            list_interaction_reduced,
            matched_eval_list_mask,
            "linear_regression",
        ))
        targets.append((
            "expected_list_interaction_legacy_max_score",
            expected_list_interaction,
            list_interaction_reduced,
            legacy_max_score_eval_list_mask,
            "linear_regression",
        ))
        targets.append((
            "expected_list_interaction_occurrence",
            expected_list_interaction / (1.0 + expected_list_interaction),
            tf.cast(tf.greater(list_interaction_reduced, 0.0), tf.float32),
            matched_eval_list_mask,
            "auc",
        ))
        targets.append((
            "expected_list_interaction_occurrence_legacy_max_score",
            expected_list_interaction / (1.0 + expected_list_interaction),
            tf.cast(tf.greater(list_interaction_reduced, 0.0), tf.float32),
            legacy_max_score_eval_list_mask,
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

    model_class._training = False
    list_value_output_dict = model_class.model(rerank_list_item_idx_flat_list)
    targets = [
        ("expected_list_watch_time", list_value_output_dict["expected_list_watch_time"]),
        ("expected_list_effective_vv", list_value_output_dict["expected_list_effective_vv"]),
        ("expected_list_interaction", list_value_output_dict["expected_list_interaction"]),
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
