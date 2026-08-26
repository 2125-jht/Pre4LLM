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

# 训练阶段开关，直接修改此处：
# - "pretrain": 只优化事实拟合目标；
# - "preference_posttrain": 只优化构造正负 List 的 Preference 目标。
TRAINING_STAGE = "preference_posttrain"
if TRAINING_STAGE not in ("pretrain", "preference_posttrain"):
    raise ValueError(
        "TRAINING_STAGE must be 'pretrain' or 'preference_posttrain'"
    )
IS_PREFERENCE_POSTTRAIN = TRAINING_STAGE == "preference_posttrain"
print("list value training stage: ", TRAINING_STAGE)
if is_training and IS_PREFERENCE_POSTTRAIN:
    print(
        "WARNING: preference_posttrain must load a converged pretrain checkpoint; "
        "do not cold-start this stage."
    )

# Standalone List Value 的全部优化目标；不存在 point-wise loss 或共享参数。
LENGTH_LOSS_WEIGHT = 1.0
PREFIX_WT_LOSS_WEIGHT = 1.0
PREFIX_ENGAGEMENT_LOSS_WEIGHT = 0.5
LIST_WT_LOSS_WEIGHT = 0.5
LIST_ENGAGEMENT_LOSS_WEIGHT = 0.2
PREFIX_MONOTONIC_LOSS_WEIGHT = 0.1
ENGAGEMENT_INTERACTION_POSITIVE_SAMPLE_WEIGHT = 3.0

# Engagement 中的显式互动部分融合四类行为。相对比例沿用原单点 LTR 中
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
MAX_ITEM_ENGAGEMENT_VALUE = 1.0 + MAX_ITEM_INTERACTION_VALUE
MAX_ENGAGEMENT_VALUE = LIST_SIZE * MAX_ITEM_ENGAGEMENT_VALUE

# GReF-inspired 用户反馈顺序偏好同步使用两个业务目标：绝对 WT 占 70%，
# Engagement（EVV + 显式互动价值）占 30%。
PREFERENCE_WT_WEIGHT = 0.70
PREFERENCE_ENGAGEMENT_WEIGHT = 0.30
PREFERENCE_UTILITY_WEIGHT_SUM = (
    PREFERENCE_WT_WEIGHT
    + PREFERENCE_ENGAGEMENT_WEIGHT
)
PREFERENCE_ITEM_WT_CAP_SECONDS = 400.0
PREFERENCE_LIST_WT_CAP_SECONDS = LIST_SIZE * PREFERENCE_ITEM_WT_CAP_SECONDS

# S_i = alpha / position_i + gamma * feedback_utility_i。连续反馈通常达不到
# 二值 click 的满分 1，因此 gamma=2；仍需明确反馈优势才能越过位置先验。
PREFERENCE_POSITION_PRIOR_WEIGHT = 1.0
PREFERENCE_FEEDBACK_WEIGHT = 2.0

# evaluator 直接比较 List Value，不使用生成概率或 reference policy。
PREFERENCE_TEMPERATURE = 0.1
# 反向反馈 List 是刻意构造的较容易负样本，只作为顺序敏感性辅助约束；
# 主约束仍是更贴近线上决策边界的 Y_w > Y_l。
PREFERENCE_REVERSE_LOSS_WEIGHT = 0.3
# 外层 Preference 权重维持 0.4；新增辅助 pair 后的实际总占比由
# loss_contribution/preference_ratio 监控。
PREFERENCE_LOSS_WEIGHT = 0.4

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

    y_evv 用作 Prefix/List Engagement 标签中的稠密基础价值。
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
    max_engagement_value=MAX_ENGAGEMENT_VALUE,
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
    # fountain_fulllink_rerank_index 是 item 在统一候选池中的 1-based 坐标，
    # 并不表达真实曝光顺序。真实顺序由 real_show_index 给出：先筛出
    # real_show item，再按 real_show_index 升序排列，最后取对应的候选池坐标。
    raw_fountain_rerank_index = tf.cast(
        tf.reshape(
            config.get_extra_param(
                "fountain_fulllink_rerank_index_list",
                size=CANDIDATES_SIZE,
            ),
            [-1, CANDIDATES_SIZE],
        ),
        tf.int32,
    )
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
    raw_show_mask = tf.greater(raw_show_label, 0.0)
    raw_fountain_index_valid = tf.logical_and(
        tf.greater(raw_fountain_rerank_index, 0),
        tf.less_equal(raw_fountain_rerank_index, CANDIDATES_SIZE),
    )
    # real_show_index 在不同样本链路可能从 0 或 1 起计；show_label 已经负责
    # 区分曝光与未曝光，因此这里接受非负位次。
    raw_real_show_index_valid = tf.greater_equal(raw_real_show_index, 0)
    raw_exposure_mapping_valid = tf.logical_and(
        raw_show_mask,
        tf.logical_and(
            raw_real_show_index_valid,
            raw_fountain_index_valid,
        ),
    )
    exposure_sort_key = tf.where(
        raw_exposure_mapping_valid,
        raw_real_show_index,
        tf.fill(
            tf.shape(raw_real_show_index),
            tf.constant(2147483647, dtype=tf.int32),
        ),
    )
    exposure_order = tf.argsort(exposure_sort_key, axis=-1)
    ordered_exposure_rerank_index = tf.gather(
        raw_fountain_rerank_index,
        exposure_order,
        axis=1,
        batch_dims=1,
    )[:, :LIST_SIZE]
    ordered_real_show_index = tf.gather(
        raw_real_show_index,
        exposure_order,
        axis=1,
        batch_dims=1,
    )[:, :LIST_SIZE]
    ordered_exposure_mapping_valid = tf.gather(
        raw_exposure_mapping_valid,
        exposure_order,
        axis=1,
        batch_dims=1,
    )[:, :LIST_SIZE]
    factual_prefix_position_mask = tf.sequence_mask(
        realshow_num,
        maxlen=LIST_SIZE,
        dtype=tf.bool,
    )
    ordered_exposure_rerank_index = tf.where(
        factual_prefix_position_mask,
        ordered_exposure_rerank_index,
        tf.zeros_like(ordered_exposure_rerank_index),
    )
    ordered_real_show_index = tf.where(
        factual_prefix_position_mask,
        ordered_real_show_index,
        tf.zeros_like(ordered_real_show_index),
    )
    real_show_rerank_indices = tf.expand_dims(
        ordered_exposure_rerank_index,
        axis=1,
    )  # (?, 1, LIST_SIZE)
    label_value_dict["fountain_fulllink_rerank_index_list"] = tf.cast(
        raw_fountain_rerank_index,
        tf.float32,
    )
    index_indices = tf.argsort(
        tf.reshape(
            label_value_dict['fountain_fulllink_rerank_index_list'],
            [-1, CANDIDATES_SIZE],
        ),
        axis=-1,
    )
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

    # -------- 候选池对真实曝光 Prefix 的覆盖诊断 --------
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
        rerank_list_score_matrix,
        tf.fill([batch_size, LIST_NUM], tf.constant(-1e9, dtype=tf.float32)),
    )
    matched_list_index = tf.argmax(masked_match_score, axis=-1, output_type=tf.int32)
    max_score_list_index = tf.argmax(
        rerank_list_score_list,
        axis=-1,
        output_type=tf.int32,
    )
    # 直接验证上游约束“旧分最高 List 就是实际曝光 List”。原有
    # prefix_match_rate 只表示任意候选中存在匹配项，不能回答最高分项是否匹配。
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
    # 与 fountain_rerank_eval_listwise_v2_new 保持相同的主训练/评估口径：
    # 每个请求选择旧分最高的完整 6-item List。真实曝光 Prefix 只提供 K、
    # Prefix 反馈以及匹配诊断，不再以 Prefix+padding 代替完整 List 输入。
    false_mask = tf.fill([batch_size, LIST_NUM], False)
    batch_indices = tf.tile(
        tf.expand_dims(tf.range(batch_size), axis=1),
        [1, LIST_NUM],
    )
    mask_indices = tf.stack(
        [tf.range(batch_size), max_score_list_index],
        axis=1,
    )
    rerank_list_mask = tf.tensor_scatter_nd_update(
        false_mask,
        mask_indices,
        tf.fill([batch_size], True),
    )
    rerank_list_mask = tf.cast(rerank_list_mask, tf.float32)
    factual_prefix_index_valid = tf.reduce_all(
        tf.logical_or(
            tf.logical_and(
                tf.expand_dims(ordered_exposure_mapping_valid, axis=1),
                tf.logical_and(
                    tf.greater(real_show_rerank_indices, 0),
                    tf.less_equal(real_show_rerank_indices, CANDIDATES_SIZE),
                ),
            ),
            tf.logical_not(observed_position_mask),
        ),
        axis=-1,
    )
    adjacent_observed_position_mask = tf.sequence_mask(
        tf.maximum(realshow_num - 1, 0),
        maxlen=LIST_SIZE - 1,
        dtype=tf.bool,
    )
    factual_exposure_order_unique = tf.reduce_all(
        tf.logical_or(
            tf.greater(
                ordered_real_show_index[:, 1:],
                ordered_real_show_index[:, :-1],
            ),
            tf.logical_not(adjacent_observed_position_mask),
        ),
        axis=-1,
    )
    factual_candidate_index_unique = tf.reduce_all(
        tf.less_equal(
            tf.reduce_sum(
                tf.one_hot(
                    ordered_exposure_rerank_index,
                    depth=CANDIDATES_SIZE + 1,
                    dtype=tf.float32,
                ) * tf.expand_dims(
                    tf.cast(factual_prefix_position_mask, tf.float32),
                    axis=-1,
                ),
                axis=1,
            ),
            1.0,
        ),
        axis=-1,
    )
    factual_observed_prefix_valid = tf.logical_and(
        factual_prefix_index_valid,
        tf.expand_dims(
            tf.logical_and(
                factual_exposure_order_unique,
                factual_candidate_index_unique,
            ),
            axis=-1,
        ),
    )
    listwise_match_mask = rerank_list_mask

    # 原先的等价写法保留，当前不使用。
    # listwise_match_mask = tf.one_hot(
    #     max_score_list_index,
    #     depth=LIST_NUM,
    #     dtype=tf.float32,
    # )

    full_observed_mask = tf.equal(realshow_num, LIST_SIZE)
    full_observed_count = tf.reduce_sum(tf.cast(full_observed_mask, tf.float32))
    full_list_match_count = tf.reduce_sum(
        tf.cast(
            tf.logical_and(full_observed_mask, has_prefix_match),
            tf.float32,
        )
    )
    full_list_match_rate = full_list_match_count / (full_observed_count + 1e-8)

    # 两项均为数据诊断，不再决定实际训练样本：prefix_match_rate 观察候选中
    # 是否存在事实 Prefix；full_list_match_rate 观察 K=LIST_SIZE 时的严格匹配。
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

    # 按真实消费长度 K 拆开观察：
    # - request_rate_before/after_match 对比当前训练分布与旧 Prefix 过滤会保留
    #   的分布；after_match 仅是反事实诊断，不再代表实际训练样本；
    # - any_prefix_match_rate 判断候选中能否找到事实 Prefix；
    # - max_score_prefix_match_rate 直接验证最高分 List 是否就是曝光 List。
    # K=LIST_SIZE 包含 realshow_num_raw >= LIST_SIZE 的截断样本，与长度标签一致。
    for consume_k in range(1, LIST_SIZE + 1):
        k_request_weight = tf.cast(
            tf.equal(realshow_num, consume_k),
            tf.float32,
        ) * observed_request_weight
        k_request_count = tf.reduce_sum(k_request_weight)
        tf.summary.scalar(
            "list_value/match/by_k/k{}_request_rate_before_match".format(
                consume_k
            ),
            k_request_count / (tf.reduce_sum(observed_request_weight) + 1e-8),
        )
        tf.summary.scalar(
            "list_value/match/by_k/k{}_request_rate_after_match".format(
                consume_k
            ),
            tf.reduce_sum(k_request_weight * matched_request_weight)
            / (tf.reduce_sum(matched_request_weight) + 1e-8),
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
    engagement_value = evtr_label + interaction_value

    # -------- GReF-inspired 用户反馈正负 List --------
    # Y_l 使用旧分 Top1 的完整 6-item List；真实曝光 Prefix 只提供前 K 位反馈。
    # item 反馈效用与两个业务输出对齐：绝对 log-WT 占 70%，由 EVV 和显式
    # 互动组成的 Engagement 占 30%。
    logged_list_index = tf.gather_nd(
        rerank_list_item_idx_flat_list,
        tf.stack([tf.range(batch_size), max_score_list_index], axis=1),
    )  # (?, LIST_SIZE)
    observed_feedback_mask = tf.sequence_mask(
        realshow_num,
        maxlen=LIST_SIZE,
        dtype=tf.float32,
    )

    def gather_logged_item_value(item_value):
        return tf.gather(
            tf.concat([zeros, item_value], axis=-1),
            logged_list_index,
            axis=1,
            batch_dims=1,
        )

    logged_play_time_s = gather_logged_item_value(play_time_s)
    logged_engagement_value = gather_logged_item_value(engagement_value)
    preference_wt_feedback = tf.clip_by_value(
        tf.math.log1p(logged_play_time_s)
        / tf.math.log1p(
            tf.constant(PREFERENCE_ITEM_WT_CAP_SECONDS, dtype=tf.float32)
        ),
        0.0,
        1.0,
    )
    preference_engagement_feedback = tf.clip_by_value(
        tf.math.log1p(logged_engagement_value)
        / tf.math.log1p(
            tf.constant(MAX_ITEM_ENGAGEMENT_VALUE, dtype=tf.float32)
        ),
        0.0,
        1.0,
    )
    preference_item_feedback_utility = (
        PREFERENCE_WT_WEIGHT * preference_wt_feedback
        + PREFERENCE_ENGAGEMENT_WEIGHT * preference_engagement_feedback
    ) / PREFERENCE_UTILITY_WEIGHT_SUM * observed_feedback_mask
    observed_feedback_count = tf.maximum(
        tf.cast(realshow_num_raw, tf.float32),
        1.0,
    )
    preference_wt_feedback_contribution = (
        PREFERENCE_WT_WEIGHT
        * tf.reduce_sum(
            preference_wt_feedback * observed_feedback_mask,
            axis=-1,
        )
        / (observed_feedback_count * PREFERENCE_UTILITY_WEIGHT_SUM)
    )
    preference_engagement_feedback_contribution = (
        PREFERENCE_ENGAGEMENT_WEIGHT
        * tf.reduce_sum(
            preference_engagement_feedback * observed_feedback_mask,
            axis=-1,
        )
        / (observed_feedback_count * PREFERENCE_UTILITY_WEIGHT_SUM)
    )

    # S_i = alpha / P_i + gamma * U_i。未曝光 suffix 的 U_i 恒为 0，且其
    # position prior 严格递减，所以重排只会发生在事实曝光 Prefix 内部，
    # suffix 的内容和相对顺序保持不变。
    preference_position = tf.cast(
        tf.range(1, LIST_SIZE + 1),
        tf.float32,
    )
    preference_personalization_score = (
        PREFERENCE_POSITION_PRIOR_WEIGHT / preference_position
        + PREFERENCE_FEEDBACK_WEIGHT * preference_item_feedback_utility
    )
    personalized_order = tf.argsort(
        -preference_personalization_score,
        axis=-1,
    )
    personalized_list_index = tf.gather(
        logged_list_index,
        personalized_order,
        axis=1,
        batch_dims=1,
    )

    # 辅助负样本 Y_b：在真实曝光 Prefix 内反向使用同一反馈效用，令高反馈
    # item 更难排在前面。给 Prefix 统一加 2.0 只为保证其始终位于未曝光
    # suffix 之前；该常数不改变 Prefix 内部的相对顺序。这里显式构造
    # [batch_size, LIST_SIZE]，兼容不支持 Select 广播的线上 TensorFlow 1。
    reverse_position_prior = tf.ones_like(
        preference_item_feedback_utility,
    ) * (
        PREFERENCE_POSITION_PRIOR_WEIGHT / preference_position
    )
    reverse_personalization_score = tf.where(
        tf.greater(observed_feedback_mask, 0.0),
        2.0
        + reverse_position_prior
        - PREFERENCE_FEEDBACK_WEIGHT * preference_item_feedback_utility,
        reverse_position_prior,
    )
    reverse_order = tf.argsort(
        -reverse_personalization_score,
        axis=-1,
    )
    reverse_list_index = tf.gather(
        logged_list_index,
        reverse_order,
        axis=1,
        batch_dims=1,
    )

    preference_list_changed = tf.reduce_any(
        tf.not_equal(personalized_list_index, logged_list_index),
        axis=-1,
    )
    reverse_list_changed = tf.reduce_any(
        tf.not_equal(reverse_list_index, logged_list_index),
        axis=-1,
    )
    positive_reverse_distinct = tf.reduce_any(
        tf.not_equal(personalized_list_index, reverse_list_index),
        axis=-1,
    )
    preference_list_valid = tf.reduce_all(
        tf.logical_and(
            tf.greater(logged_list_index, 0),
            tf.less_equal(logged_list_index, CANDIDATES_SIZE),
        ),
        axis=-1,
    )
    preference_has_comparable_prefix = tf.greater_equal(realshow_num_raw, 2)
    preference_pair_eligibility_mask = tf.cast(
        tf.logical_and(
            tf.logical_and(
                preference_list_changed,
                preference_list_valid,
            ),
            preference_has_comparable_prefix,
        ),
        tf.float32,
    )
    # 只有旧分 Top1 的前 K 位确实匹配真实曝光 Prefix，才能将前 K 位反馈
    # 归因到这条完整 List，并据此构造 Preference pair。
    factual_observed_prefix_valid_flat = tf.cast(
        tf.squeeze(factual_observed_prefix_valid, axis=-1),
        tf.float32,
    )
    preference_pair_mask = preference_pair_eligibility_mask \
        * tf.cast(max_score_prefix_match, tf.float32) \
        * factual_observed_prefix_valid_flat
    observed_utility_max = tf.reduce_max(
        tf.where(
            tf.greater(observed_feedback_mask, 0.0),
            preference_item_feedback_utility,
            tf.fill(
                tf.shape(preference_item_feedback_utility),
                tf.constant(-1e9, dtype=tf.float32),
            ),
        ),
        axis=-1,
    )
    observed_utility_min = tf.reduce_min(
        tf.where(
            tf.greater(observed_feedback_mask, 0.0),
            preference_item_feedback_utility,
            tf.fill(
                tf.shape(preference_item_feedback_utility),
                tf.constant(1e9, dtype=tf.float32),
            ),
        ),
        axis=-1,
    )
    preference_has_feedback_gap = tf.greater(
        observed_utility_max - observed_utility_min,
        1e-6,
    )
    reverse_pair_eligibility_mask = preference_pair_eligibility_mask \
        * tf.cast(reverse_list_changed, tf.float32) \
        * tf.cast(positive_reverse_distinct, tf.float32) \
        * tf.cast(preference_has_feedback_gap, tf.float32)
    reverse_preference_pair_mask = reverse_pair_eligibility_mask \
        * tf.cast(max_score_prefix_match, tf.float32) \
        * factual_observed_prefix_valid_flat
    preference_position_grid = tf.reshape(
        tf.range(LIST_SIZE, dtype=tf.int32),
        [1, LIST_SIZE],
    )
    preference_reorder_distance = tf.reduce_sum(
        tf.abs(
            tf.cast(personalized_order - preference_position_grid, tf.float32)
        ) * observed_feedback_mask,
        axis=-1,
    ) / tf.cast(realshow_num, tf.float32)
    reverse_reorder_distance = tf.reduce_sum(
        tf.abs(
            tf.cast(reverse_order - preference_position_grid, tf.float32)
        ) * observed_feedback_mask,
        axis=-1,
    ) / tf.cast(realshow_num, tf.float32)
    personalized_item_feedback_utility = tf.gather(
        preference_item_feedback_utility,
        personalized_order,
        axis=1,
        batch_dims=1,
    )
    preference_feedback_gain = tf.reduce_sum(
        (
            personalized_item_feedback_utility
            - preference_item_feedback_utility
        ) / preference_position,
        axis=-1,
    )

    # 训练时在 30 条候选后追加 Y_w/Y_l/Y_b，一次前向共享 list_backbone。
    # 三条 List 都保留完整 6-item suffix；只在真实曝光 Prefix 内按反馈重排。
    # 主 factual 监督仍读取前 LIST_NUM 条中的旧分 Top1，不读取合成 pair。
    preference_list_index = tf.stack(
        [personalized_list_index, logged_list_index, reverse_list_index],
        axis=1,
    )
    augmented_list_index = tf.concat(
        [rerank_list_item_idx_flat_list, preference_list_index],
        axis=1,
    )
    model_class._training = True
    augmented_list_value_output_dict = model_class.model(
        list_index=augmented_list_index,
        list_num=LIST_NUM + 3,
    )
    candidate_list_value_output_dict = {
        name: value[:, :LIST_NUM]
        for name, value in augmented_list_value_output_dict.items()
    }
    preference_output_dict = {
        name: value[:, LIST_NUM:]
        for name, value in augmented_list_value_output_dict.items()
    }
    list_value_output_dict = candidate_list_value_output_dict
    print(f"====> train standalone list model, gen...")
    # 请求级互动正样本率在 Prefix 过滤前后的变化，用于检查匹配失败是否
    # 与稀疏互动行为相关。只统计真实曝光 item，避免未曝光候选干扰。
    request_interaction_positive = tf.reduce_any(
        tf.logical_and(
            tf.greater(interaction_value, 0.0),
            tf.greater(label_value_dict["show_label"], 0.0),
        ),
        axis=-1,
    )
    request_interaction_positive_float = tf.cast(
        request_interaction_positive,
        tf.float32,
    )
    tf.summary.scalar(
        "list_value/match/interaction_positive_rate_before_match",
        tf.reduce_sum(
            request_interaction_positive_float * observed_request_weight
        ) / (tf.reduce_sum(observed_request_weight) + 1e-8),
    )
    tf.summary.scalar(
        "list_value/match/interaction_positive_rate_after_match",
        tf.reduce_sum(
            request_interaction_positive_float * matched_request_weight
        ) / (tf.reduce_sum(matched_request_weight) + 1e-8),
    )

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
    list_engagement_value = list_evtr_label + list_interaction_value

    # -------- List Value 标签 --------
    # 对每个真实曝光位置做累加，构造 Prefix[1:k] 的累计 WT/Engagement。
    # Prefix 反馈只在旧分 Top1 确认匹配真实曝光 Prefix 时监督；K 之后虽保留
    # 完整 item 作为模型输入，但没有事实反馈，不进入 Prefix loss。
    prefix_watch_time_label = tf.cumsum(list_play_time_s, axis=-1)
    prefix_engagement_label = tf.cumsum(list_engagement_value, axis=-1)
    prefix_label_mask = tf.cast(observed_position_mask, tf.float32) \
        * tf.expand_dims(listwise_match_mask, axis=-1) \
        * tf.reshape(
            tf.cast(max_score_prefix_match, tf.float32)
            * factual_observed_prefix_valid_flat,
            [batch_size, 1, 1],
        )

    list_play_time_s_reduced = tf.reduce_sum(list_play_time_s, axis=-1)  # (?, list_num)
    list_effective_vv_reduced = tf.reduce_sum(list_evtr_label, axis=-1)  # (?, list_num)
    list_interaction_reduced = tf.reduce_sum(list_interaction_value, axis=-1)  # (?, list_num)
    list_engagement_reduced = tf.reduce_sum(list_engagement_value, axis=-1)  # (?, list_num)

    with tf.control_dependencies(print_ops):
        targets = []
        # -------- List Value 损失 --------
        continue_logits = list_value_output_dict["continue_logits"]
        continue_probs = list_value_output_dict["continue_probs"]
        length_probs = list_value_output_dict["length_probs"]
        prefix_watch_time = list_value_output_dict["prefix_watch_time"]
        prefix_watch_time_log = list_value_output_dict["prefix_watch_time_log"]
        prefix_engagement = list_value_output_dict["prefix_engagement"]
        prefix_engagement_log = list_value_output_dict["prefix_engagement_log"]
        # 两项业务价值输出；expected_consume_length 是计算期望价值的结构输出：
        # - expected_consume_length：由 continue 概率得到的预期消费 item 数；
        # - expected_list_watch_time：sum_k P(K=k) * PrefixWT(k)，单位为秒；
        # - expected_list_engagement：sum_k P(K=k) * PrefixEngagement(k)，
        #   Engagement = EVV + like + 10*comment + 10*follow + 2.5*forward。
        expected_list_watch_time = list_value_output_dict["expected_list_watch_time"]
        expected_list_engagement = list_value_output_dict["expected_list_engagement"]
        expected_consume_length = list_value_output_dict["expected_consume_length"]

        # 与 fountain_rerank_eval_listwise_v2_new 完全相同：在旧分 Top1 的
        # 完整 6-item List 上累加，不能用真实 K 截断 content_pwtd。
        list_wt_from_context_pwtd_sum = tf.reduce_sum(
            list_context_pwtd,
            axis=-1,
        )
        # 复刻 backbone 线上 item 聚合的固定位置衰减。权重只依赖
        # List 位置，不使用当前样本的真实曝光 K，因而可与 List
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
        relative_hazard_mask = hazard_observed_mask * candidate_list_mask_3d * tf.cast(
            tf.greater(valid_list_count, 1.0),
            tf.float32,
        )

        hazard_ce = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=continue_labels,
            logits=continue_logits,
        )
        # 对每条 List 累加所有已观测决策，得到右截断的负对数似然；
        # 再按参与训练的旧分 Top1 List 数量求均值，使每个请求权重一致。
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
        # WT 和综合 Engagement 都使用 log1p 压缩长尾。
        prefix_watch_time_label_log = tf.math.log1p(prefix_watch_time_label)
        prefix_wt_loss = tf.losses.huber_loss(
            labels=prefix_watch_time_label_log,
            predictions=prefix_watch_time_log,
            weights=prefix_label_mask,
            delta=0.5,
        )
        prefix_engagement_label_log = tf.math.log1p(prefix_engagement_label)
        prefix_interaction_positive = tf.greater(
            tf.cumsum(list_interaction_value, axis=-1),
            0.0,
        )
        prefix_engagement_sample_weight = prefix_label_mask * tf.where(
            prefix_interaction_positive,
            tf.fill(
                tf.shape(prefix_engagement_label),
                tf.constant(
                    ENGAGEMENT_INTERACTION_POSITIVE_SAMPLE_WEIGHT,
                    tf.float32,
                ),
            ),
            tf.ones_like(prefix_engagement_label),
        )
        prefix_engagement_loss = tf.losses.huber_loss(
            labels=prefix_engagement_label_log,
            predictions=prefix_engagement_log,
            weights=prefix_engagement_sample_weight,
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
        list_engagement_sample_weight = listwise_match_mask * tf.where(
            tf.greater(list_interaction_reduced, 0.0),
            tf.fill(
                tf.shape(list_engagement_reduced),
                tf.constant(
                    ENGAGEMENT_INTERACTION_POSITIVE_SAMPLE_WEIGHT,
                    tf.float32,
                ),
            ),
            tf.ones_like(list_engagement_reduced),
        )
        list_engagement_loss = tf.losses.huber_loss(
            labels=tf.math.log1p(list_engagement_reduced),
            predictions=tf.math.log1p(expected_list_engagement),
            weights=list_engagement_sample_weight,
            delta=0.5,
        )

        # -------- GReF-inspired WT-primary preference loss --------
        # evaluator 没有生成概率，直接对综合 List Value 使用 RankNet/BPR：
        # 主约束比较 Y_w > Y_l；辅助约束以较低权重比较 Y_w > Y_b。
        # V 与正负样本构造统一使用 0.70 WT + 0.30 Engagement。
        preference_pred_wt_utility = tf.math.log1p(
            preference_output_dict["expected_list_watch_time"]
        ) / tf.math.log1p(
            tf.constant(PREFERENCE_LIST_WT_CAP_SECONDS, dtype=tf.float32)
        )
        preference_pred_engagement_utility = tf.math.log1p(
            preference_output_dict["expected_list_engagement"]
        ) / tf.math.log1p(
            tf.constant(MAX_ENGAGEMENT_VALUE, dtype=tf.float32)
        )
        preference_pred_utility = (
            PREFERENCE_WT_WEIGHT * preference_pred_wt_utility
            + PREFERENCE_ENGAGEMENT_WEIGHT
            * preference_pred_engagement_utility
        ) / PREFERENCE_UTILITY_WEIGHT_SUM
        # 第 0/1/2 条分别为 Y_w/Y_l/Y_b。保留各分量 margin，便于判断
        # preference 是否只靠某一个 head 投机满足。
        preference_wt_margin = (
            preference_pred_wt_utility[:, 0]
            - preference_pred_wt_utility[:, 1]
        )
        preference_engagement_margin = (
            preference_pred_engagement_utility[:, 0]
            - preference_pred_engagement_utility[:, 1]
        )
        preference_margin = (
            preference_pred_utility[:, 0]
            - preference_pred_utility[:, 1]
        )
        reverse_preference_wt_margin = (
            preference_pred_wt_utility[:, 0]
            - preference_pred_wt_utility[:, 2]
        )
        reverse_preference_engagement_margin = (
            preference_pred_engagement_utility[:, 0]
            - preference_pred_engagement_utility[:, 2]
        )
        reverse_preference_margin = (
            preference_pred_utility[:, 0]
            - preference_pred_utility[:, 2]
        )
        preference_anchor_loss_per_request = tf.nn.softplus(
            -preference_margin / PREFERENCE_TEMPERATURE
        )
        preference_anchor_loss = tf.reduce_sum(
            preference_anchor_loss_per_request * preference_pair_mask
        ) / (tf.reduce_sum(preference_pair_mask) + 1e-8)
        reverse_preference_loss_per_request = tf.nn.softplus(
            -reverse_preference_margin / PREFERENCE_TEMPERATURE
        )
        reverse_preference_loss = tf.reduce_sum(
            reverse_preference_loss_per_request
            * reverse_preference_pair_mask
        ) / (tf.reduce_sum(reverse_preference_pair_mask) + 1e-8)
        preference_loss = preference_anchor_loss \
            + PREFERENCE_REVERSE_LOSS_WEIGHT * reverse_preference_loss

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

        raw_weighted_length_loss = LENGTH_LOSS_WEIGHT * length_loss
        raw_weighted_prefix_wt_loss = PREFIX_WT_LOSS_WEIGHT * prefix_wt_loss
        raw_weighted_prefix_engagement_loss = (
            PREFIX_ENGAGEMENT_LOSS_WEIGHT * prefix_engagement_loss
        )
        raw_weighted_list_wt_loss = LIST_WT_LOSS_WEIGHT * list_wt_loss
        raw_weighted_list_engagement_loss = (
            LIST_ENGAGEMENT_LOSS_WEIGHT * list_engagement_loss
        )
        raw_weighted_preference_anchor_loss = (
            PREFERENCE_LOSS_WEIGHT * preference_anchor_loss
        )
        raw_weighted_reverse_preference_loss = (
            PREFERENCE_LOSS_WEIGHT
            * PREFERENCE_REVERSE_LOSS_WEIGHT
            * reverse_preference_loss
        )
        raw_weighted_preference_loss = raw_weighted_preference_anchor_loss \
            + raw_weighted_reverse_preference_loss
        raw_weighted_monotonic_loss = (
            PREFIX_MONOTONIC_LOSS_WEIGHT * monotonic_loss
        )
        factual_pretrain_loss = raw_weighted_length_loss \
            + raw_weighted_prefix_wt_loss \
            + raw_weighted_prefix_engagement_loss \
            + raw_weighted_list_wt_loss \
            + raw_weighted_list_engagement_loss \
            + raw_weighted_monotonic_loss

        # 两阶段严格互斥：预训练不接收 Preference 梯度，后训练只接收
        # Preference 梯度。原有 loss 在另一阶段仍计算，仅供退化监控。
        if IS_PREFERENCE_POSTTRAIN:
            list_value_loss = raw_weighted_preference_loss
            weighted_length_loss = tf.zeros_like(raw_weighted_length_loss)
            weighted_prefix_wt_loss = tf.zeros_like(raw_weighted_prefix_wt_loss)
            weighted_prefix_engagement_loss = tf.zeros_like(
                raw_weighted_prefix_engagement_loss
            )
            weighted_list_wt_loss = tf.zeros_like(raw_weighted_list_wt_loss)
            weighted_list_engagement_loss = tf.zeros_like(
                raw_weighted_list_engagement_loss
            )
            weighted_monotonic_loss = tf.zeros_like(raw_weighted_monotonic_loss)
            weighted_preference_anchor_loss = raw_weighted_preference_anchor_loss
            weighted_reverse_preference_loss = (
                raw_weighted_reverse_preference_loss
            )
        else:
            list_value_loss = factual_pretrain_loss
            weighted_length_loss = raw_weighted_length_loss
            weighted_prefix_wt_loss = raw_weighted_prefix_wt_loss
            weighted_prefix_engagement_loss = (
                raw_weighted_prefix_engagement_loss
            )
            weighted_list_wt_loss = raw_weighted_list_wt_loss
            weighted_list_engagement_loss = raw_weighted_list_engagement_loss
            weighted_monotonic_loss = raw_weighted_monotonic_loss
            weighted_preference_anchor_loss = tf.zeros_like(
                raw_weighted_preference_anchor_loss
            )
            weighted_reverse_preference_loss = tf.zeros_like(
                raw_weighted_reverse_preference_loss
            )
        weighted_preference_loss = weighted_preference_anchor_loss \
            + weighted_reverse_preference_loss
        sum_loss = list_value_loss

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
        # 两个比例都只在旧分 Top1 完整 List 上统计，口径与 length loss 一致。
        label_full_length_rate = masked_mean(
            tf.cast(tf.equal(length_label, LIST_SIZE - 1), tf.float32),
            listwise_match_mask,
        )
        predicted_full_length_rate = masked_mean(
            tf.cast(tf.equal(length_prediction, LIST_SIZE - 1), tf.float32),
            listwise_match_mask,
        )
        # argmax K=6 只说明第六类经常是最大单类；直接监控平均 P(K=6)
        # 才能判断模型是否真的对满长度过度自信。
        predicted_full_length_probability_mean = masked_mean(
            length_probs[:, :, -1],
            listwise_match_mask,
        )
        # 第 5 位的 continue 决策直接决定是否到达第 6 位。只在真实到达
        # 第 5 位、该 hazard 标签可观测的训练 List 上比较预测与真实比例。
        continue_pos5_mask = hazard_observed_mask[:, :, -1]
        continue_pos5_pred_rate = masked_mean(
            continue_probs[:, :, -1],
            continue_pos5_mask,
        )
        continue_pos5_label_rate = masked_mean(
            continue_labels[:, :, -1],
            continue_pos5_mask,
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
            "list_value/length/continue_pos5_pred_rate",
            continue_pos5_pred_rate,
        )
        tf.summary.scalar(
            "list_value/length/continue_pos5_label_rate",
            continue_pos5_label_rate,
        )
        tf.summary.scalar(
            "list_value/length/non_k6_accuracy",
            non_full_length_accuracy,
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
        expected_engagement_pred_mean = masked_mean(
            expected_list_engagement,
            listwise_match_mask,
        )
        expected_engagement_label_mean = masked_mean(
            list_engagement_reduced,
            listwise_match_mask,
        )
        expected_engagement_mae = masked_mean(
            tf.abs(expected_list_engagement - list_engagement_reduced),
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
        evv_only_pred_mean = masked_mean(
            expected_list_engagement,
            evv_only_mask,
        )
        evv_only_label_mean = masked_mean(
            list_engagement_reduced,
            evv_only_mask,
        )
        interaction_positive_pred_mean = masked_mean(
            expected_list_engagement,
            interaction_positive_mask,
        )
        interaction_positive_label_mean = masked_mean(
            list_engagement_reduced,
            interaction_positive_mask,
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
            expected_engagement_pred_mean
            / (expected_engagement_label_mean + 1e-8),
        )
        tf.summary.scalar(
            "list_value/calibration/expected_engagement_mae",
            expected_engagement_mae,
        )
        tf.summary.scalar(
            "list_value/engagement/evv_value_share",
            tf.reduce_sum(list_effective_vv_reduced * listwise_match_mask)
            / (
                tf.reduce_sum(list_engagement_reduced * listwise_match_mask)
                + 1e-8
            ),
        )
        tf.summary.scalar(
            "list_value/engagement/interaction_value_share",
            tf.reduce_sum(list_interaction_reduced * listwise_match_mask)
            / (
                tf.reduce_sum(list_engagement_reduced * listwise_match_mask)
                + 1e-8
            ),
        )
        tf.summary.scalar(
            "list_value/engagement/evv_only_pred_label_ratio",
            evv_only_pred_mean / (evv_only_label_mean + 1e-8),
        )
        tf.summary.scalar(
            "list_value/engagement/evv_only_mae",
            masked_mean(
                tf.abs(expected_list_engagement - list_engagement_reduced),
                evv_only_mask,
            ),
        )
        tf.summary.scalar(
            "list_value/engagement/interaction_positive_pred_label_ratio",
            interaction_positive_pred_mean
            / (interaction_positive_label_mean + 1e-8),
        )
        tf.summary.scalar(
            "list_value/engagement/interaction_positive_mae",
            masked_mean(
                tf.abs(expected_list_engagement - list_engagement_reduced),
                interaction_positive_mask,
            ),
        )
        tf.summary.scalar(
            "list_value/engagement/interaction_positive_rate",
            masked_mean(interaction_positive_mask, listwise_match_mask),
        )
        tf.summary.scalar(
            "list_value/engagement/interaction_positive_weighted_sample_share",
            tf.reduce_sum(
                interaction_positive_mask
                * ENGAGEMENT_INTERACTION_POSITIVE_SAMPLE_WEIGHT
            ) / (tf.reduce_sum(list_engagement_sample_weight) + 1e-8),
        )

        # -------- TensorBoard 监控：用户反馈顺序偏好 --------
        preference_comparable_request_weight = tf.cast(
            tf.logical_and(
                preference_has_comparable_prefix,
                preference_list_valid,
            ),
            tf.float32,
        )
        comparable_request_count = tf.reduce_sum(
            preference_comparable_request_weight
        )
        preference_pair_count = tf.reduce_sum(preference_pair_mask)
        reverse_preference_pair_count = tf.reduce_sum(
            reverse_preference_pair_mask
        )
        tf.summary.scalar(
            "list_value/preference/pair_rate",
            preference_pair_count / (comparable_request_count + 1e-8),
        )
        tf.summary.scalar(
            "list_value/preference/reverse/pair_rate",
            reverse_preference_pair_count
            / (comparable_request_count + 1e-8),
        )
        tf.summary.scalar(
            "list_value/factual/observed_prefix_valid_rate",
            masked_mean(
                tf.cast(
                    tf.squeeze(factual_observed_prefix_valid, axis=-1),
                    tf.float32,
                ),
                observed_request_weight,
            ),
        )
        tf.summary.scalar(
            "list_value/factual/exposed_item_mapping_rate",
            tf.reduce_sum(tf.cast(raw_exposure_mapping_valid, tf.float32))
            / (tf.reduce_sum(tf.cast(raw_show_mask, tf.float32)) + 1e-8),
        )
        tf.summary.scalar(
            "list_value/factual/exposure_order_unique_rate",
            masked_mean(
                tf.cast(factual_exposure_order_unique, tf.float32),
                observed_request_weight,
            ),
        )
        tf.summary.scalar(
            "list_value/factual/candidate_index_unique_rate",
            masked_mean(
                tf.cast(factual_candidate_index_unique, tf.float32),
                observed_request_weight,
            ),
        )
        tf.summary.scalar(
            "list_value/preference/observed_sequence_valid_rate",
            masked_mean(
                tf.cast(preference_list_valid, tf.float32),
                observed_request_weight,
            ),
        )
        tf.summary.scalar(
            "list_value/preference/reorder_distance_mean",
            masked_mean(preference_reorder_distance, preference_pair_mask),
        )
        tf.summary.scalar(
            "list_value/preference/reverse/reorder_distance_mean",
            masked_mean(
                reverse_reorder_distance,
                reverse_preference_pair_mask,
            ),
        )
        tf.summary.scalar(
            "list_value/preference/feedback_gain_mean",
            masked_mean(preference_feedback_gain, preference_pair_mask),
        )
        tf.summary.scalar(
            "list_value/preference/feedback_contribution/wt",
            masked_mean(
                preference_wt_feedback_contribution,
                preference_pair_mask,
            ),
        )
        tf.summary.scalar(
            "list_value/preference/feedback_contribution/engagement",
            masked_mean(
                preference_engagement_feedback_contribution,
                preference_pair_mask,
            ),
        )
        tf.summary.scalar(
            "list_value/preference/accuracy",
            masked_mean(
                tf.cast(tf.greater(preference_margin, 0.0), tf.float32),
                preference_pair_mask,
            ),
        )
        tf.summary.scalar(
            "list_value/preference/margin/composite",
            masked_mean(preference_margin, preference_pair_mask),
        )
        tf.summary.scalar(
            "list_value/preference/margin/wt",
            masked_mean(preference_wt_margin, preference_pair_mask),
        )
        tf.summary.scalar(
            "list_value/preference/margin/engagement",
            masked_mean(preference_engagement_margin, preference_pair_mask),
        )
        tf.summary.scalar(
            "list_value/preference/reverse/accuracy",
            masked_mean(
                tf.cast(
                    tf.greater(reverse_preference_margin, 0.0),
                    tf.float32,
                ),
                reverse_preference_pair_mask,
            ),
        )
        tf.summary.scalar(
            "list_value/preference/reverse/margin/composite",
            masked_mean(
                reverse_preference_margin,
                reverse_preference_pair_mask,
            ),
        )
        tf.summary.scalar(
            "list_value/preference/reverse/margin/wt",
            masked_mean(
                reverse_preference_wt_margin,
                reverse_preference_pair_mask,
            ),
        )
        tf.summary.scalar(
            "list_value/preference/reverse/margin/engagement",
            masked_mean(
                reverse_preference_engagement_margin,
                reverse_preference_pair_mask,
            ),
        )

        # -------- TensorBoard 监控：加权损失贡献 --------
        # loss_contribution 只记录当前阶段真正参与反传的贡献；raw_loss 同时
        # 观察被关闭的另一阶段目标，便于识别后训练引起的 factual 能力退化。
        tf.summary.scalar(
            "list_value/training_stage/is_preference_posttrain",
            tf.constant(
                1.0 if IS_PREFERENCE_POSTTRAIN else 0.0,
                dtype=tf.float32,
            ),
        )
        tf.summary.scalar(
            "list_value/raw_loss/factual_pretrain_total",
            factual_pretrain_loss,
        )
        tf.summary.scalar(
            "list_value/raw_loss/preference_total",
            raw_weighted_preference_loss,
        )
        tf.summary.scalar("list_value/loss_contribution/length", weighted_length_loss)
        tf.summary.scalar(
            "list_value/loss_contribution/prefix_value",
            weighted_prefix_wt_loss + weighted_prefix_engagement_loss,
        )
        tf.summary.scalar(
            "list_value/loss_contribution/list_total",
            weighted_list_wt_loss + weighted_list_engagement_loss,
        )
        tf.summary.scalar(
            "list_value/loss_contribution/engagement",
            weighted_prefix_engagement_loss + weighted_list_engagement_loss,
        )
        tf.summary.scalar(
            "list_value/loss_contribution/engagement_ratio",
            (
                weighted_prefix_engagement_loss
                + weighted_list_engagement_loss
            ) / (list_value_loss + 1e-8),
        )
        tf.summary.scalar("list_value/loss_contribution/monotonic", weighted_monotonic_loss)
        tf.summary.scalar(
            "list_value/loss_contribution/preference",
            weighted_preference_loss,
        )
        tf.summary.scalar(
            "list_value/loss_contribution/preference_anchor",
            weighted_preference_anchor_loss,
        )
        tf.summary.scalar(
            "list_value/loss_contribution/preference_reverse",
            weighted_reverse_preference_loss,
        )
        tf.summary.scalar(
            "list_value/loss_contribution/preference_ratio",
            weighted_preference_loss / (list_value_loss + 1e-8),
        )
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

        flat_prefix_size = LIST_NUM * LIST_SIZE
        flat_prefix_mask = tf.reshape(
            prefix_label_mask,
            [batch_size, flat_prefix_size],
        )
        targets.append((
            "prefix_watch_time",
            tf.reshape(prefix_watch_time, [batch_size, flat_prefix_size]),
            tf.reshape(prefix_watch_time_label, [batch_size, flat_prefix_size]),
            flat_prefix_mask,
            "linear_regression",
        ))
        targets.append((
            "prefix_engagement",
            tf.reshape(prefix_engagement, [batch_size, flat_prefix_size]),
            tf.reshape(prefix_engagement_label, [batch_size, flat_prefix_size]),
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
            expected_list_watch_time,
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
            "list_wt_from_context_pwtd_position_decay",
            list_wt_from_context_pwtd_position_decay,
            list_play_time_s_reduced,
            listwise_match_mask,
            "linear_regression",
        ))
        targets.append((
            "expected_list_engagement",
            expected_list_engagement,
            list_engagement_reduced,
            listwise_match_mask,
            "linear_regression",
        ))
        targets.append((
            "expected_list_engagement_evv_only",
            expected_list_engagement,
            list_engagement_reduced,
            evv_only_mask,
            "linear_regression",
        ))
        targets.append((
            "expected_list_engagement_interaction_positive",
            expected_list_engagement,
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
        ("expected_list_engagement", list_value_output_dict["expected_list_engagement"]),
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
