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

def mae_r_squared_v2(x, y, ids=None):
    diff = tf.abs(x - y)
    mae = tf.reduce_mean(diff)
    avg = tf.reduce_mean(x)
    rss = tf.reduce_sum(tf.pow(x - y, 2.0))
    tss = tf.reduce_sum(tf.pow(x - avg, 2.0))
    r_squared = 1.0 - rss / (tss + 1e-8)
    # mMAE
    m_mae = 0.0
    if ids is not None:
        m_mae = []
        for i in range(6):
            idx = tf.where(tf.equal(ids, i))
            diff_ = tf.gather(diff, idx)
            aem = tf.reduce_mean(diff_)
            # flag = tf.cast(ids == i, tf.float32)
            # ae = tf.reduce_sum(diff * flag)
            # aem = ae / (tf.reduce_sum(flag) + 1e-8)
            m_mae.append(aem)
        m_mae = tf.reduce_mean(tf.concat([m_mae], axis=0))
    return mae, r_squared, m_mae

def mae_r_squared(x, y):
    diff = tf.abs(x - y)
    mae = tf.reduce_mean(diff)
    avg = tf.reduce_mean(x)
    rss = tf.reduce_sum(tf.pow(x - y, 2.0))
    tss = tf.reduce_sum(tf.pow(x - avg, 2.0))
    r_squared = 1.0 - rss / (tss + 1e-8)
    return mae, r_squared

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
        config.declare_reallocate_slots(share_input_slots, share_output_slots, remap=True, inplace=True)
        # 需要额外copy的特征
        config.declare_reallocate_slots(copy_input_slots, copy_output_slots, remap=True, inplace=False)
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
            feature_emb_size_dict[attr.attr_name] = 1
        else:
            if attr.attr_name in photo_fea_names or attr.attr_name in ["pid_copy", "aid_copy"]:
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
                uid = tf.RaggedTensor.from_row_splits(values=sparse_feature[0], row_splits=sparse_feature[1]).to_tensor()
                uid_dict["uid"] = uid
                #tt = tf.RaggedTensor.from_row_splits(values=uid, row_splits=offset).to_tensor()
                #print_ops.append(tf.print("[Test test] slot " + str(attr.slots[0]), tt, output_stream=sys.stdout))
        elif args.with_kai:
            offset = tf.cast(config.get_signs(attr.slots[0])[1], tf.int32)
            size_var = offset[1:] - offset[0:-1]
            feature_emb_size_dict[attr.attr_name] = size_var        
        print("--->>> feature {} = {}".format(attr.attr_name, feature_emb_dict[attr.attr_name]))
        print("--->>> feature %s normal" % attr.attr_name)

    return feature_emb_dict, feature_emb_size_dict, uid_dict


def sum_loss_tensor_dict(loss_dict):
    sum_loss = None
    for key, loss in loss_dict.items():
        if (sum_loss == None):
            sum_loss = loss
        else:
            sum_loss += loss
    return sum_loss

def focal_loss_with_neg_upsampling(labels, predictions, weights, gamma=2.0, neg_weight=5.0):
    """
    Focal loss with negative sample upsampling.
    labels: 0/1 float tensor, 1=slide(正样本), 0=no-slide(负样本)
    predictions: predicted probability of sliding (continue)
    weights: sample mask
    gamma: focal loss exponent, reduces weight of easy positives
    neg_weight: weight multiplier for negative samples (no-slide), equivalent to upsampling
    """
    predictions = tf.clip_by_value(predictions, 1e-7, 1 - 1e-7)
    # pt: probability of the true class
    pt = tf.where(tf.equal(labels, 1.0), predictions, 1.0 - predictions)
    # focal weight: down-weights easy (high-confidence) samples
    focal_weight = tf.pow(1.0 - pt, gamma)
    # cross entropy
    ce = -tf.where(tf.equal(labels, 1.0), tf.math.log(predictions), tf.math.log(1.0 - predictions))
    # negative sample upsampling: multiply weight by neg_weight when label=0
    sample_weight = tf.where(tf.equal(labels, 0.0), weights * neg_weight, weights * 1.0)
    loss = tf.reduce_mean(focal_weight * ce * sample_weight)
    return loss

def ordinal_regression_loss(pred, show_weight, mask, scale=5.0, training=True, train_cutpoints=False, name="ordinal_regression_loss"):
    # 消耗内存大，容易 OOM，减小 batch size 解决
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        pred = tf.reshape(pred, [-1, 1])
        bins = tf.constant([0, 5, 7, 9, 12, 17, 20, 40, 58, 120, 300], dtype=tf.float32, shape=[1, 11])
        num_classes = 11
        initial_cut_points = tf.cast(tf.range(num_classes - 1, dtype=tf.int32), dtype=tf.float32) * scale / (num_classes - 2.0) - scale / 2.0
        print("initial_cut_points", initial_cut_points)
        # cut_points = tf.get_variable("cut_points", initializer=initial_cut_points, trainable=train_cutpoints)
        cut_points = initial_cut_points
        # 计算sigmoid边界概率
        sigmoids = tf.sigmoid(cut_points - pred)  # shape: (batch_size, num_cutpoints)
        # 计算累积概率分布
        first_col = tf.slice(sigmoids, [0,0], [-1,1])              # 首列保持不变
        middle_cols = sigmoids[:,1:] - sigmoids[:,:-1]            # 中间列差分
        last_col = 1 - tf.slice(sigmoids, [0,num_classes-2], [-1,1])  # 最后一列
        link_mat = tf.concat([first_col, middle_cols, last_col], axis=1) # shape: (batch_size, num_cutpoints)
        print("link_mat", link_mat)
        likelihoods = tf.clip_by_value(link_mat, 1e-8, 1 - 1e-8)
        if training:
            mask = tf.reshape(mask, [-1, 1])
            show_weight = tf.reshape(show_weight, [-1, 1])
            indices = tf.searchsorted(bins, tf.reshape(show_weight, [-1]), side='right', out_type=tf.int32) - 1 # 落到哪个桶
            indices = tf.reshape(indices, [-1, 1])
            labels = tf.clip_by_value(indices, 0, tf.size(bins) - 1)

            loss = -tf.reduce_sum(
                # indices: (?,2), get (?, num_calsses - 1)
                tf.gather_nd(tf.log(likelihoods), tf.concat([tf.range(tf.shape(labels)[0])[:, None], labels], axis=1)) # 最大化正样本对应的累计概率分布
                *
                mask
            )
            return loss, likelihoods
        else:
            # bins_reward = tf.constant([2.5, 6, 8, 10.5, 14.5, 18.5, 30, 49, 79, 210], dtype=tf.float32, shape=[1, 10])
            reward = tf.reduce_sum(likelihoods * bins, axis=-1, keepdims=True)
            print("wtd reward", reward)
            return reward

def pairwise_bpr_loss_v2(output, score, score2, threshold, weight, mask):
    # output: (?,len) score: (?,len) threshold: (?,len) mask: (?,len)
    output_i = tf.expand_dims(output, 2)
    output_j = tf.expand_dims(output, 1)
    score_i = tf.expand_dims(score, 2)
    score_j = tf.expand_dims(score, 1)
    weight_i = tf.expand_dims(weight, 2)
    weight_j = tf.expand_dims(weight, 1)
    score_diff = score_i - score_j
    score2_diff = tf.expand_dims(score2, 2) - tf.expand_dims(score2, 1)
    final_score_diff = tf.where(tf.equal(score_diff, 0.0), score2_diff, score_diff)
    threshold = tf.expand_dims(threshold, -1)
    pairwise_labels = tf.cast(tf.greater(final_score_diff, threshold), tf.float32)
    weight_expand = tf.where(tf.greater(pairwise_labels, 0.0), tf.tile(weight_i, [1, 1, pairwise_labels.shape[2]]), tf.tile(weight_j, [1, pairwise_labels.shape[1], 1]))
    pairwise_weight = weight_expand
    # pairwise_weight = tf.where(tf.logical_and(tf.equal(score_diff, 0.0), tf.greater(score2_diff, 0.0)), weight_expand * 0.2, weight_expand)
    # pairwise_label_mask = tf.logical_or(final_score_diff > threshold, final_score_diff < -threshold) # 在阈值内的pair不计算loss
    logit_diff = tf.sigmoid(output_i - output_j)
    # 生成有效掩码
    mask = tf.cast(mask, tf.bool)
    mask_i = tf.expand_dims(mask, 2)
    mask_j = tf.expand_dims(mask, 1)
    valid_pair_mask = tf.logical_and(mask_i, mask_j)
    final_mask = valid_pair_mask
    print("final_mask ", final_mask)
    # 计算BPR损失
    bpr_loss = -tf.math.log(logit_diff + 1e-9) * pairwise_labels * pairwise_weight
    print("bpr_loss", bpr_loss)
    bpr_loss = tf.where(final_mask, bpr_loss, tf.zeros_like(bpr_loss, dtype=tf.float32))
    return bpr_loss

def calc_advantage(reward, mask, user_id):
    user_id = tf.reshape(user_id, [-1, 1])
    item_size = tf.shape(reward)[1]
    
    user_ids_flat = tf.reshape(tf.tile(user_id, [1, item_size]), [-1])    # 展平为(batch_size * item_size, )
    reward_flat = tf.reshape(reward, [-1])                 # (batch_size * item_size, )
    mask_flat = tf.reshape(mask, [-1])                     # (batch_size * item_size, )
    mask_flat = tf.cast(mask_flat, reward.dtype)
    
    unique_users, group_idx = tf.unique(user_ids_flat)
    print("group_idx", group_idx)
    num_segments = tf.shape(unique_users)[0]  # 分组数量
    
    valid_cnt_per_group = tf.math.unsorted_segment_sum(mask_flat, group_idx, num_segments) # (group_num)
    sum_reward_per_group = tf.math.unsorted_segment_sum(reward_flat * mask_flat, group_idx, num_segments) # (group_num)
    print("sum_reward_per_group", sum_reward_per_group)
    mean_per_group = sum_reward_per_group / (valid_cnt_per_group + 1e-8)
    
    diff = reward_flat - tf.gather(mean_per_group, group_idx)
    variance_per_group = tf.math.unsorted_segment_sum((diff ** 2) * mask_flat, group_idx, num_segments)
    std_per_group = tf.sqrt(variance_per_group / (valid_cnt_per_group + 1e-8))  # 分组标准差
    
    mean_flat = tf.gather(mean_per_group, group_idx)       # 每个item对应的组均值
    std_flat = tf.gather(std_per_group, group_idx)
    print("std_flat", std_flat)
    mean = tf.reshape(mean_flat, tf.shape(reward))         # (batch_size, item_size)
    std = tf.reshape(std_flat, tf.shape(reward))
    
    # 计算标准化后的advantages
    advantages = (reward - mean) / (std + 1e-8) * mask  # (batch_size, item_size)
    return advantages

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


def get_vtr_from_watch_time(watch_time_s, max_watch_time):
    return tf.cast(watch_time_s, tf.float32) / (tf.cast(max_watch_time, tf.float32) + 1e-8)


# 适合直接用 play_time_s 作为监督目标，避免 sigmoid 截断导致的系统性低估
_WT_BUCKETS = [126.143,37.273,37.273,37.273,49.909,73.636,108.556,116.71,115.661,112.282,117.694,120.773,113.152,113.58,116.71,120.994,117.205,114.166,114.916,110.194,104.811,102.394,100.992,105.644,107.073,110.415,110.693,105.249,108.215,106.411,110.046,103.66,107.075,107.948,102.366,106.835,104.614,106.755,107.392,103.63,98.364,98.318,101.976,97.505,99.748,99.906,101.857,100.387,102.698,103.719,104.998,103.746,106.468,108.6,106.418,107.294,110.825,112.583,113.497,113.473,114.885,110.998,113.476,114.182,110.493,112.166,112.849,115.205,113.069,116.622,115.864,116.927,112.597,116.769,114.353,115.245,115.381,114.476,113.123,118.325,120.576,117.788,115.617,119.428,119.337,121.104,121.076,121.622,123.891,122.986,119.524,121.759,124.767,126.54,122.851,123.598,123.747,121.141,126.368,122.234,124.698,123.941,122.459,125.179,128.054,124.017,123.927,127.821,126.8,125.761,129.136,126.184,128.474,130.522,132.295,131.511,130.809,129.382,132.497,131.264,134.051,134.566,132.249,135.828,135.531,131.979,137.039,136.273,138.381,138.364,139.18,139.395,139.402,142.823,141.631,142.814,141.64,141.355,140.215,141.915,140.216,142.513,143.464,146.272,146.592,145.636,147.262,144.395,149.201,146.603,146.636,146.351,147.59,151.337,147.944,149.681,149.202,149.958,146.294,154.688,150.646,153.921,153.576,153.557,149.261,148.648,152.067,150.784,150.381,155.05,155.099,155.092,149.341,149.552,156.568,158.64,155.796,157.338,153.212,155.447,153.174,151.656,155.98,155.608,149.921,157.445,158.027,159.689,156.586,155.805,149.556,156.661,161.279,156.972,160.079,158.68,156.277,157.08,156.773,154.777,200.0]
def wt_encode(playtime_s, duration_s):
    buckets = tf.constant(_WT_BUCKETS, dtype=tf.float32)  # (201,)
    duration_s = tf.cast(duration_s, tf.int32)
    vtr_max = tf.ones_like(duration_s) * 200
    indices = tf.where(duration_s > 200, vtr_max, duration_s)
    max_time = tf.gather(buckets, indices)  # (...)
    ratio = tf.cast(playtime_s, tf.float32) / (max_time + 1e-8)
    return ratio

def wt_decode(ratio, duration_s):
    buckets = tf.constant(_WT_BUCKETS, dtype=tf.float32)  # (201,)
    duration_s = tf.cast(duration_s, tf.int32)
    vtr_max = tf.ones_like(duration_s) * 200
    indices = tf.where(duration_s > 200, vtr_max, duration_s)
    max_time = tf.gather(buckets, indices)  # (...)
    wt = tf.cast(ratio, tf.float32) * max_time
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
    # next
    # buckets = [0.895817357, 0.905050505, 0.6, 0.905050505, 0.922092376, 0.934030418, 0.936088747, 0.933280213, 0.936280704, 0.934754067, 0.928623448, 0.928116922, 0.92952844, 0.927857488, 0.932499712, 0.933391215, 0.935053925, 0.930077087, 0.929321329, 0.927606488, 0.920189224, 0.920173347, 0.920250631, 0.920553222, 0.920644443, 0.91933691, 0.919796758, 0.918985622, 0.915262843, 0.919708752, 0.920290931, 0.918810723, 0.918406316, 0.921082137, 0.920121129, 0.919771795, 0.920678496, 0.922153008, 0.919535568, 0.916980178, 0.916276144, 0.911466559, 0.914521932, 0.914167177, 0.913313756, 0.91433831, 0.917019199, 0.91489732, 0.913605185, 0.916629946, 0.916713078, 0.914060258, 0.912235172, 0.915915637, 0.90991776, 0.911291355, 0.915339353, 0.908603751, 0.912122874, 0.905425901, 0.890743306, 0.91476909, 0.911766203, 0.915087123, 0.911249179, 0.917259265, 0.91483871, 0.912066006, 0.913249124, 0.909195734, 0.907884418, 0.911205947, 0.907952021, 0.905669145, 0.904551769, 0.901966834, 0.910812416, 0.907463823, 0.902424404, 0.891333925, 0.904492665, 0.91055665, 0.90817954, 0.901208544, 0.909442126, 0.902182923, 0.902955692, 0.912196335, 0.912607624, 0.913892588, 0.89988764, 0.902971888, 0.897947214, 0.9011944577, 0.90133303, 0.902633486, 0.896875683, 0.902227171, 0.898063319, 0.892121996, 0.892064271, 0.89552737, 0.89430605, 0.88677226, 0.896503981, 0.899070892, 0.898685137, 0.894245142, 0.899490446, 0.897534115, 0.899038878, 0.898928025, 0.897762238, 0.895570549, 0.895464025, 0.891058855, 0.884786593, 0.895377704, 0.900013226, 0.874490153, 0.896, 0.892301608, 0.886182207, 0.895833333, 0.895603251, 0.8912888, 0.893961219, 0.894481236, 0.89409244, 0.891970379, 0.882799325, 0.889921794, 0.890653917, 0.894205277, 0.881937683, 0.89025894, 0.887626263, 0.876626826, 0.892604308, 0.895187166, 0.884055398, 0.891406637, 0.893619934, 0.888287715, 0.880108992, 0.886534757, 0.891637578, 0.881565657, 0.886541943, 0.885923515, 0.893367347, 0.889042722, 0.888813097, 0.893871218, 0.88660888, 0.891277641, 0.887620547, 0.88710342, 0.889072848, 0.881834215, 0.885019711, 0.882376396, 0.890969776, 0.882232518, 0.884368308, 0.887913104, 0.88616025, 0.887373619, 0.883941165, 0.88778626, 0.888519426, 0.886474501, 0.890849282, 0.891545524, 0.887762157, 0.893176134, 0.889138883, 0.885261194, 0.878142741, 0.887688642, 0.884532865, 0.888843145, 0.881319954, 0.890608132, 0.8798151, 0.888237945, 0.88358209, 0.899141883, 0.892132608, 0.890986819, 0.886575736, 0.877800791, 0.879748729, 0.876371421, 0.884218289, 0.882070549, 0.886446148, 0.889412689, 0.894187261, 0.885549609, 0.884533026, 0.88787234, 0.886071297, 0.887532285, 0.874334601, 0.878553685, 0.882252142, 0.889397255, 0.878754171, 0.87609293, 0.883552801, 0.885557032, 0.836985101, 0.883209746, 0.884635692, 0.885600707, 0.878287177, 0.88589398, 0.866447867, 0.875205255, 0.883646683, 0.879831611, 0.881463628, 0.87953438, 0.863648329, 0.884519351, 0.881635143, 0.865125241, 0.887080868, 0.876517516, 0.874955468, 0.885282184, 0.887091757, 0.874358974, 0.886126305, 0.88751926, 0.881293642, 0.881343556, 0.883418223, 0.876833656]
    # buckets = tf.constant(buckets, dtype=tf.float32) # (240,)
    # max_value = tf.ones_like(duration_s, dtype=tf.int32) * 239
    # duration_s = tf.cast(duration_s, dtype=tf.int32)
    # indices = tf.where(duration_s > 239, max_value, duration_s)
    # next_rate = tf.cast(tf.gather(buckets, indices), dtype=tf.float32) # (cand_size, 1)
    # base_next_weight = 10.0 / next_rate - 9.7
    return base_ltr_label, base_lvtr_label, base_finish_label, None

def get_play_labels(duration, play_time):
    # boundaries = [0.0, 3.0, 9.0, 14.0, 22.0, 40.0, 67.0, 165.0]
    # thresholds = [[7.0, 18.0], [7.0, 18.0], [7.0, 18.0], [7.0, 18.0], [7.0, 20.0], [7.0, 28.0], [7.0, 27.0], [7.0, 23.0]]
    # boundaries = [0.0, 9.0, 14.0, 22.0, 40.0, 67.0, 165.0]
    # thresholds = [[14.0, 18.0], [10.0, 18.0], [11.0, 18.0], [12.0, 20.0], [14.0, 28.0], [13.0, 27.0], [9.0, 23.0]]
    # 0506版本，有效播放50%分位数，长播65分位数
    boundaries = [0, 8.366, 10.3, 12.433, 15.066, 17.3, 20.431, 24.833, 29.333, 33.916, 39.033, 46.566, 54.7, 62.933, 76.366, 99.166, 178.266, 235, 360.433, 1108.266]
    thresholds = [[4.529, 8.124], [8.56, 10.781], [10.154, 11.684], [11.228, 12.894], [12.009, 14.768], [13.51, 16.661], [13.406, 18.912], [13.038, 21.209], [14.57, 25.462], [15.108, 28.599], [16.205, 31.515], [17.891, 35.461], [18.748, 38.867], [18.451, 41.105], [19.012, 44.266], [17.148, 43.874], [15.472, 44.873], [13.181, 38.552], [10.074, 27.184], [8.925, 19.689], [9.554, 19.591]]
    boundaries_tensor = tf.constant(boundaries, dtype=tf.float32)
    boundaries_tensor = tf.tile(tf.expand_dims(boundaries_tensor, axis=0), [tf.shape(duration)[0], 1])
    bucket_idx = tf.searchsorted(sorted_sequence=boundaries_tensor, values=tf.cast(duration, tf.float32), side="left") # 左开右闭
    max_idx = tf.constant(len(thresholds) - 1, dtype=tf.int32)
    bucket_idx = tf.clip_by_value(bucket_idx, 0, max_idx)
    evtr_threshold = tf.gather(tf.constant([t[0] for t in thresholds]), bucket_idx)
    lvtr_threshold = tf.gather(tf.constant([t[1] for t in thresholds]), bucket_idx)
    
    # 判断播放时长是否达到阈值
    evtr_label = tf.cast(tf.greater_equal(play_time, evtr_threshold), dtype=tf.float32)
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
# wtd_buckets_fountain = tf.constant(wtd_config_fountain["buckets"], dtype=tf.float32)
# wtd_configs_fountain = tf.ragged.constant(wtd_config_fountain["configs"], dtype=tf.float32)
# list_wtd_buckets_fountain = tf.constant(list_wtd_config_fountain["buckets"], dtype=tf.float32)
# list_wtd_configs_fountain = tf.ragged.constant(list_wtd_config_fountain["configs"], dtype=tf.float32)
# 获取模型output
all_param_dict, feature_emb_size_dict, _ = get_param_dict()
label_value_dict = {}
# label
label_value_dict["context_info__first_screen"] = tf.cast(tf.reshape(config.get_dense_fea("context_info__first_screen", dim=1, dtype=tf.int64), [-1, 1]), dtype=tf.float32)
if is_training:
    all_param_dict["context_info__page"] = tf.cast(tf.reshape(config.get_dense_fea("context_info__page", dim=1, dtype=tf.int64), [-1, 1]), dtype=tf.float32)
else:
    all_param_dict["context_info__page"] = tf.cast(tf.reshape(config.get_extra_param("context_info__page", size=1, default_value=0, common=True), [-1, 1]), dtype=tf.float32)
label_value_dict["context_info__real_show_index"] = tf.cast(tf.reshape(config.get_dense_fea("context_info__real_show_index", dim=CANDIDATES_SIZE, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["show_label"] = tf.cast(tf.reshape(config.get_dense_fea("context_info__real_show_list", dim=CANDIDATES_SIZE, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["play_time_s"] = tf.cast(tf.reshape(config.get_dense_fea("context_info__playing_time_list", dim=CANDIDATES_SIZE, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32) / 1000.0
label_value_dict["play_time_s"] = tf.clip_by_value(label_value_dict["play_time_s"], 0, 36000)
label_value_dict["like_label"] = tf.cast(tf.reshape(config.get_dense_fea("context_info__like_list", dim=CANDIDATES_SIZE, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["follow_label"] = tf.cast(tf.reshape(config.get_dense_fea("context_info__follow_list", dim=CANDIDATES_SIZE, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["comment_label"] = tf.cast(tf.reshape(config.get_dense_fea("context_info__comment_list", dim=CANDIDATES_SIZE, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["forward_label"] = tf.cast(tf.reshape(config.get_dense_fea("context_info__forward_list", dim=CANDIDATES_SIZE, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["slide_label"] = tf.cast(tf.reshape(config.get_dense_fea("context_info__fountain_slide_to_next_list", dim=CANDIDATES_SIZE, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["click_label"] = tf.cast(tf.reshape(config.get_label("fountain_click_label_list", dim=CANDIDATES_SIZE), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["wtd_label"] = tf.cast(tf.reshape(config.get_label("fountain_wtd_label_list", dim=CANDIDATES_SIZE), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["finish_label"] = tf.cast(tf.reshape(config.get_label("fountain_finish_label_list", dim=CANDIDATES_SIZE), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["ltr_label"] = tf.cast(tf.reshape(config.get_label("fountain_ltr_label_list", dim=CANDIDATES_SIZE), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
# dense feature
dense_dim = CANDIDATES_SIZE if is_training else 1
# label_value_dict["pctr"] = tf.cast(tf.reshape(config.get_label("context_info__pctr_list", dim=dense_dim), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
# label_value_dict["plvtr"] = tf.cast(tf.reshape(config.get_label("context_info__plvtr_list", dim=dense_dim), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
# label_value_dict["pvtr"] = tf.cast(tf.reshape(config.get_label("context_info__pvtr_list", dim=dense_dim), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["pwtd"] = tf.cast(tf.reshape(config.get_label("context_info__pwtd_list", dim=dense_dim), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
# label_value_dict["pltr"] = tf.cast(tf.reshape(config.get_label("context_info__pltr_list", dim=dense_dim), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
# label_value_dict["pcmtr"] = tf.cast(tf.reshape(config.get_label("context_info__pcmtr_list", dim=dense_dim), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
# label_value_dict["pwtr"] = tf.cast(tf.reshape(config.get_label("context_info__pwtr_list", dim=dense_dim), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["photo_info__duration_ms_list"] = tf.cast(tf.reshape(config.get_dense_fea("photo_info__duration_ms_list", dim=dense_dim, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32)
label_value_dict["photo_info__duration_ms_list"] = tf.clip_by_value(label_value_dict["photo_info__duration_ms_list"], 0, 36000 * 1000)


point_wise_tasks = ["ltr", "vtr", "click", "wtd"]
# list_wise_tasks = ["listwise_wtd"]
list_wise_tasks = []
model_class = EvaluatorModel(all_param_dict, label_value_dict, print_ops, list_size=LIST_SIZE, candidates_size=CANDIDATES_SIZE, list_num=LIST_NUM,
                             point_wise_tasks=point_wise_tasks, list_wise_tasks=list_wise_tasks)

if is_training:
    batch_size = tf.shape(label_value_dict["show_label"])[0]
    zeros = tf.zeros([batch_size, 1], dtype=tf.float32)
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
    # rerank_list_item_idx_flat_list_mask = tf.equal(rerank_list_item_idx_flat_list, real_show_rerank_indices) # (? , LIST_NUM, LIST_SIZE)
    # rerank_list_mask = tf.cast(tf.reduce_all(rerank_list_item_idx_flat_list_mask, axis=-1), tf.float32) # (?, LIST_NUM)
    max_score_list_index = tf.argmax(rerank_list_score_list, axis=-1, output_type=tf.int32)
    false_mask = tf.fill([batch_size, LIST_NUM], False)
    batch_indices = tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), [1, LIST_NUM])
    mask_indices = tf.stack([tf.range(batch_size), max_score_list_index], axis=1)
    rerank_list_mask = tf.tensor_scatter_nd_update(false_mask, mask_indices, tf.fill([batch_size], True)) # (?, LIST_NUM)
    rerank_list_mask = tf.cast(rerank_list_mask, tf.float32) # (?, LIST_NUM)

    model_class._training=True
    point_wise_output_dict, list_wise_output_dict = model_class.model(list_index=rerank_list_item_idx_flat_list, selected_list_index=max_score_list_index)
    print(f"====> train, gen...")
    show_label = label_value_dict["show_label"]
    context_pwtd = label_value_dict["pwtd"]
    play_time_s = label_value_dict["play_time_s"]
    duration_s = label_value_dict["photo_info__duration_ms_list"] / 1000
    wtd_label = wtd_encode(duration=duration_s, play_time=play_time_s, duration_bucket=wtd_buckets, play_time_buckets_ragged=wtd_configs)
    # vtr_label = label_value_dict["wtd_label"]
    vtr_label = wt_encode(playtime_s=play_time_s, duration_s=duration_s)
    click_label = label_value_dict["click_label"]
    like_label = label_value_dict["like_label"]
    follow_label = label_value_dict["follow_label"]
    comment_label = label_value_dict["comment_label"]
    forward_label = label_value_dict["forward_label"]
    ltr_label = label_value_dict["ltr_label"]
    slide_label = label_value_dict["slide_label"]
    finish_label = label_value_dict["finish_label"]
    is_first_screen = label_value_dict["context_info__first_screen"]
    evtr_label, lvtr_label, svtr_label = get_play_labels(duration_s, play_time_s)
    base_ltr_label, base_lvtr_label, base_finish_label, base_next_weight = get_base_label(duration_s, play_time_s)
    click_label2 = tf.where(tf.logical_and(tf.equal(duration_s, 0.0), tf.greater_equal(play_time_s, 3.0)), tf.ones_like(click_label), click_label)
    wt_weight = tf.math.log(1.0 + tf.clip_by_value(play_time_s, 0, 120)) / tf.math.log(2.0)
    linear_wt_weight = tf.clip_by_value(play_time_s, 0, 120) * 0.035
    effective_action = tf.cast(tf.greater((1 - svtr_label) + like_label + follow_label + forward_label + comment_label, 0.0), tf.float32)
    is_miss = effective_action * tf.cast(tf.equal(play_time_s, 0.0), tf.float32)
    normal_value = tf.minimum(play_time_s / 3.0, 80.0)
    scale_wt = tf.where(tf.equal(is_miss, 1.0), tf.fill(tf.shape(play_time_s), 2.3), normal_value)
    evtr_weight = 1.0 + scale_wt * 0.5 + (1 - svtr_label) * 1.0
    base_ltr_weight = 1.0 + slide_label * 2.0 + base_finish_label * 10.0

    action_once = tf.where((like_label + follow_label + comment_label + forward_label) > 0.0,  tf.ones_like(like_label, tf.bool), tf.zeros_like(like_label, tf.bool))
    negative_label = tf.where(tf.logical_or(tf.less(play_time_s, 3.0), tf.less(slide_label, 1.0)), tf.ones_like(like_label), tf.zeros_like(like_label))
    negative_label = tf.cast(negative_label, tf.float32)
    reward_label = tf.ones_like(like_label)
    reward_label = tf.where(tf.greater(evtr_label, 0.0), tf.ones_like(reward_label) * 2.0, reward_label)
    reward_label = tf.where(tf.greater(base_ltr_label, 0.0), tf.ones_like(reward_label) * 3.0, reward_label)
    reward_label = tf.where(action_once, tf.ones_like(reward_label) * 4.0, reward_label)
    reward_label = tf.where(tf.logical_and(tf.greater(evtr_label, 0.0), action_once), tf.ones_like(reward_label) * 5.0, reward_label)
    reward_label = tf.where(tf.greater(base_finish_label, 0.0), tf.ones_like(reward_label) * 6.0, reward_label)
    reward_label = tf.where(tf.logical_and(tf.greater(base_ltr_label, 0.0), action_once), tf.ones_like(reward_label) * 7.0, reward_label)
    reward_label = tf.where(tf.logical_and(tf.greater(base_finish_label, 0.0), action_once), tf.ones_like(reward_label) * 8.0, reward_label)
    reward_label = tf.where(tf.greater(negative_label, 0.0), tf.zeros_like(like_label), reward_label)

    reward_weight = tf.ones_like(like_label)

    advantage_reward = play_time_s + slide_label * 3.0 + like_label * 40.0 + follow_label * 100.0 + comment_label * 100.0 + forward_label * 100
    advantage_reward = tf.where(advantage_reward > 0.0, advantage_reward, tf.zeros_like(advantage_reward, dtype=tf.float32))
    advantage = cal_batch_advantage(advantage_reward, mask=show_label)
    advantage = tf.where(advantage > 0.0, advantage + 1.0, tf.ones_like(advantage, dtype=tf.float32))

    # 只 gather max_score list 对应的 label，与 PLE 输出对齐 (?, list_size)
    # max_score_list_index: (?,) int32，gather_nd 需要 (?, 2) 索引
    sel_label_idx = tf.expand_dims(max_score_list_index, axis=1)  # (?, 1)

    def gather_selected_list(flat_label):
        # flat_label: (?, CANDIDATES_SIZE)ï¼concat [0] 后变 (?, CANDIDATES_SIZE+1)，再按 selected list 的 index gather
        padded = tf.concat([zeros, flat_label], axis=-1)  # (?, CANDIDATES_SIZE+1)
        # rerank_list_item_idx_flat_list: (?, LIST_NUM, LIST_SIZE)
        all_list_labels = tf.gather(padded, rerank_list_item_idx_flat_list, axis=1, batch_dims=1)  # (?, LIST_NUM, LIST_SIZE)
        return tf.squeeze(tf.gather(all_list_labels, sel_label_idx, axis=1, batch_dims=1), axis=1)  # (?, LIST_SIZE)

    sel_show_label     = gather_selected_list(show_label)
    sel_context_pwtd   = gather_selected_list(context_pwtd)
    sel_play_time_s    = gather_selected_list(play_time_s)
    sel_duration_s     = gather_selected_list(duration_s)
    sel_vtr_label      = gather_selected_list(vtr_label)
    sel_wtd_label      = gather_selected_list(wtd_label)
    sel_click_label2   = gather_selected_list(click_label2)
    sel_slide_label    = gather_selected_list(slide_label)
    sel_evtr_weight    = gather_selected_list(evtr_weight)
    sel_advantage      = gather_selected_list(advantage)

    # 用 show_label 做样本 mask
    item_mask = sel_show_label  # (?, LIST_SIZE)

    # 反解 vtr -> watch time，用于监控
    # sel_vtr_wt = get_watch_time_from_vtr(point_wise_output_dict["vtr"], tf.cast(sel_duration_s, tf.int32))
    sel_vtr_wt = wt_decode(point_wise_output_dict["vtr"], tf.cast(sel_duration_s, tf.int32))
    # sel_vtr_max_wt = get_watch_time_from_vtr(tf.ones_like(sel_play_time_s), tf.cast(sel_duration_s, tf.int32))
    # sel_vtr_scale：归一化尺度（buckets[duration_s]），不是物理上界，play_time 允许超过此值
    sel_vtr_scale = wt_decode(tf.ones_like(sel_play_time_s), tf.cast(sel_duration_s, tf.int32))
    continue_pred = list_wise_output_dict["continue"]
    p_reach = tf.concat([tf.ones_like(continue_pred[:, :1]), tf.cumprod(continue_pred, axis=1)], axis=1)
    expected_vtr_wt = tf.reduce_sum(p_reach * sel_vtr_wt, axis=-1, keepdims=True)
    list_play_time_s = tf.reduce_sum(sel_play_time_s * item_mask, axis=-1, keepdims=True)
    continue_mask = item_mask[:, :-1] * item_mask[:, 1:]
    pair_mask = item_mask[:, :-1] * item_mask[:, 1:]
    tri_mask = item_mask[:, :-2] * item_mask[:, 1:-1] * item_mask[:, 2:]
    pair_scale = sel_vtr_scale[:, :-1] + sel_vtr_scale[:, 1:]
    tri_scale = sel_vtr_scale[:, :-2] + sel_vtr_scale[:, 1:-1] + sel_vtr_scale[:, 2:]
    pair_play_time_s = sel_play_time_s[:, :-1] + sel_play_time_s[:, 1:]
    tri_play_time_s = sel_play_time_s[:, :-2] + sel_play_time_s[:, 1:-1] + sel_play_time_s[:, 2:]
    pair_vtr_label = get_vtr_from_watch_time(pair_play_time_s - tf.stop_gradient(sel_vtr_wt[:, :-1] + sel_vtr_wt[:, 1:]), pair_scale)
    tri_vtr_label = get_vtr_from_watch_time(tri_play_time_s - tf.stop_gradient(sel_vtr_wt[:, :-2] + sel_vtr_wt[:, 1:-1] + sel_vtr_wt[:, 2:]), tri_scale)
    # clip 上界放宽为 scale * 3.0，允许 play_time > scale（循环播放等场景）
    pair_pvtr_wt = tf.clip_by_value(sel_vtr_wt[:, :-1] + sel_vtr_wt[:, 1:] + list_wise_output_dict["pair_vtr_gain"] * pair_scale, 0.0, pair_scale * 3.0)
    tri_pvtr_wt = tf.clip_by_value(sel_vtr_wt[:, :-2] + sel_vtr_wt[:, 1:-1] + sel_vtr_wt[:, 2:] + list_wise_output_dict["tri_vtr_gain"] * tri_scale, 0.0, tri_scale * 3.0)

    list_pvtr_wt = tf.reduce_sum(sel_vtr_wt * item_mask, axis=-1, keepdims=True)  # (?, 1)
    pair_item_wt = tf.concat([sel_vtr_wt[:, :1], pair_pvtr_wt - sel_vtr_wt[:, :-1]], axis=1)  # (?, LIST_SIZE)
    list_pair_pvtr_wt = tf.reduce_sum(pair_item_wt * item_mask, axis=-1, keepdims=True)  # (?, 1)
    tri_item_wt = tf.concat([sel_vtr_wt[:, :2], tri_pvtr_wt - sel_vtr_wt[:, :-2] - sel_vtr_wt[:, 1:-1]], axis=1)  # (?, LIST_SIZE)
    list_tri_pvtr_wt = tf.reduce_sum(tri_item_wt * item_mask, axis=-1, keepdims=True)  # (?, 1)

    with tf.control_dependencies(print_ops):
        # loss calc：PLE 输出已对齐到 selected list，shape 均为 (?, LIST_SIZE)
        targets = []
        sum_loss = 0.0
        for loss_name in point_wise_output_dict:
            output = point_wise_output_dict[loss_name]  # (?, LIST_SIZE)
            print(loss_name, output)

            if loss_name == "ltr":
                loss = tf.losses.log_loss(labels=sel_click_label2, predictions=output, weights=item_mask * sel_evtr_weight)
                targets.append((loss_name, output, sel_click_label2, item_mask, "auc"))
            elif loss_name == "vtr":
                loss = tf.losses.huber_loss(labels=sel_vtr_label, predictions=output, weights=item_mask, delta=0.05)
                loss = loss * 500.0
                # 绝对时长校准：log 空间 huber，直接对 vtr 反解后的秒数做端到端监督
                vtr_abs_loss = tf.losses.huber_loss(labels=tf.log(sel_play_time_s + 1.0), predictions=tf.log(sel_vtr_wt + 1.0), weights=item_mask, delta=0.2) * 20
                loss = loss + vtr_abs_loss
                tf.summary.scalar('loss_vtr_abs', vtr_abs_loss)
                targets.append((loss_name, output, sel_vtr_label, item_mask, "linear_regression"))
                targets.append(("list_vtr_wt", sel_vtr_wt, sel_play_time_s, item_mask, "linear_regression"))
            elif loss_name == "click":
                loss = tf.losses.log_loss(labels=sel_click_label2, predictions=output, weights=item_mask)
                targets.append((loss_name, output, sel_click_label2, item_mask, "auc"))

            sum_loss += loss
            tf.summary.scalar('loss_' + loss_name, loss)
        continue_loss = focal_loss_with_neg_upsampling(labels=sel_slide_label[:, :-1], predictions=continue_pred, weights=continue_mask, gamma=2.0, neg_weight=1.0)
        expected_wt_huber_loss = tf.losses.huber_loss(labels=tf.log(list_play_time_s + 1.0), predictions=tf.log(expected_vtr_wt + 1.0), weights=tf.ones_like(list_play_time_s), delta=0.2) * 0.5
        pair_vtr_loss = tf.losses.huber_loss(labels=pair_vtr_label, predictions=list_wise_output_dict["pair_vtr_gain"], weights=pair_mask, delta=0.1)
        tri_vtr_loss = tf.losses.huber_loss(labels=tri_vtr_label, predictions=list_wise_output_dict["tri_vtr_gain"], weights=tri_mask, delta=0.1)
        # 绝对时长校准 loss：log 空间 huber，直接对 pair/tri 的还原秒数做端到端监督
        # 与 gain ratio loss 不等价（分母为常数 1 而非 pair_max_wt），能改善跨 request 绝对量级对比
        pair_abs_loss = tf.losses.huber_loss(labels=tf.log(pair_play_time_s + 1.0), predictions=tf.log(pair_pvtr_wt + 1.0), weights=pair_mask, delta=0.2) * 0.5
        tri_abs_loss = tf.losses.huber_loss(labels=tf.log(tri_play_time_s + 1.0), predictions=tf.log(tri_pvtr_wt + 1.0), weights=tri_mask, delta=0.2) * 0.25
        sum_loss += continue_loss + expected_wt_huber_loss + pair_vtr_loss + pair_abs_loss + tri_vtr_loss + tri_abs_loss
        tf.summary.scalar('loss_continue', continue_loss)
        tf.summary.scalar('expected_wt_huber_loss', expected_wt_huber_loss)
        tf.summary.scalar('loss_pair_vtr_gain', pair_vtr_loss)
        tf.summary.scalar('loss_pair_abs', pair_abs_loss)
        tf.summary.scalar('loss_tri_vtr_gain', tri_vtr_loss)
        tf.summary.scalar('loss_tri_abs', tri_abs_loss)
        targets.append(("continue", continue_pred, sel_slide_label[:, :-1], continue_mask, "auc"))
        targets.append(("pair_pvtr_wt", pair_pvtr_wt, pair_play_time_s, pair_mask, "linear_regression"))
        targets.append(("tri_pvtr_wt", tri_pvtr_wt, tri_play_time_s, tri_mask, "linear_regression"))
        # 三种目标换算为 list 整体预估时长 vs 真实时长，量纲统一，AUC 可比
        targets.append(("list_wt_from_pair_pvtr", list_pair_pvtr_wt, list_play_time_s, tf.ones_like(list_play_time_s), "linear_regression"))
        targets.append(("list_wt_from_tri_pvtr", list_tri_pvtr_wt, list_play_time_s, tf.ones_like(list_play_time_s), "linear_regression"))
        targets.append(("list_wt_from_pvtr", list_pvtr_wt, list_play_time_s, tf.ones_like(list_play_time_s), "linear_regression"))
        targets.append(("expected_vtr_wt", expected_vtr_wt, list_play_time_s, tf.ones_like(list_play_time_s), "linear_regression"))
        targets.append(("expected_context_pwtd", tf.reduce_sum(sel_context_pwtd * item_mask, axis=-1, keepdims=True), list_play_time_s, tf.ones_like(list_play_time_s), "linear_regression"))
        targets.append(("list_context_pwtd", sel_context_pwtd, sel_play_time_s, item_mask, "linear_regression"))

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
        sparse_optimizer = config.optimizer.Adam(0.0009)
        dense_optimizer = config.optimizer.Adam(0.0003)
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
    point_wise_output_dict, list_wise_output_dict = model_class.model(rerank_list_item_idx_flat_list)
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
    duration_s_item = tf.reshape(tf.cast(list_duration_s, tf.int32), [-1, LIST_SIZE])
    duration_s_flat = tf.reshape(duration_s_item, [-1, LIST_NUM * LIST_SIZE])
    # pvtr_wt_item = get_watch_time_from_vtr(point_wise_output_dict["vtr"], duration_s_item) # (?*LIST_NUM, LIST_SIZE)
    pvtr_wt_item = wt_decode(point_wise_output_dict["vtr"], duration_s_item) # (?*LIST_NUM, LIST_SIZE)
    pvtr_wt = tf.reshape(pvtr_wt_item, [-1, LIST_NUM * LIST_SIZE]) # (? , LIST_NUM * LIST_SIZE)
    pwtd_wt = wtd_decode(pwtd, duration_s_flat, duration_bucket=wtd_buckets, play_time_buckets_ragged=wtd_configs) # (? , LIST_NUM * LIST_SIZE)
    rerank_list_item_idx_flat_list_print = tf.reshape(rerank_list_item_idx_flat_list, [-1, LIST_NUM * LIST_SIZE])

    continue_pred = tf.reshape(list_wise_output_dict["continue"], [-1, LIST_NUM * (LIST_SIZE - 1)])
    # vtr_max_wt_item = get_watch_time_from_vtr(tf.ones_like(point_wise_output_dict["vtr"]), duration_s_item)
    vtr_max_wt_item = wt_decode(tf.ones_like(point_wise_output_dict["vtr"]), duration_s_item)
    pair_scale_item = vtr_max_wt_item[:, :-1] + vtr_max_wt_item[:, 1:]
    tri_scale_item = vtr_max_wt_item[:, :-2] + vtr_max_wt_item[:, 1:-1] + vtr_max_wt_item[:, 2:]
    pair_pvtr_wt_item = tf.clip_by_value(pvtr_wt_item[:, :-1] + pvtr_wt_item[:, 1:] + list_wise_output_dict["pair_vtr_gain"] * pair_scale_item, 0.0, pair_scale_item * 3.0)  # (?*LIST_NUM, LIST_SIZE-1)
    tri_pvtr_wt_item = tf.clip_by_value(pvtr_wt_item[:, :-2] + pvtr_wt_item[:, 1:-1] + pvtr_wt_item[:, 2:] + list_wise_output_dict["tri_vtr_gain"] * tri_scale_item, 0.0, tri_scale_item * 3.0)  # (?*LIST_NUM, LIST_SIZE-2)
    pair_pvtr_wt = tf.reshape(pair_pvtr_wt_item, [-1, LIST_NUM * (LIST_SIZE - 1)])
    tri_pvtr_wt = tf.reshape(tri_pvtr_wt_item, [-1, LIST_NUM * (LIST_SIZE - 2)])
    # 换算为 list 整体预估时长：item[0]=pvtr_wt[0], item[i+1]=pair[i]-pvtr_wt[i]
    pair_item_wt = tf.concat([pvtr_wt_item[:, :1], pair_pvtr_wt_item - pvtr_wt_item[:, :-1]], axis=1)  # (?*LIST_NUM, LIST_SIZE)
    list_wt_from_pair_pvtr = tf.reshape(tf.reduce_sum(pair_item_wt, axis=-1), [-1, LIST_NUM])  # (?, LIST_NUM)
    # 换算为 list 整体预估时长：item[0,1]=pvtr_wt[0,1], item[i+2]=tri[i]-pvtr_wt[i]-pvtr_wt[i+1]
    tri_item_wt = tf.concat([pvtr_wt_item[:, :2], tri_pvtr_wt_item - pvtr_wt_item[:, :-2] - pvtr_wt_item[:, 1:-1]], axis=1)  # (?*LIST_NUM, LIST_SIZE)
    list_wt_from_tri_pvtr = tf.reshape(tf.reduce_sum(tri_item_wt, axis=-1), [-1, LIST_NUM])  # (?, LIST_NUM)
    p_reach_item = tf.concat([tf.ones_like(list_wise_output_dict["continue"][:, :1]), tf.cumprod(list_wise_output_dict["continue"], axis=1)], axis=1)
    expected_pvtr_wt = tf.reshape(tf.reduce_sum(p_reach_item * pair_item_wt, axis=-1), [-1, LIST_NUM])
    expected_ltr = tf.reduce_sum(p_reach_item * tf.reshape(pltr, [-1, LIST_NUM, LIST_SIZE]), axis=-1)  # (1, list_num)
    expected_pctr = tf.reduce_sum(p_reach_item * tf.reshape(pctr, [-1, LIST_NUM, LIST_SIZE]), axis=-1)  # (1, list_num)
    # expected_like = tf.reduce_sum(p_reach * tf.reshape(context_pltr, [-1, LIST_NUM, LIST_SIZE]), axis=-1)  # (1, list_num)
    expected_like = tf.reduce_sum(tf.reshape(context_pltr, [-1, LIST_NUM, LIST_SIZE]), axis=-1)  # (1, list_num)
    # expected_comment = tf.reduce_sum(p_reach * tf.reshape(context_pcmtr, [-1, LIST_NUM, LIST_SIZE]), axis=-1)  # (1, list_num)
    expected_comment = tf.reduce_sum(tf.reshape(context_pcmtr, [-1, LIST_NUM, LIST_SIZE]), axis=-1)  # (1, list_num)
    # expected_follow = tf.reduce_sum(p_reach * tf.reshape(context_pwtr, [-1, LIST_NUM, LIST_SIZE]), axis=-1)  # (1, list_num)
    expected_follow = tf.reduce_sum(tf.reshape(context_pwtr, [-1, LIST_NUM, LIST_SIZE]), axis=-1)  # (1, list_num)

    targets = []
    targets.append((f"pctr", pctr))
    # targets.append((f"pslide", pslide))
    targets.append((f"pwtd", pwtd_wt))
    targets.append((f"pvtr", pvtr_wt))
    targets.append((f"pltr", pltr))
    targets.append((f"continue", continue_pred))
    targets.append((f"pair_pvtr_wt", pair_pvtr_wt))
    targets.append((f"tri_pvtr_wt", tri_pvtr_wt))
    targets.append((f"expected_pvtr_wt_v2", list_wt_from_pair_pvtr))
    targets.append((f"expected_pvtr_wt_v3", list_wt_from_tri_pvtr))
    targets.append((f"expected_pvtr_wt", expected_pvtr_wt))
    targets.append((f"expected_ltr", expected_ltr))
    targets.append((f"expected_pctr", expected_pctr))
    targets.append((f"expected_like", expected_like))
    targets.append((f"expected_comment", expected_comment))
    targets.append((f"expected_follow", expected_follow))
    targets.append((f"context_pctr", context_pctr))
    targets.append((f"context_pwtd", context_pwtd))
    targets.append((f"context_pltr", context_pltr))
    targets.append((f"context_pcmtr", context_pcmtr))
    targets.append((f"context_pwtr", context_pwtr))
    targets.append((f"context_pftr", context_pftr))
    targets.append((f"context_plvtr", context_plvtr))
    targets.append((f"context_psvtr", context_psvtr))
    # targets.append((f"context_lv_count", context_lv_count))
    # targets.append((f"context_sv_count", context_sv_count))
    targets.append((f"duration_s", duration_s))
    targets.append((f"rerank_list_item_idx_flat_list_print", rerank_list_item_idx_flat_list_print))

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