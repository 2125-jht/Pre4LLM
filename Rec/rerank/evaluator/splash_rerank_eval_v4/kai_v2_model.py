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
parser.add_argument('--with_kai', default=False)
# parser.add_argument('--with_kai', default=True)
parser.add_argument('--text', default=False)
parser.add_argument('--tower', choices=None, dest='tower', default='False')
parser.add_argument('--with_kai_v2', default=True) #False True 
# parser.add_argument('--with_kai_v2', default=False) #False True 
args = parser.parse_known_args()[0]
is_training = args.mode == "train"

CANDIDATES_SIZE = 30
LIST_NUM = 50
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
            # rerank_list_item_idx_flat_list = tf.RaggedTensor.from_row_splits(batch["rerank_list_item_idx_flat_list"][0], row_splits= batch["rerank_list_item_idx_flat_list"][1])
            # rerank_list_item_idx_flat_list = rerank_list_item_idx_flat_list.to_tensor()
            # fountain_fulllink_rerank_index_list = tf.RaggedTensor.from_row_splits(batch["fountain_fulllink_rerank_index_list"][0], row_splits= batch["fountain_fulllink_rerank_index_list"][1])
            # fountain_fulllink_rerank_index_list = fountain_fulllink_rerank_index_list.to_tensor()

            context_page = batch['context_info__first_screen']
            print(f"realshow shape: {realshow.shape}")
            mask = tf.math.logical_or(
                tf.math.not_equal(context_page, 1),
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
            feature_emb_size_dict[attr.attr_name] = 1
        else:
            # if not attr.expand:
            #     attr.expand = 1
            # if attr.attr_name in photo_fea_names + ["photo_id_v2"]:
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

def pairwise_bpr_loss_v2(output, score, threshold, mask):
    # 生成配对矩阵
    output_i = tf.expand_dims(output, 2)
    output_j = tf.expand_dims(output, 1)
    score_i = tf.expand_dims(score, 2)
    score_j = tf.expand_dims(score, 1)
    score_diff = score_i - score_j
    pairwise_labels = tf.cast(score_diff >= threshold, tf.float32)
    pairwise_label_mask = tf.logical_or(score_diff > threshold, score_diff < -threshold) # 在阈值内的pair不计算loss
    logit_diff = tf.sigmoid(output_i - output_j)
    # 生成有效掩码
    mask_i = tf.expand_dims(mask, 2)
    mask_j = tf.expand_dims(mask, 1)
    valid_pair_mask = tf.logical_and(mask_i, mask_j)
    # 生成混合掩码
    random_mask = tf.random.uniform(tf.shape(valid_pair_mask)) < 0.05
    activated_mask = tf.logical_and(random_mask, tf.logical_not(valid_pair_mask))
    final_mask = tf.logical_or(valid_pair_mask, activated_mask)
    final_mask = tf.logical_and(final_mask, pairwise_label_mask)
    # 计算BPR损失
    bpr_loss = -tf.log(logit_diff) * pairwise_labels
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

def get_play_labels(duration, play_time):
    # boundaries = [0.0, 7.0, 14.0, 24.0, 53.0, 71.0]
    # 有效播、长播
    # thresholds = [[14.0, 18.0], [9.0, 14.0], [11.0, 18.0], [13.0, 24.0], [12.0, 27.0], [7.0, 21.0]]
    boundaries = [0.0, 3.0, 9.0, 14.0, 24.0, 53.0, 71.0]
    thresholds = [[9.0, 20.0], [7.0, 18.0], [7.0, 18.0], [7.0, 18.0], [7.0, 18.0], [7.0, 18.0]]
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
# 获取模型output
all_param_dict, feature_emb_size_dict, _ = get_param_dict()
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
label_value_dict["photo_info__duration_ms_list"] = tf.cast(tf.reshape(config.get_dense_fea("photo_info__duration_ms_list", dim=dense_dim, dtype=tf.int64), [-1, CANDIDATES_SIZE]), dtype=tf.float32)

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

# 需要 check infer 时传进来的 shape
if not is_training:
    duration_s = tf.reshape(config.get_extra_param("duration_ms_infer", size=1, dtype=tf.float32), [-1, CANDIDATES_SIZE]) / 1000.0 # (?, CANDIDATES_SIZE)
    print("duration_s ", tf.reshape(duration_s, [-1, CANDIDATES_SIZE]))
    batch_size = tf.shape(duration_s)[0]
    rerank_list_item_idx_flat_list = config.get_extra_param("rerank_list_item_idx_flat_list_double", size=LIST_NUM * LIST_SIZE, default_value=-1.0, common=True) + 1.0
    rerank_list_item_idx_flat_list = tf.cast(tf.reshape(rerank_list_item_idx_flat_list, [-1, LIST_NUM, LIST_SIZE]), tf.int32)
    print("rerank_list_item_idx_flat_list ", rerank_list_item_idx_flat_list)
    label_value_dict['rerank_list_item_idx_flat_list'] = rerank_list_item_idx_flat_list
    zeros = tf.zeros_like(duration_s, dtype=tf.float32)
    print("zeros ", zeros)

# point_wise_tasks = ["ltr", "vtr", "click", "label_30s", "label_17s", "wtd_level"]
point_wise_tasks = ["ltr", "vtr", "click", "wtd"]
list_wise_tasks = []
model_class = EvaluatorModel(all_param_dict, label_value_dict, print_ops, list_size=LIST_SIZE, candidates_size=CANDIDATES_SIZE, list_num=LIST_NUM,
                             point_wise_tasks=point_wise_tasks, list_wise_tasks=list_wise_tasks)
pred_dict = {}


if is_training:
    batch_size = tf.shape(label_value_dict["pwtd"])[0]
    zeros = tf.zeros([batch_size, 1], dtype=tf.float32)

    model_class._training=True
    point_wise_output_dict, list_wise_output_dict = model_class.model()
    print(f"====> train, gen...")
    show_label = label_value_dict["show_label"]
    context_pwtd = label_value_dict["pwtd"]
    context_pctr = label_value_dict["pctr"]
    context_pvtr = label_value_dict["pvtr"]
    context_pltr = label_value_dict["pltr"]
    vtr_label = label_value_dict["wtd_label"]
    play_time_s = label_value_dict["play_time_s"]
    duration_s = label_value_dict["photo_info__duration_ms_list"] / 1000
    wtd_label = wtd_encode(duration=duration_s, play_time=play_time_s, duration_bucket=wtd_buckets, play_time_buckets_ragged=wtd_configs)
    click_label = label_value_dict["click_label"]
    like_label = label_value_dict["like_label"]
    follow_label = label_value_dict["forward_label"]
    comment_label = label_value_dict["comment_label"]
    forward_label = label_value_dict["forward_label"]
    ltr_label = label_value_dict["ltr_label"]
    slide_label = label_value_dict["slide_label"]
    finish_label = label_value_dict["finish_label"]
    evtr_label, lvtr_label = get_play_labels(duration_s, play_time_s)
    # click weight TODO:profileEnter*20.0 comment_stay*2.0
    wt_weight = tf.math.log(1.0 + tf.clip_by_value(play_time_s, 0, 120)) / tf.math.log(2.0)
    # evtr_weight = 1.0 + lvtr_label * 3.0 + like_label * 30.0 + follow_label * 200.0 + comment_label * 80.0 + forward_label * 150.0 \
    #     + slide_label * 5.0 + wt_weight + finish_label * 10.0
    # evtr_weight = 1.0 + lvtr_label * 5.0 + like_label * 30.0 + follow_label * 200.0 + comment_label * 150.0 + forward_label * 150.0 \
    #     + slide_label * 10.0
    evtr_weight = 1.0 + lvtr_label * 1.0 + finish_label * 10.0 + like_label * 250.0 + follow_label * 200.0 + comment_label * 200.0 + forward_label * 50.0
    lvtr_weight = 1.0 + like_label * 30.0 + follow_label * 200.0 + comment_label * 150.0 + forward_label * 150.0 \
        + slide_label * 10.0 + wt_weight * 0.5
    # 反解
    pvtr_wt = get_watch_time_from_vtr(point_wise_output_dict["vtr"], tf.cast(duration_s, dtype=tf.int32)) # (? , CANDIDATES_SIZE)
    pwtd_wt = wtd_decode(point_wise_output_dict["wtd"], duration_s, duration_bucket=wtd_buckets, play_time_buckets_ragged=wtd_configs) # (? , CANDIDATES_SIZE)

    # 时长向
    advantage_reward = play_time_s + slide_label * 0.0 + finish_label * 2.0 + like_label * 150.0 + follow_label * 150.0 + comment_label * 150.0 + forward_label * 100.0
    advantage_reward = tf.where(advantage_reward > 0.0, advantage_reward, tf.zeros_like(advantage_reward, dtype=tf.float32))
    advantage = cal_batch_advantage(advantage_reward, mask=show_label) # (?, list_size)
    advantage = tf.where(advantage > 0.0, advantage + 1.0, tf.ones_like(advantage, dtype=tf.float32))

    with tf.control_dependencies(print_ops):
        # adn loss calc
        targets = []
        sum_loss = 0.0
        for loss_name in point_wise_output_dict:
            output = point_wise_output_dict[loss_name]
            print(loss_name, output)

            if loss_name == "vtr":
                weight_with_mask = show_label # only realshow sample
                loss = tf.losses.huber_loss(labels=vtr_label, predictions=output, weights=weight_with_mask, reduction=tf.losses.Reduction.SUM, delta=0.05)
                loss = loss * 15.0
                targets.append((loss_name, output, vtr_label, weight_with_mask, "linear_regression"))
                targets.append(("pvtr_wt", pvtr_wt, play_time_s, weight_with_mask, "linear_regression"))
            elif loss_name == "wtd":
                weight_with_mask = show_label # only realshow sample
                loss = tf.losses.log_loss(labels=wtd_label, predictions=output, weights=weight_with_mask, reduction=tf.losses.Reduction.SUM)
                loss = loss * 15.0
                targets.append((loss_name, output, wtd_label, weight_with_mask, "auc"))
                targets.append(("pwtd_wt", pwtd_wt, play_time_s, weight_with_mask, "linear_regression"))
            elif loss_name == "fountain_finish":
                weight_with_mask = tf.ones_like(finish_label, dtype=tf.float32) * show_label # only realshow sample
                loss = tf.losses.huber_loss(labels=finish_label, weights=weight_with_mask, predictions=output, reduction=tf.losses.Reduction.SUM, delta=0.05)
                targets.append((loss_name + "reg", output, finish_label, weight_with_mask, "linear_regression"))
            elif loss_name == "ltr":
                weight_with_mask = show_label # only realshow sample
                # lvtr_weight = tf.where(lvtr_label > 0.0, lvtr_weight, tf.ones_like(lvtr_weight))
                evtr_weight = tf.where(evtr_label > 0.0, evtr_weight, tf.ones_like(evtr_weight))
                loss = tf.losses.log_loss(labels=evtr_label, predictions=output, weights=weight_with_mask, reduction=tf.losses.Reduction.SUM)
                #loss = loss / 100.0
                targets.append((loss_name, output, evtr_label, weight_with_mask, "auc"))
            elif loss_name == "slide":
                loss = tf.losses.log_loss(labels=slide_label, predictions=output, weights=show_label, reduction=tf.losses.Reduction.SUM)
                targets.append((loss_name, output, slide_label, show_label, "auc"))
            elif loss_name == "click":
                weight_with_mask = show_label # only realshow sample
                loss = tf.losses.log_loss(labels=click_label, predictions=output, weights=weight_with_mask * advantage, reduction=tf.losses.Reduction.SUM)
                targets.append((loss_name, output, click_label, weight_with_mask, "auc"))

            sum_loss += loss
            tf.summary.scalar('loss_' + loss_name, loss)
        targets.append(("context_pwtd", context_pwtd, play_time_s, show_label, "linear_regression"))
        # targets.append(("context_pvtr", context_pvtr, play_time_s, show_label, "linear_regression"))

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
        sparse_optimizer = config.optimizer.Adam(0.001)
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
    model_class._training = False
    point_wise_output_dict, list_wise_output_dict = model_class.model()
    context_info__pctr_infer = tf.reshape(config.get_extra_param("context_info__pctr_infer", size=1, dtype=tf.float32), [-1, CANDIDATES_SIZE]) # (?, CANDIDATES_SIZE)
    context_info__pwtd_infer = tf.reshape(config.get_extra_param("context_info__pwtd_infer", size=1, dtype=tf.float32), [-1, CANDIDATES_SIZE]) # (?, CANDIDATES_SIZE)
    context_info__pltr_infer = tf.reshape(config.get_extra_param("context_info__pltr_infer", size=1, dtype=tf.float32), [-1, CANDIDATES_SIZE]) # (?, CANDIDATES_SIZE)
    context_info__pcmtr_infer = tf.reshape(config.get_extra_param("context_info__pcmtr_infer", size=1, dtype=tf.float32), [-1, CANDIDATES_SIZE]) # (?, CANDIDATES_SIZE)
    context_info__pwtr_infer = tf.reshape(config.get_extra_param("context_info__pwtr_infer", size=1, dtype=tf.float32), [-1, CANDIDATES_SIZE]) # (?, CANDIDATES_SIZE)
    context_info__pftr_infer = tf.reshape(config.get_extra_param("context_info__pftr_infer", size=1, dtype=tf.float32), [-1, CANDIDATES_SIZE]) # (?, CANDIDATES_SIZE)
    context_info__plvtr_infer = tf.reshape(config.get_extra_param("context_info__plvtr_infer", size=1, dtype=tf.float32), [-1, CANDIDATES_SIZE]) # (?, CANDIDATES_SIZE)
    pctr = tf.reshape(point_wise_output_dict["click"], [-1, CANDIDATES_SIZE])
    pwtd = tf.reshape(point_wise_output_dict["wtd"], [-1, CANDIDATES_SIZE])
    pvtr = tf.reshape(point_wise_output_dict["vtr"], [-1, CANDIDATES_SIZE])
    pltr = tf.reshape(point_wise_output_dict["ltr"], [-1, CANDIDATES_SIZE])
    # pslide = tf.reshape(point_wise_output_dict["slide"], [-1, CANDIDATES_SIZE])
    duration_s = tf.cast(tf.reshape(duration_s, [-1, CANDIDATES_SIZE]), tf.int32)
    pwtd_wt = wtd_decode(pwtd, duration_s, duration_bucket=wtd_buckets, play_time_buckets_ragged=wtd_configs) # (? , CANDIDATES_SIZE)
    pvtr_wt = get_watch_time_from_vtr(pvtr, duration_s) # (? , CANDIDATES_SIZE)
    zeros = tf.tile(tf.zeros(shape=[1, 1]), [tf.shape(pctr)[0], 1]) # (?, 1)
    list_index = label_value_dict['rerank_list_item_idx_flat_list']
    rerank_list_item_idx_flat_list_print = tf.reshape(list_index, [-1, LIST_NUM * LIST_SIZE])
    pctr = tf.gather(tf.concat([zeros, pctr], axis=-1), list_index, axis=1, batch_dims=1) # (?, list_num, list_size)
    pctr = tf.reshape(pctr, [-1, LIST_NUM * LIST_SIZE])
    pwtd = tf.gather(tf.concat([zeros, pwtd], axis=-1), list_index, axis=1, batch_dims=1) # (?, list_num, list_size)
    pvtr = tf.gather(tf.concat([zeros, pvtr], axis=-1), list_index, axis=1, batch_dims=1) # (?, list_num, list_size)
    pvtr = tf.reshape(pvtr, [-1, LIST_NUM * LIST_SIZE])
    pltr = tf.gather(tf.concat([zeros, pltr], axis=-1), list_index, axis=1, batch_dims=1) # (?, list_num, list_size)
    pltr = tf.reshape(pltr, [-1, LIST_NUM * LIST_SIZE])
    # pslide = tf.gather(tf.concat([zeros, pslide], axis=-1), list_index, axis=1, batch_dims=1) # (?, list_num, list_size)
    # pslide = tf.reshape(pslide, [-1, LIST_NUM * LIST_SIZE])
    # print("pslide ", pslide)
    pvtr_wt = tf.gather(tf.concat([zeros, pvtr_wt], axis=-1), list_index, axis=1, batch_dims=1) # (?, list_num, list_size)
    pvtr_wt = tf.reshape(pvtr_wt, [-1, LIST_NUM * LIST_SIZE])
    pwtd_wt = tf.gather(tf.concat([zeros, pwtd_wt], axis=-1), list_index, axis=1, batch_dims=1) # (?, list_num, list_size)
    pwtd_wt = tf.reshape(pwtd_wt, [-1, LIST_NUM * LIST_SIZE])
    
    list_context_pctr = tf.gather(tf.concat([zeros, context_info__pctr_infer], axis=-1), list_index, axis=1, batch_dims=1) # (?, list_num, list_size)
    list_context_pwtd = tf.gather(tf.concat([zeros, context_info__pwtd_infer], axis=-1), list_index, axis=1, batch_dims=1) # (?, list_num, list_size)
    list_context_pltr = tf.gather(tf.concat([zeros, context_info__pltr_infer], axis=-1), list_index, axis=1, batch_dims=1) # (?, list_num, list_size)
    list_context_pcmtr = tf.gather(tf.concat([zeros, context_info__pcmtr_infer], axis=-1), list_index, axis=1, batch_dims=1) # (?, list_num, list_size)
    list_context_pwtr = tf.gather(tf.concat([zeros, context_info__pwtr_infer], axis=-1), list_index, axis=1, batch_dims=1) # (?, list_num, list_size)
    list_context_pftr = tf.gather(tf.concat([zeros, context_info__pftr_infer], axis=-1), list_index, axis=1, batch_dims=1) # (?, list_num, list_size)
    list_context_plvtr = tf.gather(tf.concat([zeros, context_info__plvtr_infer], axis=-1), list_index, axis=1, batch_dims=1) # (?, list_num, list_size)
    context_pctr = tf.reshape(list_context_pctr, [-1, LIST_NUM * LIST_SIZE])
    context_pwtd = tf.reshape(list_context_pwtd, [-1, LIST_NUM * LIST_SIZE])
    context_pltr = tf.reshape(list_context_pltr, [-1, LIST_NUM * LIST_SIZE])
    context_pcmtr = tf.reshape(list_context_pcmtr, [-1, LIST_NUM * LIST_SIZE])
    context_pwtr = tf.reshape(list_context_pwtr, [-1, LIST_NUM * LIST_SIZE])
    context_pftr = tf.reshape(list_context_pftr, [-1, LIST_NUM * LIST_SIZE])
    context_plvtr = tf.reshape(list_context_plvtr, [-1, LIST_NUM * LIST_SIZE])

    targets = []
    targets.append((f"pctr", pctr))
    targets.append((f"pwtd", pwtd_wt))
    targets.append((f"pvtr", pvtr_wt))
    targets.append((f"pltr", pltr))
    # targets.append((f"pslide", pslide))
    targets.append((f"context_pctr", context_pctr))
    targets.append((f"context_pwtd", context_pwtd))
    targets.append((f"context_pltr", context_pltr))
    targets.append((f"context_pcmtr", context_pcmtr))
    targets.append((f"context_pwtr", context_pwtr))
    targets.append((f"context_pftr", context_pftr))
    targets.append((f"context_plvtr", context_plvtr))
    targets.append((f"duration_s", duration_s))
    # targets.append((f"pwtd_gather_list_print", pwtd_gather_list_print))
    # targets.append((f"pltr_gather_list_print", pltr_gather_list_print))
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
