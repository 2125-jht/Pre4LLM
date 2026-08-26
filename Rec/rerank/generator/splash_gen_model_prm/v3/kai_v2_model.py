from __future__ import print_function

import click
from kess import get_version
from numpy import dtype
MODEL_TRANS_ORIGIN='cpp'

import json
import yaml
import logging
import os
import sys

import argparse
import tensorflow as tf

from feature_attr_extract import * 
from model import FountainDeepLtrMultiTaskModel

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

CANDIDATES_SIZE = 60

# 目前这段逻辑功能未知
if args.with_kai_v2:
    import kai.tensorflow as config
    import tensorflow.compat.v1 as tf
    default_param_attr = config.nn.ParamAttr(initializer=config.nn.UniformInitializer(0.0001),
                                             access_method=config.nn.ProbabilityAccess(100.0),
                                             recycle_method=config.nn.UnseendaysRecycle(delete_after_unseen_days=30, delete_threshold=1.0, allow_dynamic_delete=True))
    config.nn.set_default_param_attr(default_param_attr)
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
    "fountain_click_label_list",
    "fountain_wtd_label_list",
    "fountain_finish_label_list",
    # "fountain_slide",
    "fountain_ltr_label_list"
]

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


# def get_label(name):
#     # assert name in all_model_labels, name
#     return config.get_label("%s_label" % name)

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

def ordinal_regression_loss(pred, labels, num_classes, mask, scale=5.0, train_cutpoints=False, name="ordinal_regression_loss"):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        num_cut_points = num_classes - 1
        initial_cut_points = tf.range(num_cut_points, dtype=tf.float32) * scale / (num_classes - 2) - scale / 2
        cut_points = tf.get_variable("cut_points", initializer=initial_cut_points, trainable=train_cutpoints)
        pred = tf.reshape(pred, [-1, 1])
        # 计算sigmoid边界概率
        sigmoids = tf.sigmoid(cut_points - pred)  # shape: (batch_size, num_cutpoints)
        # 构建概率链接矩阵
        first_col = tf.slice(sigmoids, [0,0], [-1,1])              # 首列保持不变
        middle_cols = sigmoids[:,1:] - sigmoids[:,:-1]            # 中间列差分
        last_col = 1 - tf.slice(sigmoids, [0,num_classes-2], [-1,1])  # 最后一列
        link_mat = tf.concat([first_col, middle_cols, last_col], axis=1)
        likelihoods = tf.clip_by_value(link_mat, 1e-8, 1 - 1e-8)
        neg_log_likelihood = tf.log(likelihoods)
        # label需要对应 num classes
        labels = tf.reshape(labels, [-1, 1])
        mask = tf.reshape(mask, [-1, 1])
        loss = -tf.reduce_mean(
            # indices: (?,2), get (?, num_calsses - 1)
            tf.gather_nd(neg_log_likelihood, tf.concat([tf.range(tf.shape(labels)[0])[:, None], labels], axis=1))
            *
            mask
        )

        return loss, likelihoods

def pairwise_bpr_loss(logits, score, threshold, mask):
    # 生成配对矩阵
    logits_i = tf.expand_dims(logits, 2)
    logits_j = tf.expand_dims(logits, 1)
    logit_diff = logits_i - logits_j
    score_i = tf.expand_dims(score, 2)
    score_j = tf.expand_dims(score, 1)
    pairwise_labels = tf.cast(score_i - threshold >= score_j, tf.float32)
    # 生成有效掩码
    mask_i = tf.expand_dims(mask, 2)
    mask_j = tf.expand_dims(mask, 1)
    valid_pair_mask = tf.logical_and(mask_i, mask_j)
    # 计算BPR损失
    bpr_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=pairwise_labels,
        logits=logit_diff
    )
    # bpr_loss = -tf.log(score_i - score_j + 1e-8)
    print("bpr_loss", bpr_loss)
    bpr_loss = tf.where(valid_pair_mask, bpr_loss, tf.zeros_like(bpr_loss, dtype=tf.float32))
    # 归一化损失
    # num_valid = tf.reduce_sum(tf.cast(valid_pair_mask, tf.float32))
    # bpr_loss = tf.reduce_sum(bpr_loss) / (num_valid + 1e-8)
    return bpr_loss

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

def get_view_label(playing_time, duration_ms):
    eff_threshold = tf.ones_like(duration_ms) * 12700
    long_threshold = tf.ones_like(duration_ms) * 79700

    long_threshold = tf.where(duration_ms <= 195000, 92500, long_threshold)
    eff_threshold = tf.where(duration_ms <= 195000, 17600, eff_threshold)
    
    long_threshold = tf.where(duration_ms <= 118200, 74900, long_threshold)
    eff_threshold = tf.where(duration_ms <= 118200, 18300, eff_threshold)
    
    long_threshold = tf.where(duration_ms <= 71800, 46600, long_threshold)
    eff_threshold = tf.where(duration_ms <= 71800, 13100, eff_threshold)
    
    long_threshold = tf.where(duration_ms <= 38800, 28800, long_threshold)
    eff_threshold = tf.where(duration_ms <= 38800, 11400, eff_threshold)
    
    long_threshold = tf.where(duration_ms <= 20300, 18400, long_threshold)
    eff_threshold = tf.where(duration_ms <= 20300, 9900, eff_threshold)
    
    long_threshold = tf.where(duration_ms <= 12700, 13600, long_threshold)
    eff_threshold = tf.where(duration_ms <= 12700, 8700, eff_threshold)
    
    long_threshold = tf.where(duration_ms <= 8700, 12000, long_threshold)
    eff_threshold = tf.where(duration_ms <= 8700, 7200, eff_threshold)
    
    long_threshold = tf.where(tf.equal(duration_ms, 0), 13100, long_threshold)
    eff_threshold = tf.where(tf.equal(duration_ms, 0), 4500, eff_threshold)

    effective_view = playing_time >= eff_threshold
    effective_view = tf.cast(effective_view, dtype=tf.float32)
    long_view = playing_time >= long_threshold
    long_view = tf.cast(long_view, dtype=tf.float32)

    return effective_view, long_view

# print info
worker_global_step = config.get_step()
ops = [tf.print("====> step", worker_global_step, summarize=-1, output_stream=sys.stdout)]

############################################################

# print info
worker_global_step = config.get_step()
ops = [tf.print("====> step", worker_global_step, summarize=-1, output_stream=sys.stdout)]

############################################################

# 获取模型output
all_param_dict, feature_emb_size_dict, uid_dict = get_param_dict()

loss_names = ["ltr", "vtr", "click", "lvtr", "svtr", "show", "next"]
model_class = FountainDeepLtrMultiTaskModel(loss_names, all_param_dict, CANDIDATES_SIZE, print_ops=print_ops, training=True)
pred_dict = {}

if is_training:
    uid = uid_dict["uid"]
    index_label = tf.reshape(config.get_label("fountain_fulllink_rerank_index_list", dim=CANDIDATES_SIZE), [-1, CANDIDATES_SIZE])
    index_label = tf.cast(index_label, dtype=tf.int32)
    model_class._training=True
    output_dict = model_class.model(index_label)
    print(f"====> train, gen...")

    show_label = tf.reshape(config.get_dense_fea("context_info__real_show_list", dim=CANDIDATES_SIZE, dtype=tf.int64), [-1, CANDIDATES_SIZE])
    show_label = tf.cast(show_label, dtype=tf.float32)

    playtime = tf.reshape(config.get_dense_fea("context_info__playing_time_list", dim=CANDIDATES_SIZE, dtype=tf.int64), [-1, CANDIDATES_SIZE])
    playtime = tf.cast(playtime, dtype=tf.float32)

    item_weight = tf.clip_by_value(playtime - 1.0, 0, 1000) # 得到真实时长

    click_label = tf.reshape(config.get_label("fountain_click_label_list", dim=CANDIDATES_SIZE), [-1, CANDIDATES_SIZE])
    # click_label = get_view_label(item_weight, duration_ms)
    wtd_label = tf.reshape(config.get_label("fountain_wtd_label_list", dim=CANDIDATES_SIZE), [-1, CANDIDATES_SIZE])
    finish_label = tf.reshape(config.get_label("fountain_finish_label_list", dim=CANDIDATES_SIZE), [-1, CANDIDATES_SIZE])
    ltr_label = tf.reshape(config.get_label("fountain_ltr_label_list", dim=CANDIDATES_SIZE), [-1, CANDIDATES_SIZE])
    ltr_weight = config.get_extra_param("fountain_ltr_weight_list", size=CANDIDATES_SIZE)
    ltr_weight = tf.cast(tf.reshape(ltr_weight, [-1, CANDIDATES_SIZE]), dtype=tf.float32)

    next_label = tf.reshape(config.get_dense_fea("context_info__fountain_slide_to_next_list", dim=CANDIDATES_SIZE, dtype=tf.int64), [-1, CANDIDATES_SIZE])
    next_label = tf.cast(next_label, dtype=tf.float32)

    advantage = calc_advantage(item_weight - 1, show_label, uid)
    show_advantage = tf.where(advantage > 0, advantage + 1, tf.ones_like(advantage))
    advantage = calc_advantage(item_weight - 1, click_label, uid)
    click_advantage = tf.where(advantage > 0, advantage + 1, tf.ones_like(advantage)) * show_label

    item_weight_clip = item_weight
    bound_neg_1 = tf.ones_like(item_weight, dtype=tf.float32) * 2.0 # [0, 3)
    bound_neg_2 = tf.ones_like(item_weight, dtype=tf.float32) * 1.5 # [3, 5)
    bound_neg_3 = tf.ones_like(item_weight, dtype=tf.float32) * 1.0 # [5, 7)
    # bound_neg_4 = tf.ones_like(item_weight, dtype=tf.float32) * 0.2 # [7, 10)
    bound1 = tf.ones_like(item_weight, dtype=tf.float32) # [7, 12)
    bound2 = (item_weight - 10) * 0.1 + 1.0 # [12, 20) max = 2
    bound3 = (item_weight - 20) * 0.025 + 2.0 # [20, 60) max = 3
    # bound4 = tf.log(item_weight - 59) / tf.math.log(3.0) / 1.5 + 2.0 # [60, 1000)
    bound4 = tf.ones_like(item_weight, dtype=tf.float32) * 2.5 # [60, 1000)
    item_weight = tf.where(item_weight_clip >= 3, bound_neg_2, bound_neg_1)
    item_weight = tf.where(item_weight_clip >= 5, bound_neg_3, item_weight)
    item_weight = tf.where(item_weight_clip >= 7, bound1, item_weight)
    item_weight = tf.where(item_weight_clip >= 10, bound2, item_weight)
    item_weight = tf.where(item_weight_clip >= 20, bound3, item_weight)
    item_weight = tf.where(item_weight_clip >= 60, bound4, item_weight)
    item_weight = item_weight * tf.cast(show_label, dtype=tf.float32) # mask掉未曝光的item
    # 增加互动权重
    # item_weight = item_weight + ltr_label * 0.5

    bpr_act_weight = tf.where(ltr_label > 0, tf.ones_like(ltr_label, dtype=tf.float32) * 1.5, tf.ones_like(ltr_label, dtype=tf.float32))
    bpr_threshold = tf.where(item_weight < 20.0, tf.ones_like(item_weight) * 3, tf.ones_like(item_weight) * 5)
    bpr_threshold = tf.where(item_weight > 60.0, tf.ones_like(item_weight) * 10, bpr_threshold)

    print_ops.append(tf.print(f"bpr threshold", bpr_threshold[2], summarize = 8, output_stream=sys.stdout))
    print_ops.append(tf.print(f"show_label", show_label[2], summarize = 8, output_stream=sys.stdout))
    print_ops.append(tf.print(f"playtime", playtime[2], summarize = 8, output_stream=sys.stdout))
    print_ops.append(tf.print(f"item_weight", item_weight[2], summarize = 8, output_stream=sys.stdout))
    print_ops.append(tf.print(f"click_label", click_label[2], summarize = 8, output_stream=sys.stdout))
    print_ops.append(tf.print(f"click_output", tf.reshape(output_dict["click"], [-1, CANDIDATES_SIZE])[2], summarize = 8, output_stream=sys.stdout))

    with tf.control_dependencies(print_ops):
        # adn loss calc
        targets = []
        sum_loss = 0.0
        for loss_name in output_dict:
            # labels_tensor = label_value_dict[loss_name]
            # weight_with_mask = label_weight_dict[loss_name] * label_mask_dict[loss_name]
            output = output_dict[loss_name]
            output = tf.reshape(output, [-1, CANDIDATES_SIZE])

            if loss_name == "fountain_wtd":
                # group_id = all_param_dict["duration_group_id"]
                weight_with_mask = tf.ones_like(wtd_label, dtype=tf.float32) * show_label # only realshow sample
                loss = tf.losses.huber_loss(labels=wtd_label,
                                            predictions=output,
                                            weights=weight_with_mask,
                                            reduction=tf.losses.Reduction.SUM,
                                            delta=0.05)
                loss = loss * 2.0
                targets.append((loss_name + "reg", output, wtd_label, weight_with_mask, "linear_regression"))
                # mae, r_squared = mae_r_squared(output, labels_tensor)
                # mae, _, m_mae = mae_r_squared_v2(output, labels_tensor, group_id)
                # tf.summary.scalar('MAE:' + loss_name, mae)
                # tf.summary.scalar('mMAE:' + loss_name, m_mae)
            elif loss_name == "fountain_finish":
                weight_with_mask = tf.ones_like(finish_label, dtype=tf.float32) * show_label # only realshow sample
                loss = tf.losses.huber_loss(labels=finish_label,
                                        weights=weight_with_mask,
                                        predictions=output,
                                        reduction=tf.losses.Reduction.SUM,
                                        delta=0.05)
                # mae, _, m_mae = mae_r_squared_v2(output, labels_tensor, group_id)
                # tf.summary.scalar('MAE:' + loss_name, mae)
                # tf.summary.scalar('mMAE:' + loss_name, m_mae)
                targets.append((loss_name + "reg", output, finish_label, weight_with_mask, "linear_regression"))
            elif loss_name == "fountain_ltr":
                weight_with_mask = ltr_weight * show_label # only realshow sample
                loss = tf.losses.log_loss(labels=ltr_label, predictions=output, weights=weight_with_mask,
                                        reduction=tf.losses.Reduction.SUM)
                #loss = loss / 100.0
                targets.append((loss_name, output, ltr_label, weight_with_mask, "auc"))
                weight = tf.ones_like(weight_with_mask)
                # targets.append((loss_name + "reg", output, weight_with_mask, weight, "linear_regression"))
            else:
                # weight_with_mask = item_weight * show_label # only realshow sample
                # weight_with_mask = show_label * click_advantage # only realshow sample
                weight_with_mask = show_label # only realshow sample
                loss = tf.losses.log_loss(labels=click_label, predictions=output, weights=weight_with_mask,
                                        reduction=tf.losses.Reduction.SUM)
                # loss = - tf.reduce_sum(tf.log(output + 1e-9) * click_label)
                #loss = loss / 100.0
                targets.append((loss_name, output, click_label, show_label, "auc"))
                bpr_loss = pairwise_bpr_loss_v2(output, (show_weight - 1) * bpr_act_weight, tf.expand_dims(bpr_threshold, axis=-1), show_label > 0)
                bpr_loss = tf.reduce_sum(bpr_loss) * 0.05 # 过大会主导学习，导致ctr auc降低
                sum_loss += bpr_loss
                tf.summary.scalar('bpr_loss', bpr_loss)

            sum_loss += loss
            tf.summary.scalar('loss_' + loss_name, loss)

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
    duration_ms = config.get_extra_param("duration_ms", size=1, default_value=0.0) / 1000
    l2r_output, vtr_output, ctr_output, next_output, pfr_output = model_class.model()
    output_dict = {
        # "ltr": l2r_output,
        # "vtr": vtr_output,
        "click": ctr_output,
        # "fountain_slide": next_output,
        # "finish": pfr_output,
    }
    targets = []
    ctr_output = tf.identity(tf.reshape(output_dict["click"], [-1, 1]))
    targets.append((f"rerank_gen_score_0", tf.reshape(ctr_output, [-1, 1])))
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
