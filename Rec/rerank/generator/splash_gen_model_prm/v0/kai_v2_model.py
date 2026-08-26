from __future__ import print_function

import click
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
            if attr.attr_name in photo_fea_names + ["photo_id_v2"]:
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
                tt = tf.RaggedTensor.from_row_splits(values=sparse_feature[0], row_splits=sparse_feature[1]).to_tensor()
                #print_ops.append(tf.print("[Test test] slot " + str(attr.slots[0]), tt, output_stream=sys.stdout))
        elif args.with_kai:
            offset = tf.cast(config.get_signs(attr.slots[0])[1], tf.int32)
            size_var = offset[1:] - offset[0:-1]
            feature_emb_size_dict[attr.attr_name] = size_var        
        print("--->>> feature {} = {}".format(attr.attr_name, feature_emb_dict[attr.attr_name]))
        print("--->>> feature %s normal" % attr.attr_name)

    return feature_emb_dict, feature_emb_size_dict


def sum_loss_tensor_dict(loss_dict):
    sum_loss = None
    for key, loss in loss_dict.items():
        if (sum_loss == None):
            sum_loss = loss
        else:
            sum_loss += loss
    return sum_loss


# print info
worker_global_step = config.get_step()
ops = [tf.print("====> step", worker_global_step, summarize=-1, output_stream=sys.stdout)]

############################################################

# 获取模型output
all_param_dict, feature_emb_size_dict = get_param_dict()
# label_value_dict, label_weight_dict, label_mask_dict = get_label_dict()

loss_names = ["ltr", "vtr", "click"]
model_class = FountainDeepLtrMultiTaskModel(loss_names, all_param_dict, CANDIDATES_SIZE, print_ops=print_ops, training=True)
pred_dict = {}


if is_training:
    index_label = tf.reshape(config.get_label("fountain_fulllink_rerank_index_list", dim=CANDIDATES_SIZE), [-1, CANDIDATES_SIZE])
    index_label = tf.cast(index_label, dtype=tf.int32)
    model_class._training=True
    output_dict = model_class.model(index_label)
    print(f"====> train, gen...")
    show_label = tf.reshape(config.get_dense_fea("context_info__real_show_list", dim=CANDIDATES_SIZE, dtype=tf.int64), [-1, CANDIDATES_SIZE])
    show_label = tf.cast(show_label, dtype=tf.float32)
    show_weight = config.get_extra_param("fountain_fulllink_rerank_realshow_label_weight_list", size=CANDIDATES_SIZE)
    show_weight = tf.cast(tf.reshape(show_weight, [-1, CANDIDATES_SIZE]), dtype=tf.float32)
    show_weight = tf.cast(show_weight, dtype=tf.float32)
    item_weight = tf.clip_by_value(show_weight, 0, 1000)
    # item_weight = show_weight / 10.0
    # item_weight = tf.where(item_weight > 7, tf.log(item_weight) / tf.math.log(1.4) - 4.6, tf.ones_like(item_weight, dtype=tf.float32))
    bound1 = tf.ones_like(item_weight, dtype=tf.float32) # [7, 12)
    bound2 = (item_weight - 12) * 0.25 + 1.0 # [12, 20)
    bound3 = (item_weight - 20) * 0.125 + 3.0 # [20, 60)
    bound4 = tf.log(item_weight - 53) / tf.math.log(1.5) + 3.0 # [60, 1000)
    item_weight = tf.where(item_weight >= 12, bound2, bound1)
    item_weight = tf.where(item_weight >= 20, bound3, item_weight)
    item_weight = tf.where(item_weight >= 60, bound4, item_weight)

    click_label = tf.reshape(config.get_label("fountain_click_label_list", dim=CANDIDATES_SIZE), [-1, CANDIDATES_SIZE])
    wtd_label = tf.reshape(config.get_label("fountain_wtd_label_list", dim=CANDIDATES_SIZE), [-1, CANDIDATES_SIZE])
    finish_label = tf.reshape(config.get_label("fountain_finish_label_list", dim=CANDIDATES_SIZE), [-1, CANDIDATES_SIZE])
    ltr_label = tf.reshape(config.get_label("fountain_ltr_label_list", dim=CANDIDATES_SIZE), [-1, CANDIDATES_SIZE])
    ltr_weight = config.get_extra_param("fountain_ltr_weight_list", size=CANDIDATES_SIZE)
    ltr_weight = tf.cast(tf.reshape(ltr_weight, [-1, CANDIDATES_SIZE]), dtype=tf.float32)
    print_ops.append(tf.print(f"index_label", index_label[2], summarize = 8, output_stream=sys.stdout))
    print_ops.append(tf.print(f"show_label", show_label[2], summarize = 8, output_stream=sys.stdout))
    print_ops.append(tf.print(f"show_weight", show_weight[2], summarize = 8, output_stream=sys.stdout))
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

            if loss_name == "show":
                weight_with_mask = tf.ones_like(show_label, dtype=tf.float32) # only realshow sample
                loss = tf.losses.log_loss(labels=show_label, predictions=output, weights=weight_with_mask,
                                          reduction=tf.losses.Reduction.SUM)
                targets.append((loss_name, output, show_label, weight_with_mask, "auc"))
            elif loss_name == "vtr":
                weight_with_mask = show_label # only realshow sample
                loss = tf.losses.log_loss(labels=wtd_label, predictions=output, weights=weight_with_mask,
                                          reduction=tf.losses.Reduction.SUM)
                targets.append((loss_name, output, wtd_label, weight_with_mask, "linear_regression"))
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
            elif loss_name == "ltr":
                # weight_with_mask = ltr_weight * show_label # only realshow sample
                weight_with_mask = show_label # only realshow sample
                loss = tf.losses.log_loss(labels=ltr_label, predictions=output, weights=weight_with_mask,
                                        reduction=tf.losses.Reduction.SUM)
                #loss = loss / 100.0
                targets.append((loss_name, output, ltr_label, weight_with_mask, "auc"))
            else:
                # weight_with_mask = item_weight * show_label
                weight_with_mask = show_label # only realshow sample
                loss = tf.losses.log_loss(labels=click_label, predictions=output, weights=weight_with_mask,
                                          reduction=tf.losses.Reduction.SUM)
                # loss = - tf.reduce_sum(tf.log(output + 1e-9) * click_label)
                #loss = loss / 100.0
                targets.append((loss_name, output, click_label, show_label, "auc"))

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
    def ensemble_sort(pxtr, rank_w, weight_w, weight_pow=1.0, cal_type="add"):
        pxtr = tf.reshape(pxtr, [-1,]) # [0.1,0.5,0.2,0.4]
        rank = tf.argsort(pxtr, direction="DESCENDING") # (?) 降序排列后元素所在原数组中的位置 [1 3 2 0]
        rank = tf.cast(tf.argsort(rank), dtype=tf.float32) # (?,) 再对位置数组排序后得到原数组索引下的对应降序rank索引 [3 0 2 1]
        if cal_type == "add":
            es_i = (rank_w / (rank + 10.0)) * tf.math.pow(weight_w * pxtr + 1.0, weight_pow)
        else:
            es_i = tf.math.pow(tf.math.pow(1 + weight_w * pxtr, weight_pow) / (10.0 + rank), rank_w)
        return es_i
    model_class._training = False
    output_dict = model_class.model()
    pctr = tf.reshape(output_dict["click"], [-1, 1])
    pwtd = tf.reshape(output_dict["vtr"], [-1, 1])
    pltr = tf.reshape(output_dict["ltr"], [-1, 1])
    batch_size = tf.shape(pwtd)[0]
    # photo_id_emb = all_param_dict["photo_id"]
    # context_cascade_pctr_emb = all_param_dict["context_cascade_pctr"]
    duration_ms = config.get_extra_param("duration_ms", size=1, default_value=0.0) / 1000
    duration_s = tf.cast(tf.reshape(duration_ms, [batch_size, -1]), dtype=tf.int32) # (cand_size, 1)
    buckets = [126.143,37.273,37.273,37.273,49.909,73.636,108.556,116.71,115.661,112.282,117.694,120.773,113.152,113.58,116.71,120.994,117.205,114.166,114.916,110.194,104.811,102.394,100.992,105.644,107.073,110.415,110.693,105.249,108.215,106.411,110.046,103.66,107.075,107.948,102.366,106.835,104.614,106.755,107.392,103.63,98.364,98.318,101.976,97.505,99.748,99.906,101.857,100.387,102.698,103.719,104.998,103.746,106.468,108.6,106.418,107.294,110.825,112.583,113.497,113.473,114.885,110.998,113.476,114.182,110.493,112.166,112.849,115.205,113.069,116.622,115.864,116.927,112.597,116.769,114.353,115.245,115.381,114.476,113.123,118.325,120.576,117.788,115.617,119.428,119.337,121.104,121.076,121.622,123.891,122.986,119.524,121.759,124.767,126.54,122.851,123.598,123.747,121.141,126.368,122.234,124.698,123.941,122.459,125.179,128.054,124.017,123.927,127.821,126.8,125.761,129.136,126.184,128.474,130.522,132.295,131.511,130.809,129.382,132.497,131.264,134.051,134.566,132.249,135.828,135.531,131.979,137.039,136.273,138.381,138.364,139.18,139.395,139.402,142.823,141.631,142.814,141.64,141.355,140.215,141.915,140.216,142.513,143.464,146.272,146.592,145.636,147.262,144.395,149.201,146.603,146.636,146.351,147.59,151.337,147.944,149.681,149.202,149.958,146.294,154.688,150.646,153.921,153.576,153.557,149.261,148.648,152.067,150.784,150.381,155.05,155.099,155.092,149.341,149.552,156.568,158.64,155.796,157.338,153.212,155.447,153.174,151.656,155.98,155.608,149.921,157.445,158.027,159.689,156.586,155.805,149.556,156.661,161.279,156.972,160.079,158.68,156.277,157.08,156.773,154.777,200.0]
    buckets = tf.constant(buckets, dtype=tf.float32) # (200,)
    vtr_max = tf.constant(200, shape=[1, 1], dtype=tf.int32)
    vtr_max = tf.tile(vtr_max, [batch_size, tf.shape(duration_s)[1]]) # (cand_size, 1)
    vtr_indices = tf.where(duration_s > 200, vtr_max, duration_s)
    print("vtr_indices ", vtr_indices)
    max_time = tf.gather(buckets, vtr_indices) # (cand_size, 1)
    print("max_time ", max_time)
    wt = pwtd * max_time
    # add_score = ensemble_sort(pctr, 1.0, 0.0, cal_type="add") + \
    #             ensemble_sort(pwtd, 1.0, 0.0, cal_type="add") + \
    #             ensemble_sort(pltr, 1.0, 0.0, cal_type="add")
    # multi_score = ensemble_sort(pctr, 1.0, 0.0, cal_type="mul") * \
    #             ensemble_sort(pwtd, 2.0, 1.0, cal_type="mul") * \
    #             ensemble_sort(pltr, 3.0, 1.0, cal_type="mul")
    multi_rank_score = ensemble_sort(pctr, 1.0, 0.0, cal_type="mul") * \
                ensemble_sort(wt, 1.5, 0.0, cal_type="mul") * \
                ensemble_sort(pltr, 0.1, 0.0, cal_type="mul")

    targets = []
    targets.append((f"rerank_gen_score_0", tf.reshape(multi_rank_score, [-1, 1])))
    # targets.append((f"rerank_gen_score_0", tf.reshape(add_score, [-1, 1])))
    # targets.append((f"rerank_gen_score_1", tf.reshape(multi_score, [-1, 1])))
    # targets.append((f"rerank_gen_score_0", tf.reshape(wt, [-1, 1])))
    targets.append((f"pctr", pctr))
    targets.append((f"pwtd", pwtd))
    targets.append((f"pltr", pltr))

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
