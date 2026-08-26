from __future__ import print_function
MODEL_TRANS_ORIGIN='cpp'

import json
import logging
import os
import sys

import argparse
from model import RevisitModel

from input import user_features, item_features, user_config, item_config, extra_param_config

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'predict'], dest='mode', default='train')
parser.add_argument('--dryrun', dest='dryrun', const=True, default=False, nargs='?')
parser.add_argument('--with_kai', default=False)
parser.add_argument('--text', default=False)
parser.add_argument('--tower', choices=None, dest='tower', default='False')
parser.add_argument('--with_kai_v2', default=True) # default=False True
args = parser.parse_known_args()[0]
is_training = args.mode == "train"

print("args ========> ")
print(args)

# https://docs.corp.kuaishou.com/k/home/VHQxhUsdPngA/fcAAxbjU-3mVHD0EdnCEMSNfr#section=h.rmlf76tvrz6l
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

all_model_labels = [
    "point_ltr_label", 
    "point_ltr_wt",
    "is_next_label",
    "is_revisit"]

predict_model_labels = [
    "pltr0", "pltr1", "pltr2", "pltr3", "pltr4", "pltr5","pltr6","pltr7", "pltr8", "pltr9",
    "pnext0","pnext1","pnext2","pnext3","pnext4","pnext5","pnext6","pnext7","pnext8","pnext9",
    "previsit"]
print("====> common_attr_names: ", [attr.attr_name for attr in user_features if attr.is_common])
print("====> itemfeature_name: ", [attr.attr_name for attr in item_features])

print_ops = []


def get_label(name, list_dim=10):
    assert name in all_model_labels, name
    return config.get_label(name, dim=list_dim)


def get_param_dict():
    """
    train and dnn infer：不需要区分common or no_common,(infer配置中对应的tensorflow_use_batching=true)
    tower infer : 需要区分attr是common or no_common
    :return:
    """
    user_feature_emb_dict = {}
    user_feature_emb_size_dict = {}
    item_feature_emb_dict = {}
    item_feature_emb_size_dict = {}
    extra_param_dict = {}
    for attr in user_features:
        print("--->>> feature %s start" % attr.attr_name)
        if not is_training:
            if not attr.expand:
                attr.expand =  1
            if attr.is_common:
                user_feature_emb_dict[attr.attr_name] = config.new_embedding(attr.attr_name, dim=attr.dim, slots=attr.slots, expand=attr.expand, compress_group='USER')
            else:
                user_feature_emb_dict[attr.attr_name] = config.new_embedding(attr.attr_name, dim=attr.dim, slots=attr.slots, expand=attr.expand)
        else:
            # if not attr.expand: #local test 的时候取消注释
            #     attr.expand = 1 #local test 的时候取消注释
            user_feature_emb_dict[attr.attr_name] = config.new_embedding(attr.attr_name, dim=attr.dim, slots=attr.slots, expand=attr.expand)
        if args.with_kai_v2:
            sparse_feature = config.get_sparse_fea(name=str(attr.slots[0]))
            offset = sparse_feature[1]
            size_var = offset[1:] - offset[0:-1]
            user_feature_emb_size_dict[attr.attr_name] = size_var
        else:
            offset = tf.cast(config.get_signs(attr.slots[0])[1], tf.int32)
            size_var = offset[1:] - offset[0:-1]
            user_feature_emb_size_dict[attr.attr_name] = size_var
        print("--->>> feature %s normal" % attr.attr_name)

    for attr in item_features:
        print("--->>> feature %s start" % attr.attr_name)
        if not is_training:
            if not attr.expand:
                attr.expand =  1
            if attr.is_common:
                item_feature_emb_dict[attr.attr_name] = config.new_embedding(attr.attr_name, dim=attr.dim, slots=attr.slots, expand=attr.expand, compress_group='USER')
            else:
                item_feature_emb_dict[attr.attr_name] = config.new_embedding(attr.attr_name, dim=attr.dim, slots=attr.slots, expand=attr.expand)
        else:
            # if not attr.expand: #local test 的时候取消注释
            #     attr.expand = 1 #local test 的时候取消注释
            item_feature_emb_dict[attr.attr_name] = config.new_embedding(attr.attr_name, dim=attr.dim, slots=attr.slots, expand=attr.expand)
        if args.with_kai_v2: 
            sparse_feature = config.get_sparse_fea(name=str(attr.slots[0])) 
            offset = sparse_feature[1]
            size_var = offset[1:] - offset[0:-1]
            item_feature_emb_size_dict[attr.attr_name] = size_var
        else:
            offset = tf.cast(config.get_signs(attr.slots[0])[1], tf.int32)
            size_var = offset[1:] - offset[0:-1]
            item_feature_emb_size_dict[attr.attr_name] = size_var
        print("--->>> feature %s normal" % attr.attr_name)
    for k, v in extra_param_config.items():
        # extra_param_dict.update({k: config.get_extra_param(v.get("attr_name"), size=v.get("size"), default_value=1.0)})
        if v.get("type") =='bigint':
            extra_param_dict[k] = config.get_dense_fea(k, dim=1, dtype=tf.int64)
        else:
            extra_param_dict[k] = config.get_dense_fea(k, dim=10, dtype=tf.float32)


    print("extra_param: ", extra_param_dict.keys())
    return user_feature_emb_dict, item_feature_emb_dict, extra_param_dict




def make_data():
    feature_config = {}
    common_features = {}
    item_features = {}
    user_config.update(item_config)
    # user_config.update(extra_param_config)
    for feature_name, attr_config in user_config.items():
        if attr_config.get("use_common_attr_only") is True:
            common_features.update({feature_name: attr_config})
        else:
            item_features.update({feature_name: attr_config})
            # if attr_config["attrs"][0]["attr"][0]=="hetu_level_two_tag":
            #         attr_config["attrs"][0]["attr"][0]="hetu_level_two_tag_rename"
            # print("hetu_level_two_tag_rename", attr_config)

    feature_config.update({"labels": predict_model_labels})  # all_model_labels
    feature_config.update({"common_features": common_features})
    feature_config.update({"item_features": item_features})
    with open("./infer_server/feature_config.json", 'w') as f:
        f.write(json.dumps(feature_config, indent=2))


def sum_loss_tensor_dict(loss_dict):
    sum_loss = None
    for key, loss in loss_dict.items():
        if (sum_loss == None):
            sum_loss = loss
        else:
            sum_loss += loss
    return sum_loss


def get_label_value_or(label_name, default_value):
    photo_attr_value = extra_param_dict[label_name]
    photo_attr_value = tf.where(tf.is_nan(photo_attr_value), default_value, photo_attr_value)
    return photo_attr_value


def get_extra_param(name, list_dim=10):
    return config.get_extra_param(name, size=list_dim)


def get_slot_dims(config_name):
    ret = []
    for attr_config in json_param_config.get(config_name).values():
        ret.append(attr_config.get("dim"))
    return ret


def variable_summaries(var, name, only_scalar=False):
  with tf.name_scope('summaries_' + name):
    if not only_scalar:
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean_' + name, mean)
      with tf.name_scope('stddev_' + name):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev_' + name, stddev)
      tf.summary.scalar('max_' + name, tf.reduce_max(var))
      tf.summary.scalar('min_' + name, tf.reduce_min(var))
      tf.summary.histogram('histogram_' + name, var)
    else:
      tf.summary.scalar('scalar_' + name, var)


# print info
worker_global_step = config.get_step()
ops = [tf.print("step", worker_global_step, summarize=-1, output_stream=sys.stdout)]

# 获取模型output
user_param_dict, item_param_dict, extra_param_dict = get_param_dict()
print(f"====> user_param_dict: {user_param_dict}")
print(f"====> item_param_dict: {item_param_dict}")
print(f"====> extra_param_dict: {extra_param_dict}")



model = RevisitModel(user_param_dict, item_param_dict, extra_param_dict)
pred_point_ltr,pred_next,pred_revisit,model_print_ops = model.model()

# 放到pred_dict中
pred_dict = {}
ltrs = tf.split(pred_point_ltr, num_or_size_splits=10, axis=-1)
nexts = tf.split(pred_next, num_or_size_splits=10, axis=-1)
pred_dict["pltr0"] = ltrs[0]
pred_dict["pltr1"] = ltrs[1]
pred_dict["pltr2"] = ltrs[2]
pred_dict["pltr3"] = ltrs[3]
pred_dict["pltr4"] = ltrs[4]
pred_dict["pltr5"] = ltrs[5]
pred_dict["pltr6"] = ltrs[6]
pred_dict["pltr7"] = ltrs[7]
pred_dict["pltr8"] = ltrs[8]
pred_dict["pltr9"] = ltrs[9]
pred_dict["pnext0"] = nexts[0]
pred_dict["pnext1"] = nexts[1]
pred_dict["pnext2"] = nexts[2]
pred_dict["pnext3"] = nexts[3]
pred_dict["pnext4"] = nexts[4]
pred_dict["pnext5"] = nexts[5]
pred_dict["pnext6"] = nexts[6]
pred_dict["pnext7"] = nexts[7]
pred_dict["pnext8"] = nexts[8]
pred_dict["pnext9"] = nexts[9]
pred_dict["previsit"] = pred_revisit

if not is_training:
    for label_name, prob in pred_dict.items():
        pred_dict[label_name] = tf.identity(prob, label_name)

if is_training:
    targets = []
    loss_tensor_dict = {}
    
    with tf.control_dependencies(print_ops + model_print_ops):
        # ltr label
        label_values, label_weight = {}, {}
        label_values = get_label("point_ltr_label")
        label_weight = get_extra_param("point_ltr_wt")
        next_label = get_extra_param("is_next_label")
        mask_label = get_extra_param("is_real_show_list")
        label_weight = mask_label*label_weight
        label_values_list = tf.split(label_values, num_or_size_splits=10, axis=-1)
        label_weight_list = tf.split(label_weight, num_or_size_splits=10, axis=-1)
        mask_label_list = tf.split(mask_label, num_or_size_splits=10, axis=-1)
        next_label_list = tf.split(next_label, num_or_size_splits=10, axis=-1)

        for i, label in enumerate(label_values_list):
            loss_tensor = 0
            weight = label_weight_list[i]
            loss_tensor = tf.losses.log_loss(labels=label, predictions=pred_dict[f"pltr{i}"], weights=weight, reduction=tf.losses.Reduction.SUM)
            targets.append((f"ltr{i}", pred_dict[f"pltr{i}"], label, weight, "auc"))
            loss_tensor_dict[f"ltr{i}"] = loss_tensor

            loss_next_tensor = tf.losses.log_loss(labels=next_label_list[i], predictions=pred_dict[f"pnext{i}"], weights=mask_label_list[i], reduction=tf.losses.Reduction.SUM)
            targets.append((f"next{i}", pred_dict[f"pnext{i}"], next_label_list[i], mask_label_list[i], "auc"))
            loss_tensor_dict[f"next{i}"] = loss_next_tensor

        # session label
        revisit = get_extra_param("is_revisit",1)
        loss_tensor_dict['revisit'] = tf.losses.log_loss(labels=revisit, predictions=pred_dict["previsit"], reduction=tf.losses.Reduction.SUM)
        targets.append(("revisit", pred_dict["previsit"], revisit, tf.ones_like(revisit), "auc"))

        # inner_time = get_extra_param("session_inner_time",1)
        # loss_tensor_dict['inner_time'] = tf.losses.huber_loss(labels=inner_time,predictions=pred_dict["pinner"], reduction=tf.losses.Reduction.SUM, delta=0.05)
        # targets.append(("inner_time", pred_dict["pinner"], inner_time, tf.ones_like(inner_time), "linear_regression"))

        # vv_len = get_extra_param("session_vv",1)
        # loss_tensor_dict['session_vv'] = tf.losses.huber_loss(labels=vv_len,predictions=pred_dict["pvv"], reduction=tf.losses.Reduction.SUM, delta=0.05)
        # targets.append(("session_vv", pred_dict["pvv"], vv_len, tf.ones_like(vv_len), "auc"))


        # outer_time = get_extra_param("session_out_time",1)
        # loss_tensor_dict['outer_time'] = tf.losses.huber_loss(labels=outer_time,predictions=pred_dict["pout"], reduction=tf.losses.Reduction.SUM, delta=0.05)
        # targets.append(("outer_time", pred_dict["pout"], outer_time, tf.ones_like(outer_time), "auc"))


    sum_loss = sum_loss_tensor_dict(loss_tensor_dict)
    for t in targets:
        print(f"====> target: {t}")

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
        config.dump_kai_training_config('./training/conf', targets, loss=loss, text=args.text, init_params_in_tf=True, extra_ops=print_ops + model_print_ops)
    elif args.with_kai_v2:
        config.build_model(optimizer=opts, metrics=targets)
    else:
        optimizer = tf.train.GradientDescentOptimizer(1, name="opt")
        opt = optimizer.minimize(loss)
        config.dump_training_config('./training/conf', targets, opts=[opt], text=args.text)

else:
    targets = []
    for label_name in predict_model_labels:
        targets.append((label_name, pred_dict[label_name]))
    q_names, preds = zip(*targets)
    config.dump_predict_config(
        "./infer_server/models/",
        targets,
        input_type=3,
        extra_preds=q_names,
    )
    print("q_name: ", q_names)
    make_data()
print("is_training %s, tower %s" % (is_training, args.tower))
