from __future__ import print_function
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

print_ops = []
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
        # dataset.add_feature('fountain_fulllink_rerank_index_list', dataset.DENSE, tf.int64, max_length=60)
        # dataset.add_feature('fountain_fulllink_rerank_index_weight_list', dataset.DENSE, tf.int64, max_length=60)
        dataset.add_feature('context_info__real_show_index_list', dataset.DENSE, tf.int64, max_length=60)
        dataset.add_feature('context_info__real_show_list', dataset.DENSE, tf.int64, max_length=60)
        dataset.add_feature('fountain_fulllink_rerank_realshow_label_weight_list', dataset.DENSE, tf.int64, max_length=60)      
        dataset.add_feature('3027', dataset.DENSE, tf.int64, max_length=60)
        dataset.add_feature('3028', dataset.DENSE, tf.int64, max_length=60)
        dataset.add_feature('context_page', dataset.DENSE, tf.int64, max_length=60)

        # 2.声明mask，batch是一个dict，key为声明的字段名，value根据特征类型分为2种情况：
        # dataset.DENSE: 值为tf.Tensor
        # dataset.SPARSE: 值为元组: (tf.Tensor, tf.Tensor)，
        #   其中第一个tensor表示feasign，第二个tensor表示cumsum
        #   可以使用tf.RaggedTensor.from_row_splits转成RaggedTensor
        def mask_fn(batch):
            realshow = batch["context_info__real_show_list"]
            realshow = tf.RaggedTensor.from_row_splits(realshow[0], row_splits= realshow[1])
            realshow = realshow.to_tensor()
            
            realshow_weight = batch["fountain_fulllink_rerank_realshow_label_weight_list"]
            realshow_weight = tf.RaggedTensor.from_row_splits(realshow_weight[0], row_splits= realshow_weight[1])
            realshow_weight = realshow_weight.to_tensor()          

            mean_play_time = tf.reduce_sum(realshow_weight, axis=-1)-60
            realshow = tf.reduce_sum(realshow, axis=-1)
            
            print_ops.append(tf.print(f"context_first_page={batch['3027']}, context_page={batch['3028']}, context_page={batch['context_page']}",
                                      output_stream=sys.stdout))
            
            # 修改过滤条件:
            # 1. 保留短视频样本(播放时长<20s)
            # 2. 对长视频样本(>20s)进行降采样,只保留部分高质量样本(点击数>=2)
            is_short_video = tf.math.less(mean_play_time, 20)
            
            fountain_click_label = batch["fountain_click_label_list"]
            fountain_click_label = tf.RaggedTensor.from_row_splits(fountain_click_label[0], row_splits=fountain_click_label[1])
            fountain_click_label = fountain_click_label.to_tensor()
            fountain_click_sum = tf.reduce_sum(fountain_click_label, axis=-1)
            
            # 长视频需要满足点击数>=2
            long_video_quality = tf.math.greater_equal(fountain_click_sum, 2)
            
            # 过滤规则:
            # 1. 曝光数要>=5
            # 2. 短视频全部保留
            # 3. 长视频只保留高质量样本
            mask = tf.math.logical_or(
                tf.math.less(realshow, 3),  # 曝光数过滤
                tf.math.logical_and(
                    tf.logical_not(is_short_video),  # 长视频
                    tf.logical_not(long_video_quality)  # 且不是高质量样本
                )
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
# label name和cofea_reader.py中各label前缀保持一致
all_model_labels = [
    "fountain_fulllink_rerank_index_list",
    "fountain_fulllink_rerank_index_weight_list",
    "fountain_fulllink_rerank_realshow_label_weight_list",
    "fountain_click_label_list",
    "fountain_finish_label_list"
]

realshow_labels = [
    "context_info__real_show_index_list",
    "context_info__real_show_list",  
]

print("common_attr_names: ", [attr.attr_name for attr in all_features if attr.is_common])
print("all_feature_name: ", [attr.attr_name for attr in all_features])

def mark_common_attr():
    common_embeddings = []
    for attr in all_features:
        if attr.is_common:
            common_embeddings.append(attr.attr_name)
    with open('./infer_server/models/dnn_model.yaml', "r+") as f:
        yaml_config = yaml.load(f.read(), Loader=yaml.FullLoader)
        print(yaml_config['embedding']['slots_config'][0])
        for idx, slot_config in enumerate(yaml_config['embedding']['slots_config']):
            if slot_config['input_name'] in common_embeddings:
                yaml_config['embedding']['slots_config'][idx]['is_common'] = True
        f.seek(0)
        yaml.dump(yaml_config, f)
        f.truncate()

def get_dense_fea(name, list_dim=60):
    assert name in realshow_labels, name
    return config.get_dense_fea(name, dim=list_dim, dtype=tf.int64)

def get_label(name, list_dim=60):
    assert name in all_model_labels, name
    return config.get_label(name, dim=list_dim)

def get_param_dict():
    """
    train and dnn infer：不需要区分common or no_common,(infer配置中对应的tensorflow_use_batching=true)
    tower infer : 需要区分attr是common or no_common
    :return:
    """
    # if args.with_kai_v2:
    #     # share embedding
    #     config.declare_reallocate_slots(share_input_slots,
    #                          share_output_slots,
    #                          remap=True,
    #                          inplace=True)
    #     # 需要额外copy的特征
    #     config.declare_reallocate_slots(copy_input_slots,
    #                          copy_output_slots,
    #                          remap=True,
    #                          inplace=False)
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

# all_model_labels里的名字和cofea_reader.py中名字对应
def get_label_dict():
    label_value_dict = {}
    for label_name in all_model_labels:
        print(f"====> get_label, name : {label_name}")
        label_value = get_label(label_name)
        label_value_dict[label_name] = label_value

    for label_name in realshow_labels:
        print(f"====> get_label, name : {label_name}")
        label_value = get_dense_fea(label_name)
        label_value_dict[label_name] = label_value

    return label_value_dict


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
print("feature_emb_size_dict ", feature_emb_size_dict)
label_value_dict = get_label_dict()

model_class = FountainDeepLtrMultiTaskModel(all_param_dict, label_value_dict, print_ops)

logits, loss, item_weight, generator_loss, generator_logits, cl_loss_states, cl_loss = model_class.model()

def cal_loss(s_logits, t_logits, temperature):
    soft_labels = tf.nn.log_softmax(t_logits / temperature, axis=-1)
    log_prob = tf.nn.log_softmax(s_logits / temperature, axis=-1)
    ori_kd_loss = -tf.exp(soft_labels) * log_prob + tf.exp(soft_labels) * soft_labels
    loss = tf.reduce_mean(tf.reduce_sum(ori_kd_loss, axis=-1))
    
    return loss

if is_training:
    print(f"====> train, gen...")

    targets = []
    sum_loss = 0.0
    list_dim  = 60

    loss_sum = 1000*(loss+cl_loss_states)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('cl_loss_states', cl_loss_states)
    tf.summary.scalar('loss_sum', loss_sum)
    # print_ops.append(tf.print("ryx loss ", loss, summarize = 10, output_stream=sys.stdout))
    print("logits shape", logits.shape)
    print("item_weight shape", item_weight.shape)

    tf.summary.scalar('generator_loss', generator_loss)
    tf.summary.scalar('cl_loss',cl_loss)
    generator_loss_sum = 1000*(generator_loss+cl_loss)
    tf.summary.scalar('generator_loss_sum', generator_loss_sum)

    with tf.control_dependencies(print_ops):
        logits = tf.reduce_sum(logits, axis=-1)
        logits = tf.expand_dims(logits, axis=-1)
        zero = tf.zeros_like(logits)
        one = tf.ones_like(logits)
        print("zero shape", zero.shape)
        print("one shape", one.shape)
        targets.append(('reward', logits, zero, one, 'linear_regression'))

    if args.with_kai_v2:
        sparse_optimizer = config.optimizer.Adam(0.001)
        dense_optimizer = config.optimizer.Adam(0.0005)
        dense_optimizer_gen = config.optimizer.Adam(0.0005)
        total_sparse_var = config.get_sparse_trainable_variables()
        total_dense_var = config.get_dense_trainable_variables()
        dense_gen_var_list = []
        dense_eval_var_list = []
        for var in total_dense_var:
            if "generator" in var.name:
                dense_gen_var_list.append(var)
            else:
                dense_eval_var_list.append(var)

        sparse_optimizer.minimize(generator_loss_sum+loss_sum, var_list=total_sparse_var)
        dense_optimizer.minimize(loss_sum, var_list=dense_eval_var_list)
        dense_optimizer_gen.minimize(generator_loss_sum, var_list=dense_gen_var_list)

        # sparse_optimizer.minimize(loss, var_list=config.get_collection(config.GraphKeys.EMBEDDING_INPUT))
        # dense_optimizer.minimize(loss, var_list=config.get_collection(config.GraphKeys.TRAINABLE_VARIABLES))
        opts = [sparse_optimizer, dense_optimizer, dense_optimizer_gen]
    else:
        optimizer = tf.train.GradientDescentOptimizer(1, name="opt")
        grad_var = optimizer.compute_gradients(loss)
        opt = optimizer.apply_gradients(grad_var)
        opts = [opt]

    if args.dryrun:
        pass  # config.mock_and_profile(opt, './training_log/', batch_sizes=[128, 288])
    elif args.with_kai:
        print(f"====> train, with kai")
        # print(f"====> dump btq, user_top: {user_top}, photo_top: {photo_top}")
        config.dump_kai_training_config('./training/conf', targets, loss=sum_loss, text=args.text, init_params_in_tf=True)
    elif args.with_kai_v2:
        config.build_model(optimizer=opts, metrics=targets)
    else:
        config.dump_training_config('./training/conf', targets, opts=opts, text=args.text)
else:
    # predict
    # predict = tf.transpose(predict,  perm=[0, 2, 1])
    print("predict shape", predict.shape)
    predict = tf.reduce_sum(predict, axis=-1)
    print("predict shape", predict.shape)
    predict = tf.reshape(predict,[-1,1])

    targets = []
    pred_output = tf.identity(predict)
    targets.append(("rerank_gen", pred_output))
    q_names, preds = zip(*targets)
    config.dump_predict_config(
        "./infer_server/models/",
        targets,
        input_type=3,
        extra_preds=q_names,
    )
    print("====> q_name: ", q_names)
    mark_common_attr()

print(f"====> is_training: {is_training}, tower: {args.tower}, dryrun: {args.dryrun}")
