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
from model import MultiInterestModel
from modules_ import *
from util import *
parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'predict'], dest='mode', default='train')
parser.add_argument('--dryrun', dest='dryrun', const=True, default=False, nargs='?')
parser.add_argument('--with_kai', default=False)
parser.add_argument('--text', default=False)
parser.add_argument('--tower', choices=None, dest='tower', default='False')
parser.add_argument('--with_kai_v2', default=True) #False True 
args = parser.parse_known_args()[0]
is_training = args.mode == "train"

# IS_DEBIAS
IS_DEBIAS = True
DEBIAS_ALPHA = 0.01
# for kai2.0 output emb
output_var_list = []
custom_grad_dict = {}


# 目前这段逻辑功能未知
if args.with_kai_v2:
    import kai.tensorflow as config
    import tensorflow.compat.v1 as tf
    import kai
    from kai.tensorflow.utils import data_table
    default_param_attr = config.nn.ParamAttr(initializer=config.nn.UniformInitializer(0.0001), access_method=config.nn.ProbabilityAccess(100.0), recycle_method=config.nn.UnseendaysRecycle(delete_after_unseen_days=30, delete_threshold=1.0, allow_dynamic_delete=True))
    config.nn.set_default_param_attr(default_param_attr)

    class DumpTensorHook(config.training.RunHookBase):
        def __init__(self, table_name, dump_tensors_dict):
            """
                本Hook用于获取tf图中dump_tensors_dict对应的tensor数据，导出到HDFS上
            Args:
                table_name (string): 表名
                dump_tensors_dict (dict): 需要导出的tensor数据，dict(tensor_name, tensor_op)
            """
            assert isinstance(dump_tensors_dict, dict)
            worker_id = kai.current_rank()
            model_path = kai.Config().save_option.model_path
            # 新建一个表
            self._dump_table = data_table.DataTable(
                table_name=table_name, worker_id=worker_id, model_path=model_path)
            self._dump_tensors_dict = dump_tensors_dict

        def before_step_run(self, step_run_context):
            """
                将 self._dump_tensors_dict 中的tensor注入fetches中
                后续step run图时会自动跑出来对应Tensor的数值

            Args:
                step_run_context (_type_): _description_

            Returns:
                _type_: _description_
            """
            return kai.training.StepRunArgs(fetches=self._dump_tensors_dict)

        def after_step_run(self, step_run_context, step_run_values):
            """
                获取run图的结果，将结果写入表中

            Args:
                step_run_context (_type_): _description_
                step_run_values (_type_): _description_
            """
            sink_data = {}
            for name, op in self._dump_tensors_dict.items():
                value = step_run_values.result[name]
                batch_size = value.shape[0]
                sink_data[name] = value.reshape(batch_size, -1)

            step_id = step_run_context.descr_list.step
            pass_id = step_run_context.descr_list.pass_id
            sink_data["step_id"] = [step_id] * batch_size
            sink_data["pass_id"] = [pass_id] * batch_size
            self._dump_table.append_batch(sink_data)

    #################################### 通过样本自带数据进行过滤 ####################################
    def filter_mask_wrapper(dataset):
        # 1. 声明字段
        #  sample_type为字段名，特征类型dataset.DENSE表示稠密，tf.int64为数据类型，dim为1
        dataset.add_feature('context_info__like', dataset.DENSE, tf.int64, 1)
        dataset.add_feature('context_info__follow', dataset.DENSE, tf.int64, 1)
        dataset.add_feature('context_info__comment', dataset.DENSE, tf.int64, 1)
        dataset.add_feature('context_info__collect', dataset.DENSE, tf.int64, 1)
        dataset.add_feature('context_info__download', dataset.DENSE, tf.int64, 1)
        dataset.add_feature('context_info__profile_enter', dataset.DENSE, tf.int64, 1)
        dataset.add_feature('context_info__playing_time', dataset.DENSE, tf.int64, 1)
        dataset.add_feature('photo_info__duration_ms', dataset.DENSE, tf.int64, 1)
        # dataset.DENSE: 值为tf.Tensor
        # dataset.SPARSE: 值为元组: (tf.Tensor, tf.Tensor)，
        #   其中第一个tensor表示feasign，第二个tensor表示cumsum
        #   可以使用tf.RaggedTensor.from_row_splits转成RaggedTensor
        def mask_fn(batch):
            label_like              = tf.cast(batch['context_info__like'], tf.float32)
            label_follow            = tf.cast(batch['context_info__follow'], tf.float32)
            label_comment           = tf.cast(batch['context_info__comment'], tf.float32)
            label_collect           = tf.cast(batch['context_info__collect'], tf.float32)
            label_download          = tf.cast(batch['context_info__download'], tf.float32)
            label_profile_enter     = tf.cast(batch['context_info__profile_enter'], tf.float32)
            playing_time            = tf.cast(batch['context_info__playing_time'], tf.float32)
            duration_ms             = tf.cast(batch['photo_info__duration_ms'], tf.float32)
            
            label_finish = tf.where(tf.greater(playing_time, duration_ms), tf.ones_like(playing_time), tf.zeros_like(playing_time))
            label_play_over_3s = tf.where(tf.greater(playing_time, 3000), tf.ones_like(playing_time), tf.zeros_like(playing_time))
            label_play_over_7s = tf.where(tf.greater(playing_time, 7000), tf.ones_like(playing_time), tf.zeros_like(playing_time))
            action_cnt = label_play_over_3s*(label_play_over_7s+label_finish+label_like+label_follow+label_comment+label_collect+label_download+label_profile_enter)
            mask = tf.less(action_cnt, 1)
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
    config = MioConfig.from_base_yaml(base_config, clear_embeddings=True, clear_params=True, dryrun=args.dryrun, label_with_kv=True, grad_no_scale=False, with_kai=args.with_kai, predict=(args.mode != "train"))

print("common_attr_names: ", [attr.attr_name for attr in all_features if attr.is_common])
print("all_feature_name: ", [attr.attr_name for attr in all_features])

print_ops = []

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
    for attr in all_features:
        print("--->>> feature %s start" % attr.attr_name)
        if not is_training:
            if attr in infer_ignore_feat:
                print("--->>> ignore feature %s at infer stage" % attr.attr_name)
                return
            if not attr.expand:
                attr.expand = 1
            embed, size_var = config.new_embedding(attr.attr_name, dim=attr.dim, slots=attr.slots, expand=attr.expand, common=attr.is_common, sized=True)
            feature_emb_dict[attr.attr_name] = embed
            feature_emb_size_dict[attr.attr_name] = size_var
        else:
            print(attr.attr_name, attr.dim, attr.slots, attr.expand)
            feature_emb_dict[attr.attr_name] = config.new_embedding(attr.attr_name, dim=attr.dim, slots=attr.slots, expand=attr.expand)
        if attr.expand is not None and attr.expand > 1:
            feature_emb_dict[attr.attr_name] = tf.reshape(feature_emb_dict[attr.attr_name], [-1, attr.expand, attr.dim])
        # 获取长度
        if args.with_kai_v2: 
            sparse_feature = config.get_sparse_fea(name=str(attr.slots[0])) 
            offset = sparse_feature[1]
            size_var = offset[1:] - offset[0:-1]
            feature_emb_size_dict[attr.attr_name] = size_var
            # print("size_var", size_var)
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

def kai_output_embedding(feature, output_emb):
    #新增加的slot
    if args.with_kai_v2:
        custom_grad_dict[feature.name] = output_emb
        output_var_list.append(feature)
    else:
        config.custom_gradients[feature]= output_emb
        #自定义优化器参数
        #AssignAdd优化器 w=decay_rate* w+ add_rate* g
        config.custom_opt[feature]={"opt_type": "AssignAdd", "decay_rate": 0.0, "add_rate": 1.0}

def gen_custom_label():
    # 通过接口获取基础label和数据 形状为[bz,1]
    label_click            = config.get_label("explore_click_label")
    label_like              = tf.cast(config.get_dense_fea("context_info__like", dim=1, dtype=tf.int64), dtype=tf.float32)
    label_follow            = tf.cast(config.get_dense_fea("context_info__follow", dim=1, dtype=tf.int64), dtype=tf.float32)
    label_comment           = tf.cast(config.get_dense_fea("context_info__comment", dim=1, dtype=tf.int64), dtype=tf.float32)
    label_collect           = tf.cast(config.get_dense_fea("context_info__collect", dim=1, dtype=tf.int64), dtype=tf.float32)
    label_download          = tf.cast(config.get_dense_fea("context_info__download", dim=1, dtype=tf.int64), dtype=tf.float32)
    label_profile_enter     = tf.cast(config.get_dense_fea("context_info__profile_enter", dim=1, dtype=tf.int64), dtype=tf.float32)
    playing_time            = tf.cast(config.get_dense_fea("context_info__playing_time", dim=1, dtype=tf.int64), dtype=tf.float32)
    duration_ms             = tf.cast(config.get_dense_fea("photo_info__duration_ms", dim=1, dtype=tf.int64), dtype=tf.float32)

    label_finish = tf.where(tf.greater(playing_time, duration_ms), tf.ones_like(playing_time), tf.zeros_like(playing_time))
    label_play_over_7s = tf.where(tf.greater(playing_time, 7000), tf.ones_like(playing_time), tf.zeros_like(playing_time))
    label_read = tf.greater(label_play_over_7s+label_finish+label_like+label_follow+label_comment+label_collect+label_download+label_profile_enter, 0)
    label_read = tf.where(label_read, tf.ones_like(label_read, dtype=tf.float32), tf.zeros_like(label_read, dtype=tf.float32))
    return label_click

# print info
worker_global_step = config.get_step()
ops = [tf.print("====> step", worker_global_step, summarize=-1, output_stream=sys.stdout)]

# 获取模型output
all_param_dict, feature_emb_size_dict = get_param_dict()
if args.with_kai_v2:
    # avg_step 初始化为 1000, last_step 初始化为 0
    param_attr1 = config.nn.ParamAttr(initializer=config.nn.ConstInitializer(0.0))
    param_attr2 = config.nn.ParamAttr(initializer=config.nn.ConstInitializer(1000.0))
    config.set_slot_param_attr([4003], param_attr1)
    config.set_slot_param_attr([4004], param_attr2)

history_size = 768
selected_size = 64
transformer_num_layer = 3
truncate_dense_feature = processColossusFeature(config, all_param_dict, feature_emb_size_dict, history_size=history_size)
# model_class = MultiTargetModel(all_param_dict, args)
model = MultiInterestModel(all_param_dict, feature_emb_size_dict, truncate_dense_feature, args, 
                        history_size=history_size, print_ops=print_ops, selected_size=selected_size, transformer_num_layer=transformer_num_layer)

if is_training:
    print(f"====> train, gen...")
    interest_embedding, readout, photo_emb, score, _ = model._multi_interest_encode_v2()
    uid = config.get_dense_fea("user_info__id", dim=1, dtype=tf.int64)
    pid = config.get_dense_fea("photo_info__photo_id", dim=1, dtype=tf.int64)
    # get user_emb photo_emb similarity
    user_emb = tf.reshape(interest_embedding, [-1, interest_embedding.shape[1] * interest_embedding.shape[2]])
    tf.summary.histogram("photo_emb", photo_emb)
    tf.summary.histogram("user_emb", user_emb)
    similarity(user_emb, "user_embedding_similarity")
    similarity(photo_emb, "photo_embedding_similarity")
    photo_emb_norm = tf.norm(photo_emb, axis=1)
    print("mxj_photo_emb_norm", photo_emb_norm)

    import kai.tensorflow as kai
    # ..............................
    # ........... 图定义 ............
    # ..............................

    kai.add_run_hook(DumpTensorHook('dump_tensors', {
        'uid': uid,
        'pid': pid,
        'colossus_channel': all_param_dict["truncate_user_colossus_channel_list"],
        'colossus_playtime': truncate_dense_feature["truncate_colossus_play_time_list"],
        'colossus_duration': truncate_dense_feature["truncate_colossus_duration_list"], 
        'colossus_label': truncate_dense_feature["truncate_colossus_label_list"],
        'colossus_channel': truncate_dense_feature['truncate_colossus_channel_list'],
        'score': score,
        "photo_emb_norm": photo_emb_norm
    }), 'custom_dump_tensor_hook')

    # print("interest_embedding", interest_embedding)
    with tf.control_dependencies(print_ops):
        kai_output_embedding(all_param_dict["user_emb"], interest_embedding)
        kai_output_embedding(all_param_dict["photo_emb"], photo_emb)
        label = gen_custom_label()
        if args.with_kai_v2:
            config.set_feature_score_attr("explore_click_label", data_source_name="train")
        targets = []
        losses = {}
        #################################### logQ 热度消偏 ####################################
        last_step = all_param_dict["last_step"]
        ave_step  = all_param_dict["ave_step"]
        tf.summary.histogram("ave_step", ave_step)
        current_batch_step = tf.cast(tf.tile(tf.reshape(config.get_step(), [1, -1]), [tf.shape(last_step)[0], 1]), last_step.dtype)
        logQ = tf.reshape(tf.log(tf.ones_like(ave_step) / tf.maximum(ave_step, tf.ones_like(ave_step))),[-1,1])
        step_diff = current_batch_step - last_step
        ave_step_grad = tf.where(tf.less(step_diff,tf.zeros_like(step_diff)), ave_step, step_diff)
        # exponential moving averge to estimate ave_step
        new_ave_step = (1-DEBIAS_ALPHA)*ave_step + DEBIAS_ALPHA*ave_step_grad
        with tf.variable_scope("logQ", reuse=tf.AUTO_REUSE) as scope:
            tf.summary.scalar('avg_step', tf.reduce_mean(ave_step))
            tf.summary.scalar('last_step', tf.reduce_mean(last_step))

        output = sigmoid_layer("ctr_click", readout, photo_emb)
        targets.append(("click", output, label, tf.ones_like(label), "auc"))
        loss, cos_mat = sampled_softmax_loss(label, readout, photo_emb, logQ=None, name="click")

        recall_at_k(cos_mat, top_k=[1, 10], indicator=label, name="click")
        with tf.variable_scope("label", reuse=tf.AUTO_REUSE) as scope:
                tf.summary.scalar('click_rate', tf.reduce_mean(label))
        with tf.variable_scope("loss", reuse=tf.AUTO_REUSE) as scope:
            tf.summary.scalar('loss', loss)

        # for label, name in zip(labels, outputs):
        #     targets.append((name, outputs[name], label, tf.ones_like(label), "auc"))
        #     loss, cos_mat = sampled_softmax_loss(label, user_embs[name], photo_embs[name], logQ=None, name=name)
        #     losses[name] = loss
        #     recall_at_k(cos_mat, indicator=label, name=name)
        #     with tf.variable_scope("label", reuse=tf.AUTO_REUSE) as scope:
        #         tf.summary.scalar('{}_rate'.format(name), tf.reduce_mean(label))

        # sum_loss = sum([losses[name] for name in losses])

        # with tf.variable_scope("loss", reuse=tf.AUTO_REUSE) as scope:
        #     tf.summary.scalar('sum_loss', sum_loss)
        #     for name in losses:
        #         tf.summary.scalar('loss_{}'.format(name), losses[name])

    if args.with_kai_v2:
        sparse_optimizer = config.optimizer.Adam(0.0001)
        dense_optimizer = config.optimizer.Adam(0.0001)
        # 取出所有待更新的 sparse 参数, 用于从中分离出使用自定义优化器更新的参数
        sparse_var_list = config.Collector().get_collection(config.GraphKeys.EMBEDDING_INPUT)
        #################################### logQ 热度消偏自定义优化器 ####################################
        last_step_optimizer = config.optimizer.AssignAddOptimizer(decay_rate=0, add_rate=1.0)
        last_step_optimizer.minimize(loss, var_list=[last_step], custom_gradient={last_step.name: current_batch_step})
        ave_step_optimizer = config.optimizer.AssignAddOptimizer(decay_rate=0, add_rate=1.0)
        ave_step_optimizer.minimize(loss, var_list=[ave_step], custom_gradient={ave_step.name: new_ave_step})
        logQ_var_list = [last_step, ave_step]
        for sparse_var in logQ_var_list:
            print("remove", sparse_var, "because logQ_var_list")
            sparse_var_list.remove(sparse_var)
        #################################### 双塔模型 top emb 自定义优化器 ####################################
        output_embedding_optimizer = config.optimizer.AssignAddOptimizer(decay_rate=0, add_rate=1)
        output_embedding_optimizer.minimize(loss, var_list=output_var_list, custom_gradient=custom_grad_dict)
        for sparse_var in output_var_list:
            print("remove", sparse_var, "because output_top_layer")
            sparse_var_list.remove(sparse_var)
        sparse_optimizer.minimize(loss, var_list=sparse_var_list)
        dense_optimizer.minimize(loss, var_list=config.get_collection(config.GraphKeys.TRAINABLE_VARIABLES))
        opts = [sparse_optimizer, dense_optimizer,last_step_optimizer,ave_step_optimizer,output_embedding_optimizer]
    else:
        #################################### logQ 热度消偏自定义优化器 ####################################
        config.custom_gradients[last_step] = current_batch_step
        config.custom_gradients[ave_step] = new_ave_step
        # 使用 AssignAdd 更新 last step 和 ave step
        # AssignAdd优化器 w=decay_rate* w+ add_rate* g
        config.custom_opt[last_step] = {"opt_type": "AssignAdd","decay_rate": 0.0,"add_rate": 1.0,"initializer": {"type": "Const","const": 0.}}
        config.custom_opt[ave_step] = {"opt_type": "AssignAdd","decay_rate": 0.0,"add_rate": 1.0,"initializer": {"type": "Const","const": 1000.0}}

        optimizer = tf.train.GradientDescentOptimizer(1, name="opt")
        grad_var = optimizer.compute_gradients(loss)
        opt = optimizer.apply_gradients(grad_var)
        opts = [opt]

    if args.dryrun:
        pass  # config.mock_and_profile(opt, './training_log/', batch_sizes=[128, 288])
    elif args.with_kai:
        print(f"====> train, with kai")
        # print(f"====> dump btq, user_top: {user_top}, photo_top: {photo_top}")
        config.dump_kai_training_config('./training/conf', targets, loss=loss, text=args.text, init_params_in_tf=True)
    elif args.with_kai_v2:
        print(f"====> train, with kai2.0")
        config.build_model(optimizer=opts, metrics=targets)
    else:
        config.dump_training_config('./training/conf', targets, opts=opts, text=args.text)
elif args.mode == 'predict':
        interest_embedding, readout, photo_emb, score, indices = model._multi_interest_encode_v2()
        print("outside model order:")
        user_emb = tf.reshape(interest_embedding, [-1, interest_embedding.shape[1] * interest_embedding.shape[2]])
        user_targets = [("user_emb", user_emb), ("indices", indices), ("scores", score)]
        q_names, preds = zip(*user_targets)
        print("====> q_name: ", q_names)
        config.dump_predict_config('./uni_retr_server_local_ann_get_topK/predict/conf', user_targets, input_type=3, extra_preds=q_names, dump_mode="user_predict")

print(f"====> is_training: {is_training}, tower: {args.tower}, dryrun: {args.dryrun}")

