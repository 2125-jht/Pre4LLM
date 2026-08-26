from __future__ import print_function
MODEL_TRANS_ORIGIN='cpp'

import json
import yaml
import logging
import os
import sys

import argparse
import tensorflow as tf
import pandas as pd
import numpy as np

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
            
            label_finish = tf.where(tf.greater_equal(playing_time, duration_ms), tf.ones_like(playing_time), tf.zeros_like(playing_time))
            label_play_over_3s = tf.where(tf.greater_equal(playing_time, 3000), tf.ones_like(playing_time), tf.zeros_like(playing_time))
            label_play_over_7s = tf.where(tf.greater_equal(playing_time, 7000), tf.ones_like(playing_time), tf.zeros_like(playing_time))
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

def my_load_dense_func(warmup_weight: dict, warmup_extra: dict, ps_weight: dict, ps_extra: dict, tf_weight: dict, load_option):
    ''' 
    https://docs.corp.kuaishou.com/k/home/VMPozW5hnQSA/fcAAXcP_sb-h0_8v1lEr7wIqa#section=h.jitvgok6c7vl
    - 参数一：warmup_weight，从base的model加载得到的weight，key为参数名，value为numpy形式的参数值 。 
    - 参数二：warmup_extra，从base的model加载得到的extra(optimizer 依赖参数)，key为参数名，value为numpy形式的参数值
    - 参数三：ps_weight，从参数服务器上拉取的weight，key为参数名，value为numpy形式的参数值
    - 参数四：ps_extra，从参数服务器上拉取的extra，key为参数名，value为numpy形式的参数值
    - 参数五：tf_weight，tensorflow本地通过初始化op生成的weight，key为参数名，value为numpy形式的参数值
    - 参数六：load_option，kai.load()的配置，包含加载参数的地址，加载模式等信息
    - 返回值一：weight(dict), 最终确定的weight组合，key为参数名，value为numpy形式的参数值
    - 返回值二：extra(dict), 最终确定的extra组合，key为参数名，value为numpy形式的参数值
    - 都是这种格式： {weight_name1 : np_array, weight_name2 : np_array, ... }
    '''
    weight = None
    extra = None
    dense_variable_nums = len(tf_weight)#新图的总参数

    if warmup_weight is not None and len(warmup_weight) > 0:
        for var_name in list(warmup_weight):
            print(var_name)
            if var_name not in tf_weight: # 表示参数存在base模型，但新模型没有。即【删除参数】
                print("加载的 dense variable({}) 在运行时不存在，其值被忽略。".format(var_name))
                del warmup_weight[var_name]
                del warmup_extra[var_name]
            elif warmup_weight[var_name].size != tf_weight[var_name].size: # base模型的参数维度和新模型不一样，即此参数被修改，需要额外处理。即【修改参数】
                print("加载的 dense variable({}) size ({} vs {}) 不匹配，进行随机初始化".format(var_name, warmup_weight[var_name].size, tf_weight[var_name].size))
                del warmup_weight[var_name]
                del warmup_extra[var_name]
                warmup_weight[var_name] = np.random.uniform(-1e-4, 1e-4,size=tf_weight[var_name].shape).astype(np.float32)
                warmup_extra[var_name] = np.random.uniform(-1e-4, 1e-4,size=tf_weight[var_name].shape).astype(np.float32)
        weight = warmup_weight
    else:
        weight = tf_weight # 冷启动。用tf初始化。若用 weight=ps_weight，表示weight用ps初始化。

    if warmup_extra is not None and len(warmup_extra) > 0: # 
        for var_name in list(warmup_extra):
            if var_name not in ps_extra:
                print("加载的 dense variable extra({}) 在运行时不存在，其值被忽略。".format(var_name))  # noqa
                del warmup_extra[var_name]
            elif warmup_extra[var_name].size != ps_extra[var_name].size:
                print("加载的 dense variable extra({}) size ({} vs {}) 不匹配，进行随机初始化".format(var_name, warmup_extra[var_name].size, ps_extra[var_name].size))
                del warmup_extra[var_name]
                warmup_extra[var_name] = np.zeros(ps_extra[var_name].shape, dtype=np.float32)
        extra = warmup_extra
    else:
        extra = ps_extra

    if len(weight) < dense_variable_nums: # 不存在base模型里，但存在新模型里，即【新增参数】
        for var_name, var in tf_weight.items():
            if var_name not in weight: # tf_weight 是新模型的所有w，而weight是前面一项项放入的。这里表示新模型有而旧模型没有。
                weight[var_name] = var # 表示：将新增的var_name 的数值arr赋予给 weight。 此处可改为自定义初始化后的arr
                                    # 或者使用 numpy 自己控制数值如何初始化
                # weight[var_name] = np.array()
                print("加载的 dense variable({}) 是新增参数".format(var_name))
            
    if len(extra) < dense_variable_nums: # 不存在base模型里，但存在新模型里，即【新增extra参数】
        for var_name, var in ps_extra.items():
            if var_name not in extra: # ps_extra是新模型的所有extra，而extra是前面一项项放入的
                extra[var_name] = var # 表示：将新增的var_name 的数值arr赋予给 extra，此处可改为自定义初始化后的arr
                print("加载的 dense variable extra({}) 是新增参数".format(var_name)) 

    assert len(weight) == dense_variable_nums
    assert len(extra) == dense_variable_nums

    return weight, extra # 返回让框架来赋值
  
#config.set_load_dense_func(my_load_dense_func)

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
            if attr.is_common:
                # embed, size_var = config.new_embedding(attr.attr_name, dim=attr.dim, slots=attr.slots, expand=attr.expand, compress_group="USER", sized=True)
                embed, size_var = config.new_embedding(attr.attr_name, dim=attr.dim, slots=attr.slots, expand=attr.expand, common=attr.is_common, sized=True)
            else:
                embed, size_var = config.new_embedding(attr.attr_name, dim=attr.dim, slots=attr.slots, expand=attr.expand, sized=True)
            # embed, size_var = config.new_embedding(attr.attr_name, dim=attr.dim, slots=attr.slots, expand=attr.expand, common=attr.is_common, sized=True)
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

    label_finish = tf.where(tf.greater_equal(playing_time, duration_ms), tf.ones_like(playing_time), tf.zeros_like(playing_time))
    label_play_over_7s = tf.where(tf.greater_equal(playing_time, 7000), tf.ones_like(playing_time), tf.zeros_like(playing_time))
    label_read = tf.greater_equal(label_play_over_7s+label_finish+label_like+label_follow+label_comment+label_collect+label_download+label_profile_enter, 0)
    label_read = tf.where(label_read, tf.ones_like(label_read, dtype=tf.float32), tf.zeros_like(label_read, dtype=tf.float32))

    duration_ms_zero_ratio = tf.reduce_mean(
        tf.where(tf.equal(duration_ms, 0), tf.ones_like(duration_ms, dtype=tf.float32), tf.zeros_like(duration_ms, dtype=tf.float32)))
    tf.summary.scalar("duration_ms_zero_ratio", duration_ms_zero_ratio)
    
    return label_click

# print info
all_param_dict, feature_emb_size_dict = get_param_dict()
worker_global_step = config.get_step()
ops = [tf.print("====> step", worker_global_step, summarize=-1, output_stream=sys.stdout)]

# 获取模型output
# all_param_dict, feature_emb_size_dict = get_param_dict()
model = MultiInterestModel(all_param_dict, feature_emb_size_dict, print_ops=print_ops)

if is_training:
    # ==================== 训练模式 ====================
    print("=== 进入训练模式 ===")
    # === 获取基础特征 ===
    uid = config.get_dense_fea("user_info__id", dim=1, dtype=tf.int64)  # 用户ID
    pid = config.get_dense_fea("photo_info__photo_id", dim=1, dtype=tf.int64)  # 视频ID

    # === 获取语义ID相关特征 ===
    photo_semantic_id_int = config.get_dense_fea("photo_semantic_id", dim=1, dtype=tf.int64)
    
    # === 数据预处理 ===
    photo_semantic_id = processInput(photo_semantic_id_int)  # 处理输入序列
    label = processLabel(photo_semantic_id_int)  # 处理标签

    # === 模型前向传播 ===
    print("=== 模型前向传播 ===")
    loss, result_dict = model.model(photo_semantic_id, label, photo_semantic_id_int)  # 计算训练损失
    
    print("=== test beam search ===")
    # _ = model.beam_search(beam_size=1)  # 束搜索推理（训练时也运行用于调试）

    # === 控制依赖和目标设置 ===
    with tf.control_dependencies(print_ops):
        targets = []
        label = gen_custom_label()  # 生成自定义标签
        label_shape = tf.shape(label)
        # === 生成随机预测值（用于测试） ===
        mask = tf.less(tf.random_uniform(label_shape), 0.8)  # 80%概率为True
        # 为mask=True的部分生成[0.5, 1.0]的随机值
        high_vals = tf.random_uniform(label_shape, minval=0.5, maxval=1.0)
        # 为mask=False的部分生成[0.0, 0.5]的随机值
        low_vals = tf.random_uniform(label_shape, minval=0.0, maxval=0.5)
        # 根据mask选择最终值
        result = tf.where(mask, high_vals, low_vals)

        # 添加评估目标：(任务名, 预测值, 真实标签, 权重, 评估指标)
        targets.append(("click", result, label, tf.ones_like(label), "auc"))
        # 添加损失的TensorBoard监控
        with tf.variable_scope("loss", reuse=tf.AUTO_REUSE) as scope:
            tf.summary.scalar('loss', loss)

    # === 优化器设置 ===
    if args.with_kai_v2:
        print("=== 使用Kai v2.0优化器 ===")
        # 分别为稀疏和稠密参数设置优化器
        sparse_optimizer = config.optimizer.Adam(0.0001)  # 稀疏参数优化器
        dense_optimizer = config.optimizer.Adam(0.0001)  # 稠密参数优化器
        # 获取待更新的参数列表
        sparse_var_list = config.Collector().get_collection(config.GraphKeys.EMBEDDING_INPUT)  # 稀疏参数
        print('sparse', sparse_var_list)
        dense_var_list = config.get_collection(config.GraphKeys.TRAINABLE_VARIABLES)  # 稠密参数
        print('dense', dense_var_list)
        # 分别优化稀疏和稠密参数
        sparse_optimizer.minimize(loss, var_list=sparse_var_list)
        dense_optimizer.minimize(loss, var_list=dense_var_list)
        # opts = [sparse_optimizer, dense_optimizer, output_embedding_optimizer]
        opts = [sparse_optimizer, dense_optimizer]
    else:
        print("=== 使用传统TensorFlow优化器 ===")
        # 使用梯度下降优化器
        optimizer = tf.train.GradientDescentOptimizer(1, name="opt")
        grad_var = optimizer.compute_gradients(loss)  # 计算梯度
        opt = optimizer.apply_gradients(grad_var)  # 应用梯度
        opts = [opt]

    # === 根据运行模式进行相应配置 ===
    if args.dryrun:
        # Dry run模式：不执行实际操作
        pass  # config.mock_and_profile(opt, './training_log/', batch_sizes=[128, 288])
    elif args.with_kai:
        print(f"====> train, with kai")
        # 使用Kai v1.0进行训练配置导出
        config.dump_kai_training_config(
            './training/conf',
            targets,
            loss=loss,
            text=args.text,
            init_params_in_tf=True
        )
    elif args.with_kai_v2:
        print(f"====> train, with kai2.0")
        # 使用Kai v2.0构建模型
        config.build_model(optimizer=opts, metrics=targets)
    else:
        print(f"====> train, with mio")
        # 使用MIO框架进行训练配置导出
        config.dump_training_config('./training/conf', targets, opts=opts, text=args.text)

elif args.mode == 'predict':
    # user_sid = model.beam_search(beam_size=256)
    user_sid, user_sid_prob = model.beam_search_fast(beam_size=512, temperature=1)
    # user_sid = tf.reshape(user_sid, [-1, tf.shape(user_sid)[1] * tf.shape(user_sid)[2]])
    user_sid_origin = tf.cast(tf.reshape(user_sid, [-1, tf.shape(user_sid)[1] * tf.shape(user_sid)[2]]), tf.float32)
    user_sid_prob = tf.reshape(user_sid_prob, [-1, tf.shape(user_sid_prob)[1] * tf.shape(user_sid_prob)[2]])
    print('user_sid:', user_sid)
    print("outside model order:")
    user_targets = [
        ("user_sid_origin", user_sid_origin), 
        ("user_sid_prob", user_sid_prob)
    ]
    q_names, preds = zip(*user_targets)
    print("====> q_name: ", q_names)
    config.dump_predict_config('./uni_retr_server_local_ann/predict/conf_gpu/', user_targets, input_type=3, extra_preds=q_names, dump_mode="user_predict")

print(f"====> is_training: {is_training}, tower: {args.tower}, dryrun: {args.dryrun}")
