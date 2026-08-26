# -*- coding: utf-8 -*-
"""
基于Transformer的多兴趣推荐模型 - Kai框架训练和推理脚本

主要功能：
1. 支持Kai v2.0和传统TensorFlow框架
2. 实现用户兴趣建模和序列生成
3. 支持训练和推理两种模式
4. 集成样本过滤、自定义优化器等高级功能
"""

from __future__ import print_function
MODEL_TRANS_ORIGIN = 'cpp'  # 模型转换来源标识

import json
import yaml
import logging
import os
import sys

import argparse
import tensorflow as tf
import pandas as pd
import numpy as np

# 导入自定义模块
from feature_attr_extract import *  # 特征属性提取相关
from model import MultiInterestModel  # 多兴趣模型
from modules_ import *  # 模型组件
from util import *  # 工具函数
# === 命令行参数解析 ===
parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'predict'], dest='mode', default='train',
                    help='运行模式：train(训练) 或 predict(推理)')
parser.add_argument('--dryrun', dest='dryrun', const=True, default=False, nargs='?',
                    help='是否为dry run模式，用于调试')
parser.add_argument('--with_kai', default=False,
                    help='是否使用Kai v1.0框架')
parser.add_argument('--text', default=False,
                    help='是否使用文本模式')
parser.add_argument('--tower', choices=None, dest='tower', default='False',
                    help='是否使用tower模式')
parser.add_argument('--with_kai_v2', default=True,  # False True
                    help='是否使用Kai v2.0框架')
args = parser.parse_known_args()[0]
is_training = args.mode == "train"  # 判断是否为训练模式

# === 去偏配置 ===
IS_DEBIAS = True  # 是否启用去偏
DEBIAS_ALPHA = 0.01  # 去偏超参数

# === Kai 2.0 相关变量 ===
# 用于输出嵌入的变量列表和自定义梯度字典
output_var_list = []
custom_grad_dict = {}

# === Kai框架初始化 ===
if args.with_kai_v2:
    # 使用Kai v2.0框架
    import kai.tensorflow as config
    import tensorflow.compat.v1 as tf
    import kai
    from kai.tensorflow.utils import data_table
    # 设置默认参数属性：初始化器、访问方法、回收策略
    default_param_attr = config.nn.ParamAttr(
        initializer=config.nn.UniformInitializer(0.0001),  # 均匀分布初始化
        access_method=config.nn.ProbabilityAccess(100.0),  # 100%概率访问
        recycle_method=config.nn.UnseendaysRecycle(  # 基于未见天数的回收策略
            delete_after_unseen_days=30,  # 30天未见后删除
            delete_threshold=1.0,  # 删除阈值
            allow_dynamic_delete=True  # 允许动态删除
        )
    )
    config.nn.set_default_param_attr(default_param_attr)

    class DumpTensorHook(config.training.RunHookBase):
        """
        Tensor数据导出Hook类
        
        用于在训练过程中将指定的tensor数据导出到HDFS表中，
        便于后续分析和调试
        """
        def __init__(self, table_name, dump_tensors_dict):
            """
            初始化Tensor导出Hook
            
            Args:
                table_name (str): HDFS表名
                dump_tensors_dict (dict): 需要导出的tensor字典，
                                        格式: {tensor_name: tensor_op}
            """
            assert isinstance(dump_tensors_dict, dict)
            worker_id = kai.current_rank()  # 获取当前worker ID
            model_path = kai.Config().save_option.model_path  # 获取模型保存路径

            # 创建数据表用于存储tensor数据
            self._dump_table = data_table.DataTable(
                table_name=table_name,
                worker_id=worker_id,
                model_path=model_path
            )
            self._dump_tensors_dict = dump_tensors_dict

        def before_step_run(self, step_run_context):
            """
            每步训练前的回调函数
            
            将需要导出的tensor注入到fetches中，
            使得在图执行时能够获取这些tensor的数值
            
            Args:
                step_run_context: 步骤运行上下文
                
            Returns:
                StepRunArgs: 包含fetches的运行参数
            """
            return kai.training.StepRunArgs(fetches=self._dump_tensors_dict)

        def after_step_run(self, step_run_context, step_run_values):
            """
            每步训练后的回调函数
            
            获取tensor的运行结果并写入数据表
            
            Args:
                step_run_context: 步骤运行上下文
                step_run_values: 步骤运行结果
            """
            sink_data = {}
            # 处理每个tensor的结果
            for name, op in self._dump_tensors_dict.items():
                value = step_run_values.result[name]
                batch_size = value.shape[0]
                # 将tensor重塑为二维：[batch_size, -1]
                sink_data[name] = value.reshape(batch_size, -1)

            # 添加步骤信息
            step_id = step_run_context.descr_list.step
            pass_id = step_run_context.descr_list.pass_id
            sink_data["step_id"] = [step_id] * batch_size
            sink_data["pass_id"] = [pass_id] * batch_size

            # 批量写入数据表
            self._dump_table.append_batch(sink_data)

    def filter_mask_wrapper(dataset):
        """
        样本过滤包装器
        
        基于用户行为数据定义样本过滤条件，过滤掉低质量样本
        
        Args:
            dataset: 数据集对象
            
        Returns:
            mask_fn: 掩码函数，返回需要过滤的样本掩码
        """
        # === 1. 声明特征字段 ===
        # 用户交互行为特征
        dataset.add_feature('context_info__like', dataset.DENSE, tf.int64, 1)  # 点赞
        dataset.add_feature('context_info__follow', dataset.DENSE, tf.int64, 1)  # 关注
        dataset.add_feature('context_info__comment', dataset.DENSE, tf.int64, 1)  # 评论
        dataset.add_feature('context_info__collect', dataset.DENSE, tf.int64, 1)  # 收藏
        dataset.add_feature('context_info__download', dataset.DENSE, tf.int64, 1)  # 下载
        dataset.add_feature('context_info__profile_enter', dataset.DENSE, tf.int64, 1)  # 进入主页
        dataset.add_feature('context_info__playing_time', dataset.DENSE, tf.int64, 1)  # 播放时长
        dataset.add_feature('photo_info__duration_ms', dataset.DENSE, tf.int64, 1)  # 视频总时长
        # 特征类型说明：
        # dataset.DENSE: 稠密特征，值为tf.Tensor
        # dataset.SPARSE: 稀疏特征，值为元组(tf.Tensor, tf.Tensor)
        #   第一个tensor表示feasign，第二个tensor表示cumsum
        #   可使用tf.RaggedTensor.from_row_splits转成RaggedTensor

        def mask_fn(batch):
            """
            定义具体的过滤逻辑
            
            Args:
                batch: 批次数据
                
            Returns:
                mask: 布尔掩码，True表示需要过滤的样本
            """
            # === 2. 提取各种用户行为特征 ===
            label_like = tf.cast(batch['context_info__like'], tf.float32)
            label_follow = tf.cast(batch['context_info__follow'], tf.float32)
            label_comment = tf.cast(batch['context_info__comment'], tf.float32)
            label_collect = tf.cast(batch['context_info__collect'], tf.float32)
            label_download = tf.cast(batch['context_info__download'], tf.float32)
            label_profile_enter = tf.cast(batch['context_info__profile_enter'], tf.float32)
            playing_time = tf.cast(batch['context_info__playing_time'], tf.float32)
            duration_ms = tf.cast(batch['photo_info__duration_ms'], tf.float32)
            # === 3. 计算衍生特征 ===
            # 是否完播：播放时长 > 视频总时长
            label_finish = tf.where(
                tf.greater(playing_time, duration_ms),
                tf.ones_like(playing_time),
                tf.zeros_like(playing_time)
            )

            # 是否播放超过7秒
            label_play_over_7s = tf.where(
                tf.greater(playing_time, 7000),
                tf.ones_like(playing_time),
                tf.zeros_like(playing_time)
            )
            # === 4. 计算综合行为得分 ===
            # 综合考虑播放时长和各种交互行为
            action_cnt = label_play_over_7s * (
                    label_play_over_7s + label_finish + label_like +
                    label_follow + label_comment + label_collect +
                    label_download + label_profile_enter
            )
            # === 5. 生成过滤掩码 ===
            # 过滤掉行为得分小于1的样本（低质量样本）
            mask = tf.less(action_cnt, 1)
            return mask

        return mask_fn
    # 注册样本过滤器到训练数据源
    config.declare_sample_filter(filter_mask_wrapper, data_source_name='train')

else:
    # 使用传统TensorFlow框架或MIO框架
    import tensorflow as tf
    from mio_tensorflow.config import MioConfig
    # 在非dry run且非Kai模式下应用MIO补丁
    if not args.dryrun and not args.with_kai:
        import mio_tensorflow.patch as mio_tensorflow_patch
        mio_tensorflow_patch.apply()
    # 配置日志
    logging.basicConfig()
    # 加载配置文件
    base_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), './base.yaml')
    config = MioConfig.from_base_yaml(
        base_config,
        clear_embeddings=True,  # 清空嵌入
        clear_params=True,  # 清空参数
        dryrun=args.dryrun,  # 干跑模式
        label_with_kv=True,  # 标签带键值
        grad_no_scale=False,  # 梯度不缩放
        with_kai=args.with_kai,  # 是否使用Kai
        predict=(args.mode != "train")  # 是否为预测模式
    )
    
def my_load_dense_func(warmup_weight: dict, warmup_extra: dict, ps_weight: dict, ps_extra: dict, tf_weight: dict, load_option):
    """
    自定义稠密参数加载函数
    
    用于处理模型参数的热启动加载，支持参数的增删改场景
    参考文档: https://docs.corp.kuaishou.com/k/home/VMPozW5hnQSA/fcAAXcP_sb-h0_8v1lEr7wIqa#section=h.jitvgok6c7vl
    
    Args:
        warmup_weight (dict): 从base模型加载的权重参数
        warmup_extra (dict): 从base模型加载的额外参数(optimizer相关)
        ps_weight (dict): 从参数服务器拉取的权重参数
        ps_extra (dict): 从参数服务器拉取的额外参数
        tf_weight (dict): TensorFlow本地初始化的权重参数
        load_option: kai.load()的配置信息
        
    Returns:
        tuple: (weight_dict, extra_dict) 最终确定的参数组合
    """
    weight = None
    extra = None
    dense_variable_nums = len(tf_weight)  # 稠密变量数量

    # === 处理权重参数 ===
    if warmup_weight is not None and len(warmup_weight) > 0:
        # 检查warmup模型中的每个参数
        for var_name in list(warmup_weight):
            print(var_name)

            if var_name not in tf_weight:
                # 场景1：【删除参数】- 参数存在于base模型但新模型中没有
                print("加载的 dense variable({}) 在运行时不存在，其值被忽略。".format(var_name))
                del warmup_weight[var_name]
                del warmup_extra[var_name]
            elif warmup_weight[var_name].size != tf_weight[var_name].size:
                # 场景2：【修改参数】- 参数维度发生变化
                print("加载的 dense variable({}) size ({} vs {}) 不匹配，进行随机初始化".format(
                    var_name, warmup_weight[var_name].size, tf_weight[var_name].size))
                del warmup_weight[var_name]
                del warmup_extra[var_name]
                # 重新随机初始化
                warmup_weight[var_name] = np.random.uniform(
                    -1e-4, 1e-4, size=tf_weight[var_name].shape
                ).astype(np.float32)
                warmup_extra[var_name] = np.random.uniform(
                    -1e-4, 1e-4, size=tf_weight[var_name].shape
                ).astype(np.float32)
        weight = warmup_weight
    else:
        # 冷启动：使用TensorFlow初始化的权重
        weight = tf_weight

    # === 处理额外参数(optimizer相关) ===
    if warmup_extra is not None and len(warmup_extra) > 0:
        for var_name in list(warmup_extra):
            if var_name not in ps_extra:
                print("加载的 dense variable extra({}) 在运行时不存在，其值被忽略。".format(var_name))
                del warmup_extra[var_name]
            elif warmup_extra[var_name].size != ps_extra[var_name].size:
                print("加载的 dense variable extra({}) size ({} vs {}) 不匹配，进行随机初始化".format(
                    var_name, warmup_extra[var_name].size, ps_extra[var_name].size))
                del warmup_extra[var_name]
                # 用零初始化额外参数
                warmup_extra[var_name] = np.zeros(ps_extra[var_name].shape, dtype=np.float32)
        extra = warmup_extra
    else:
        extra = ps_extra

    # === 处理新增参数 ===
    if len(weight) < dense_variable_nums:
        # 场景3：【新增参数】- 新模型有但旧模型没有的参数
        for var_name, var in tf_weight.items():
            if var_name not in weight:
                weight[var_name] = var  # 使用TensorFlow初始化值
                print("加载的 dense variable({}) 是新增参数".format(var_name))
    if len(extra) < dense_variable_nums:
        # 新增额外参数处理
        for var_name, var in ps_extra.items():
            if var_name not in extra:
                extra[var_name] = var
                print("加载的 dense variable extra({}) 是新增参数".format(var_name))

    # 确保参数数量一致
    assert len(weight) == dense_variable_nums
    assert len(extra) == dense_variable_nums

    return weight, extra

# 注释掉的自定义加载函数设置
# config.set_load_dense_func(my_load_dense_func)

# === 打印特征信息 ===
print("common_attr_names: ", [attr.attr_name for attr in all_features if attr.is_common])
print("all_feature_name: ", [attr.attr_name for attr in all_features])

print_ops = []  # 用于存储调试打印操作

def mark_common_attr():
    """
    标记通用属性
    
    将通用嵌入特征在配置文件中标记为is_common=True，
    用于推理服务的配置生成
    """
    common_embeddings = []
    # 收集所有通用特征名称
    for attr in all_features:
        if attr.is_common:
            common_embeddings.append(attr.attr_name)
    # 更新YAML配置文件
    with open('./infer_server/models/dnn_model.yaml', "r+") as f:
        yaml_config = yaml.load(f.read(), Loader=yaml.FullLoader)
        print(yaml_config['embedding']['slots_config'][0])
        # 为通用特征设置is_common标志
        for idx, slot_config in enumerate(yaml_config['embedding']['slots_config']):
            if slot_config['input_name'] in common_embeddings:
                yaml_config['embedding']['slots_config'][idx]['is_common'] = True
        # 写回文件
        f.seek(0)
        yaml.dump(yaml_config, f)
        f.truncate()

def get_param_dict():
    """
    获取特征参数字典
    
    根据训练/推理模式和框架类型，初始化所有特征的嵌入参数
    
    说明：
    - train和dnn infer：不需要区分common/no_common (tensorflow_use_batching=true)
    - tower infer：需要区分attr是common/no_common
    
    Returns:
        tuple: (feature_emb_dict, feature_emb_size_dict) 特征嵌入字典和大小字典
    """
    if args.with_kai_v2:
        # === Kai v2.0 框架下的特征共享配置 ===
        # 共享嵌入：多个输入slot共享同一个输出slot
        config.declare_reallocate_slots(share_input_slots, share_output_slots, remap=True, inplace=True)
        # 复制嵌入：需要额外copy的特征
        config.declare_reallocate_slots(copy_input_slots, copy_output_slots, remap=True, inplace=False)

    feature_emb_dict = {}  # 特征嵌入字典
    feature_emb_size_dict = {}  # 特征大小字典
    # === 遍历所有特征进行初始化 ===
    for attr in all_features:
        print("--->>> feature %s start" % attr.attr_name)

        if not is_training:
            # === 推理模式处理 ===
            if attr in infer_ignore_feat:
                print("--->>> ignore feature %s at infer stage" % attr.attr_name)
                return

            # 设置默认expand
            if not attr.expand:
                attr.expand = 1
            # 根据是否为通用特征创建嵌入
            if attr.is_common:
                embed, size_var = config.new_embedding(
                    attr.attr_name,
                    dim=attr.dim,
                    slots=attr.slots,
                    expand=attr.expand,
                    compress_group="USER",  # 通用特征使用USER压缩组
                    sized=True
                )
            else:
                embed, size_var = config.new_embedding(
                    attr.attr_name,
                    dim=attr.dim,
                    slots=attr.slots,
                    expand=attr.expand,
                    sized=True
                )
            feature_emb_dict[attr.attr_name] = embed
            feature_emb_size_dict[attr.attr_name] = size_var
        else:
            # === 训练模式处理 ===
            print(attr.attr_name, attr.dim, attr.slots, attr.expand)
            feature_emb_dict[attr.attr_name] = config.new_embedding(
                attr.attr_name,
                dim=attr.dim,
                slots=attr.slots,
                expand=attr.expand
            )
        # === 处理多维展开的特征 ===
        if attr.expand is not None and attr.expand > 1:
            # 重塑为3D张量: [batch_size, expand, dim]
            feature_emb_dict[attr.attr_name] = tf.reshape(
                feature_emb_dict[attr.attr_name],
                [-1, attr.expand, attr.dim]
            )

        # === 获取特征长度信息 ===
        if args.with_kai_v2:
            # Kai v2.0框架下获取稀疏特征长度
            sparse_feature = config.get_sparse_fea(name=str(attr.slots[0]))
            offset = sparse_feature[1]  # 偏移量数组
            size_var = offset[1:] - offset[0:-1]  # 计算每个样本的特征长度
            feature_emb_size_dict[attr.attr_name] = size_var
            # 调试：打印slot 16的RaggedTensor
            if attr.slots[0] == 16:
                tt = tf.RaggedTensor.from_row_splits(
                    values=sparse_feature[0],
                    row_splits=sparse_feature[1]
                ).to_tensor()
                # print_ops.append(tf.print("[Test test] slot " + str(attr.slots[0]), tt, output_stream=sys.stdout))
        elif args.with_kai:
            # Kai v1.0框架下获取特征长度
            offset = tf.cast(config.get_signs(attr.slots[0])[1], tf.int32)
            size_var = offset[1:] - offset[0:-1]
            feature_emb_size_dict[attr.attr_name] = size_var
        print("--->>> feature {} = {}".format(attr.attr_name, feature_emb_dict[attr.attr_name]))
        print("--->>> feature %s normal" % attr.attr_name)

    return feature_emb_dict, feature_emb_size_dict

def kai_output_embedding(feature, output_emb):
    """
    设置Kai框架的输出嵌入
    
    为指定特征设置自定义的输出嵌入，通常用于双塔模型的top层嵌入
    
    Args:
        feature: 特征张量
        output_emb: 输出嵌入张量
    """
    if args.with_kai_v2:
        # Kai v2.0框架：使用自定义梯度字典
        custom_grad_dict[feature.name] = output_emb
        output_var_list.append(feature)
    else:
        # Kai v1.0框架：直接设置自定义梯度和优化器
        config.custom_gradients[feature] = output_emb
        # 自定义优化器参数：AssignAdd优化器 w = decay_rate * w + add_rate * g
        config.custom_opt[feature] = {
            "opt_type": "AssignAdd",
            "decay_rate": 0.0,
            "add_rate": 1.0
        }

def gen_custom_label():
    """
    生成自定义标签
    
    基于用户的多种行为数据生成综合的阅读标签，
    用于多任务学习或辅助监督
    
    Returns:
        label_click: 点击标签张量
    """
    # === 获取基础标签和行为数据 ===
    label_click = config.get_label("explore_click_label")  # 基础点击标签
    # 各种用户行为特征 (形状为[batch_size, 1])
    label_like = tf.cast(config.get_dense_fea("context_info__like", dim=1, dtype=tf.int64), dtype=tf.float32)
    label_follow = tf.cast(config.get_dense_fea("context_info__follow", dim=1, dtype=tf.int64), dtype=tf.float32)
    label_comment = tf.cast(config.get_dense_fea("context_info__comment", dim=1, dtype=tf.int64), dtype=tf.float32)
    label_collect = tf.cast(config.get_dense_fea("context_info__collect", dim=1, dtype=tf.int64), dtype=tf.float32)
    label_download = tf.cast(config.get_dense_fea("context_info__download", dim=1, dtype=tf.int64), dtype=tf.float32)
    label_profile_enter = tf.cast(config.get_dense_fea("context_info__profile_enter", dim=1, dtype=tf.int64), dtype=tf.float32)
    playing_time = tf.cast(config.get_dense_fea("context_info__playing_time", dim=1, dtype=tf.int64), dtype=tf.float32)
    duration_ms = tf.cast(config.get_dense_fea("photo_info__duration_ms", dim=1, dtype=tf.int64), dtype=tf.float32)

    # === 生成衍生标签 ===
    # 完播标签：播放时长 > 视频总时长
    label_finish = tf.where(
        tf.greater(playing_time, duration_ms),
        tf.ones_like(playing_time),
        tf.zeros_like(playing_time)
    )
    # 超过7秒播放标签
    label_play_over_7s = tf.where(
        tf.greater(playing_time, 7000),
        tf.ones_like(playing_time),
        tf.zeros_like(playing_time)
    )
    # === 综合阅读标签 ===
    # 综合考虑播放时长和各种交互行为
    label_read = tf.greater(
        label_play_over_7s + label_finish + label_like +
        label_follow + label_comment + label_collect +
        label_download + label_profile_enter,
        0
    )
    label_read = tf.where(
        label_read,
        tf.ones_like(label_read, dtype=tf.float32),
        tf.zeros_like(label_read, dtype=tf.float32)
    )
    return label_click

##############################################################################################################

# === 初始化模型和参数 ===
print("=== 初始化特征参数字典 ===")
all_param_dict, feature_emb_size_dict = get_param_dict()
worker_global_step = config.get_step()  # 获取全局步数
ops = [tf.print("====> step", worker_global_step, summarize=-1, output_stream=sys.stdout)]

# === 创建多兴趣模型实例 ===
print("=== 创建多兴趣模型 ===")
model = MultiInterestModel(all_param_dict, feature_emb_size_dict, print_ops=print_ops)

if is_training:
    # ==================== 训练模式 ====================
    print("=== 进入训练模式 ===")
    # === 获取基础特征 ===
    uid = config.get_dense_fea("user_info__id", dim=1, dtype=tf.int64)  # 用户ID
    pid = config.get_dense_fea("photo_info__photo_id", dim=1, dtype=tf.int64)  # 视频ID

    # === 获取语义ID相关特征 ===
    photo_semantic_id_int = config.get_dense_fea("photo_semantic_id", dim=1, dtype=tf.int64)
    
    # 准备调试打印数据
    # photo_semantic_id_int_print = tf.reshape(photo_semantic_id_int, [1, -1])
    # pid_print = tf.reshape(pid, [1, -1])
    # print_tensor = tf.concat([pid_print, photo_semantic_id_int_print], axis=0)
    # print_ops.append(tf.print("photo_semantic_id_int", print_tensor, summarize=-1, output_stream=sys.stdout))
    
    # === 数据预处理 ===
    photo_semantic_id = processInput(photo_semantic_id_int)  # 处理输入序列
    label = processLabel(photo_semantic_id_int)  # 处理标签

    # === 模型前向传播 ===
    print("=== 模型前向传播 ===")
    loss = model.model(photo_semantic_id, label, photo_semantic_id_int)  # 计算训练损失
    
    print("=== test beam search ===")
    # _ = model.beam_search(beam_size=1)  # 束搜索推理（训练时也运行用于调试）

    # 获取批次大小
    batch_size = tf.cast(tf.shape(uid)[0], tf.float32)
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
        # === 双塔模型top层嵌入自定义优化器（注释掉的代码） ===
        # output_embedding_optimizer = config.optimizer.AssignAddOptimizer(decay_rate=0, add_rate=1)
        # output_embedding_optimizer.minimize(loss, var_list=output_var_list, custom_gradient=custom_grad_dict)
        # for sparse_var in output_var_list:
        #     print("remove", sparse_var, "because output_top_layer")
        #     sparse_var_list.remove(sparse_var)
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
    # ==================== 推理模式 ====================
    print("=== 进入推理模式 ===")
    # === 模型推理 ===
    user_sid = model.beam_search()  # 束搜索生成用户兴趣序列
    user_sid = user_sid[:, :, 1:]  # 去掉起始token

    # 数据后处理
    # user_sid = tf.reshape(user_sid, [-1, tf.shape(user_sid)[1] * tf.shape(user_sid)[2]])
    user_sid_int = processOutputV2(user_sid)  # 处理输出格式
    user_sid_origin = tf.reshape(user_sid, [-1, tf.shape(user_sid)[1] * tf.shape(user_sid)[2]])

    print('user_sid:', user_sid)
    print("outside model order:")
    # === 定义推理输出目标 ===
    user_targets = [
        ("user_sid_origin", user_sid_origin),  # 原始用户兴趣序列
        ("user_sid_int", user_sid_int)  # 处理后的用户兴趣序列
    ]
    q_names, preds = zip(*user_targets)
    print("====> q_name: ", q_names)
    # === 导出推理配置 ===
    config.dump_predict_config(
        './uni_retr_server_local_ann/predict/conf',  # 配置输出路径
        user_targets,  # 推理目标
        input_type=3,  # 输入类型
        extra_preds=q_names,  # 额外预测输出
        dump_mode="user_predict"  # 导出模式
    )
    
# === 打印运行信息 ===
print(f"====> is_training: {is_training}, tower: {args.tower}, dryrun: {args.dryrun}")
