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
# parser.add_argument('--with_kai', default=False)
parser.add_argument('--with_kai', default=True)
# parser.add_argument('--with_kai_v2', default=True) #False True 
parser.add_argument('--with_kai_v2', default=False) #False True 
parser.add_argument('--text', default=False)
parser.add_argument('--tower', choices=None, dest='tower', default='False')

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
        dataset.add_feature('fountain_click_label_list',dataset.DENSE,tf.int64,max_length=60)
        dataset.add_feature('fountain_finish_label_list',dataset.DENSE,tf.int64,max_length=60)
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
            is_short_video = tf.math.less(mean_play_time, 20) #短视频定义为<20s
            
            # 点击率:
            fountain_click_label = batch["fountain_click_label_list"]
            fountain_click_label = tf.RaggedTensor.from_row_splits(fountain_click_label[0], row_splits=fountain_click_label[1])
            fountain_click_label = fountain_click_label.to_tensor()
            fountain_click_sum = tf.reduce_sum(fountain_click_label, axis=-1)
            
            # 长视频需要满足点击数>=为高质量长视频
            long_video_quality = tf.math.greater_equal(fountain_click_sum, 2)
            
            # 完播率:
            fountain_finish_label = batch["fountain_finish_label_list"]
            fountain_finish_label = tf.RaggedTensor.from_row_splits(fountain_finish_label[0], row_splits=fountain_finish_label[1])
            fountain_finish_label = fountain_finish_label.to_tensor()
            fountain_is_finish = tf.cast(tf.math.greater(fountain_finish_label, 0.5),tf.int64) #if is_finish>0.5, set 1; else 0 
            fountain_is_finish_sum = tf.reduce_sum(fountain_is_finish, axis=-1) #>0.5的总和
            
            # 过滤规则:
            mask = tf.math.logical_or(tf.math.less(realshow,5), #曝光数<5的过滤
                                      tf.math.logical_and(
                                          tf.logical_not(is_short_video), #非短视频
                                          tf.logical_not(long_video_quality) #非高质量长视频
                                        ) 
                                      )
            
            return mask
        # 3.返回mask_fn
        return mask_fn
    # 注册过滤条件
    config.declare_sample_filter(filter_mask_wrapper, data_source_name='train')
    import kai.tensorflow as kai
    from kai.tensorflow.utils import data_table


    class DumpTensorHook(kai.training.RunHookBase):
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
            #! kaiwork时注释掉
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
# batch_size = tf.cast(tf.size(all_param_dict["pId_KAI"][:, 0]), dtype=tf.float32)

model_class = FountainDeepLtrMultiTaskModel(all_param_dict, label_value_dict, print_ops)

# predictv1, predictv2, cl_loss = model_class.model()
predict, cl_loss = model_class.model()

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
    gen_model_label = label_value_dict['context_info__real_show_index_list']
    gen_model_label = tf.reshape(gen_model_label, [-1, list_dim])
    gen_model_weight = label_value_dict['fountain_fulllink_rerank_realshow_label_weight_list']
    gen_model_weight = tf.reshape(gen_model_weight, [-1, list_dim])
    real_show_label = label_value_dict['context_info__real_show_list']
    real_show_label = tf.reshape(real_show_label, [-1, list_dim])
    
    click_label = label_value_dict['fountain_click_label_list']
    click_label = tf.reshape(click_label, [-1, list_dim])
    is_finish_label = label_value_dict['fountain_finish_label_list']
    is_finish_label = tf.reshape(is_finish_label, [-1, list_dim])
    # print_ops.append(tf.print("is_finish_label origin:",is_finish_label,summarize=10,output_stream=sys.stdout))
    fountain_is_finish = tf.cast(tf.math.greater(is_finish_label, 0.5),tf.int64) #if>0.5, set 1
    
    gen_label_top6 = gen_model_label[:,:6] #(?,6)
    real_show_top6 = real_show_label[:,:6] #(?,6)
    indices_shape = tf.shape(real_show_top6)
    
    col_indices = tf.tile(tf.expand_dims(tf.range(indices_shape[1]),0),[indices_shape[0],1]) #(?,6)
    # print_ops.append(tf.print("yqy col_indices ", col_indices, summarize=10, output_stream=sys.stdout)) 
    col_indices = tf.cast(col_indices, dtype=tf.int64)
    real_show_top6 = tf.cast(real_show_top6, dtype=tf.int64)
    masked_indices = tf.cast(col_indices*real_show_top6,dtype=tf.int64) #(?,6)
    # print_ops.append(tf.print("yqy masked_indices ", masked_indices, summarize=10, output_stream=sys.stdout))
    
    gen_model_weight = tf.batch_gather(gen_model_weight, masked_indices)
    masked_indices = tf.expand_dims(masked_indices, axis=2) #(?,6,1)

    # predict:?,6,60; pos_label_index:?,6,1
    # 根据indice取出模型预估值
    # pos_output = tf.batch_gather(predict, pos_label_index) #(?,6,1)
    pos_output = tf.batch_gather(predict, masked_indices) #(?,6,1)
    pos_output = tf.squeeze(pos_output, axis=-1) #(?,6)
    # print_ops.append(tf.print("pos_output:",pos_output,summarize=20,output_stream=sys.stdout))
    real_show_top6 = tf.cast(real_show_top6, dtype=tf.float32)
    
    # 只对曝光位置计算loss
    valid_pos_output = tf.log(pos_output+1e-9)*real_show_top6 #(?,6)
    # print_ops.append(tf.print("valid_pos_output:",valid_pos_output,summarize=20,output_stream=sys.stdout))
    valid_counts = tf.reduce_sum(real_show_top6, axis=-1)+1e-9 #避免除0
    
    # 对每个样本，只计算有效位置的平均loss
    gen_loss = tf.reduce_sum(valid_pos_output, axis=-1)/valid_counts #(?,)
    # print_ops.append(tf.print("gen_loss each sample:",gen_loss,summarize=20,output_stream=sys.stdout))
    
    # batch平均loss
    gen_loss_mean = -tf.reduce_mean(gen_loss)
    # print_ops.append(tf.print("gen_loss_mean each batch:",gen_loss_mean,summarize=20,output_stream=sys.stdout))
    
    # kd_loss = cal_loss(tf.stop_gradient(predictv1), predictv2, 1.0)
    # predictv1 = tf.nn.softmax(predictv1, axis=-1)
    
    # finish_sum = tf.reduce_sum(gen_model_weight, axis=-1)
    # true_diffs = finish_sum[:, None] - finish_sum[None, :]

    # dpo loss
    # 每个样本的平均播放时长
    mean_play_time = tf.reduce_sum(gen_model_weight, axis=-1)-6 # #(bs,)
    # print_ops.append(tf.print("ryx mean_play_time ", mean_play_time, summarize = 10, output_stream=sys.stdout))
    mean_play_time = tf.minimum(tf.ceil(mean_play_time/10),30) # 将播放时长限制在30以内
    # 构建偏好对
    true_diffs = mean_play_time[:, None] - mean_play_time[None, :] #任意两个样本之间的播放时长差值，构建偏好矩阵,[bs,bs]
    padded_pairs_mask = tf.less(true_diffs, 0) #只保留有效偏好对
    padded_pairs_mask = tf.cast(padded_pairs_mask, dtype=tf.float32)
    # print_ops.append(tf.print("ryx padded_pairs_mask sum ", tf.reduce_sum(padded_pairs_mask), summarize = 10, output_stream=sys.stdout))
    # print_ops.append(tf.print("ryx padded_pairs_mask shape ", tf.shape(padded_pairs_mask), summarize = 10, output_stream=sys.stdout))

    # 计算模型预测分数的差值
    scores_diffs = gen_loss[:, None] - gen_loss[None, :] #(bs,bs)
    # print_ops.append(tf.print("ryx scores_diffs shape ", tf.shape(scores_diffs), summarize = 10, output_stream=sys.stdout))
    beta = 0.3
    # hinge loss
    gen_pair_loss = tf.reduce_mean(tf.maximum(0.0, beta+scores_diffs)*tf.cast(padded_pairs_mask, dtype=tf.float32)) #对播放时长更长的样本给出更高的预测分数，差异足够大,>beta
    
    tf.summary.scalar('gen_pair_loss', gen_pair_loss)
    tf.summary.scalar('gen_loss', gen_loss_mean)
    tf.summary.scalar('cl_loss', cl_loss)

    distill_label = tf.cast(tf.greater(gen_model_label, 0), tf.float32)
    print(f"distill_label shape {distill_label.shape}")

    loss = 1000*(gen_loss_mean+cl_loss)
    tf.summary.scalar('loss', loss)

    print(f"predict.shape: {predict.shape}") # (?, 6, 60)
    y_pred_mask = tf.reduce_sum(predict, axis=1) # (?, 60) 
    print(f"y_pred_mask.shape: {y_pred_mask.shape}")
    predict = tf.identity(predict)
    _, rank_index = tf.math.top_k(y_pred_mask, 60, sorted=True)
    print(f"rank_index.shape: {rank_index.shape}")
    sorted_label = tf.batch_gather(distill_label, rank_index)
    print(f"sorted_label.shape: {sorted_label.shape}")

    recall_6_th = tf.reduce_sum(sorted_label[:,:6] , -1, keep_dims=True) / (tf.reduce_sum(distill_label, -1, keep_dims=True)+1e-9)
    print_ops.append(tf.print('predict recall_{}th'.format(6), tf.reduce_mean(recall_6_th), summarize = 10, output_stream=sys.stdout))
    recall_10_th = tf.reduce_sum(sorted_label[:,:10] , -1, keep_dims=True) / (tf.reduce_sum(distill_label, -1, keep_dims=True)+1e-9)
    print_ops.append(tf.print('predict recall_{}th'.format(10), tf.reduce_mean(recall_10_th), summarize = 10, output_stream=sys.stdout))

    # kai.add_run_hook(DumpTensorHook('dump_tensors', {
    #     'predict': predict,
    #     'cl_loss': cl_loss, #new add item_embedding
    # }), 'custom_dump_tensor_hook')

    with tf.control_dependencies(print_ops):
        predict = tf.identity(predict)
        print("predict shape", predict.shape)
        # neg_output = tf.identity(neg_output)
        pos_output = tf.identity(pos_output)
        pos_output_shape = tf.expand_dims(tf.reduce_sum(pos_output,axis=-1),axis=-1)


        recall_num = [6, 10]
        for p in recall_num:
            recall = tf.reduce_sum(sorted_label[:,:p] , -1, keep_dims=True) / (tf.reduce_sum(distill_label, -1, keep_dims=True)+1e-9)
            one = tf.ones_like(pos_output_shape)
            zero = tf.zeros_like(pos_output_shape)
            print("one shape", one.shape)
            print("zero shape", zero.shape)
            targets.append(('recall_{}th'.format(p), pos_output_shape, zero, one, 'linear_regression'))

    if args.with_kai_v2:
        sparse_optimizer = config.optimizer.Adam(0.001)
        dense_optimizer = config.optimizer.Adam(0.00005)
        sparse_optimizer.minimize(loss, var_list=config.get_collection(config.GraphKeys.EMBEDDING_INPUT))
        dense_optimizer.minimize(loss, var_list=config.get_collection(config.GraphKeys.TRAINABLE_VARIABLES))
        opts = [sparse_optimizer, dense_optimizer]
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
