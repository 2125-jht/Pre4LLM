# -*- coding: utf-8 -*-
'''
------------------------------------------------------------------------
@Description :  
@Author :  邓英杰
@Time :  2025/01/17 17:39:15
------------------------------------------------------------------------
'''

MODEL_TRANS_ORIGIN='cpp'

import yaml
import logging
import os
import sys

import argparse
import tensorflow as tf

from feature_attr_extract import * 
from context_nce_model import FountainDeepLtrMultiTaskModel

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
LIST_SIZE = 6
CANDIDATES_SIZE = 60

print_ops = []
# 目前这段逻辑功能未知
if args.with_kai_v2:
    import kai.tensorflow as config
    import tensorflow.compat.v1 as tf
    default_param_attr = config.nn.ParamAttr(initializer=config.nn.UniformInitializer(0.0001),
                                             access_method=config.nn.ProbabilityAccess(100.0),
                                             recycle_method=config.nn.UnseendaysRecycle(delete_after_unseen_days=30, delete_threshold=0.1, allow_dynamic_delete=True))
    config.nn.set_default_param_attr(default_param_attr)

    def filter_mask_wrapper(dataset):
        # 1. 声明字段
        #  sample_type为字段名，特征类型dataset.DENSE表示稠密，tf.int64为数据类型，dim为1
        # dataset.add_feature('fountain_fulllink_rerank_index_list', dataset.DENSE, tf.int64, max_length=60)
        # dataset.add_feature('fountain_fulllink_rerank_index_weight_list', dataset.DENSE, tf.int64, max_length=60)
        dataset.add_feature('context_info__real_show_index_list', dataset.DENSE, tf.int64, max_length=60)
        dataset.add_feature('context_info__real_show_list', dataset.DENSE, tf.int64, max_length=60)
        dataset.add_feature('fountain_fulllink_rerank_realshow_label_weight_list', dataset.DENSE, tf.int64, max_length=60)
        dataset.add_feature('context_info__first_screen', dataset.DENSE, tf.int64, max_length=60)
        dataset.add_feature('tab', dataset.DENSE, tf.int64, max_length=60)
        dataset.add_feature('3017', dataset.DENSE, tf.int64, max_length=60)
        dataset.add_feature('3019', dataset.DENSE, tf.int64, max_length=60)
        dataset.add_feature('3030', dataset.DENSE, tf.int64, max_length=60)
        dataset.add_feature('1031', dataset.DENSE, tf.int64, max_length=60)


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
            
            # photo_hetu_tag_level5_list = tf.RaggedTensor.from_row_splits(batch["1031"][0], row_splits= batch["1031"][1]).to_tensor()
            # print_ops.append(tf.print("photo_hetu_tag_level5_list=", photo_hetu_tag_level5_list[0], summarize = 10, output_stream=sys.stdout))

            total_play_time = tf.reduce_sum(realshow_weight, axis=-1) - CANDIDATES_SIZE
            realshow = tf.reduce_sum(realshow, axis=-1)
            
            context_page = batch['context_info__first_screen']
            tab = batch['tab']
        
            # print_ops.append(tf.print("tab=", tab, output_stream=sys.stdout))
            # print_ops.append(tf.print("context_page=", context_page, output_stream=sys.stdout))
            # print_ops.append(tf.print("context_page len=", len(context_page), output_stream=sys.stdout))
            # print_ops.append(tf.print("realshow_weight", realshow_weight, output_stream=sys.stdout))
            print(f"realshow shape: {realshow.shape}")
            print(f"realshow_weight shape: {realshow_weight.shape}")
            
            is_short_request = tf.math.less(total_play_time, 20) # 60%分位数
            
            fountain_click_label = batch["fountain_click_label_list"]
            fountain_click_label = tf.RaggedTensor.from_row_splits(fountain_click_label[0], row_splits=fountain_click_label[1])
            fountain_click_label = fountain_click_label.to_tensor()
            fountain_click_sum = tf.reduce_sum(fountain_click_label, axis=-1)

            # 长视频需要满足点击数>=2
            # long_video_quality = tf.math.greater_equal(fountain_click_sum, 2)
            
            # 过滤规则:
            # 0. 只保留首屏样本
            # 1. 曝光数要>=2
            # 2. 至少要有一条有效点击
            # mask = tf.math.logical_or(
            #     tf.math.not_equal(context_page, 1),
            #     tf.math.less(realshow, 1),
            # )
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
# label name和cofea_reader.py中各label前缀保持一致
all_model_labels = [
    "fountain_fulllink_rerank_index_list",
    "fountain_fulllink_rerank_index_weight_list",
    "fountain_fulllink_rerank_realshow_label_weight_list",
    "fountain_click_label_list",
    "fountain_wtd_label_list",
    "fountain_ltr_label_list",
    "fountain_ltr_weight_list",
]

realshow_labels = [
    "context_info__real_show_index_list",
    "context_info__real_show_list",
    "context_info__playing_time_list",
    "context_info__click_list",
    "context_info__like_list",
    "context_info__follow_list",
    "context_info__comment_list",
    "context_info__forward_list",
    "context_info__fountain_slide_to_next_list",
]

print("common_attr_names: ", [attr.attr_name for attr in all_features if attr.is_common])
print("all_feature_name: ", [attr.attr_name for attr in all_features])

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

def get_dense_fea(name, list_dim):
    assert name in realshow_labels, name
    return config.get_dense_fea(name, dim=list_dim, dtype=tf.int64)

def get_label(name, list_dim):
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
                tt = tf.RaggedTensor.from_row_splits(values=sparse_feature[0], row_splits=sparse_feature[1]).to_tensor()
                #print_ops.append(tf.print("[Test test] slot " + str(attr.slots[0]), tt, output_stream=sys.stdout))
        elif args.with_kai:
            offset = tf.cast(config.get_signs(attr.slots[0])[1], tf.int32)
            size_var = offset[1:] - offset[0:-1]
            feature_emb_size_dict[attr.attr_name] = size_var        
        print("--->>> feature {} = {}, shape={}".format(attr.attr_name, feature_emb_dict[attr.attr_name], feature_emb_dict[attr.attr_name].shape))
        # print("--->>> feature %s normal" % attr.attr_name)

    return feature_emb_dict, feature_emb_size_dict

# all_model_labels里的名字和cofea_reader.py中名字对应
def get_label_dict(list_dim=CANDIDATES_SIZE):
    label_value_dict = {}
    for label_name in all_model_labels:
        print(f"====> get_label, name : {label_name}")
        label_value = get_label(label_name, list_dim)
        label_value_dict[label_name] = label_value

    for label_name in realshow_labels:
        print(f"====> get_label, name : {label_name}")
        label_value = get_dense_fea(label_name, list_dim)
        label_value_dict[label_name] = label_value

    return label_value_dict

def get_dense_dict(dense_feas, list_dim):
    dense_value_dict = {}
    for name in dense_feas:
        # dense_value_dict[name] = config.get_extra_param(name, size=list_dim, default_value=0.0)
        dense_value_dict[name] = config.get_extra_param(name, default_value=0.0)

    return dense_value_dict


def sum_loss_tensor_dict(loss_dict):
    sum_loss = None
    for key, loss in loss_dict.items():
        if (sum_loss == None):
            sum_loss = loss
        else:
            sum_loss += loss
    return sum_loss

def set_zero_topk(pred, indices):
    batch_size, seq_len, vocab_len = tf.shape(pred)[0],pred.shape[1],pred.shape[2]
    # 计算展平后的 batch 和 seq 偏移位置
    index = tf.expand_dims(tf.range(0,batch_size),axis=1) * seq_len * vocab_len + tf.expand_dims(tf.range(0, seq_len),axis=0) * vocab_len # (?, seq_len)
    index = tf.expand_dims(index, axis=2) # (?, seq_len, 1)
    # print("index shape ",index.shape)
    selected_token = tf.expand_dims(tf.expand_dims(indices, axis=1), axis=2) # (?, 1, 1)
    selected_token = tf.cast(selected_token, tf.int32)
    selected_token = tf.tile(selected_token, [1, seq_len, 1]) + index # (?, seq_len, 1)
    # print("selected token shape ", selected_token.shape)
    pred = tf.reshape(pred, (batch_size * seq_len * vocab_len, 1))
    selected_token = tf.reshape(selected_token, (batch_size * seq_len * 1, 1))

    output_tensor = tf.tensor_scatter_nd_update(pred, selected_token, tf.expand_dims(tf.ones(batch_size * seq_len) * float("-inf"), axis=1))
    output_tensor = tf.reshape(output_tensor, [batch_size, seq_len, vocab_len])
    return output_tensor

def greedy_search(prediction):
    generated_tokens = []
    for i in range(LIST_SIZE):
        logits = prediction[:, i, :] # (?, candidates_size)
        _, sampled_token = tf.nn.top_k(logits, k=1) # (?, k=1)
        sampled_token = tf.squeeze(sampled_token, axis=-1) # (?,)
        prediction = set_zero_topk(prediction, sampled_token)
        generated_tokens.append(sampled_token)
    generated_tokens = tf.stack(generated_tokens, axis=-1) # (?, list_size)
    return generated_tokens

def list_recall(predict, label_value_dict):
    predict = tf.identity(predict) # (?, list_size+1, candidates_size+3)
    predict = predict[:,:-1,2:-1] # (?, list_size, candidates_size)，下标从0开始
    print('list_recall predict', predict.shape)
    
    gen_model_label = label_value_dict['context_info__real_show_list'] # 形如 [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    print('gen_model_label', gen_model_label.get_shape().as_list())
    gen_model_label = tf.reshape(gen_model_label, [-1, CANDIDATES_SIZE])
    show_label = tf.cast(tf.greater(gen_model_label, 0), tf.float32) # [None, candidates_size]
    true_label = show_label[:,:LIST_SIZE]
    print("true_label shape", true_label.get_shape().as_list())
    indices_matrix = tf.tile(tf.expand_dims(tf.range(0, CANDIDATES_SIZE), 0), [tf.shape(gen_model_label)[0], 1])
    print("indices_matrix shape ", indices_matrix.shape)
    true_index = tf.where(tf.greater(gen_model_label, 0), indices_matrix, tf.fill(tf.shape(gen_model_label), 0))
    true_index = true_index[:,:LIST_SIZE]
    print("true_index shape ", true_index.shape)

    _, rank_index = tf.math.top_k(predict, 1, sorted=True) # 返回最后一维最大值index, (?, list_size, 1)
    rank_index = tf.squeeze(rank_index, -1)

    # print_ops.append(tf.print('[train] true_label ', true_label[2], summarize = -1, output_stream=sys.stdout))
    # print_ops.append(tf.print('[train] true_index ', true_index[2], summarize = -1, output_stream=sys.stdout))
    # print_ops.append(tf.print('[train] select_index ', rank_index[2], summarize = -1, output_stream=sys.stdout))
    # print_ops.append(tf.print('[train] predict ', predict[2][:8], summarize = -1, output_stream=sys.stdout))

    hit_cnt = tf.reduce_sum(tf.cast(tf.equal(true_index, rank_index), tf.float32) * true_label, axis=-1, keep_dims=True)
    avg_precision = hit_cnt / (tf.reduce_sum(true_label, -1, keep_dims=True)+1e-9)
    tf.summary.scalar('avg_precision', tf.reduce_mean(avg_precision))
    # print_ops.append(tf.print('[train] avg_precision ', tf.reduce_mean(avg_precision), summarize = 8, output_stream=sys.stdout))

    # 不重复选取
    greedy_indices = greedy_search(predict) #bs,6
    greedy_hit = tf.batch_gather(show_label, greedy_indices)
    recall_6_th_greedy = tf.reduce_sum(greedy_hit, -1, keep_dims=True) / (tf.reduce_sum(true_label, -1, keep_dims=True)+1e-9)
    tf.summary.scalar('recall_6_th_greedy', tf.reduce_mean(recall_6_th_greedy))
    # print_ops.append(tf.print('[train] recall_6_th_greedy ', tf.reduce_mean(recall_6_th_greedy), summarize = 8, output_stream=sys.stdout))

# print info
worker_global_step = config.get_step()
ops = [tf.print("====> step", worker_global_step, summarize=-1, output_stream=sys.stdout)]

############################################################

# 获取模型output
all_param_dict, feature_emb_size_dict = get_param_dict()
print("feature_emb_size_dict ", feature_emb_size_dict)
label_value_dict = get_label_dict()
# batch_size = tf.shape(all_param_dict["context_pctr"])[0]
dense_value_dict = {}
dense_value_dict = get_dense_dict(["context_info__pctr", "context_info__pltr", "context_info__plvtr", "context_info__pwtr", "context_info__pvtr", "context_info__pwtd"], CANDIDATES_SIZE)


model_class = FountainDeepLtrMultiTaskModel(all_param_dict, label_value_dict, dense_value_dict, print_ops, list_size=LIST_SIZE, candidates_size=CANDIDATES_SIZE)


if is_training:
    print(f"====> train, gen...")

    predict, nce_loss, gen_loss = model_class.model(training=True)
    nce_loss = nce_loss / 1000 # 对齐量级
    print("return predict ",predict.shape)
    print_ops = model_class.print_ops

    targets = []
    loss = gen_loss + nce_loss * 0.5
    tf.summary.scalar('gen_loss', gen_loss)
    tf.summary.scalar('nce_loss', nce_loss)
    
    list_recall(predict, label_value_dict)
    # 打印、监控
    # for attr_name, emb in all_param_dict.items():
    #     tf.summary.histogram(attr_name, emb)
    #     tf.summary.scalar(f"{attr_name}/mean", tf.reduce_mean(emb))
    #     tf.summary.scalar(f"{attr_name}/abs_mean", tf.reduce_mean(tf.abs(emb)))
    #     tf.summary.scalar(f"{attr_name}/norm2", tf.reduce_mean(tf.norm(emb, axis=1)))
    #     embedding_gradients = tf.gradients(loss, emb)
    #     tf.summary.scalar(f"{attr_name}/gradient/mean", tf.reduce_mean(embedding_gradients))
    #     tf.summary.scalar(f"{attr_name}/gradient/abs_mean", tf.reduce_mean(tf.abs(embedding_gradients)))
    #     tf.summary.scalar(f"{attr_name}/gradient/norm2", tf.reduce_mean(tf.norm(embedding_gradients, axis=1)))

    with tf.control_dependencies(print_ops):
        logits = tf.reduce_sum(predict, axis=-1)
        logits = tf.expand_dims(logits, axis=-1)
        zero = tf.zeros_like(logits)
        one = tf.ones_like(logits)
        print("zero shape", zero.shape)
        print("one shape", one.shape)
        targets.append(('recall', logits, zero, one, 'linear_regression'))

    if args.with_kai_v2:
        sparse_optimizer = config.optimizer.Adam(0.001)
        # dense_optimizer = config.optimizer.Adam(0.00005)
        dense_optimizer = config.optimizer.Adam(0.0001)
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
        config.dump_kai_training_config('./training', targets, loss=loss, text=args.text, init_params_in_tf=True)
    elif args.with_kai_v2:
        config.build_model(optimizer=opts, metrics=targets)
    else:
        config.dump_training_config('./training', targets, opts=opts, text=args.text)
else:
    model_class._training = False
    # logits: [batch_size, beam_size, seq_length, vocab_size], 
    # generated_sequence: [batch_size, beam_size, seq_length]
    # probs[0]: [batch_size, beam_size, vocab_size]
    logits, generated_sequence, preward, best_sequences, probs = model_class.model(training=False)
    photo_id_emb = all_param_dict["photo_id"]
    # context_cascade_pctr_emb = all_param_dict["context_cascade_pctr"]

    targets = []

    # best_sequence = best_sequences[0] - 2 # (4,)
    # size = tf.shape(best_sequence)[0]
    # values = tf.range(size, 0, -1) # 递减score [4, 3, 2, 1]
    # vocab_size = tf.shape(logits)[2]-3
    # tensor_zeros = tf.zeros(vocab_size, dtype=tf.int32)
    # scores = tf.tensor_scatter_nd_update(tensor_zeros, tf.expand_dims(best_sequence, axis=1), values)
    # scores = tf.cast(tf.reshape(scores, [-1, 1]), dtype=tf.float32)
    # print(f"scores shape: {scores.shape}")
    # targets.append((f"rerank_gen_score_0",scores))
    selected_indices = generated_sequence[0] - 2 # [beam_size, seq_length]
    vocab_size = tf.shape(logits)[-1] - 3
    # for i in range(10):
    for i in range(1):
        selected_indices_i = selected_indices[i, :] # [seq_length]
        tensor_zeros = tf.zeros(vocab_size, dtype=tf.int32)
        output_tensor = tf.tensor_scatter_nd_update(tensor_zeros, tf.expand_dims(selected_indices_i, 1), tf.range(selected_indices_i.shape[0], 0, -1))
        output_tensor = tf.reshape(output_tensor, [-1, 1])
        output_tensor = tf.cast(output_tensor, dtype=tf.float32)
        # print("output_tensor shape", output_tensor.shape)
        pred_output = tf.identity(output_tensor) #(?,1)
        print("pred_output shape",pred_output.shape)
        targets.append((f"rerank_gen_score_{i}",pred_output))

    probs = [tf.transpose(x[0][:, 2:-1], perm=[1, 0]) for x in probs] # (30, beam_size)
    targets.append(("photo_id_emb", tf.identity(photo_id_emb)))
    # targets.append(("context_cascade_pctr_emb", tf.identity(context_cascade_pctr_emb)))
    # targets.append(("preward", tf.identity(tf.reshape(preward[:,2:-1], [-1, 1]))))
    targets.append(("logits_0", tf.identity(probs[0])))
    targets.append(("logits_1", tf.identity(probs[1])))
    targets.append(("logits_2", tf.identity(probs[2])))
    targets.append(("logits_3", tf.identity(probs[3])))
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
