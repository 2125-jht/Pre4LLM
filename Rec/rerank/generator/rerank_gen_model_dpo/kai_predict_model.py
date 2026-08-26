from __future__ import print_function
MODEL_TRANS_ORIGIN='cpp'

import json
import yaml
import logging
import os
import sys

import argparse
import tensorflow as tf

from feature_predict_attr_extract import * 
from predict_model import FountainDeepLtrPredictModel

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
        dataset.add_feature('sample_type', dataset.DENSE, tf.int64, 1)
        dataset.add_feature('sample_type', dataset.DENSE, tf.int64, 1)

        # 2.声明mask，batch是一个dict，key为声明的字段名，value根据特征类型分为2种情况：
        # dataset.DENSE: 值为tf.Tensor
        # dataset.SPARSE: 值为元组: (tf.Tensor, tf.Tensor)，
        #   其中第一个tensor表示feasign，第二个tensor表示cumsum
        #   可以使用tf.RaggedTensor.from_row_splits转成RaggedTensor
        def mask_fn(batch):
            play_time_list = [5, 6, 6, 7, 8, 9, 11, 11, 12, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 15, 15, 15, 16, 16, 16, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15]
            play_time_list = tf.constant(play_time_list)
            duration_tensor = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0])

            mean_value = tf.reduce_mean(duration_tensor)

            # Convert the mean value to an integer
            mean_int_value = tf.cast(mean_value, dtype=tf.int32)
            value = tf.gather(play_time_list, mean_int_value)
            sample_type = batch['sample_type']

            mask = tf.math.logical_or(tf.math.logical_and(tf.math.less(mean_play_time, 90),
                                    tf.math.greater_equal(mean_play_time, 0)),
                                    tf.math.less(realshow, 6))
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
]

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


def get_label(name, list_dim=60):
    assert name in all_model_labels, name
    return config.get_label(name, dim=list_dim)

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

    return label_value_dict


def sum_loss_tensor_dict(loss_dict):
    sum_loss = None
    for key, loss in loss_dict.items():
        if (sum_loss == None):
            sum_loss = loss
        else:
            sum_loss += loss
    return sum_loss

def set_zero_by_idx(pred, indices):  
  batch_size, beam_len, vocab_len = tf.shape(pred)[0],tf.shape(pred)[1],tf.shape(pred)[2]
  batch_size, beam_len, seq_len = tf.shape(indices)[0],tf.shape(indices)[1],tf.shape(indices)[2]
  pred = tf.reshape(pred,(batch_size*beam_len, -1))
  indices = tf.reshape(indices,(batch_size*beam_len, -1))
  flat_indices = indices + tf.reshape(tf.range(0, batch_size*beam_len*vocab_len, vocab_len), (batch_size*beam_len, 1))
  flat_indices = tf.reshape(flat_indices, (-1,1))
  pred = tf.reshape(pred, (batch_size*beam_len*vocab_len,))
  pred = tf.tensor_scatter_nd_update(pred, flat_indices, tf.zeros(batch_size*beam_len*seq_len))
  pred = tf.reshape(pred,(batch_size, beam_len, vocab_len))
  
  return pred

def beam_search(prediction, k=10):
    batch_size, seq_length, vocab_size = tf.shape(prediction)[0],tf.shape(prediction)[1],tf.shape(prediction)[2]
    
    log_prob, indices = tf.nn.top_k(prediction[:,0,:], k, sorted=True)
    indices = tf.expand_dims(indices, -1)
    for n1 in range(1, 6):
        predict_temp = tf.expand_dims(prediction[:,n1,:], 1)
        predict_temp = tf.tile(predict_temp, [1, k, 1])      
        if n1 >= 2:
            predict_temp = set_zero_by_idx(predict_temp, new_indices)
        else: 
            predict_temp = set_zero_by_idx(predict_temp, indices)
        log_prob_temp = tf.expand_dims(log_prob, -1) + predict_temp
        
        log_prob, index_temp = tf.nn.top_k(tf.reshape(log_prob_temp, [batch_size, -1]), k, sorted=True)
        idx_begin = index_temp // vocab_size
        idx_concat = index_temp % vocab_size
        
        new_indices = tf.zeros([batch_size, k, n1+1], dtype=tf.int32)
        idx_expand = tf.expand_dims(idx_begin, -1)
        new_indices = tf.concat([tf.batch_gather(indices, idx_begin), tf.expand_dims(idx_concat, -1)], axis=-1)
        indices = new_indices

    return indices, log_prob

def extract_embeddings(item_embedding, indices):
    batch_size, seq_len, embedding_dim = tf.shape(item_embedding)[0], tf.shape(item_embedding)[1], tf.shape(item_embedding)[2]
    indices = tf.range(0, batch_size)[:, tf.newaxis] * seq_len + indices
    extracted_embeddings = tf.gather(tf.reshape(item_embedding, [-1, embedding_dim]), indices)
    return extracted_embeddings

def set_zero_topk(pred, indices):
    batch_size, seq_len, vocab_len = tf.shape(pred)[0],tf.shape(pred)[1],tf.shape(pred)[2]
    index = tf.expand_dims(tf.range(0,batch_size),axis=1)*seq_len*vocab_len+ tf.expand_dims(tf.range(0, seq_len),axis=0)*vocab_len
    index = tf.expand_dims(index, axis=2)
    print("index shape ",index.shape)
    # selected_token = tf.constant([3,5], dtype=tf.int32)
    selected_token = tf.expand_dims(tf.expand_dims(indices,axis=1),axis=2)
    selected_token = tf.cast(selected_token, tf.int32)
    # print("tensor shape ",tensor.shape)
    selected_token = tf.tile(selected_token,[1,seq_len,1])+index
    print("selected token shape ", selected_token.shape)
    # input = tf.reshape(tf.cast(tf.range(0,batch_size*seq_len*vocab_len),tf.float32),(batch_size,seq_len,vocab_len))
    print("input tensor ", input)
    pred = tf.reshape(pred,(batch_size*seq_len*vocab_len, -1))
    selected_token = tf.reshape(selected_token, (batch_size*seq_len*1,-1))

    output_tensor = tf.tensor_scatter_nd_update(pred, selected_token, tf.expand_dims(tf.ones(batch_size*seq_len)*float("-inf"),axis=1))
    output_tensor = tf.reshape(output_tensor, [batch_size, seq_len, vocab_len])
    return output_tensor

def contrastive_search(prediction, item_embedding, k=30, alpha=0.5):
    batch_size, seq_len, vocab_len = tf.shape(prediction)[0],tf.shape(prediction)[1],tf.shape(prediction)[2]
    k = tf.minimum(vocab_len, k)

    selected_indices = None
    # alpha = 0.5

    for i in range(6):
        logits = prediction[:, i, :]
        log_prob, candidate_indices = tf.nn.top_k(prediction[:,i,:], k, sorted=True)
        candidate_embedding = extract_embeddings(item_embedding, candidate_indices)
        if selected_indices is not None:
            history_embedding = extract_embeddings(item_embedding, selected_indices)
            # print("history_embedding shape ", history_embedding.shape)
            # print("candidate_embedding shape ", candidate_embedding.shape)
            norm_candiate = candidate_embedding / tf.norm(candidate_embedding, axis=2, keep_dims=True)
            norm_history = history_embedding / tf.norm(history_embedding, axis=2, keep_dims=True)
            cosine_matrix = tf.matmul(norm_candiate, tf.transpose(norm_history, perm=[0,2,1]))
            scores = tf.reduce_max(cosine_matrix, axis=-1)
            # print("scores shape ", scores.shape)

            scores = (1.0 - alpha)*log_prob-alpha*scores
            # print("scores shape ", scores.shape)
            _, selected_idx = tf.nn.top_k(scores, k=1, sorted=True)
            selected_idx = tf.batch_gather(candidate_indices, selected_idx)

            # print("selected_idx shape ", selected_idx.shape)
            selected_idx = tf.squeeze(selected_idx, axis=-1)
            prediction = set_zero_topk(prediction, selected_idx)
            # print("selected_indices shape ",selected_indices.shape)
            # print("selected_idx shape ",selected_idx.shape)
            selected_idx = tf.expand_dims(selected_idx, axis=1)

            selected_indices = tf.concat([selected_indices, selected_idx],axis=-1)
            # print("selected_indices shape ", selected_indices.shape)
        else:
            selected_indices = candidate_indices[:,0]
            # print("selected_indices shape ",selected_indices.shape)
            prediction = set_zero_topk(prediction, selected_indices)
            selected_indices = tf.expand_dims(selected_indices, axis=1)
    return selected_indices

def greedy_search(prediction):
    seq_len = 6
    generated_tokens = []
    for i in range(seq_len):
        logits = prediction[:, i, :]
        _, sampled_token = tf.nn.top_k(logits, k=1)
        print("prediction shape ",prediction.shape)
        print("sampled_token shape ",sampled_token.shape)
        sampled_token = tf.squeeze(sampled_token, axis=-1)
        prediction = set_zero_topk(prediction, sampled_token)
        generated_tokens.append(sampled_token)
    generated_tokens = tf.stack(generated_tokens, axis=-1)

    return generated_tokens

def topk_sampling(prediction, k=10):
    generated_tokens = []
    seq_len = 6

    # Loop through each decoding step
    for i in range(seq_len):
        # Get the logits for the current decoding step
        print("seq_len ", i)
        logits = prediction[:, i, :]

        # Sample from the logits using top-k sampling
        _, top_k_indices = tf.nn.top_k(logits, k=3)
        top_k_indices = tf.transpose(tf.random.shuffle(tf.transpose(top_k_indices)))
        print("top_k_indices shape ",top_k_indices.shape)
        sampled_token = top_k_indices[:,0]
        print("sampled_token ", sampled_token.shape)
        prediction = set_zero_topk(prediction, sampled_token)
        generated_tokens.append(sampled_token)

    # Reshape the list to a tensor of shape (batch_size, seq_len)
    generated_tokens = tf.stack(generated_tokens, axis=-1)

    return generated_tokens


# print info
worker_global_step = config.get_step()
ops = [tf.print("====> step", worker_global_step, summarize=-1, output_stream=sys.stdout)]

############################################################

# 获取模型output
all_param_dict, feature_emb_size_dict = get_param_dict()
print("feature_emb_size_dict ", feature_emb_size_dict)
label_value_dict = get_label_dict()
# batch_size = tf.cast(tf.size(all_param_dict["pId_KAI"][:, 0]), dtype=tf.float32)

model_class = FountainDeepLtrPredictModel(all_param_dict, label_value_dict, print_ops)
pred_dict = {}

predict, item_embedding = model_class.model()
output_dict = {
    "rerank_gen": predict,
}
if is_training:
    print(f"====> train, gen...")
    targets = []
    sum_loss = 0.0
    list_dim  = 60
    gen_model_label = label_value_dict['fountain_fulllink_rerank_index_list']
    print('gen_model_label', gen_model_label.get_shape().as_list())
    print('gen_model_label shape', tf.shape(gen_model_label))
    gen_model_label = tf.reshape(gen_model_label, [-1, list_dim])
    gen_model_weight = label_value_dict['fountain_fulllink_rerank_index_weight_list']

    # print_ops.append(tf.print("ryx gen_model_weight", gen_model_weight, summarize = 10, output_stream=sys.stdout))
    # print_ops.append(tf.print("ryx gen_model_label ", gen_model_label, summarize = 10, output_stream=sys.stdout))   
    gen_model_weight = tf.reshape(gen_model_weight, [-1, list_dim])
    _, rank_index = tf.math.top_k(gen_model_label, 60, sorted=True)
    shuffled_label_index = rank_index[:,:6]
    print("shuffled_label_index shape")

    
    # print_ops.append(tf.print("ryx shuffled_label_index ", shuffled_label_index, summarize = 10, output_stream=sys.stdout))
    # print_ops.append(tf.print("ryx shuffled_label_index", tf.shape(shuffled_label_index), summarize = 10, output_stream=sys.stdout))
    shuffled_label_index = tf.reverse(shuffled_label_index, axis=[1])
    # print_ops.append(tf.print("ryx shuffled_label_index reverse", shuffled_label_index, summarize = 10, output_stream=sys.stdout))

    # print("shuffled_label_index shape", tf.shape(shuffled_label_index))
    gen_model_weight = tf.batch_gather(gen_model_weight, shuffled_label_index)
    print("ryx distill_label shape", tf.shape(gen_model_weight))
    print("ryx logits shape", tf.shape(shuffled_label_index))
    wtd_loss = tf.losses.mean_squared_error(wtd_estimation, gen_model_weight)
    tf.summary.scalar('wtd_loss', wtd_loss)
    
    distill_label = tf.cast(tf.greater(gen_model_label, 0), tf.float32)
    print("distill_label shape", tf.shape(distill_label))
    print("logits shape", tf.shape(logits))
    logits_loss = tf.losses.log_loss(labels=distill_label, predictions=logits, weights = gen_model_weight, reduction=tf.losses.Reduction.SUM)
    tf.summary.scalar('logits_loss', logits_loss)

    shuffled_label_index = tf.expand_dims(shuffled_label_index, axis=2)
    output = tf.batch_gather(predict, shuffled_label_index)
    print(f"output shape {output.shape}")
    gen_loss = -tf.reduce_sum(tf.log(output))
    print_ops.append(tf.print("ryx logits_loss ", logits_loss, summarize = 10, output_stream=sys.stdout))
    tf.summary.scalar('gen_loss', gen_loss)
    loss = logits_loss+gen_loss+wtd_loss
    tf.summary.scalar('loss', loss)

    y_pred_mask = tf.reduce_sum(predict, axis=1)
    _, rank_index = tf.math.top_k(y_pred_mask, 60, sorted=True)
    sorted_label = tf.batch_gather(distill_label, rank_index)

    recall_6_th = tf.reduce_sum(sorted_label[:,:6] , -1, keep_dims=True) / (tf.reduce_sum(distill_label, -1, keep_dims=True)+1e-9)
    # print_ops.append(tf.print("ryx sorted_label reduce sum", tf.shape(tf.reduce_sum(sorted_label[:,:6] , -1, keep_dims=True)), summarize = 10, output_stream=sys.stdout))
    # print_ops.append(tf.print("ryx label reduce sum shape", tf.shape(tf.reduce_sum(distill_label, -1, keep_dims=True)), summarize = 10, output_stream=sys.stdout))
    print_ops.append(tf.print('recall_{}th'.format(6), tf.reduce_mean(recall_6_th), summarize = 10, output_stream=sys.stdout))
    recall_10_th = tf.reduce_sum(sorted_label[:,:10] , -1, keep_dims=True) / (tf.reduce_sum(distill_label, -1, keep_dims=True)+1e-9)
    print_ops.append(tf.print('recall_{}th'.format(10), tf.reduce_mean(recall_10_th), summarize = 10, output_stream=sys.stdout))
    
    with tf.control_dependencies(print_ops):
        predict = tf.identity(predict)
        output = tf.identity(output)

        recall_num = [6, 10]
        for p in recall_num:
            recall = tf.reduce_sum(sorted_label[:,:p] , -1, keep_dims=True) / (tf.reduce_sum(distill_label, -1, keep_dims=True)+1e-9)
            one = tf.ones_like(recall)
            zero = tf.zeros_like(recall)
            print(':recall shape:' + str(recall.get_shape().as_list()))
            print_ops.append(tf.print('recall_{}th'.format(p), recall, summarize = 10, output_stream=sys.stdout))
            targets.append(('recall_{}th'.format(p), recall, zero, one, 'linear_regression'))
    
        
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
else:#11
    print("predict shape ", predict.shape)
    print("item_embedding shape ", item_embedding.shape)
    # predict_temp = tf.reduce_mean(predict, axis=1)
    # print("predict temp shape ", predict_temp.shape)
    # output_tensor = tf.reshape(predict_temp,[-1,1])

    # selected_indices = contrastive_search(predict, item_embedding)
    # selected_indices = selected_indices[0,:]
    # print("selected_indices shape", selected_indices.shape)

    # tensor_zeros = tf.zeros(tf.shape(predict)[2],dtype=tf.int32)
    # output_tensor = tf.tensor_scatter_nd_update(tensor_zeros, tf.expand_dims(selected_indices, 1), tf.range(selected_indices.shape[0],0,-1))
    # output_tensor = tf.reshape(output_tensor,[-1,1])
    # output_tensor = tf.cast(output_tensor, dtype=tf.float32)
    
    # targets = []
    # pred_output = tf.identity(output_tensor)
    # targets.append(("rerank_gen", pred_output))
    
    # v3 alternative
    targets = []    
    alphas = [0,0.3,0.5,0.7,0.9]
    for i,alpha in enumerate(alphas): 
        predict_ = tf.identity(predict)
        item_embedding_ = tf.identity(item_embedding)
        if i%2==0:
            top_k_indices = topk_sampling(predict_)
            top_k_indices = top_k_indices[0,:]
            # print("selected_indices shape", top_k_indices.shape)
            tensor_zeros = tf.zeros(tf.shape(predict_)[2],dtype=tf.int32)
            output_tensor = tf.tensor_scatter_nd_update(tensor_zeros, tf.expand_dims(top_k_indices, 1), tf.range(top_k_indices.shape[0],0,-1))
            output_tensor = tf.reshape(output_tensor,[-1,1]) 
            output_tensor = tf.cast(output_tensor, dtype=tf.float32)
            # print("output_tensor shape", output_tensor.shape)
            pred_output = tf.identity(output_tensor) #
            targets.append((f"rerank_gen_score_{i}", pred_output))
        else:
            selected_indices = contrastive_search(predict_, item_embedding_,alpha=alpha) #batch,6
            selected_indices = selected_indices[0,:] #1st batch
            # print("yqy selected_indices shape", selected_indices.shape) #(6,)
            tensor_zeros = tf.zeros(tf.shape(predict_)[2],dtype=tf.int32)
            output_tensor = tf.tensor_scatter_nd_update(tensor_zeros, tf.expand_dims(selected_indices, 1), tf.range(selected_indices.shape[0],0,-1))
            output_tensor = tf.reshape(output_tensor,[-1,1]) 
            output_tensor = tf.cast(output_tensor, dtype=tf.float32)
            # print("output_tensor shape", output_tensor.shape)
            pred_output = tf.identity(output_tensor) #(?,1)
            # print("yqy pred shape",pred_output.shape)
            targets.append((f"rerank_gen_score_{i}", pred_output))
    
    # beam
    predict_=tf.identity(predict)
    indices, log_prob = beam_search(predict_) #(?,10,6)
    selected_score, selected_index = tf.nn.top_k(log_prob[0, :], 5, sorted=True) #(5,)
    indices=indices[0] #(10,6)
    selected_indices_beam = tf.gather(indices,selected_index) #(5,6)
    for i in range(5):#0-4
        selected_indices_i = selected_indices_beam[i,:] #(6,)
        tensor_zeros = tf.zeros(tf.shape(predict_)[2], dtype=tf.int32)
        output_tensor = tf.tensor_scatter_nd_update(tensor_zeros, tf.expand_dims(selected_indices_i, 1), tf.range(selected_indices_i.shape[0], 0, -1))
        output_tensor = tf.reshape(output_tensor, [-1, 1])
        output_tensor = tf.cast(output_tensor, dtype=tf.float32)
        # print("output_tensor shape", output_tensor.shape)
        pred_output = tf.identity(output_tensor) #(?,1)
        # print("pred_output shape",pred_output.shape)
        targets.append((f"rerank_gen_score_{i+5}",pred_output))

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
