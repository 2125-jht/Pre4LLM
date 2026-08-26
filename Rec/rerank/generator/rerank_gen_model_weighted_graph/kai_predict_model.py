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

            mask = tf.math.logical_or(tf.math.equal(sample_type, 1),
                                    tf.math.greater(sample_type, 2))
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
    indices = tf.expand_dims(indices, -1) #(?,k,1)
    for n1 in range(1, 6):
        predict_temp = tf.expand_dims(prediction[:,n1,:], 1) #第i个位置的所有分数 (?,1,60)
        predict_temp = tf.tile(predict_temp, [1, k, 1]) #(?,24,60)
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
    batch_size, seq_len, embedding_dim = tf.shape(item_embedding)[0], tf.shape(item_embedding)[1], tf.shape(item_embedding)[2] #(?,24,128)
    indices = tf.range(0, batch_size)[:, tf.newaxis] * seq_len + indices #(?,24)
    extracted_embeddings = tf.gather(tf.reshape(item_embedding, [-1, embedding_dim]), indices) #(?,24,128)
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

def contrastive_search(prediction, item_embedding, k=6):
    batch_size, seq_len, vocab_len = tf.shape(prediction)[0],tf.shape(prediction)[1],tf.shape(prediction)[2]

    selected_indices = None
    alpha = 0.7
    # alpha = 1 #only consider the scores

    for i in range(6):
        logits = prediction[:, i, :]
        log_prob, candidate_indices = tf.nn.top_k(prediction[:,i,:], k, sorted=True)
        candidate_embedding = extract_embeddings(item_embedding, candidate_indices)
        if selected_indices is not None:
            history_embedding = extract_embeddings(item_embedding, selected_indices)
            candidate_embedding = extract_embeddings(item_embedding, candidate_indices)
            # print("history_embedding shape ", history_embedding.shape)
            # print("candidate_embedding shape ", candidate_embedding.shape)
            norm_candiate = candidate_embedding / tf.norm(candidate_embedding, axis=2, keep_dims=True)
            norm_history = history_embedding / tf.norm(history_embedding, axis=2, keep_dims=True)
            cosine_matrix = tf.matmul(norm_candiate, tf.transpose(norm_history, perm=[0,2,1])) #similarity of candidate and history
            scores = tf.reduce_max(cosine_matrix, axis=-1)
            # print("scores shape ", scores.shape)

            # scores = log_prob-alpha*scores
            # scores = (1.0 - alpha)*log_prob-alpha*scores
            
            # yqy:only consider scores
            scores = scores
            # print("scores shape ", scores.shape)
            _, selected_idx = tf.nn.top_k(scores, k=1, sorted=True)
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
        logits = prediction[:, i, :] #each item's logits
        _, sampled_token = tf.nn.top_k(logits, k=1) #get the top-1 token index
        print("prediction shape ",prediction.shape)
        print("sampled_token shape ",sampled_token.shape)
        sampled_token = tf.squeeze(sampled_token, axis=-1)  #remove the last dimension
        prediction = set_zero_topk(prediction, sampled_token)  #set the logits of the sampled token to -inf
        generated_tokens.append(sampled_token)  #append the sampled token to the list
    generated_tokens = tf.stack(generated_tokens, axis=-1)  #stack the list of tokens into a tensor

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
        _, top_k_indices = tf.nn.top_k(logits, k=3) # Get the top-k indices
        top_k_indices = tf.transpose(tf.random.shuffle(tf.transpose(top_k_indices))) # Shuffle the top-k indices
        print("top_k_indices shape ",top_k_indices.shape) 
        sampled_token = top_k_indices[:,0] # Get the first sampled token
        print("sampled_token ", sampled_token.shape)
        prediction = set_zero_topk(prediction, sampled_token)
        generated_tokens.append(sampled_token)

    # Reshape the list to a tensor of shape (batch_size, seq_len)
    generated_tokens = tf.stack(generated_tokens, axis=-1)

    return generated_tokens

# dat decode
def inference_lookahead_repeatprevent(transition_matrix, token_distribution, target_length=6, lookahead=False):
    if lookahead:
        # E jointly considers P and E
        token_distribution_ = tf.reduce_max(token_distribution, axis=-1)  # 每个节点最大token对应的概率值 (?, 24)
        token_distribution_ = tf.expand_dims(token_distribution_, axis=1)  # (?, 1, 24)
        transition_matrix = transition_matrix + token_distribution_  # (?, 24, 24) 结合当前位置的分数和转移到下一个位置的分数

    batch_size = tf.shape(transition_matrix)[0]
    num_tokens = tf.shape(token_distribution)[-1]

    # 初始化输出token列表和已使用token的mask
    output_tokens = []
    used_mask = tf.zeros([batch_size, num_tokens], dtype=tf.bool)  # 初始化mask

    # 选择第一个节点的token
    tokens = tf.argmax(token_distribution, axis=-1)  # (?, 24) 每个节点对应的最大的token
    first_token = tf.cast(tokens[:, 0],dtype=tf.int32) #第一个token
    output_tokens.append(first_token)
    used_mask = tf.tensor_scatter_nd_update(
        used_mask, #(?,60)
        tf.stack([tf.range(batch_size), first_token], axis=1), #(?,2) 索引
        tf.ones([batch_size], dtype=tf.bool)
    ) #(?,60)
    cur_vertex = tf.cast(tf.argmax(transition_matrix[:, 0, :], axis=-1),dtype=tf.int32)   # 从第一个顶点开始选择下一个顶点 #(?,)
    
    for i in range(1, target_length):
        # 创建索引来从token_distribution中提取数据
        indices = tf.stack([tf.range(batch_size), cur_vertex], axis=1)
        # indices = tf.stack([tf.range(batch_size), cur_vertex, tf.zeros_like(cur_vertex, dtype=tf.int32)], axis=1)
        # 获取当前顶点的token分布，并应用mask
        current_token_distribution = tf.gather_nd(token_distribution, indices) #(?,60)
        masked_token_distribution = tf.where(
            used_mask,
            tf.fill([batch_size, num_tokens], -float('inf')),  # 已使用的token设置为负无穷
            current_token_distribution
        )
        # 选择当前顶点的最优token
        next_token = tf.cast(tf.argmax(masked_token_distribution, axis=-1),dtype=tf.int32)
        output_tokens.append(tf.cast(next_token,dtype=tf.int32))
        # 更新已使用token的mask
        used_mask = tf.tensor_scatter_nd_update(
            used_mask,
            tf.stack([tf.range(batch_size), next_token], axis=1),
            tf.ones([batch_size], dtype=tf.bool)
        )
        # 根据转移概率选择下一个顶点
        # cur_vertex = tf.gather(tf.argmax(transition_matrix, axis=-1), cur_vertex, batch_dims=1)
        # cur_vertex = tf.map_fn(lambda x: tf.argmax(transition_matrix[x, :, :], axis=-1)[cur_vertex[x]], tf.range(batch_size), dtype=tf.int32)
        cur_vertex = tf.map_fn(
                            lambda x: tf.cast(tf.argmax(transition_matrix[x, :, :], axis=-1), tf.int32)[cur_vertex[x]], 
                            tf.range(batch_size), 
                            dtype=tf.int32
                        )
    generated_tokens = tf.stack(output_tokens, axis=1)  # (?, 6)
    return generated_tokens

def inference_lookahead_greedy(transition_matrix,p,target_length=6,lookahead=False):    
    if lookahead:
        # E jointly considers P and E
        token_distribution = tf.reduce_max(p,axis=-1) #最大token对应的概率值 (?,24)
        token_distribution = tf.expand_dims(token_distribution,axis=1) #(?,1,24)
        transition_matrix = transition_matrix + token_distribution #(?,24,24) 结合当前位置的分数和转移到下一个位置的分数
    # simple greedy
    output_tokens = []
    # 1 parallel argmax to attain most likely transition and token for each vertex
    edges = tf.argmax(transition_matrix,axis=-1) #(?,24) 24个顶点对应的prob最高的下一个顶点
    tokens = tf.argmax(p,axis=-1) #(?,24) 24个顶点对应的prob最高的token
    # 2 generate the output_tokens by collecting the predicted tokens along the chosen path
    output_tokens.append(tokens[:,0]) #1st vertex's token  vertex0
    for i in range(target_length-1):        
        if i==0:
            cur_vertex = edges[:,i] #(?,1) 当前顶点 edges[:,0] eg. 0->3>6>10... vertex0下一个vertex3
        else:
            cur_vertex = tf.gather(edges,cur_vertex,batch_dims=1) #vertex3对应的下一个vertex6; edges[:,cur_vertex] #(?,1)
        next_token = tf.gather(tokens,cur_vertex,batch_dims=1) #vertex3对应的token; tokens[:,cur_vertex] #(?,1)
        output_tokens.append(next_token) 
    generated_tokens = tf.stack(output_tokens,axis=1) #(?,6)
    return generated_tokens
    
def mask_token_distribution(token_distribution,batch_size,num_vertices,sampled_token):
    # 生成batch索引
    batch_indices = tf.repeat(tf.range(batch_size), num_vertices) #0-10，每个重复24次
    # 生成vertex索引
    vertex_indices = tf.tile(tf.range(num_vertices), [batch_size]) #0-23, 共10次循环
    # 生成token索引
    token_indices = tf.repeat(sampled_token,num_vertices) #10个token，各自重复10次  
    # 堆叠样本索引、节点索引和token索引
    indices = tf.stack([batch_indices, vertex_indices, token_indices], axis=1)
    indices = tf.reshape(indices, [batch_size, num_vertices, 3])
    token_distribution = tf.tensor_scatter_nd_update(
        token_distribution,
        indices,
        tf.fill([batch_size,num_vertices], float('-inf')) #对每个batch的每个vertex的某一token填充-inf
    )
    return token_distribution

def inference_topk_sampling(transition_matrix,token_distribution,target_length=6,k=3,temperature=1.0):
    batch_size = tf.shape(token_distribution)[0]
    num_vertices = tf.shape(token_distribution)[1]
    num_tokens = tf.shape(token_distribution)[2]
    
    generated_tokens = []
    cur_vertex = tf.zeros((batch_size,), dtype=tf.int32)  # 从第一个节点开始
    for i in range(target_length):
        # 获取当前节点的token分布
        logits = tf.gather_nd(token_distribution, tf.stack([tf.range(batch_size), cur_vertex], axis=1)) #(?,60)
        # 获取topk
        top_k_values, top_k_indices = tf.nn.top_k(logits, k=k) #(?,k)
        scaled_logits = top_k_values / temperature 
        probs = tf.nn.softmax(scaled_logits, axis=-1) #(?,k)
        
        # 根据概率分布进行采样
        sampled_indices = tf.random.categorical(tf.math.log(probs), num_samples=1) #(?,1)
        # sampled_token = tf.batch_gather(top_k_indices, sampled_indices) #(?,1) 
        sampled_token = tf.gather(top_k_indices, sampled_indices, batch_dims=1) #(?,1)
        # mask
        token_distribution = mask_token_distribution(token_distribution,batch_size,num_vertices,sampled_token)
        # append
        generated_tokens.append(tf.reshape(sampled_token, [-1])) #(?,)
        # 根据转移概率矩阵选择下一个节点
        transition_probs = tf.gather_nd(transition_matrix, tf.stack([tf.range(batch_size), cur_vertex], axis=1))
        next_vertex = tf.argmax(transition_probs, axis=-1)
        cur_vertex = tf.cast(next_vertex, dtype=tf.int32)
    generated_tokens = tf.stack(generated_tokens, axis=-1) #(?,target_len)
    return generated_tokens

def inference_beam_search(transition_matrix, token_distribution, target_length=6, beam_width=3):
    batch_size = tf.cast(tf.shape(token_distribution)[0], dtype=tf.int32)
    num_vertices = tf.cast(tf.shape(token_distribution)[1], dtype=tf.int32)
    num_tokens = tf.cast(tf.shape(token_distribution)[2], dtype=tf.int32)
    
    # 初始化第一个节点的beam
    initial_logits = tf.gather_nd(token_distribution, 
                                tf.stack([tf.range(batch_size, dtype=tf.int32), 
                                tf.zeros((batch_size,), dtype=tf.int32)], axis=1))
    log_probs, initial_tokens = tf.nn.top_k(initial_logits, k=beam_width)
    
    # 初始化beam状态
    beam_tokens = tf.expand_dims(initial_tokens, axis=2)
    beam_scores = log_probs
    beam_vertices = tf.zeros((batch_size, beam_width), dtype=tf.int32)

    used_tokens = tf.zeros((batch_size,beam_width,num_tokens),dtype=tf.bool) #(?,3,60)
     # 标记初始token为已使用
    batch_indices = tf.range(batch_size)[:, tf.newaxis]  # (?, 1)
    beam_indices = tf.range(beam_width)[tf.newaxis, :]   # (1, beam_width)
    
    # 创建正确的索引形状
    indices = tf.stack([
        tf.tile(batch_indices, [1, beam_width]),         # (?, beam_width)
        tf.tile(beam_indices, [batch_size, 1]),          # (?, beam_width)
        initial_tokens                                    # (?, beam_width)
    ], axis=-1)  # (?, beam_width, 3)
    
    # 更新used_tokens
    used_tokens = tf.tensor_scatter_nd_update(
        used_tokens,
        tf.reshape(indices, [-1, 3]),  # 展平索引为 (? * beam_width, 3)
        tf.ones([batch_size * beam_width], dtype=tf.bool)  # 对应数量的True值
    )
    
    for i in range(1, target_length):
        # 展开所有可能的下一步
        flat_vertices = tf.reshape(beam_vertices, [-1])
        
        # 获取转移概率和token分布
        batch_indices = tf.repeat(tf.range(batch_size), beam_width)
        flat_transition_probs = tf.gather_nd(transition_matrix,
                                           tf.stack([batch_indices, flat_vertices], axis=1))
        
        flat_token_dist = tf.gather_nd(token_distribution,
                                     tf.stack([batch_indices, flat_vertices], axis=1))
        
        # 计算下一步得分
        next_vertex_scores = tf.reshape(flat_transition_probs, [batch_size, beam_width, num_vertices])
        next_token_scores = tf.reshape(flat_token_dist, [batch_size, beam_width, num_tokens])
        
        # mask掉已使用的token
        next_token_scores = tf.where(
            used_tokens,
            tf.fill(tf.shape(next_token_scores), float('-inf')),
            next_token_scores
        )
        
        # 选择top_k的下一步
        combined_scores = tf.expand_dims(beam_scores, axis=2) + next_token_scores
        flat_scores = tf.reshape(combined_scores, [batch_size, -1])
        top_scores, top_indices = tf.nn.top_k(flat_scores, k=beam_width)
        
        # 计算选中的token和beam索引
        beam_indices = top_indices // num_tokens
        token_indices = top_indices % num_tokens
        
        # 更新beam状态 - 使用gather with batch_dims
        new_tokens = tf.gather(beam_tokens, beam_indices, batch_dims=1)
        new_tokens = tf.concat([new_tokens, tf.expand_dims(token_indices, axis=2)], axis=2)
        
        # 选择下一个顶点
        next_vertices = tf.argmax(next_vertex_scores, axis=-1)
        next_vertices = tf.gather(next_vertices, beam_indices, batch_dims=1)
        
        # 更新状态
        beam_tokens = tf.cast(new_tokens, dtype=tf.int32)
        beam_scores = top_scores
        beam_vertices = tf.cast(next_vertices, dtype=tf.int32)
        
        # 更新used_tokens
        used_tokens = tf.gather(used_tokens, beam_indices, batch_dims=1)
        # 创建正确的索引
        batch_indices = tf.range(batch_size)
        indices = tf.stack([
            tf.repeat(batch_indices, beam_width),  # (batch_size * beam_width,)
            tf.tile(tf.range(beam_width), [batch_size]),  # (batch_size * beam_width,)
            tf.reshape(token_indices, [-1])  # (batch_size * beam_width,)
        ], axis=1)  # (batch_size * beam_width, 3)
        
        used_tokens = tf.tensor_scatter_nd_update(
            used_tokens,
            indices,  # (batch_size * beam_width, 3)
            tf.ones([batch_size * beam_width], dtype=tf.bool)
        )
    
    # 返回最高分的序列
    # best_tokens = beam_tokens[:, 0, :]
    return beam_tokens

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

predict, item_embedding, transition_matrix, token_distribution = model_class.model()

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

    gen_model_weight = tf.reshape(gen_model_weight, [-1, list_dim])
    _, rank_index = tf.math.top_k(gen_model_label, 60, sorted=True)
    shuffled_label_index = rank_index[:,:6] #TOP 6 index
    shuffled_label_index = tf.reverse(shuffled_label_index, axis=[1])

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
else:    
    targets=[]
    for i in range(10):
        print(f"====> i: {i}")
        transition_matrix_ = tf.identity(transition_matrix) #(?,24,24)
        token_distribution_ = tf.identity(token_distribution) #(?,24,60)        
        if i<2:
            selected_indices = inference_lookahead_repeatprevent( #(?,target_len)
                transition_matrix_,
                token_distribution_,
                target_length=12,
                lookahead= (i%2==0)
            )
        elif i<5:
            selected_indices = inference_topk_sampling( #(?,target_len)
                transition_matrix_,
                token_distribution_,
                target_length=12,
                k=i
            )
        else:
            if i==5:
                beam_tokens = inference_beam_search( #(?,beam_width,target_len)
                transition_matrix_,
                token_distribution_,
                target_length=12,
                beam_width=5
            )
            # i=6-9
            beam_idx = i-5 #0,1,2...4
            selected_indices = tf.cast(beam_tokens[:, beam_idx, :], dtype=tf.int32) #(?,target_len)

        selected_indices = tf.cast(selected_indices[0,:], dtype=tf.int32) #(target_len,)
        tensor_zeros = tf.zeros(tf.shape(token_distribution)[2],dtype=tf.int32) #(60,)
        update_index = tf.range(12,0,-1, dtype=tf.int32) #(12,)
        output_tensor = tf.tensor_scatter_nd_update(
            tensor_zeros, 
            tf.expand_dims(selected_indices, 1), 
            update_index
        ) #(60,)
        output_tensor = tf.reshape(output_tensor,[-1,1]) #(60,1)
        output_tensor = tf.cast(output_tensor, dtype=tf.float32)
        pred_output = tf.identity(output_tensor) #
        targets.append((f"rerank_gen_score_{i}", pred_output))

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
