import tensorflow as tf
import numpy as np
import sys

def calc_sim_cos(a):
    a_norm = tf.nn.l2_normalize(a, axis=1)  # [bs, dim]
    
    # 2. 计算余弦相似度矩阵（点积）
    sim_matrix = tf.matmul(a_norm, a_norm, transpose_b=True)  # [bs, bs]
    
    # 3. 去除对角元素（自身与自身的相似度为1）
    bs = tf.shape(a)[0]
    mask = tf.ones_like(sim_matrix) - tf.eye(bs)
    sim_matrix_no_diag = sim_matrix * mask  # [bs, bs]
    
    # 4. 计算两两相似度的平均值（不包括对角线）
    total_pairs = tf.cast(bs * (bs - 1), tf.float32)
    avg_cos_sim = tf.reduce_sum(sim_matrix_no_diag) / total_pairs
    return avg_cos_sim

def sigmoid_layer(loss_name, left_input, right_input):
  with tf.variable_scope("{}_loss".format(loss_name), reuse=tf.AUTO_REUSE):
      output = tf.reduce_sum(tf.multiply(
          left_input, right_input), axis=1, keepdims=True)
      output = tf.sigmoid(output)
  return output


def get_duplicate(name, ids):
    one_hot = tf.cast(tf.eye(tf.shape(ids)[0]), tf.float64)
    duplicate_matrix = tf.cast(tf.equal(ids, tf.transpose(ids)), tf.float64) - one_hot
    tf.summary.scalar("id_duplicate/{}".format(name), tf.reduce_mean(tf.reduce_sum(duplicate_matrix, axis=1)))

def print_tensor(name, tensor):
    tf.summary.scalar(name, tf.reduce_mean(tensor))
    tf.summary.histogram(name, tensor)

def log_tensor(name, tensor):
    tf.summary.scalar(name, tf.reduce_mean(tensor))
    tf.summary.histogram(name, tensor)
    zero_ratio = tf.reduce_mean(tf.where(tf.equal(tensor, 0), tf.ones_like(tensor, dtype=tf.float32), tf.zeros_like(tensor, dtype=tf.float32)))
    tf.summary.scalar(name + "zero_ratio", zero_ratio)

def similarity(emb, name="default", epsilon=1e-8):
    norms = tf.norm(emb, axis=1, keepdims=True)
    normalized_embeddings = emb / (norms + epsilon)
    cosine_similarity_matrix = tf.matmul(normalized_embeddings, normalized_embeddings, transpose_b=True)

    num_row = tf.shape(emb)[0]
    num_col = tf.shape(emb)[0]
    
    lower_matrix_mask = tf.linalg.band_part(tf.ones((num_row, num_col)),-1, 0)
    lower_matrix_mask = tf.where(tf.eye(num_row) > 0, tf.zeros_like(lower_matrix_mask), lower_matrix_mask)

    masked_cos_matrix = tf.boolean_mask(cosine_similarity_matrix, lower_matrix_mask > 0.5)

    tf.summary.scalar("cos_similarity/{}".format(name), tf.reduce_mean(masked_cos_matrix))
    tf.summary.histogram("cos_similarity/{}".format(name), masked_cos_matrix)

# def recall_at_k(predict, label, loss_mask, print_ops, top_k=[5, 15], name="default"):
#     true_label_expanded = tf.expand_dims(label, axis=1)
#     for k in top_k:
#         top_k_values, top_k_indices = tf.nn.top_k(predict, k=k)
#         correct = tf.equal(top_k_indices, true_label_expanded)
#         correct_any = tf.reduce_any(correct, axis=1)

#         # print_ops.append(tf.print("correct_any_shape", tf.shape(correct_any), summarize=-1, output_stream=sys.stdout))

#         recall_at_k = tf.reduce_sum(tf.cast(correct_any, tf.float32) * loss_mask) / (tf.reduce_sum(loss_mask) + 0.001)

#         tf.summary.scalar("top_k/{}_{}".format(name, k), recall_at_k)

def recall_at_k(predictions, top_k=[5, 15], indicator=None, name="default"):
    max_k = max(top_k)

    _, indices = tf.nn.top_k(predictions, k=max_k, sorted=True)
    labels = tf.reshape(tf.range(0, tf.shape(predictions)[0]), [-1, 1])
    for k in top_k:
        top_k_indices = tf.slice(indices, [0, 0], [-1, k])
        tp = tf.reduce_any(tf.equal(top_k_indices, labels), axis=1)
        num =  tf.cast(tf.shape(predictions)[0],dtype=tf.float32)
        if indicator is not None:
            indicator = tf.reshape(indicator, tf.shape(tp))
            tp = tf.boolean_mask(tp, indicator)
            num = tf.reduce_sum(tf.cast(indicator, dtype=tf.float32))
        recall_at_k = tf.reduce_sum(tf.cast(tp, dtype=tf.float32))/num
        tf.summary.scalar("top_k/{}_{}".format(name, k), recall_at_k)

def mlp(name, net, hidden_units, output_unit=None, activation=tf.nn.relu):
  scope = name + '_mlp'
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
      for i, k in enumerate(hidden_units):
          layer_name = scope + '_{}'.format(i)
          net = tf.layers.dense(net, k, activation=activation, name=layer_name)
      if output_unit != None:
          net = tf.layers.dense(net, output_unit, activation=None, name=scope + '_final')
  return net

def sampled_softmax_loss(label, left_emb, right_emb, t=0.05, logQ=None, name="Default"):
    cos_mat = tf.matmul(left_emb, right_emb, transpose_b=True)                   # [bz, bz]
    if logQ is not None:
        sim_mat = cos_mat / t - tf.reshape(logQ, [1, -1])
    else:
        sim_mat = cos_mat / t
    bz = tf.shape(left_emb)[0]
    fake_label = tf.eye(bz)
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=sim_mat, labels=fake_label)
    loss = tf.reshape(loss, [-1, 1])
    with tf.variable_scope("sampled_softmax_{}".format(name), reuse=tf.AUTO_REUSE) as scope:
        num = tf.cast(bz, dtype=tf.float32)
        tf.summary.scalar('mean_pos_cosine', tf.reduce_sum(fake_label * cos_mat) / num)
        tf.summary.scalar('mean_neg_cosine', tf.reduce_sum((cos_mat - fake_label * cos_mat)) / (num*num-num))
    return tf.reduce_sum(loss*label), cos_mat
