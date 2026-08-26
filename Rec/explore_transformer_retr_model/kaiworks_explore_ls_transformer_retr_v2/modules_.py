import tensorflow as tf
import numpy as np

def calc_sim_cos_both(x, y):
  return tf.reduce_mean(tf.reduce_sum(tf.nn.l2_normalize(x, axis=-1) * tf.nn.l2_normalize(y, axis=-1), axis=-1))

def calc_sim_cos_singel(a):
    """
    计算批次内向量的平均余弦相似度
    
    用于衡量批次内向量表示的多样性，相似度越高说明向量越相似，
    多样性越低。常用于监控模型是否出现表示坍塌问题。
    
    Args:
        a: 输入张量，形状为[batch_size, embedding_dim]
        
    Returns:
        avg_cos_sim: 标量，批次内所有向量对的平均余弦相似度
    """
    # 1. L2归一化，将向量归一化到单位球面上
    a_norm = tf.nn.l2_normalize(a, axis=1)  # [batch_size, embedding_dim]
    
    # 2. 计算余弦相似度矩阵（归一化后的点积就是余弦相似度）
    sim_matrix = tf.matmul(a_norm, a_norm, transpose_b=True)  # [batch_size, batch_size]
    
    # 3. 去除对角元素（自身与自身的相似度恒为1，不参与统计）
    bs = tf.shape(a)[0]
    mask = tf.ones_like(sim_matrix) - tf.eye(bs)  # 对角线为0，其他位置为1的掩码
    sim_matrix_no_diag = sim_matrix * mask  # [batch_size, batch_size]
    
    # 4. 计算非对角线元素的平均相似度
    total_pairs = tf.cast(bs * (bs - 1), tf.float32)  # 总的向量对数量
    avg_cos_sim = tf.reduce_sum(sim_matrix_no_diag) / total_pairs
    return avg_cos_sim

def calc_sim_cos(x, y=None):
    if y is None:
        return calc_sim_cos_singel(x)
    else:
        return calc_sim_cos_both(x,y)

def sigmoid_layer(loss_name, left_input, right_input):
  with tf.variable_scope("{}_loss".format(loss_name), reuse=tf.AUTO_REUSE):
      output = tf.reduce_sum(tf.multiply(
          left_input, right_input), axis=1, keepdims=True)
      output = tf.sigmoid(output)
  return output

def get_interest_similarity(name, user_interest):
   # user_interest
   for i in range(4):
       for j in range(i + 1, 4):
           first_embedding = user_interest[:, i, :]
           second_embedding = user_interest[:, j, :]
           sim = calc_sim_cos(first_embedding, second_embedding)
           print_tensor(name + "/{}_{}".format(i, j), sim)

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


# def recall_at_k(predictions, top_k=[5, 15], indicator=None, name="default"):
#     max_k = max(top_k)

#     _, indices = tf.nn.top_k(predictions, k=max_k, sorted=True)
#     labels = tf.reshape(tf.range(0, tf.shape(predictions)[0]), [-1, 1])
#     for k in top_k:
#         top_k_indices = tf.slice(indices, [0, 0], [-1, k])
#         tp = tf.reduce_any(tf.equal(top_k_indices, labels), axis=1)
#         num =  tf.cast(tf.shape(predictions)[0],dtype=tf.float32)
#         if indicator is not None:
#             indicator = tf.reshape(indicator, tf.shape(tp))
#             tp = tf.boolean_mask(tp, indicator)
#             num = tf.reduce_sum(tf.cast(indicator, dtype=tf.float32))
#         recall_at_k = tf.reduce_sum(tf.cast(tp, dtype=tf.float32))/num
#         tf.summary.scalar("top_k/{}_{}".format(name, k), recall_at_k)

# def recall_at_k(predict, label, loss_mask, print_ops, top_k=[5, 15], name="default"):
#     """
#     计算Top-K召回率
    
#     评估模型在Top-K预测中包含正确答案的比例。
#     召回率是推荐系统中的重要评估指标。
    
#     Args:
#         predict: 预测logits，形状为[batch_size, vocab_size]
#         label: 真实标签，形状为[batch_size]，包含正确的类别索引
#         loss_mask: 损失掩码，形状为[batch_size]，1表示有效样本，0表示无效样本
#         print_ops: 打印操作列表（此处未使用）
#         top_k: 要计算的K值列表，如[5, 15]表示计算Top-5和Top-15召回率
#         name: 指标名称，用于TensorBoard显示
#     """
#     # 将标签扩展为[batch_size, 1]，便于后续比较
#     true_label_expanded = tf.expand_dims(label, axis=1)
    
#     # 对每个K值计算召回率
#     for k in top_k:
#         # 1. 获取Top-K预测结果
#         top_k_values, top_k_indices = tf.nn.top_k(predict, k=k)  # [batch_size, k]
        
#         # 2. 检查Top-K中是否包含正确标签
#         correct = tf.equal(top_k_indices, true_label_expanded)  # [batch_size, k]
        
#         # 3. 每个样本只要Top-K中有一个正确即算命中
#         correct_any = tf.reduce_any(correct, axis=1)  # [batch_size]

#         # 4. 计算加权召回率（只考虑有效样本）
#         recall_at_k_value = tf.reduce_sum(tf.cast(correct_any, tf.float32) * loss_mask) / (tf.reduce_sum(loss_mask) + 0.001)

#         # 5. 记录到TensorBoard
#         tf.summary.scalar("top_k/{}_{}".format(name, k), recall_at_k_value)

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


def rms_norm(x, eps=1e-8, p=-1., bias=False, scope=None):
    """
        Root Mean Square Layer Normalization
    :param x: input tensor, with shape [batch, ..., dimension]
    :param eps: epsilon value, default 1e-8
    :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
    :param bias: whether use bias term for RMSNorm, disabled by
        default because RMSNorm doesn't enforce re-centering invariance.
    :param scope: the variable scope
    :return: a normalized tensor, with shape as `x`
    """
    with tf.variable_scope(scope or "rms_norm"):
        layer_size = x.get_shape().as_list()[-1]

        scale = tf.get_variable("scale", [layer_size], initializer=tf.ones_initializer())
        if bias:
            offset = tf.get_variable("offset", [layer_size], initializer=tf.zeros_initializer())
        else:
            offset = 0.

        if p < 0. or p > 1.:
            ms = tf.reduce_mean(x ** 2, -1, keepdims=True)
        else:
            partial_size = int(layer_size * p)
            partial_x, _ = tf.split(x, [partial_size, layer_size - partial_size], axis=-1)

            ms = tf.reduce_mean(partial_x ** 2, -1, keepdims=True)

        return scale * x * tf.rsqrt(ms + eps) + offset

def gelu(x):
    return 0.5 * x * (1 + tf.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * tf.pow(x,3))))

def multihead_attention(queries, 
                        keys,
                        values, 
                        num_units=None, 
                        num_heads=8, 
                        dropout_rate=0,
                        mask=None,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention", 
                        reuse=None,
                        with_qk=False,
                        num_blocks=0):
    '''Applies multihead attention.
    
    Args:
        queries: A 3d tensor with shape of [N, T_q, C_q].
        keys: A 3d tensor with shape of [N, T_k, C_k].
        num_units: A scalar. Attention size.
        dropout_rate: A floating point number.
        is_training: Boolean. Controller of mechanism for dropout.
        causality: Boolean. If true, units that reference the future are masked. 
        num_heads: An int. Number of heads.
        scope: Optional scope for `variable_scope`.
        reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.
        
    Returns
        A 3d tensor with shape of (N, T_q, C)  
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]
        
        # Linear projections
        # Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu) # (N, T_q, C)
        # K = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        # V = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        Q = tf.layers.dense(queries, num_units, activation=None) # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=None) # (N, T_k, C)
        V = tf.layers.dense(values, num_units, activation=None) # (N, T_k, C)
        
        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, C/h) 
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)
        
        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
        tf.summary.histogram("attentions_weight_before_softmax/%d_layer" % num_blocks, outputs)
        
        # Key Masking
        # key_masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1)) # (N, T_k)
        # key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
        # key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)
        
        # paddings = tf.ones_like(outputs)*(-2**32+1)
        # outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)
        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :]) # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense() # (T_q, T_k)
            cau_masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]) # (h*N, T_q, T_k)
            cau_masks = tf.cast(cau_masks, outputs.dtype)
            paddings = tf.ones_like(cau_masks)*(-2**32+1)
            outputs = tf.where(tf.equal(cau_masks, 0), paddings, outputs) # (h*N, T_q, T_k)

        
        # if mask is not None:
        #     # origin mask # [bs, seq_len], transform mask #[bs, seq_len, seq_len]
        #     print('mask', mask)
        #     mask = tf.tile(tf.expand_dims(mask, axis=1), [num_heads, tf.shape(mask)[1], 1])
        #     print('mask', mask)
        #     print("cau_masks", cau_masks)
        #     print("outputs", outputs)
        #     paddings = tf.ones_like(mask)*(-2**32+1)
        #     outputs = tf.where(tf.equal(mask, 0), paddings, outputs)

        # Activation
        outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)
        tf.summary.histogram("attentions_weight/%d_layer" % num_blocks, outputs)
        # # Query Masking
        # query_masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1)) # (N, T_q)
        # query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
        # query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # (h*N, T_q, T_k)
        # outputs *= query_masks # broadcasting. (N, T_q, C)
        
        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        # Weighted sum
        outputs = tf.matmul(outputs, V_) # ( h*N, T_q, C/h)
        
        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2) # (N, T_q, C)
        # Residual connection
        # outputs += keys
        # Normalize
        #outputs = normalize(outputs) # (N, T_q, C)
        outputs = tf.layers.dense(outputs, num_units, activation=None) # (N, T_q, C)
        
    if with_qk: return Q,K
    else: return outputs


def feedforward(inputs, 
                num_units=[2048, 512],
                scope="feedforward", 
                dropout_rate=0.2,
                is_training=True,
                reuse=None):
    '''Point-wise feed forward net.
    
    Args:
        inputs: A 3d tensor with shape of [N, T, C].
        num_units: A list of two integers.
        scope: Optional scope for `variable_scope`.
        reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.
        
    Returns:
        A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                    "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        outputs = gelu(outputs)
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                    "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        
        # Residual connection
        # outputs += inputs
        
        # Normalize
        #outputs = normalize(outputs)
    
    return outputs

def transformer_encoder(seq_input_embeddings, num_layer, num_units, dropout_rate, num_heads=8, mask=None, training=True):
    for i in range(num_layer):
        with tf.variable_scope("encoder_num_blocks_%d" % i):
            with tf.variable_scope("atten"):
                attn_out = multihead_attention(queries=rms_norm(seq_input_embeddings, scope="rms_atten"),
                                                            keys=seq_input_embeddings,
                                                            values=seq_input_embeddings,
                                                            num_units=num_units,
                                                            num_heads=num_heads,
                                                            dropout_rate=dropout_rate,
                                                            mask=mask,
                                                            is_training=training,
                                                            causality=False,
                                                            scope="self_attention",
                                                            num_blocks=i)
                seq_input_embeddings += attn_out
            with tf.variable_scope("ffn"):
                ffn_out = feedforward(rms_norm(seq_input_embeddings, scope="rms_ffn"),num_units=[num_units*4,num_units],dropout_rate=dropout_rate,is_training=training)
                seq_input_embeddings += ffn_out
    return rms_norm(seq_input_embeddings, scope="final_rms_enc")


def transformer_decoder(decoder_embedding, seq_input_embeddings, num_layer, num_units, dropout_rate, num_heads=8, mask=None, training=True):
    for i in range(num_layer):
        with tf.variable_scope("decoder_num_blocks_%d" % i):
            with tf.variable_scope("self_atten"):
                self_atten_out = multihead_attention(queries=rms_norm(decoder_embedding, scope="rms_self_atten"),
                                                            keys=decoder_embedding,
                                                            values=decoder_embedding,
                                                            num_units=num_units,
                                                            num_heads=num_heads,
                                                            dropout_rate=dropout_rate,
                                                            mask=mask,
                                                            is_training=training,
                                                            causality=True,
                                                            scope="self_attention",
                                                            num_blocks=i)
                decoder_embedding += self_atten_out
                
            with tf.variable_scope("cross_atten"):
                cross_atten_out = multihead_attention(queries=rms_norm(decoder_embedding, scope="rms_cross_atten"),
                                                            keys=seq_input_embeddings,
                                                            values=seq_input_embeddings,
                                                            num_units=num_units,
                                                            num_heads=num_heads,
                                                            dropout_rate=dropout_rate,
                                                            mask=mask,
                                                            is_training=training,
                                                            causality=False,
                                                            scope="cross_attention",
                                                            num_blocks=i)
                decoder_embedding += cross_atten_out
                
            with tf.variable_scope("ffn"):
                ffn_out = feedforward(rms_norm(decoder_embedding, scope="rms_ffn"),num_units=[num_units*4,num_units],dropout_rate=dropout_rate,is_training=training)
                decoder_embedding += ffn_out
                
    return rms_norm(decoder_embedding, scope="final_rms_dec")



def gumbel_softmax(logits, temperature=1.0, hard=False, eps=1e-20):
    # logits [batch_size, history_size]
    U = tf.random_uniform(tf.shape(logits), minval=0, maxval=1)
    gumbel_noise = -tf.log(-tf.log(U + eps) + eps) * 0.01

    print_tensor("origin_logits", logits)
    print_tensor("gumbel_noise", gumbel_noise)
    logits_with_noise = logits + gumbel_noise
    y = tf.nn.softmax(logits_with_noise / temperature, axis=-1)

    if hard:
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, axis=-1, keepdims=True)), y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y

def gumbel_top_k(logits, k, temperature):
    gumbel_probs = gumbel_softmax(logits, temperature, hard=False)
    _, indices_1 = tf.nn.top_k(gumbel_probs, k=k)
    indices_2, _ = tf.nn.top_k(indices_1, k=k, sorted=False)
    indices = tf.reverse(indices_2, axis=[1])
    return indices

def watch_time_encoding(play_time, duration, embedding_table, embedding_dim=64, num_buckets=10,):
    values = play_time / duration
    # 将 values 归一化到 [0, num_buckets) 之间
    bucket_indices = tf.cast(tf.floor(values * num_buckets), tf.int32)
    # 确保最大值索引不会超过 num_buckets-1
    bucket_indices = tf.minimum(bucket_indices, num_buckets - 1)
    # 查找与桶索引对应的 embedding
    embedded_tensor = tf.nn.embedding_lookup(embeddings, bucket_indices)

    return embedded_tensor

class Gate_NU():
    def __init__(self, hidden_dim, output_dim):
        self.dense1 = tf.layers.Dense(units=hidden_dim, activation=tf.nn.relu, name='dense1')
        self.dense2 = tf.layers.Dense(units=output_dim, activation=tf.nn.softmax, name='dense2')

    def __call__(self, inputs):
        mid = self.dense1(inputs)
        output = 2 * self.dense2(mid)
        return output

class MLP():
    def __init__(self, hidden_dim, output_dim, activation_one, activation_two=None):
        super(MLP, self).__init__()
        self.dense1 = tf.layers.Dense(units=hidden_dim, activation=activation_one, name='dense1')
        self.dense2 = tf.layers.Dense(units=output_dim, activation=activation_two, name='dense2')
    
    def __call__(self, inputs):
        # 使用 tf.layers.dense API 进行全连接层的计算
        mid = self.dense1(inputs)
        output = self.dense2(mid)
        return output

def label_encoding(label_tensor, embedding_dim=64):
    # input: lable_tensor [batch_size, history_size]
    # output: result_encoding [batch_size, history_size, embedding_dim]
    batch_size = tf.shape(label_tensor)[0]
    like = tf.not_equal(tf.bitwise.bitwise_and(label_tensor, 1), 0)
    follow = tf.not_equal(tf.bitwise.bitwise_and(label_tensor, 1 << 1), 0)
    forward = tf.not_equal(tf.bitwise.bitwise_and(label_tensor, 1 << 2), 0)
    hate = tf.not_equal(tf.bitwise.bitwise_and(label_tensor, 1 << 3), 0)
    comment = tf.not_equal(tf.bitwise.bitwise_and(label_tensor, 1 << 4), 0)
    has_entered_profile = tf.not_equal(tf.bitwise.bitwise_and(label_tensor, 1 << 6), 0)
    has_comment_stay_time_value = tf.not_equal(tf.bitwise.bitwise_and(label_tensor, 1 << 8), 0)

    positive_1 = tf.logical_or(like, tf.logical_or(follow, forward))
    positive_2 = tf.logical_or(comment, tf.logical_or(has_entered_profile, has_comment_stay_time_value))
    positive = tf.logical_or(positive_1, positive_2)

    negative = hate
    neutral = tf.logical_and(tf.logical_not(positive), tf.logical_not(negative))

    ones = tf.ones_like(label_tensor)
    twos = ones + ones

    result = tf.zeros_like(label_tensor)
    result = tf.where(positive, ones, result)
    result = tf.where(negative, twos, result)

    label_encoding = dense_encoding(3, "label", embedding_dim)
    result_encoding = label_encoding(result)
    result_encoding = tf.reshape(result_encoding, [batch_size, -1, embedding_dim])
    return result_encoding


def play_time_encoding(play_time, duration, embedding_dim=64):
    # input: play_time duration [batch_size, history_size]
    # output: completion_encoding [batch_size, history_size, embedding_dim]
    batch_size = tf.shape(play_time)[0]
    play_time = tf.reshape(play_time, [-1])
    duration = tf.reshape(duration, [-1])

    completion = tf.cast(tf.cast(play_time, tf.float32) / (tf.cast(duration, tf.float32) + 1e-5) * 10, tf.int32)
    completion = tf.where(completion > 10, tf.fill(tf.shape(play_time), 10), completion)

    completion_encoding_layer = dense_encoding(11, "completion", embedding_dim)
    completion_encoding = tf.reshape(completion_encoding_layer(completion), [batch_size, -1, embedding_dim])
    return completion_encoding


class dense_encoding():
    def __init__(self, vocab_size, name, embedding_dim=64):
        self._embedding_matrix = tf.Variable(tf.random_normal([vocab_size, embedding_dim]), name=name+"_embedding_matrix")

    def __call__(self, input_indices):
        return tf.nn.embedding_lookup(self._embedding_matrix, input_indices)

def photo_score_log_loss(label, pred, logQ=None, name="Default", tab_mask = None):
    loss = tf.losses.log_loss(labels=label, predictions=pred, weights=tf.ones_like(pred),
                            reduction=tf.losses.Reduction.NONE)
    final_loss = loss * tab_mask

    batch_size = tf.reduce_sum(tf.ones_like(tab_mask))
    tab_mask_count = tf.reduce_sum(tab_mask)
    tab_mask_rate = tab_mask_count / batch_size

   
    tf.summary.scalar('quality_label_rate/' + 'batch_size', batch_size)
    tf.summary.scalar('quality_label_rate/' + 'tab_mask_count', tab_mask_count)
    tf.summary.scalar('quality_label_rate/' + 'tab_mask_rate', tab_mask_rate)

    num_positive = tf.reduce_sum(label * tab_mask)
    num_negative = tab_mask_count - num_positive
    pos_neg_rate = num_positive / num_negative
    
    tf.summary.scalar('pos_neg_rate/' + "num_positive", num_positive)
    tf.summary.scalar('pos_neg_rate/' + "num_negative", num_negative)
    tf.summary.scalar('pos_neg_rate/' + "pos_neg_rate", pos_neg_rate)
   

    emp_xtr = tf.reduce_sum(label * tab_mask) / tf.reduce_sum(tab_mask)
    pred_xtr = tf.reduce_sum(pred * tab_mask) / tf.reduce_sum(tab_mask)
    tf.summary.scalar('eval_xtr/' + name + "_emp_xtr", emp_xtr)
    tf.summary.scalar('eval_xtr/' + name + "_pred_xtr", pred_xtr)
    tf.summary.scalar('eval_xtr/' + name + "_xtr_gap", emp_xtr - pred_xtr)

    return tf.reduce_sum(final_loss)