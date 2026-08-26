# -*- coding: utf-8 -*-
"""
深度学习模型工具函数库

包含常用的模型组件和工具函数：
1. 相似度计算函数
2. 损失函数和激活函数
3. 评估指标计算
4. 通用网络层定义
5. 张量统计和可视化工具
"""

import tensorflow as tf
import numpy as np
import sys

def calc_sim_cos(a):
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

def calc_sim_cos_btd(a, eps=1e-12):
    """
    [B, T, D] 相似度统计：
    1) 对每个 b：计算该样本内所有 (t_i, t_j), i!=j 的余弦相似度均值
    2) 再对 batch 维做平均，得到标量

    Args:
        a: tf.Tensor, shape [B, T, D]
        eps: 数值稳定项
    Returns:
        avg_cos_sim: scalar，batch 内整体平均相似度
        per_sample_avg: [B]，每条样本内部的平均相似度（可用于监控）
    """
    a = tf.convert_to_tensor(a)
    # 维度：B,T,D
    B = tf.shape(a)[0]
    T = tf.shape(a)[1]

    # 1) 每个 token 向量做 L2 normalize（沿 D 归一化）
    a_norm = tf.nn.l2_normalize(a, axis=2, epsilon=eps)  # [B, T, D]

    # 2) 每个样本内做相似度矩阵：S[b] = a_norm[b] @ a_norm[b]^T
    # 结果 [B, T, T]
    sim = tf.matmul(a_norm, a_norm, transpose_b=True)

    # 3) 去对角线（i==j）
    mask = tf.ones([T, T], dtype=sim.dtype) - tf.eye(T, dtype=sim.dtype)  # [T, T]
    sim_no_diag = sim * mask[None, :, :]  # [B, T, T]

    # 4) 每条样本内部平均：sum(sim_no_diag) / (T*(T-1))
    total_pairs_per_sample = tf.cast(T * (T - 1), sim.dtype)  # scalar
    # 防止 T<2 时除0：这里用最大值保护（也可以按你需求返回0）
    denom = tf.maximum(total_pairs_per_sample, tf.cast(1.0, sim.dtype))

    per_sample_sum = tf.reduce_sum(sim_no_diag, axis=[1, 2])  # [B]
    per_sample_avg = per_sample_sum / denom                  # [B]

    # 5) batch 平均
    avg_cos_sim = tf.reduce_mean(per_sample_avg)  # scalar

    return avg_cos_sim


def sigmoid_layer(loss_name, left_input, right_input):
    """
    Sigmoid层实现
    
    计算两个向量的点积并通过sigmoid激活，常用于二分类任务
    或者计算两个向量的相似度评分
    
    Args:
        loss_name: 损失名称，用于变量作用域命名
        left_input: 左侧输入向量，形状为[batch_size, dim]
        right_input: 右侧输入向量，形状为[batch_size, dim]
        
    Returns:
        output: sigmoid激活后的输出，形状为[batch_size, 1]，值域为(0,1)
    """
    with tf.variable_scope("{}_loss".format(loss_name), reuse=tf.AUTO_REUSE):
        # 计算元素级乘积并求和（等价于点积）
        output = tf.reduce_sum(tf.multiply(left_input, right_input), axis=1, keepdims=True)
        # 通过sigmoid激活函数映射到(0,1)区间
        output = tf.sigmoid(output)
    return output

def get_duplicate(name, ids):
    """
    计算ID重复率统计
    
    统计批次内ID的重复情况，用于监控数据质量。
    重复率高可能表示数据采样存在偏差。
    
    Args:
        name: 统计名称，用于TensorBoard显示
        ids: ID张量，形状为[batch_size, 1]或[batch_size]
    """
    # 创建单位矩阵，用于排除自身比较
    one_hot = tf.cast(tf.eye(tf.shape(ids)[0]), tf.float64)
    
    # 创建重复矩阵：相同ID位置为1，不同ID位置为0
    duplicate_matrix = tf.cast(tf.equal(ids, tf.transpose(ids)), tf.float64) - one_hot
    
    # 计算每行重复的平均数量，并记录到TensorBoard
    tf.summary.scalar("id_duplicate/{}".format(name), 
                     tf.reduce_mean(tf.reduce_sum(duplicate_matrix, axis=1)))

def print_tensor(name, tensor):
    """
    张量统计记录
    
    将张量的统计信息记录到TensorBoard，包括均值和分布直方图
    
    Args:
        name: 张量名称，用于TensorBoard显示
        tensor: 要统计的张量
    """
    # 记录张量的均值
    tf.summary.scalar(name, tf.reduce_mean(tensor))
    # 记录张量值的分布直方图
    tf.summary.histogram(name, tensor)

def log_tensor(name, tensor):
    """
    详细的张量统计记录
    
    除了基本的均值和分布外，还统计零值占比，
    用于监控稀疏性或激活情况
    
    Args:
        name: 张量名称，用于TensorBoard显示
        tensor: 要统计的张量
    """
    # 记录基本统计
    tf.summary.scalar(name, tf.reduce_mean(tensor))
    tf.summary.histogram(name, tensor)
    
    # 计算零值占比
    zero_ratio = tf.reduce_mean(
        tf.where(tf.equal(tensor, 0), 
                tf.ones_like(tensor, dtype=tf.float32), 
                tf.zeros_like(tensor, dtype=tf.float32))
    )
    tf.summary.scalar(name + "zero_ratio", zero_ratio)

def similarity(emb, name="default", epsilon=1e-8):
    """
    计算嵌入向量的余弦相似度分布
    
    计算批次内所有向量对的余弦相似度，并统计分布情况。
    只考虑下三角矩阵以避免重复计算和自相似度。
    
    Args:
        emb: 嵌入向量张量，形状为[batch_size, embedding_dim]
        name: 统计名称，用于TensorBoard显示
        epsilon: 防止除零的小常数
    """
    # 1. 计算L2范数并归一化
    norms = tf.norm(emb, axis=1, keepdims=True)  # [batch_size, 1]
    normalized_embeddings = emb / (norms + epsilon)  # [batch_size, embedding_dim]
    
    # 2. 计算余弦相似度矩阵
    cosine_similarity_matrix = tf.matmul(normalized_embeddings, 
                                       normalized_embeddings, 
                                       transpose_b=True)  # [batch_size, batch_size]

    # 3. 创建矩阵维度
    num_row = tf.shape(emb)[0]
    num_col = tf.shape(emb)[0]
    
    # 4. 创建下三角掩码（不包括对角线）
    # band_part(-1, 0)创建下三角矩阵（包括对角线）
    lower_matrix_mask = tf.linalg.band_part(tf.ones((num_row, num_col)), -1, 0)
    # 去除对角线元素
    lower_matrix_mask = tf.where(tf.eye(num_row) > 0, 
                               tf.zeros_like(lower_matrix_mask), 
                               lower_matrix_mask)

    # 5. 应用掩码，只保留下三角部分的相似度值
    masked_cos_matrix = tf.boolean_mask(cosine_similarity_matrix, 
                                      lower_matrix_mask > 0.5)

    # 6. 记录相似度统计
    tf.summary.scalar("cos_similarity/{}".format(name), tf.reduce_mean(masked_cos_matrix))
    tf.summary.histogram("cos_similarity/{}".format(name), masked_cos_matrix)
def recall_at_k(predict, label, loss_mask, print_ops, top_k=[5, 15], name="default"):
    """
    计算Top-K召回率
    
    评估模型在Top-K预测中包含正确答案的比例。
    召回率是推荐系统中的重要评估指标。
    
    Args:
        predict: 预测logits，形状为[batch_size, vocab_size]
        label: 真实标签，形状为[batch_size]，包含正确的类别索引
        loss_mask: 损失掩码，形状为[batch_size]，1表示有效样本，0表示无效样本
        print_ops: 打印操作列表（此处未使用）
        top_k: 要计算的K值列表，如[5, 15]表示计算Top-5和Top-15召回率
        name: 指标名称，用于TensorBoard显示
    """
    # 将标签扩展为[batch_size, 1]，便于后续比较
    true_label_expanded = tf.expand_dims(label, axis=1)
    
    # 对每个K值计算召回率
    for k in top_k:
        # 1. 获取Top-K预测结果
        top_k_values, top_k_indices = tf.nn.top_k(predict, k=k)  # [batch_size, k]
        
        # 2. 检查Top-K中是否包含正确标签
        correct = tf.equal(top_k_indices, true_label_expanded)  # [batch_size, k]
        
        # 3. 每个样本只要Top-K中有一个正确即算命中
        correct_any = tf.reduce_any(correct, axis=1)  # [batch_size]

        # 4. 计算加权召回率（只考虑有效样本）
        recall_at_k_value = tf.reduce_sum(tf.cast(correct_any, tf.float32) * loss_mask) / (tf.reduce_sum(loss_mask) + 0.001)

        # 5. 记录到TensorBoard
        tf.summary.scalar("top_k/{}_{}".format(name, k), recall_at_k_value)

# 注释掉的备用recall_at_k实现
# def recall_at_k(predictions, top_k=[5, 15], indicator=None, name="default"):
#     """
#     备用的recall_at_k实现
#
#     这个版本假设标签就是样本的索引（自监督学习场景）
#     """
#     max_k = max(top_k)
# 
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

def mlp(name, net, hidden_units, output_unit=None, activation=tf.nn.relu):
    """
    多层感知机（MLP）实现
    
    构建标准的全连接神经网络，支持多个隐藏层和自定义激活函数
    
    Args:
        name: MLP名称，用于变量作用域
        net: 输入张量，形状为[batch_size, input_dim]
        hidden_units: 隐藏层单元数列表，如[128, 64]表示两个隐藏层
        output_unit: 输出层单元数，如果为None则不添加输出层
        activation: 激活函数，默认为ReLU
        
    Returns:
        net: 输出张量，形状取决于最后一层的单元数
    """
    scope = name + '_mlp'
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # 构建隐藏层
        for i, k in enumerate(hidden_units):
            layer_name = scope + '_{}'.format(i)
            net = tf.layers.dense(net, k, activation=activation, name=layer_name)
        
        # 可选的输出层（通常不使用激活函数）
        if output_unit != None:
            net = tf.layers.dense(net, output_unit, activation=None, name=scope + '_final')
    return net

def sampled_softmax_loss(label, left_emb, right_emb, t=0.05, logQ=None, name="Default"):
    """
    采样softmax损失函数
    
    用于对比学习或双塔模型训练。计算批次内的成对相似度，
    将同一批次的其他样本作为负样本，实现高效的对比学习。
    
    Args:
        label: 样本权重，形状为[batch_size, 1]，用于加权损失
        left_emb: 左塔嵌入，形状为[batch_size, embedding_dim]
        right_emb: 右塔嵌入，形状为[batch_size, embedding_dim]
        t: 温度参数，用于控制softmax的平滑度，越小越sharp
        logQ: 负采样校正项（可选），用于处理采样偏差
        name: 损失名称，用于TensorBoard显示
    Returns:
        loss: 加权损失值
        cos_mat: 余弦相似度矩阵，可用于进一步分析
    """
    # 1. 计算余弦相似度矩阵
    cos_mat = tf.matmul(left_emb, right_emb, transpose_b=True)  # [batch_size, batch_size]
    
    # 2. 应用温度缩放和可选的负采样校正
    if logQ is not None:
        # 使用负采样校正，减去采样概率的对数
        sim_mat = cos_mat / t - tf.reshape(logQ, [1, -1])
    else:
        # 标准温度缩放
        sim_mat = cos_mat / t
    
    # 3. 构建标签：对角线为1（正样本），其他为0（负样本）
    bz = tf.shape(left_emb)[0]
    fake_label = tf.eye(bz)  # [batch_size, batch_size]
    
    # 4. 计算softmax交叉熵损失
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=sim_mat, labels=fake_label)
    loss = tf.reshape(loss, [-1, 1])  # [batch_size, 1]
    
    # 5. 记录正负样本相似度统计
    with tf.variable_scope("sampled_softmax_{}".format(name), reuse=tf.AUTO_REUSE) as scope:
        num = tf.cast(bz, dtype=tf.float32)
        
        # 正样本平均余弦相似度（对角线元素）
        tf.summary.scalar('mean_pos_cosine', 
                         tf.reduce_sum(fake_label * cos_mat) / num)
        
        # 负样本平均余弦相似度（非对角线元素）
        tf.summary.scalar('mean_neg_cosine',
                         tf.reduce_sum((cos_mat - fake_label * cos_mat)) / (num*num-num))
    
    # 6. 返回加权损失和相似度矩阵
    return tf.reduce_sum(loss*label), cos_mat


def masked_trimmed_mean(x, mask, trim_ratio=0.05, name=None):
    """
    x:    [B] float32/float64
    mask: [B] float32 0/1
    trim_ratio: 去掉两端比例（0.05 => 去掉最小5%和最大5%）
    return: 标量，截尾后的均值
    """
    with tf.name_scope(name or "masked_trimmed_mean"):
        x = tf.cast(x, tf.float32)
        mask = tf.cast(mask, tf.float32)

        # 取出有效样本（mask=1）的值：用一个很大的正/负数把无效位置排除掉
        # 为了后续 top_k，构造两个版本：
        #   x_for_small: 计算“最大(负x)”以拿到最小值的topk
        #   x_for_large: 直接 topk 拿到最大值topk
        very_neg = tf.fill(tf.shape(x), tf.constant(-1e30, tf.float32))
        very_pos = tf.fill(tf.shape(x), tf.constant( 1e30, tf.float32))

        # 统计有效样本数 n
        n = tf.cast(tf.reduce_sum(mask), tf.int32)  # [1]
        n_f = tf.cast(n, tf.float32)

        # 需要剔除的个数 k = floor(n * trim_ratio)
        k = tf.cast(tf.floor(n_f * trim_ratio), tf.int32)

        # 剩余个数 m = n - 2k，防止过小（至少1）
        m = tf.maximum(n - 2 * k, 1)

        # 如果 n 很小（比如 <20），k 可能为 0，这时就是普通均值
        # 先构造“只包含有效值”的向量，但不改变shape：无效的填极小/极大保证不会进入topk
        x_keep_large = tf.where(mask > 0, x, very_neg)  # 无效变很小，选最大时不会被选到
        x_keep_small = tf.where(mask > 0, x, very_pos)  # 无效变很大，选最小时不会被选到

        # 去掉最大 k：保留其余 => 取 top (n-k) 的最大（相当于丢掉最大的k个）
        # 注意 top_k 的 k 必须 <= B，所以这里用 B 做上限也没问题，因为无效被 very_neg 排除
        keep_after_drop_max, _ = tf.nn.top_k(x_keep_large, k=tf.maximum(n - k, 1), sorted=False)  # [n-k]

        # 在 “已去掉最大k”的集合里，再去掉最小 k：
        # 等价于：从 keep_after_drop_max 里取最小 (m) 个
        # 用 top_k(-x) 取最大 m 个，再取负回来
        neg_vals = -keep_after_drop_max
        smallest_m_neg, _ = tf.nn.top_k(neg_vals, k=m, sorted=False)  # 最大m个neg => 最小m个x
        trimmed_vals = -smallest_m_neg  # [m]

        return tf.reduce_mean(trimmed_vals)