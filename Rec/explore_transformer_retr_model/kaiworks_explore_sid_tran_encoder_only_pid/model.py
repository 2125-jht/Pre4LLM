# -*- coding: utf-8 -*-
"""
多兴趣推荐模型 - 基于Transformer的编码器-解码器架构
用于用户兴趣建模和内容推荐

主要功能：
1. 基于用户静态特征和点击行为建模用户兴趣
2. 使用Transformer编码器提取用户兴趣表示
3. 使用Transformer解码器生成推荐sid
4. 支持束搜索(Beam Search)进行sid生成
"""
import tensorflow as tf
import sys
from feature_attr_extract import *
from modulesV2 import *
from modules_ import *

# 用户静态特征名称列表
# 包含用户的基本属性信息
user_static_fea_names = [
    "user_id",          # 用户ID
    "user_gender",      # 用户性别
    "user_age_segment", # 用户年龄段
    "user_level"        # 用户等级
]

# 用户点击行为特征名称列表
# 包含用户的历史交互行为数据
user_click_fea_names = [
    "user_profile_v1_click_pid_list",  # 用户点击的视频ID列表
    "user_profile_v1_click_aid_list"   # 用户点击的作者ID列表
]

class MultiInterestModel(object):
    """
    多兴趣推荐模型类
    
    该模型使用Transformer架构，通过编码器-解码器结构：
    1. 编码器：将用户特征和行为序列编码为多个兴趣表示
    2. 解码器：基于兴趣表示生成推荐的语义ID序列
    """
    
    def __init__(self, feature_emb_dict, feature_emb_size_dict, dim=512, vocab_sizes=[8192, 8192, 8192], print_ops=None):
        """
        初始化多兴趣模型
        
        Args:
            feature_emb_dict: 特征嵌入字典，存储各特征的嵌入向量
            feature_emb_size_dict: 特征嵌入维度字典
            dim: 模型隐藏层维度，默认64
            vocab_sizes: 各个词汇表大小的列表，对应不同语义层级，默认[8192, 8192, 8192]
            print_ops: 用于调试的打印操作列表
        """
        self._feature_emb_dict = feature_emb_dict
        self._feature_emb_size_dict = feature_emb_size_dict
        self._print_ops = print_ops
        self._vocab_sizes = vocab_sizes  # 三个语义层级的词汇表大小
        self._total_vocab_size = sum(self._vocab_sizes)  # 总词汇表大小
        
        # 创建统一的嵌入矩阵，包含所有语义ID的嵌入向量
        # 使用均匀分布初始化，范围为[-1/dim, 1/dim]
        self._embedding = tf.get_variable(
            shape=[self._total_vocab_size+1, dim], 
            name='embedding',
            initializer=tf.random_uniform_initializer(minval=-1.0/dim, maxval=1.0/dim), 
            trainable=True
        )
        # self._embedding = tf.get_variable(
        #     name="embedding",                       
        #     shape=[self._total_vocab_size + 1, dim],
        #     initializer=tf.glorot_uniform_initializer(),           # ★ Xavier Uniform ★
        #     trainable=True
        # )
        
        self._dim = dim

    def model(self, photo_sid, label, photo_semantic_id_int):
        """
        主训练模型前向传播
        
        Args:
            photo_sid: 视频语义ID序列，shape=[batch_size, seq_len]
            label: 真实标签，shape=[batch_size, 3]，对应三个语义层级
            photo_semantic_id_int: 视频语义ID整数序列，用于计算loss mask
            
        Returns:
            loss: 训练损失值
        """
        
        # === 1. 用户静态特征处理 ===
        # 拼接所有用户静态特征
        user_static_fea = tf.concat([self._feature_emb_dict[fea] for fea in user_static_fea_names], axis=1)
        # 通过MLP将静态特征映射到指定维度
        user_static_emb = mlp('user_static_emb', user_static_fea, [2*self._dim], self._dim, activation=tf.nn.leaky_relu)
        batch_size = tf.shape(user_static_emb)[0]
        # 调整形状为[batch_size, 1, dim]，作为序列的一个元素
        user_static_emb = tf.reshape(user_static_emb, [batch_size, 1, self._dim])

        # === 2. 用户点击行为特征处理 ===
        # 拼接用户点击特征（视频ID和作者ID）
        user_click_fea = tf.concat([self._feature_emb_dict[fea] for fea in user_click_fea_names], axis=2)
        # 通过MLP处理点击特征
        user_click_emb = mlp('user_click_emb', user_click_fea, [4*self._dim], self._dim, activation=tf.nn.leaky_relu)
        
        # === 3. 构建编码器输入 ===
        # 将静态特征和点击行为特征拼接作为编码器输入
        encoder_input = tf.concat([user_static_emb, user_click_emb], axis=1)
        
        # need_pe = True
        
        # if need_pe:
        # # 获取位置编码
        #     seq_len = tf.shape(encoder_input)[1]  # 获取序列长度
        #     position_encoding = get_encoder_position_encoding(seq_len, self._dim)  # 获取位置编码
        #     # 添加位置编码到输入嵌入
        #     encoder_input += position_encoding  # 添加位置编码
        
        # === 4. 构建解码器输入 ===
        # 添加起始token（使用总词汇表大小作为特殊标记）
        start_token_indice = tf.tile(tf.constant(self._total_vocab_size, shape=(1, 1), dtype=tf.int32), [batch_size, 1])
        # 将起始token与视频语义ID拼接
        photo_with_start_token = tf.concat([start_token_indice, photo_sid], axis=1)
        # 查找嵌入向量
        decoder_input = tf.nn.embedding_lookup(self._embedding, photo_with_start_token)

        # if need_pe:
        #     # 获取解码器位置编码
        #     decoder_seq_len = tf.shape(decoder_input)[1]  # 获取解码器序列长度
        #     decoder_position_encoding = get_decoder_position_encoding(decoder_seq_len, self._dim)
        #     # 添加位置编码到解码器输入
        #     decoder_input += decoder_position_encoding
        
        # === 5. Transformer编码器 ===
        # 使用4层Transformer编码器处理用户特征
        encoder_output = transformer_encoder_layer(encoder_input, 4, dim=self._dim)  # [batch_size, seq_len, dim]
        # encoder_output = hstu_encoder_layer(encoder_input, 4, dim=self._dim)  # [batch_size, seq_len, dim]
        
        # 计算编码器输出的余弦相似度（用于调试）
        encoder_output_sim = tf.reshape(encoder_output, [batch_size, -1])
        print_tensor("encoder_output_sim", calc_sim_cos(encoder_output_sim))
        
        # === 6. Transformer解码器 ===
        # 使用4层Transformer解码器生成序列表示
        decoder_output = transformer_decoder_layer(encoder_output, decoder_input, 4, dim=self._dim)
        # decoder_output = hstu_decoder_layer(encoder_output, decoder_input, 4, dim=self._dim)
        # 计算解码器各步输出的余弦相似度（用于调试）
        for i in range(len(self._vocab_sizes)):
            similarity = calc_sim_cos(decoder_output[:, i, :])
            print_tensor('decoder_sim/decoder_output_%d' % i, similarity)

        # === 7. 损失计算 ===
        losses = []
        # 创建损失掩码，只对有效的语义ID计算损失
        loss_mask = tf.where(
            photo_semantic_id_int > 0,  
            tf.ones_like(photo_semantic_id_int, dtype=tf.float32), 
            tf.zeros_like(photo_semantic_id_int, dtype=tf.float32)
        )
        loss_mask = tf.reshape(loss_mask, [-1])

        # 对每个语义层级分别计算损失
        for step in range(len(self._vocab_sizes)):
            with tf.variable_scope('proj_%d' % step):
                # 使用MLP将解码器输出映射到对应词汇表大小的logits
                # pred_logit = mlp("pred", decoder_output[:, step, :], [self._vocab_sizes[step]], self._vocab_sizes[step], activation=tf.nn.leaky_relu)
                pred_logit = tf.layers.dense(decoder_output[:, step, :], self._vocab_sizes[step], name='pred') # [batch_size, vocab_size]
                print_tensor("logits/pred_logit_%d" % step, pred_logit)
                # 转换标签为one-hot编码
                one_hot_labels = tf.one_hot(label[:, step], self._vocab_sizes[step])
                # 计算交叉熵损失，使用温度缩放(temperature=2.0)
                loss_i = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels, logits=pred_logit)
                losses.append(loss_i)
                
                # 计算各种recall指标
                recall_at_k(pred_logit, label[:, step], loss_mask, self._print_ops, top_k=[1, 16, 128], name="predict_recall_%d" % step)
        print_tensor("loss_mask", loss_mask)
        # 计算加权平均损失
        loss = tf.reduce_sum((losses[0] + losses[1] + losses[2]) * loss_mask) / tf.reduce_sum(loss_mask + 0.001)
        return loss
    
    def beam_search(self, beam_size=512):
        """
        束搜索推理方法 - 版本1
        
        使用固定beam_size进行序列生成，适用于推理阶段
        
        Args:
            beam_size: 束搜索的beam大小，默认512
            
        Returns:
            selected_sequences: 生成的序列，shape=[batch_size, beam_size, seq_len]
        """
        
        # === 1. 用户特征处理（与训练时相同） ===
        # 拼接所有用户静态特征
        user_static_fea = tf.concat([self._feature_emb_dict[fea] for fea in user_static_fea_names], axis=1)
        # 通过MLP将静态特征映射到指定维度
        user_static_emb = mlp('user_static_emb', user_static_fea, [2*self._dim], self._dim, activation=tf.nn.leaky_relu)
        batch_size = tf.shape(user_static_emb)[0]
        # 调整形状为[batch_size, 1, dim]，作为序列的一个元素
        user_static_emb = tf.reshape(user_static_emb, [batch_size, 1, self._dim])
        # 拼接用户点击特征（视频ID和作者ID）
        user_click_fea = tf.concat([self._feature_emb_dict[fea] for fea in user_click_fea_names], axis=2)
        # 通过MLP处理点击特征
        user_click_emb = mlp('user_click_emb', user_click_fea, [4*self._dim], self._dim, activation=tf.nn.leaky_relu)  
              
        encoder_input = tf.concat([user_static_emb, user_click_emb], axis=1)

        # === 2. 编码器处理 ===
        encoder_output = transformer_encoder_layer(encoder_input, 4, dim=self._dim)
        # encoder_output = hstu_encoder_layer(encoder_input, 4, dim=self._dim)  # [batch_size, seq_len, dim]
        batch_size = tf.shape(encoder_output)[0]

        # === 3. 初始化束搜索状态 ===
        scores = tf.zeros([batch_size, beam_size])  # 累积得分
        # 初始序列：所有beam都以起始token开始
        selected_sequences = tf.tile(tf.constant(self._total_vocab_size, shape=[1, 1, 1]), [batch_size, beam_size, 1])
        # === 4. 逐步生成序列 ===
        for step in range(len(self._vocab_sizes)):
            seq_len = tf.shape(selected_sequences)[2]

            # 获取当前序列的嵌入表示
            decoder_input = tf.nn.embedding_lookup(self._embedding, selected_sequences)
            decoder_input_beam_size = tf.shape(decoder_input)[1]
            
            # 扩展编码器输出以匹配beam维度
            encoder_output_expand = tf.tile(tf.expand_dims(encoder_output, axis=1), [1, decoder_input_beam_size, 1, 1])
            
            # 解码器前向传播
            decoder_output = transformer_decoder_layer(encoder_output_expand, decoder_input, 4, dim=self._dim)
            # decoder_output = hstu_decoder_layer(encoder_output_expand, decoder_input, 4, dim=self._dim)
            self._print_ops.append(tf.print("decoder_output_shape_%d" % step, tf.shape(decoder_output), summarize=-1, output_stream=sys.stdout))

            # 预测下一个token的概率分布
            with tf.variable_scope('proj_%d' % step, reuse=tf.AUTO_REUSE):
                # pred_logit = mlp("pred", decoder_output[:, :, step, :], [self._vocab_sizes[step]], self._vocab_sizes[step], activation=tf.nn.leaky_relu)
                pred_logit = tf.layers.dense(decoder_output[:, :, step, :], self._vocab_sizes[step], name='pred') # [batch_size, vocab_size]
            # 计算概率和对数概率
            next_token_probs = tf.nn.softmax(pred_logit / 2.0, axis=-1)  # [batch_size, beam_size, vocab_size]
            log_probs = tf.math.log(next_token_probs)

            # === 5. 束搜索逻辑 ===
            # 束搜索是一种贪婪的序列生成算法，在每一步保持top-k个最优候选序列
            
            # 步骤5.1: 计算所有可能的候选分数
            # scores形状: [batch_size, beam_size] - 当前每个beam的累积对数概率
            # log_probs形状: [batch_size, beam_size, vocab_size] - 下一个token的对数概率分布
            # expand_dims将scores扩展为[batch_size, beam_size, 1]，便于与log_probs广播相加
            candidate_scores = tf.expand_dims(scores, -1) + log_probs  # [batch_size, beam_size, vocab_size]
            
            # 步骤5.2: 将候选分数重塑为二维，便于后续top_k操作
            # 从[batch_size, beam_size, vocab_size]重塑为[batch_size, beam_size*vocab_size]
            # 这样每个batch有beam_size*vocab_size个候选分数
            candidate_scores = tf.reshape(candidate_scores, [batch_size, beam_size*self._vocab_sizes[step]])
            
            # 步骤5.3: 构建对应的候选序列
            # 5.3.1: 为当前序列添加新的维度，准备复制
            # selected_sequences形状: [batch_size, beam_size, current_seq_length]
            # 添加维度后: [batch_size, beam_size, 1, current_seq_length]
            candidate_sequences = tf.expand_dims(selected_sequences, axis=2)
            
            # 5.3.2: 为每个可能的下一个token复制当前序列
            # tile操作将每个序列复制vocab_size次，对应vocab_size个可能的下一个token
            # 结果形状: [batch_size, beam_size, vocab_size, current_seq_length]
            candidate_sequences = tf.tile(candidate_sequences, [1, 1, self._vocab_sizes[step], 1])
            
            # 5.3.3: 生成要添加的新token
            # tf.range(vocab_size)生成[0, 1, 2, ..., vocab_size-1]
            # 通过多次expand_dims将其变为4D张量: [1, 1, vocab_size, 1]
            add_token = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.range(self._vocab_sizes[step]), axis=1), axis=0), axis=0)
            
            # 5.3.4: 将新token张量扩展到与候选序列匹配的形状
            # tile操作复制到: [batch_size, beam_size, vocab_size, 1]
            add_token = tf.tile(add_token, [batch_size, beam_size, 1, 1])
            
            # 5.3.5: 将新token拼接到序列末尾
            # concat在最后一个维度（序列长度维度）上拼接
            # 结果形状: [batch_size, beam_size, vocab_size, current_seq_length + 1]
            candidate_sequences = tf.concat([candidate_sequences, add_token], axis=-1)
            
            # 5.3.6: 重塑候选序列以匹配候选分数的形状
            # 从[batch_size, beam_size, vocab_size, seq_length]重塑为[batch_size, beam_size*vocab_size, seq_length]
            # 现在候选分数和候选序列在第二个维度上都有beam_size*vocab_size个元素，一一对应
            candidate_sequences = tf.reshape(candidate_sequences, [batch_size, beam_size*self._vocab_sizes[step], -1])
            
            # 调试输出：打印候选序列的形状
            self._print_ops.append(tf.print("selected_sequences_shape_%d" % step, tf.shape(candidate_sequences), summarize=-1, output_stream=sys.stdout))
            # 步骤5.4: 从所有候选中选择top-k个最优候选
            # top_k操作返回最大的k个分数及其对应的索引
            # top_k_scores形状: [batch_size, beam_size] - 最优的beam_size个分数
            # top_k_indices形状: [batch_size, beam_size] - 对应的索引位置
            top_k_scores, top_k_indices = tf.math.top_k(candidate_scores, k=beam_size, sorted=True)
            # 步骤5.5: 构建gather操作所需的索引
            # gather_nd需要完整的坐标索引，包括batch维度和beam维度
            # 5.5.1: 生成batch索引
            # tf.range(batch_size)生成[0, 1, 2, ..., batch_size-1]
            # reshape为[-1, 1]得到[[0], [1], [2], ...]
            # tile复制beam_size次得到[[0, 0, ...], [1, 1, ...], ...]形状为[batch_size, beam_size]
            batch_indces = tf.tile(tf.reshape(tf.range(batch_size), [-1, 1]), [1, beam_size])
            
            # 5.5.2: 组合batch索引和beam索引
            # stack操作将batch_indices和top_k_indices沿着新的维度组合
            # 结果形状: [batch_size, beam_size, 2]
            # 每个元素[batch_idx, beam_idx]指定了candidate_sequences中的一个位置
            gather_indices = tf.stack([batch_indces, top_k_indices], axis=2)

            # 步骤5.6: 根据索引获取对应的最优序列
            # gather_nd根据gather_indices从candidate_sequences中提取对应的序列
            # 结果形状: [batch_size, beam_size, seq_length]
            new_sequence = tf.gather_nd(candidate_sequences, gather_indices)

            # 步骤5.7: 更新束搜索的状态，准备下一轮迭代
            # 5.7.1: 更新累积分数
            # 将top_k_scores重塑为标准的[batch_size, beam_size]形状
            scores = tf.reshape(top_k_scores, [batch_size, beam_size])
            
            # 5.7.2: 更新选中的序列
            # 将new_sequence重塑为标准的[batch_size, beam_size, seq_length]形状
            # 这些序列将作为下一步的输入序列
            selected_sequences = tf.reshape(new_sequence, [batch_size, beam_size, -1])
        return selected_sequences
    
    