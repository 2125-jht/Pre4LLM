import tensorflow as tf
import sys
from feature_attr_extract import *
from modules_ import *

user_common_fea_names = [
    "user_id"
]

user_colossus_fea_names = [
    "truncate_user_colossus_pid_list",
    "truncate_user_colossus_aid_list",
    "truncate_user_colossus_channel_list"
]

user_action_fea_names = {
    "long_view": ["user_profile_v1_play18s_pid_list", "user_profile_v1_play18s_aid_list"]
}

photo_common_fea_names = [
    "photo_id",
    "photo_author_id"
]

photo_quality_fea_names = [
    "photo_author_id_v2",
    "photo_author_fans_count",
    "photo_author_fans_count_2",
    "photo_author_upload_count",
    "photo_author_upload_count_2",
    "photo_author_click_count",
    "photo_author_click_count_2",
    "photo_author_like_count",
    "photo_author_like_count_2",
    "photo_author_follow_count",
    "photo_author_follow_count_2",
    "photo_author_long_view_count",
    "photo_author_long_view_count_2",
    "photo_author_emp_ctr",
    "photo_author_emp_ltr",
    "photo_author_emp_wtr",
    "photo_author_emp_lvtr",
    "photo_author_emp_svtr",
    "photo_author_emp_watch_time",
    "photo_mmu_embedding"
]

class MultiInterestModel(object):
    def __init__(self, feature_emb_dict, feature_emb_size_dict, truncate_dense_feature, args, history_size=768, selected_size=64,
    num_interest=4, add_pos=True, dim=64, print_ops=[], transformer_num_layer=3, vocab_sizes=[8192, 8192, 8192]):
        self._feature_emb_dict = feature_emb_dict
        self._feature_emb_size_dict = feature_emb_size_dict
        self._truncate_dense_feature = truncate_dense_feature
        self._history_size = history_size
        self._selected_size = selected_size
        self._num_interest = num_interest
        self._add_pos = add_pos
        self._dim = dim
        self._batch_id = None
        self._aux_tensor = []
        self._args = args
        self._print_ops = print_ops
        self._transformer_num_layer = transformer_num_layer
        
        self._vocab_sizes = vocab_sizes  # 三个语义层级的词汇表大小
        self._total_vocab_size = sum(self._vocab_sizes)  # 总词汇表大小
        
        self._embedding = tf.get_variable(
            shape=[self._total_vocab_size+1, dim], 
            name='embedding',
            initializer=tf.random_uniform_initializer(minval=-1.0/dim, maxval=1.0/dim), 
            trainable=True
        )

    # sum pooling like (uId + dId) ; (pId + aId)
    def _pooling(self, embeddings, item_num, name):
        embeddings = [tf.reshape(embed, [-1, item_num, self._dim]) for embed in embeddings]
        # tf.add_n([embed for embed in embeddings])
        return tf.add_n(embeddings)
        # return tf.reduce_sum(tf.concat(embeddings, axis=1), axis=1)
        
    def _select_feature(self, input_feature, indices):
        # input_feature [batch_size, history_size, dim], indices [batch_size, selected_size, dim]
        # output_feature [batch_size, selected_size, dim]
        batch_size = tf.shape(input_feature)[0]
        row = tf.range(batch_size) * self._history_size
        row = tf.reshape(row, [-1, 1])
        indices = indices + row
        
        input_feature_flatten = tf.reshape(input_feature, [-1, self._dim])
        indices_flatten = tf.reshape(indices, [-1])
        output_feature = tf.gather(input_feature_flatten, indices_flatten)
        output_feature = tf.reshape(output_feature, [batch_size, self._selected_size, self._dim])

        return output_feature
    
    def _single_score_module(self, score_feature_list):
        score_features = tf.concat(score_feature_list, axis=-1)
        score_features = tf.reshape(score_features, [-1, self._history_size, self._dim * len(score_feature_list)])
        score_mlp = MLP(64, 1, tf.nn.relu, tf.nn.sigmoid)
        score = score_mlp(score_features)
        return score

    # def _multi_interest_encode(self):
    #     # [-1, 1, dim]
    #     user_parameters = [self._feature_emb_dict[name] for name in user_common_fea_names]
    #     # [b, dim]
    #     user_embedding = self._pooling(user_parameters, 1, "user_attrs")
        
    #     photo_parameters = [self._feature_emb_dict[name] for name in photo_common_fea_names]
    #     #[b, dim]
    #     photo_embedding = self._pooling(photo_parameters, 1, "photo_attrs")

    #     action_list_dict = {}
    #     for action, action_params in user_action_fea_names.items():
    #         # [b, h, dim]
    #         action_lengths = self._feature_emb_size_dict[action_params[0]]
    #         mask = tf.sequence_mask(action_lengths, maxlen=self._history_size, dtype=tf.float32)
    #         list_embedding = self._pooling([self._feature_emb_dict[i] for i in action_params], self._history_size, "user_history")
    #         action_list_dict[action] = (list_embedding, mask)
    #     interest_embedding, readout = self._transformer_with_special_token(action_list_dict, photo_embedding, user_embedding)
    #     return interest_embedding, readout, tf.squeeze(photo_embedding, axis=1)

    def _multi_interest_encode_v2(self, photo_sid, label, photo_semantic_id_int):
        # ------------------------------------------------------------------
        # ❶ 用户静态特征 → pooling
        user_parameters = [self._feature_emb_dict[name]               # 每个都是 [B, 1, D] 或 [B, D]
                        for name in user_common_fea_names]
        user_embedding = self._pooling(user_parameters, 1, "user_attrs")   # [B, D]
        batch_size = tf.shape(user_embedding)[0]                           # ()  标量

        # ------------------------------------------------------------------
        # ❷ 当前候选内容静态特征 → pooling
        photo_parameters = [self._feature_emb_dict[name]              # 每个都是 [B, 1, D] 或 [B, D]
                            for name in photo_common_fea_names]
        photo_embedding = self._pooling(photo_parameters, 1, "photo_attrs")  # [B, D]

        # ------------------------------------------------------------------
        # ❸ 用户历史交互序列（稀疏 id Embedding）
        colossus_pid     = self._feature_emb_dict["truncate_user_colossus_pid_list"]      # [B, H, D]
        colossus_aid     = self._feature_emb_dict["truncate_user_colossus_aid_list"]      # [B, H, D]
        colossus_channel = self._feature_emb_dict["truncate_user_colossus_channel_list"]  # [B, H, D]

        # ❹ 用户历史交互序列（数值特征）
        truncate_colossus_play_time = tf.cast(self._truncate_dense_feature
                                            ["truncate_colossus_play_time_list"], tf.int32)  # [B, H]
        truncate_colossus_duration  = tf.cast(self._truncate_dense_feature
                                            ["truncate_colossus_duration_list"],  tf.int32)  # [B, H]
        truncate_colossus_label     = tf.cast(self._truncate_dense_feature
                                            ["truncate_colossus_label_list"],     tf.int32)  # [B, H]
        completion = tf.cast(tf.cast(truncate_colossus_play_time, tf.float32) / 
                             (tf.cast(truncate_colossus_duration, tf.float32) + 1e-5) * 10, tf.int32)     # [B, H]

        # ------------------------------------------------------------------
        # ❺ 行为类型 & 完播率离散化 → embedding
        interact_feature   = label_encoding(truncate_colossus_label)                 # [B, H, D]
        completion_feature = play_time_encoding(truncate_colossus_play_time,
                                                truncate_colossus_duration)          # [B, H, D]

        # ------------------------------------------------------------------
        # ❻ 打分网络（逐条历史→1 分数）
        score_feature_list = [colossus_channel, interact_feature, completion_feature] # 3×[B, H, D]
        score_unsqueezed   = self._single_score_module(score_feature_list)            # [B, H, 1]
        score              = tf.squeeze(score_unsqueezed, axis=2)                    # [B, H]
        print_tensor("score", score)

        # ------------------------------------------------------------------
        # ❼ 历史内容拼接特征
        colossus_photo_feature = self._pooling([colossus_pid, colossus_aid],
                                            self._history_size, "photo_feature")   # [B, H, D]

        # ❽ Gumbel-Top-k 选择 S 条历史
        indices = gumbel_top_k(score, self._selected_size, 1.0)   # [B, S]
        print_tensor("indices", indices)

        # ❾ 打分与特征相乘（让 score 可反传梯度）
        colossus_photo_feature = score_unsqueezed * colossus_photo_feature            # [B, H, D]

        # ❿ 根据 indices 取出被选中的历史条目
        selected_photo_feature = self._select_feature(colossus_photo_feature,
                                                    indices)                        # [B, S, D]

        # ------------------------------------------------------------------
        # ⓫ Transformer 编码，追加 “特殊 token” 表示用户
        encoder_output = self._transformer_with_special_token(
                            selected_photo_feature,          # [B, S, D]
                            None,
                            user_embedding)                  

        encoder_output_sim = tf.reshape(encoder_output, [batch_size, -1])             # [B, (S+1)*D]
        print_tensor("encoder_output_sim", calc_sim_cos(encoder_output_sim))

        # ------------------------------------------------------------------
        # ⓬ 最后一位（special token）即用户顶层兴趣表示
        user_top_embedding = tf.squeeze(encoder_output[:, -1:, ], axis=1)             # [B, D]

        # photo_embedding 已在 ❷ 得到                                          # [B, D]

        return user_top_embedding, tf.squeeze(photo_embedding, axis=1)

        

    def _transformer_with_special_token(self, item_list_emb, item_emb, user_emb):
        # input: item_list_emb
        batch_size = tf.shape(user_emb)[0]
        print("item_list_emb", item_list_emb)
        with tf.variable_scope("add_pos", reuse=tf.AUTO_REUSE) as scope:
            if self._add_pos:
                position_embedding = tf.get_variable(shape=[1, self._selected_size, self._dim], name='position_embedding')
                item_list_add_pos = item_list_emb + tf.tile(position_embedding, [batch_size, 1, 1])
            else:
                item_list_add_pos = item_list_emb

            # [bs, num_interest, dim]
        # special_token = tf.tile(tf.get_variable(shape=[1, self._num_interest, self._dim], name='special_token'), [batch_size, 1, 1])
        # special_token = tf.get_variable(shape=item_emb.shape, name='special_token')
        # transformer_input = tf.concat([item_list_add_pos, special_token], axis=1)
        transformer_input = item_list_add_pos
        # mask_ones = tf.ones([batch_size, self._selected_size])
        # mask_zeros = tf.zeros([batch_size, self._num_interest])
        # mask = tf.concat([mask_ones, mask_zeros], axis=1)
        start_token_indice = tf.tile(tf.constant(self._total_vocab_size, shape=(1, 1), dtype=tf.int32), [batch_size, 1])
        token_emb = tf.nn.embedding_lookup(self._embedding, start_token_indice) #[batch, 1, dim]
        transformer_input = tf.concat([transformer_input, token_emb], axis=1)
        
        encoder_output = transformer_encoder(seq_input_embeddings=transformer_input,
                                                    num_layer=self._transformer_num_layer, 
                                                    num_units=64, 
                                                    dropout_rate=0.1, 
                                                    num_heads=8, 
                                                    mask=None, 
                                                    training=True)
        return encoder_output
        
        # interest_emb = transformer_output[:, -4:]
        # get_interest_similarity("user_interest_without_uid_sim", interest_emb)
        # interest_emb = interest_emb + user_emb
        # get_interest_similarity("user_interest_sim", interest_emb)
        # atten = tf.matmul(interest_emb, item_emb, transpose_b=True)
        # # atten = [bs, total_heads]
        # atten = tf.nn.softmax(tf.pow(tf.reshape(atten, [-1, self._num_interest]), 1))
        # max_interest = tf.argmax(atten, axis=1, output_type=tf.int32)
        # for i in range(self._num_interest):
        #     tf.summary.scalar('interest/active_{}'.format(i), tf.reduce_mean(tf.cast(tf.equal(max_interest, i), tf.float32)))
        # readout = tf.gather(
        #     tf.reshape(interest_emb, [-1, self._dim]), tf.argmax(atten, axis=1, output_type=tf.int32) + tf.range(tf.shape(item_emb)[0]) * self._num_interest)
        # # readout = tf.reshape(readout, [-1, 1, self._dim])
        # readout = tf.reshape(readout, [-1, self._dim])
        # return interest_emb, readout


    # def _score_module(self, user_emb, context_emb, photo_info):
    #     hidden_dim = 64

    #     cross_feature = context_emb * photo_info
    #     input_feature = tf.add_n([context_emb, photo_info, cross_feature])

    #     gate_nu_one = Gate_NU(hidden_dim, hidden_dim)
    #     uid_input_one = gate_nu(user_emb)
    #     gate_nu_two = Gate_NU(hidden_dim, hidden_dim)
    #     uid_input_two = gate_nu(user_emb)

    #     cross_mlp_one = MLP(hidden_dim, hidden_dim, tf.nn.relu)
    #     cp_mid = cross_mlp_one(uid_input_one * cross_feature)
    #     cross_mlp_two = MLP(hidden_dim, hidden_dim, tf.nn.relu)
    #     cp_output = cross_mlp_two(uid_input_two * cp_mid)

    #     photo_mlp = MLP(hidden_dim, hidden_dim, tf.nn.relu)
    #     photo_output = photo_mlp(photo_info)

    #     final_input = cp_output * photo_output
    #     score_mlp = MLP(hidden_dim, 1, tf.nn.relu, tf.nn.sigmoid)
    #     score = score_mlp(final_input)

    #     return score


    # def _photo_score_module(self):
    #     photo_parameters = [self._feature_emb_dict[name] for name in photo_quality_fea_names]
    #     photo_input = tf.concat(photo_parameters, axis=1)

    #     photo_quality_emb = mlp("photo_quality_tower", photo_input, [256], 64)
    #     quality_score = tf.sigmoid(tf.reduce_sum(photo_quality_emb, axis=1, keepdims=True))
    #     print_tensor("qualtiy/quality_score", quality_score)
    #     print_tensor("qualtiy/photo_mmu_emebdding", self._feature_emb_dict["photo_mmu_embedding"])
    #     print_tensor("qualtiy/photo_mmu_embedding_none", tf.reduce_sum(tf.abs(self._feature_emb_dict["photo_mmu_embedding"]), axis=-1))

    #     photo_cover_emb = mlp("photo_cover_tower", photo_input, [256], 64)
    #     cover_score = tf.sigmoid(tf.reduce_sum(photo_cover_emb, axis=1, keepdims=True))
    #     print_tensor("qualtiy/cover_score", cover_score)
    #     return quality_score, cover_score
    
    
    # def model(self, photo_sid, label, photo_semantic_id_int):
    #     """
    #     主训练模型前向传播
        
    #     Args:
    #         photo_sid: 视频语义ID序列，shape=[batch_size, seq_len]
    #         label: 真实标签，shape=[batch_size, 3]，对应三个语义层级
    #         photo_semantic_id_int: 视频语义ID整数序列，用于计算loss mask
            
    #     Returns:
    #         loss: 训练损失值
    #     """
        
    #     # === 1. 用户静态特征处理 ===
    #     # 拼接所有用户静态特征
    #     user_static_fea = tf.concat([self._feature_emb_dict[fea] for fea in user_static_fea_names], axis=1)
    #     # 通过MLP将静态特征映射到指定维度
    #     user_static_emb = mlp('user_static_emb', user_static_fea, [2*self._dim], self._dim, activation=tf.nn.leaky_relu)
    #     batch_size = tf.shape(user_static_emb)[0]
    #     # 调整形状为[batch_size, 1, dim]，作为序列的一个元素
    #     user_static_emb = tf.reshape(user_static_emb, [batch_size, 1, self._dim])

    #     # === 2. 用户点击行为特征处理 ===
    #     # 拼接用户点击特征（视频ID和作者ID）
    #     user_click_fea = tf.concat([self._feature_emb_dict[fea] for fea in user_click_fea_names], axis=2)
    #     # 通过MLP处理点击特征
    #     user_click_emb = mlp('user_click_emb', user_click_fea, [4*self._dim], self._dim, activation=tf.nn.leaky_relu)
        
    #     # === 3. 构建编码器输入 ===
    #     # 将静态特征和点击行为特征拼接作为编码器输入
    #     encoder_input = tf.concat([user_static_emb, user_click_emb], axis=1)
        
    #     # need_pe = True
        
    #     # if need_pe:
    #     # # 获取位置编码
    #     #     seq_len = tf.shape(encoder_input)[1]  # 获取序列长度
    #     #     position_encoding = get_encoder_position_encoding(seq_len, self._dim)  # 获取位置编码
    #     #     # 添加位置编码到输入嵌入
    #     #     encoder_input += position_encoding  # 添加位置编码
        
    #     # === 4. 构建解码器输入 ===
    #     # 添加起始token（使用总词汇表大小作为特殊标记）
    #     start_token_indice = tf.tile(tf.constant(self._total_vocab_size, shape=(1, 1), dtype=tf.int32), [batch_size, 1])
    #     # 将起始token与视频语义ID拼接
    #     photo_with_start_token = tf.concat([start_token_indice, photo_sid], axis=1)
    #     # 查找嵌入向量
    #     decoder_input = tf.nn.embedding_lookup(self._embedding, photo_with_start_token)

    #     # if need_pe:
    #     #     # 获取解码器位置编码
    #     #     decoder_seq_len = tf.shape(decoder_input)[1]  # 获取解码器序列长度
    #     #     decoder_position_encoding = get_decoder_position_encoding(decoder_seq_len, self._dim)
    #     #     # 添加位置编码到解码器输入
    #     #     decoder_input += decoder_position_encoding
        
    #     # === 5. Transformer编码器 ===
    #     # 使用4层Transformer编码器处理用户特征
    #     encoder_model = EncoderModel(num_layers=4, dim=self._dim, num_heads=8, dropout_rate=0.1, hidden_dim=self._dim*2)
    #     encoder_output = encoder_model.forward(encoder_input, training=True) # [batch_size, seq_len, dim]
        
    #     # 计算编码器输出的余弦相似度（用于调试）
    #     encoder_output_sim = tf.reshape(encoder_output, [batch_size, -1])
    #     print_tensor("encoder_output_sim", calc_sim_cos(encoder_output_sim))
        
    #     # === 6. Transformer解码器 ===
    #     # 使用4层Transformer解码器生成序列表示
    #     decoder_model = DecoderModel(num_layers=4, dim=self._dim, num_heads=8, dropout_rate=0.1, hidden_dim=self._dim*2)
    #     decoder_output = decoder_model.forward(decoder_input, encoder_output, training=True) # [batch_size, seq_len, dim]
        
    #     # 计算解码器各步输出的余弦相似度（用于调试）
    #     for i in range(len(self._vocab_sizes)):
    #         similarity = calc_sim_cos(decoder_output[:, i, :])
    #         print_tensor('decoder_sim/decoder_output_%d' % i, similarity)

    #     # === 7. 损失计算 ===
    #     losses = []
    #     # 创建损失掩码，只对有效的语义ID计算损失
    #     loss_mask = tf.where(
    #         photo_semantic_id_int > 0,  
    #         tf.ones_like(photo_semantic_id_int, dtype=tf.float32), 
    #         tf.zeros_like(photo_semantic_id_int, dtype=tf.float32)
    #     )
    #     loss_mask = tf.reshape(loss_mask, [-1])

    #     # 对每个语义层级分别计算损失
    #     for step in range(len(self._vocab_sizes)):
    #         with tf.variable_scope('proj_%d' % step):
    #             # 使用MLP将解码器输出映射到对应词汇表大小的logits
    #             # pred_logit = mlp("pred", decoder_output[:, step, :], [self._vocab_sizes[step]], self._vocab_sizes[step], activation=tf.nn.leaky_relu)
    #             pred_logit = tf.layers.dense(decoder_output[:, step, :], self._vocab_sizes[step], name='pred') # [batch_size, vocab_size]
    #             print_tensor("logits/pred_logit_%d" % step, pred_logit)
    #             # 转换标签为one-hot编码
    #             one_hot_labels = tf.one_hot(label[:, step], self._vocab_sizes[step])
    #             # 计算交叉熵损失，使用温度缩放(temperature=2.0)
    #             loss_i = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels, logits=pred_logit)
    #             losses.append(loss_i)
                
    #             # 计算各种recall指标
    #             recall_at_k(pred_logit, label[:, step], loss_mask, self._print_ops, top_k=[1, 16, 128], name="predict_recall_%d" % step)
    #     print_tensor("loss_mask", loss_mask)
    #     # 计算加权平均损失
    #     loss = tf.reduce_sum((losses[0] + losses[1] + losses[2]) * loss_mask) / tf.reduce_sum(loss_mask + 0.001)
    #     return loss