import sys
import tensorflow as tf
from modules import *
from feature_attr_extract import user_fea_names,explore_profile_fea_names,photo_fea_names,source_fea_names,fountain_seq_pid_names,fountain_seq_aid_names

def dnn_layer(inputs, hidden_units, activation=tf.nn.relu, batch_normalization=False, training=True,
              dropout=None, last_layer_no_activation=False, last_layer_no_batch_norm=False,
              last_layer_no_dropout=False, scope="mlp", **kwargs):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        x = inputs
        for i, units in enumerate(hidden_units):
            if (i == len(hidden_units) - 1) and last_layer_no_activation:
                activation = None
            x = tf.layers.dense(x, units, activation, name="layer_{}".format(i), **kwargs)

            if batch_normalization is True and ((i < len(hidden_units) - 1) or not last_layer_no_batch_norm):
                # 训练阶段，要保证均值和方差的正确更新；预测阶段，则要保证所有参数与训练阶段的一致，其实主要就4个，训练阶段全局的gamma beta 均值 方差
                x = tf.layers.batch_normalization(x, training=training, name="{}_bn_{}".format(i))

            if dropout is not None and ((i < len(hidden_units) - 1) or not last_layer_no_dropout):
                if training:
                    x = tf.nn.dropout(x, rate=dropout, name="layer_dropout_{}".format(i))
                else: x = x
    return x

def din_attn(queries, keys, keys_length, scope_pre, training, hidden_units=[64,32,1], activation=tf.nn.relu, use_prelu=True):
    '''
        queries:     [B, H]    [batch_size,embedding_size]
        keys:        [B, T, H]   [batch_size,T,embedding_size]
        keys_length: [B]        [batch_size] 真实长度
        # T为历史行为序列长度
        return: B * 1 * H
    '''
    with tf.variable_scope(f"{scope_pre}_din_attn", reuse=tf.AUTO_REUSE):
        def prelu(_x):
            alphas = tf.get_variable('prelu_alpha', _x.get_shape()[-1],
                                initializer=tf.constant_initializer(0.0),
                                dtype=tf.float32)
            pos = tf.nn.relu(_x)
            neg = alphas * (_x - abs(_x)) * 0.5
            return pos + neg

        T = tf.shape(keys)[1]
        H = queries.get_shape().as_list()[-1]
        queries = tf.tile(queries, [1, T])
        queries = tf.reshape(queries, [-1, T, H])
        din_all = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1) # B*T*4H

        activation = prelu if use_prelu else activation
        din_output = dnn_layer(din_all, hidden_units=hidden_units,activation=activation, training=training,
                            last_layer_no_activation=True, last_layer_no_batch_norm=True) # B*T*1

        # 为了让outputs维度和keys的维度一致
        outputs = tf.reshape(din_output, [-1, 1, T]) # B*1*T
        
        key_masks = tf.sequence_mask(keys_length, T) # B*T
        key_masks = tf.expand_dims(key_masks,1) # B*1*T
        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(key_masks,outputs,paddings) # B * 1 * T

        # Scale（缩放）
        # outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)
        outputs = tf.nn.softmax(outputs) # B * 1 * T
        # Weighted Sum outputs=g(Vi,Va)   keys=Vi
        #这步为公式中的g(Vi*Va)*Vi
        outputs = tf.matmul(outputs,keys) # B * 1 * H 三维矩阵相乘，相乘发生在后两维，即 B * (( 1 * T ) * ( T * H ))

    return outputs

class FountainDeepLtrMultiTaskModel:
    def __init__(self, parameters_dict, label_value_dict, dense_value_dict, print_ops, list_size, candidates_size, dim=32, extra_param_dict= None, training=True):
        self._parameters_dict = parameters_dict
        self._label_value_dict = label_value_dict
        self._dense_value_dict = dense_value_dict
        self.list_size = list_size
        self.candidates_size = candidates_size
        self.dim = dim
        self.training = training
        self.position_embeddings = tf.get_variable(
            name='position_embeddings', 
            shape=[6, 32], 
            initializer=tf.random_normal_initializer()
        )
        # Create [sos] and [eos] embeddings
        self.sos_embedding = tf.get_variable(
            "sos_embedding", shape=[1, 32], initializer=tf.random_uniform_initializer()
        )
        self.eos_embedding = tf.get_variable(
            "eos_embedding", shape=[1, 32], initializer=tf.random_uniform_initializer()
        )
        self.pad_embedding = tf.get_variable(
            "pad_embedding", shape=[1, 32], initializer=tf.random_uniform_initializer()
        )
        self.print_ops = print_ops

    def _calc_point_reward(self):
        # 计算单点价值
        pctr = tf.cast(self._dense_value_dict["context_info__pctr"], dtype=tf.float32)
        pvtr = tf.cast(self._dense_value_dict["context_info__pvtr"], dtype=tf.float32)
        pltr = tf.cast(self._dense_value_dict["context_info__pltr"], dtype=tf.float32)
        plvtr = tf.cast(self._dense_value_dict["context_info__plvtr"], dtype=tf.float32)
        pwtr = tf.cast(self._dense_value_dict["context_info__pwtr"], dtype=tf.float32)
        psvtr = tf.cast(self._dense_value_dict["context_info__psvtr"], dtype=tf.float32)
        preward = 1+ pctr + pvtr * 2 + pltr + plvtr * 0.8 + pwtr * 0.5 + psvtr * 0.1
        return preward

    def _mlp_layer(self,
                  scope_name,
                  hidden_states: tf.Tensor,
                  hidden_units: list,
                  activation=tf.nn.relu) -> tf.Tensor:
        with tf.variable_scope(f"{scope_name}_mlp_layer", reuse=tf.AUTO_REUSE):
            for i, hidden_unit in enumerate(hidden_units):
                hidden_states = tf.layers.dense(hidden_states, hidden_unit, activation=activation, use_bias=True)
        return hidden_states

    def _fea_seq_attn(self, query, input_dicts, fea_names, list_dim, merge_type="concat"):
        # 计算序列特征attn, query shape: (?,list_len,d)
        rt_seq_mha_list = []
        for fea in fea_names:
            # keys shape: (?, keys_len, d)
            keys = input_dicts[fea] if self.training else input_dicts[fea][:, 0, :, :]
            # 这里假设最后一维全0是padding
            mask = tf.logical_not(tf.reduce_all(tf.equal(keys, 0.0), axis=-1)) # (?,T)
            mask = tf.cast(mask, dtype=tf.float32)
            print(f"{fea} mask shape: {mask.shape}, query shape: {query.shape}")
            mha_out = multi_head_attention(query, input_dicts[fea], input_dicts[fea], 2, dropout_rate=0.1, training=self.training,
                                            use_seq_mask=True, seq_mask=mask)
            rt_seq_mha_list.append(mha_out)
        if merge_type == "concat":
            rt_seq_mha = tf.concat(rt_seq_mha_list, axis=-1) # (?,list_len,d*n)
        elif merge_type == "add":
            rt_seq_mha = tf.add_n(rt_seq_mha_list) # (?,list_len,d)
        return rt_seq_mha

    def _get_shared_features(self, input_dicts, list_dim) -> tuple:
        with tf.variable_scope("share_bottom", reuse=tf.AUTO_REUSE):
            # print("user_id embeding shape", input_dicts['user_id'].shape)
            # user_embs = tf.concat([input_dicts[k] for k in input_dicts if k in user_fea_names], axis=-1)
            # user_embs = tf.tile(tf.expand_dims(user_embs, axis=1),[1,list_dim,1]) if self.training else user_embs
            # print("user_embs shape ", user_embs.shape) # train: (?, 30, 120), infer: (?, 120)
            photo_embs    = tf.concat([input_dicts[k] for k in input_dicts if k in photo_fea_names], axis=-1)
            print("photo_embs shape ", photo_embs.shape) # (?, 30, 380)
            source_embs    = tf.concat([input_dicts[k] for k in input_dicts if k in source_fea_names], axis=-1)
            source_embs = tf.tile(tf.expand_dims(source_embs, axis=1),[1,list_dim,1]) if self.training else source_embs
            print("source_embs shape ", source_embs.shape) # 
            # rt_seq_mha    = tf.concat([input_dicts[k] for k in input_dicts if k in explore_profile_fea_names], axis=-1)
            # rt_seq_mha  = tf.reduce_mean(rt_seq_mha, axis=1)
            # rt_seq_mha = tf.tile(tf.expand_dims(rt_seq_mha, axis=1),[1,list_dim,1]) if self.training else rt_seq_mha
            # print("rt_seq_mha shape ", rt_seq_mha.shape) # 
            # fountain_seq_pid_mha    = tf.concat([input_dicts[k] for k in input_dicts if k in fountain_seq_pid_names], axis=-1)
            # fountain_seq_pid_mha  = tf.reduce_mean(fountain_seq_pid_mha, axis=1)
            # fountain_seq_pid_mha = tf.tile(tf.expand_dims(fountain_seq_pid_mha, axis=1),[1,list_dim,1]) if self.training else fountain_seq_pid_mha
            # print("fountain_seq_pid_mha shape ", fountain_seq_pid_mha.shape) # 
            # fountain_seq_aid_mha    = tf.concat([input_dicts[k] for k in input_dicts if k in fountain_seq_aid_names], axis=-1)
            # fountain_seq_aid_mha  = tf.reduce_mean(fountain_seq_aid_mha, axis=1)
            # fountain_seq_aid_mha = tf.tile(tf.expand_dims(fountain_seq_aid_mha, axis=1),[1,list_dim,1]) if self.training else fountain_seq_aid_mha
            # print("fountain_seq_aid_mha shape ", fountain_seq_aid_mha.shape) # 
            # 计算序列attn
            # source_id = tf.tile(tf.expand_dims(input_dicts["context_source_pid"], axis=1), [1, list_dim, 1]) if self.training else input_dicts["context_source_pid"]
            # source_pid = tf.tile(tf.expand_dims(input_dicts["context_source_aid"], axis=1), [1, list_dim, 1]) if self.training else input_dicts["context_source_pid"]
            # id_query = source_id + input_dicts["photo_id"]
            # aid_query = source_pid + input_dicts["photo_author_id"]
            # rt_seq_mha = self._fea_seq_attn(id_query, input_dicts, explore_profile_fea_names, list_dim, merge_type="concat")
            # print("rt_seq_mha shape ", rt_seq_mha.shape) # train: (?, 30, dim), infer: (?, dim)
            # fountain_seq_pid_mha = self._fea_seq_attn(id_query, input_dicts, fountain_seq_pid_names, list_dim, merge_type="concat")
            # print("fountain_seq_pid_mha shape ", fountain_seq_pid_mha.shape) # train: (?, 30, dim), infer: (?, dim)
            # fountain_seq_aid_mha = self._fea_seq_attn(aid_query, input_dicts, fountain_seq_aid_names, list_dim, merge_type="concat")
            # print("fountain_seq_aid_mha shape ", fountain_seq_aid_mha.shape) # train: (?, 30, dim), infer: (?, dim)

            # common_embs = tf.concat([user_embs, source_embs, photo_embs, rt_seq_mha, fountain_seq_pid_mha, fountain_seq_aid_mha], axis=-1)
            common_embs = tf.concat([source_embs, photo_embs], axis=-1)
            if not self.training:
                '''
                    特征工程中每一条样本中 common attr 已经填充, 每个item特征对应一个user特征无需tile;
                    训练时, 此处是 item 维度的 list sample, 因此 common attr 需要复制 list 长度才能对齐一个 list sample;
                    infer 时, 非 list sample, batch size 实际上对应 list 长度, 因此需要处理
                '''
                emb_dim = common_embs.shape[-1]
                common_embs = tf.reshape(common_embs, [1, -1, emb_dim]) # (1, ?, 532)
            print("common_embs shape", common_embs.shape) # (?, 30, 532)
            return common_embs
        
    def _contrastive_loss(self, score_matrix, margin=0.5, seqlen=6):
       gold_score = tf.linalg.diag_part(score_matrix)
       gold_score = tf.expand_dims(gold_score, axis=2)

       difference_matrix = gold_score - score_matrix
       loss_matrix = margin - difference_matrix
       loss_matrix = tf.nn.relu(loss_matrix)

       base_mask = tf.ones((seqlen, seqlen)) - tf.linalg.diag(tf.ones(seqlen))
       base_mask = tf.expand_dims(base_mask, axis=0)
       base_mask = tf.tile(base_mask,[tf.shape(score_matrix)[0],1,1])

       cl_loss = tf.reduce_mean(loss_matrix*base_mask)
       
       return cl_loss
    
    def gumbel_softmax(self, logits, tau=1.0, hard=False, dim=-1):
        def sample_gumbel(shape):
            """Sample from Gumbel(0, 1)"""
            uniform_samples = tf.random_uniform(shape, minval=0, maxval=1)
            return -tf.log(-tf.log(uniform_samples + 1e-20) + 1e-20)
        
        # Sample Gumbel noise
        gumbels = sample_gumbel(tf.shape(logits))
        gumbels = (logits + gumbels) / tau 
        y_soft = tf.nn.softmax(gumbels, axis=dim)

        if hard:
            # Straight through.
            index = tf.argmax(y_soft, axis=dim)
            y_hard = tf.one_hot(index, depth=tf.shape(logits)[dim], dtype=logits.dtype)
            y_hard = tf.reshape(y_hard, tf.shape(logits))
            ret = tf.stop_gradient(y_hard - y_soft) + y_soft
        else:
            ret = y_soft
        return ret

    def choose_item(self, decoder_emb, method=0, use_gumbel_softmax=False, tau=1.0, hard=True):
        # 端到端的情况下需要打开 gumbel softmax，具备采样能力
        if method == 0:
            # MLP 预测每个step对应选择哪个item [0, 1, 2, 3, EOT]
            with tf.variable_scope("predict_nn", reuse=tf.AUTO_REUSE):
                decoder_emb = tf.tile(tf.expand_dims(decoder_emb, axis=2), [1, 1, self.candidates_size + 3, 1]) # (?,list_size+1,candidates_size+3,32)
                vocab_emb = tf.tile(tf.expand_dims(self.photo_embs, axis=1), [1, tf.shape(decoder_emb)[1], 1, 1]) # (?,list_size+1,candidates_size+3,32)
                concat_emb = tf.concat([decoder_emb, vocab_emb], axis=-1) # (?,list_size+1,candidates_size+3,64)
                predict = self._mlp_layer("mlp_layer", concat_emb, [128, 64])
                predict = tf.layers.dense(predict, 1, activation=tf.nn.sigmoid, name="prob_layer") # (?,list_size+1,candidates_size+3,1)
                predict = tf.squeeze(predict, axis=-1) # (?,list_size+1,candidates_size+3)
        elif method == 1:
            # cosine 选取
            with tf.variable_scope("predict_cosine", reuse=tf.AUTO_REUSE):
                predict = tf.matmul(decoder_emb, tf.transpose(self.photo_embs,  perm=[0, 2, 1])) # (?, list_size+1, candidates_size+3)
                predict = tf.nn.softmax(predict, axis=-1)
        if use_gumbel_softmax:
            predict = self.gumbel_softmax(predict, tau=tau, hard=hard)
        else:
            predict = tf.nn.softmax(predict, axis=-1) # (?,list_size+1,candidates_size+3)
        return predict

    def model(self, decode_method="beam_search"):
        def beam_search(model, encoder_output, sos_token, eos_token, pad_token, beam_size=3, max_length=6):
            """
            实现 Beam Search 的自回归解码，同时添加每条生成路径中 token 不能重复的约束。
            Args:
                model: 生成模型，包含 encoder 和 decoder。
                encoder_output: 编码器的输出，形状 [batch_size, vocab_size, dim]。
                sos_token: 起始 token 的 ID，形状 [batch_size, 1]。
                eos_token: 结束 token 的 ID，形状 [batch_size, 1]。
                pad_token: 填充 token 的 ID，形状 [batch_size, 1]。
                beam_size: Beam Search 的宽度，表示保留的候选路径数量。
                max_length: 最大生成长度。

            Returns:
                best_sequences: 形状 [batch_size, max_length]，表示生成的序列。
            """
            batch_size, vocab_size = tf.shape(encoder_output)[0],tf.shape(encoder_output)[1]
            
            # 初始化每束的生成序列、分数和完成状态
            sequences = tf.tile(tf.expand_dims(sos_token, axis=1), [1, beam_size, 1])  # [batch_size, beam_size, 1]
            scores = tf.zeros([batch_size, beam_size])  # [batch_size, beam_size]
            reward = tf.zeros([batch_size, beam_size])  # [batch_size, beam_size]
            
            # repeat encoder_output for each beam 
            encoder_output = tf.tile(tf.expand_dims(encoder_output, axis=1), [1, beam_size, 1, 1])  # [batch_size, beam_size, vocab_size, dim])
            # 单点价值
            preward = self._calc_point_reward() # [?, vocab_size - 3]
            sos_reward = tf.zeros([batch_size, 1])
            pad_reward = tf.zeros([batch_size, 1])
            eos_reward = tf.zeros([batch_size, 1])
            preward = tf.concat([sos_reward, pad_reward, preward, eos_reward], axis=-1) # [?, vocab_size]
            print(f"preward shape: {preward.shape}")
            probs = []
            # 开始 Beam Search
            for step in range(max_length):
                # 当前所有序列的 embedding
                batch_indices = tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), [1, beam_size]) # [batch_size, beam_size]
                beam_indices = tf.tile(tf.expand_dims(tf.range(beam_size), axis=0), [batch_size, 1]) # [batch_size, beam_size]
                # 扩展batch和beam维度以匹配sequences的每个位置
                batch_indices = tf.expand_dims(batch_indices, axis=2)  # [batch_size, beam_size, 1]
                beam_indices = tf.expand_dims(beam_indices, axis=2)    # [batch_size, beam_size, 1]
                # 复制到已生成序列长度
                seq_length = tf.shape(sequences)[2]
                batch_indices = tf.tile(batch_indices, [1, 1, seq_length])  # [batch_size, beam_size, seq_length]
                beam_indices = tf.tile(beam_indices, [1, 1, seq_length])    # [batch_size, beam_size, seq_length]
                # 提取已生成序列的embedding
                gather_indices = tf.stack([batch_indices, beam_indices, sequences], axis=-1) # [batch_size, beam_size, seq_length, 3]
                decoder_input = tf.gather_nd(encoder_output, gather_indices)  # [batch_size, beam_size, seq_length, dim]
                # decoder forward
                decoder_output = model.forward_decoder(encoder_output, decoder_input, training=self.training)  # [batch_size, beam_size, seq_length, dim]
                decoder_output = self._mlp_layer("decoder_output", decoder_output, [32], activation=None) # (?,candidates_size,32)
                # 计算 logits
                logits = tf.matmul(encoder_output, tf.transpose(decoder_output, perm=[0, 1, 3, 2])) # [batch_size, beam_size, vocab_size, seq_length]
                next_token_logits = logits[:, :, :, -1]  # [batch_size, beam_size, vocab_size]
                # 选择下一个token

                tau = 5
                next_token_probs = tf.nn.softmax(next_token_logits/tau, axis=-1)  # [batch_size, beam_size, vocab_size]
                probs.append(next_token_probs)
                log_probs = tf.math.log(next_token_probs+1e-9)  # 转换为 log 概率
                # cur_reward = next_token_probs * tf.tile(tf.expand_dims(preward, axis=1), [1, beam_size, 1])  # 转换为 log 概率 [batch_size, beam_size, vocab_size]

                # 初始化 used_tokens，每个束内的 token 初始状态为未使用
                used_token = tf.zeros([batch_size, beam_size, vocab_size], dtype=tf.bool)  # [batch_size, beam_size, vocab_size]
                # 将每束的sos、eos、pad token 的 used_tokens 置为 True
                batch_indices, beam_indices = tf.repeat(tf.range(batch_size), beam_size), tf.tile(tf.range(beam_size), [batch_size])  # [batch_size * beam_size]

                # 特殊token索引
                sos_indices = tf.repeat(tf.squeeze(sos_token, axis=1), beam_size)  # [batch_size * beam_size]
                eos_indices = tf.repeat(tf.squeeze(eos_token, axis=1), beam_size)  # [batch_size * beam_size]
                pad_indices = tf.repeat(tf.squeeze(pad_token, axis=1), beam_size)  # [batch_size * beam_size]
                
                # 更新used_tokens
                used_token = tf.tensor_scatter_nd_update(
                    used_token,
                    tf.stack([batch_indices, beam_indices, sos_indices], axis=1),
                    tf.ones([batch_size * beam_size], dtype=tf.bool)
                )
                used_token = tf.tensor_scatter_nd_update(
                    used_token,
                    tf.stack([batch_indices, beam_indices, eos_indices], axis=1),
                    tf.ones([batch_size * beam_size], dtype=tf.bool)
                )
                used_token = tf.tensor_scatter_nd_update(
                    used_token,
                    tf.stack([batch_indices, beam_indices, pad_indices], axis=1),
                    tf.ones([batch_size * beam_size], dtype=tf.bool)
                )
                print("used_token shape",used_token.shape)

                one_hot = tf.one_hot(sequences, vocab_size, on_value=True, off_value=False)  # shape: [batch_size, beam_size, seq_len, vocab_size]
                used_token_tmp = tf.reduce_any(one_hot, axis=2)  # shape: [batch_size, beam_size, vocab_size]
                used_token = tf.logical_or(used_token_tmp, used_token)
                
                # 根据used_token将已生成tokn的分数设为-inf
                log_probs = tf.where(used_token, tf.fill(tf.shape(log_probs), float('-inf')), log_probs)  # [batch_size, beam_size, vocab_size]
                # used_reward = tf.where(used_token, tf.fill(tf.shape(log_probs), float('-inf')), cur_reward)  # [batch_size, beam_size, vocab_size]
                # 计算总分数  (当前路径分数 + 新 token 的分数)
                scores = tf.expand_dims(scores, axis=-1) + log_probs  # [batch_size, beam_size, vocab_size]
                # reward = tf.expand_dims(reward, axis=-1) + cur_reward + used_reward  # [batch_size, beam_size, vocab_size]
                
                # topk最高分数 
                if step == 0:
                    # 第一步直接取 topk 不同 idx
                    top_k_scores, top_k_indices = tf.math.top_k(scores[:, 0, :], k=beam_size, sorted=True)  # [batch_size,beam_size)
                    # top_k_reward, top_k_indices = tf.math.top_k(reward[:, 0, :], k=beam_size, sorted=True)  # [batch_size,beam_size)
                else:
                    scores_flat = tf.reshape(scores, [batch_size, -1]) # [batch_size, beam_size * vocab_size]
                    top_k_scores, top_k_indices = tf.math.top_k(scores_flat, k=beam_size, sorted=True)  # [batch_size,beam_size)
                    # reward_flat = tf.reshape(reward, [batch_size, -1]) # [batch_size, beam_size * vocab_size]
                    # top_k_reward, top_k_indices = tf.math.top_k(reward_flat, k=beam_size, sorted=True)  # [batch_size,beam_size)
                # 更新序列和分数
                beam_indices = top_k_indices // vocab_size  # [batch_size, beam_size]# 当前topk路径上一步的idx
                token_indices = top_k_indices % vocab_size  # [batch_size, beam_size]# 当前topk路径当前步的idx

                # 更新生成的序列的beam索引
                batch_indices = tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), [1, beam_size])  # [batch_size, beam_size]
                gather_indices = tf.stack([batch_indices, beam_indices], axis=-1)  # [batch_size, beam_size, 2]
                # 获取要更新的beam
                selected_sequences = tf.gather_nd(sequences, gather_indices)  # [batch_size, beam_size, seq_length]
                # 将新token添加到beam序列末尾
                sequences = tf.concat([selected_sequences, tf.expand_dims(token_indices, axis=-1)], axis=-1) # [batch_size, beam_size, seq_length + 1]
                scores = top_k_scores
                # reward = top_k_reward

            # 从 beam_size 条路径中选择分数最高的路径
            best_sequence_indices = tf.expand_dims(tf.argmax(scores, axis=1), axis=-1)  # [batch_size,1]
            # best_sequence_indices = tf.expand_dims(tf.argmax(reward, axis=1), axis=-1)  # [batch_size,1]
            best_sequences = tf.gather_nd(sequences, tf.concat([tf.expand_dims(tf.cast(tf.range(batch_size),dtype=tf.int64), axis=-1), best_sequence_indices], axis=-1)) # [batch_size, max_length]
            best_sequences = best_sequences[:, 1:]
            generated_sequence = sequences[:,:,1:]
            # print("beam search end!")
            # self.print_ops.append(tf.print("used_token ", used_token, summarize=10, output_stream=sys.stdout))
            return logits, generated_sequence, preward, best_sequences, probs

        with tf.variable_scope("prepare", reuse=tf.AUTO_REUSE):
            common_embs = self._get_shared_features(self._parameters_dict, self.candidates_size) # (?,30,532)
            photo_embs    = tf.concat([self._parameters_dict[k] for k in self._parameters_dict if k in photo_fea_names], axis=-1)
            self.photo_embs = tf.layers.dense(photo_embs, 32, activation=tf.nn.relu) # 作为 emb dict
            batch_size = tf.shape(common_embs)[0]
            hidden_states_in = self._mlp_layer("mlp_layer_1", common_embs, [64, 32]) # (?,candidates_size,32)
            # 添加特殊token的embedding
            pad_embedding = tf.tile(tf.expand_dims(self.pad_embedding, axis=0), #(?,1,32)
                                [batch_size, 1, 1])
            sos_embedding = tf.tile(tf.expand_dims(self.sos_embedding, axis=0),  #(?,1,32)
                                [batch_size, 1, 1])
            eos_embedding = tf.tile(tf.expand_dims(self.eos_embedding, axis=0), #(?,1,32)
                                [batch_size, 1, 1])

            sos_token = tf.tile(tf.constant(1, shape=[1, 1], dtype=tf.int32), [batch_size, 1]) #(?,1)
            eos_token = tf.tile(tf.constant(self.candidates_size + 2, shape=[1, 1], dtype=tf.int32), [batch_size, 1])
            pad_token = tf.tile(tf.constant(0, shape=[1, 1], dtype=tf.int32), [batch_size, 1]) #(?,1) all 0
            self.photo_embs = tf.concat([pad_embedding, sos_embedding, self.photo_embs, eos_embedding], axis=1) # (?,candidates_size+3,32)
            print("hidden_states_in shape", hidden_states_in.shape)

            if self.training:
                label_dicts = self._label_value_dict
                realshow_label = label_dicts['context_info__real_show_list']
                realshow_label = tf.reshape(realshow_label, [-1, self.candidates_size])
                realshow_label = realshow_label[:,:self.list_size] # (?,list_size)，截断为 list_size 个
                realshow_weight = label_dicts['fountain_fulllink_rerank_realshow_label_weight_list']
                realshow_weight = tf.reshape(realshow_weight, [-1, self.candidates_size])
                realshow_weight = realshow_weight[:,:self.list_size] # (?,list_size)
                realshow_label = tf.cast(realshow_label,dtype=tf.int32) #(?,list_size)

                indices_shape = tf.shape(realshow_label)
                col_indices = tf.tile(tf.expand_dims(tf.range(indices_shape[1]),0),[indices_shape[0],1])+2 # (?,list_size) 从第2个起
                # self.print_ops.append(tf.print("rerank shape ", tf.shape(realshow_label), summarize=10, output_stream=sys.stdout))
                print("realshow_label shape ",realshow_label.shape)
                print("col indices shape ",col_indices.shape)
                realshow_indices = tf.cast(col_indices * realshow_label,dtype=tf.int32) # label为0 1，过滤了未曝光的index
                inputs = tf.concat([sos_token, realshow_indices], axis=1) # (?,list_size+1)
                outputs = tf.concat([realshow_indices, eos_token], axis=1) # (?,list_size+1)
                print("inputs ", inputs)
                print("outputs ", outputs)
                batch_indices = tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), [1, self.list_size+1]) # (?, self.list_size+1)
                # batch_indices = tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), [1, self.candidates_size+1]) # after shuffle
                gather_indices = tf.stack([batch_indices, inputs], axis=-1) # (?, self.list_size+1, 2)

        # with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
        #     model = StackedTransformerModel(num_layers=1, dim=32, num_heads=4, hidden_dim=128, dropout_rate=0.0, k=6)
        #     item_hidden = model.forward(hidden_states_in, training=True)
        #     item_hidden = tf.layers.dense(item_hidden, 128, activation=tf.nn.leaky_relu)
        #     item_hidden = tf.layers.dense(item_hidden, 64, activation=tf.nn.leaky_relu)
        #     item_hidden = tf.layers.dense(item_hidden, 1, activation=tf.nn.sigmoid) # (?, cand_size, 1)

        with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
            # 初始化transformer模型
            # model = StackedTransformerModel(num_layers=1, dim=32, num_heads=4, hidden_dim=128, dropout_rate=0.0, k=6)
            # hidden_states = model.forward(hidden_states_in, training=True)
            model = SetTransformerModel(dim=32, hidden_dim=128, num_inds=30, num_seeds=30, num_heads=4, dropout_rate=0.0, training=self.training)
            encoder_output = model.ISAB("isab_0", hidden_states_in, training=self.training)
            print("encoder output shape ",encoder_output.shape) # (?,candidates_size+3,32)
            if self.training:
                item_embeddings = tf.gather_nd(self.photo_embs, gather_indices) # (?,candidates_size+3,32) 中查找对应 list idx 的 emb
                print("item_embeddings shape", item_embeddings.shape) # (?,list_size+1,32)

                # decoder
                decoder = DecoderLayer("decoder_layer_0", 32, 2, 32, dropout_rate=0.0, training=self.training)
                item_embedding = decoder.forward(item_embeddings, encoder_output, training=self.training) # (?,list_size+1,32)
                print("item_embedding shape ", item_embedding.shape)

                # 选取 item, 0: nn, 1: cosine; 是否进行采样
                predict = self.choose_item(item_embedding, method=0, use_gumbel_softmax=False, tau=0.1, hard=True) # (?,list_size+1,candidates_size+3)
                output_indices = tf.expand_dims(outputs, axis=2) # (?,list_size+1,1)
                pos_output = tf.batch_gather(predict, output_indices) # (?,list_size+1,1) 拿到真实index对应的score, 非全局emb Matrix 需要使用batch_gather
                vector = tf.zeros([batch_size, 1], dtype=tf.int32) # mask EOS token
                realshow_label = tf.concat([realshow_label, vector], axis=1)
                print("pos_output shape", pos_output.shape)
                print("outputs shape", outputs.shape)

                pos_output = tf.squeeze(pos_output, axis=-1) #(?,list_size+1)
                realshow_label = tf.cast(realshow_label,dtype=tf.float32)
                valid_pos_output = -tf.log(pos_output+1e-9)*realshow_label
                self.print_ops.append(tf.print("realshow_label ", realshow_label[2], summarize=8, output_stream=sys.stdout))
                self.print_ops.append(tf.print("pos_output ", pos_output[2], summarize=8, output_stream=sys.stdout))

                valid_counts = tf.reduce_sum(realshow_label, axis=-1)+1e-9
                item_weight = tf.clip_by_value(realshow_weight, 0, 600) # (?,cand_size)
                vector = tf.ones([batch_size, 1], dtype=tf.float32)
                item_weight = tf.concat([item_weight, vector], axis=1) # (?, list_size + 1) 单点reward
                # seq_weight = tf.reduce_sum(item_weight * realshow_label,axis=-1) / valid_counts # (?,)
                # seq_weight = tf.where(seq_weight > 7, tf.log(seq_weight) - 0.9, tf.ones_like(seq_weight, dtype=tf.float32))
                print("item_weight ", item_weight)
                item_weight = tf.where(item_weight > 7, tf.log(item_weight) / tf.math.log(1.4) - 4.6, tf.ones_like(item_weight, dtype=tf.float32))
                item_weight *= realshow_label
                # self.print_ops.append(tf.print("seq_weight ", seq_weight[2], summarize=8, output_stream=sys.stdout))
                self.print_ops.append(tf.print("item_weight ", item_weight[2], summarize=8, output_stream=sys.stdout))
                # gen_loss = tf.reduce_sum(valid_pos_output, axis=-1)*seq_weight/valid_counts
                gen_loss = tf.reduce_sum(valid_pos_output * item_weight, axis=-1)/valid_counts
                print("gen_loss shape", gen_loss.shape)
                gen_loss = tf.reduce_mean(gen_loss)
                
                return None, gen_loss, None, pos_output, predict
            
            else:
                # 初始化解码过程
                max_length = 4  # 目标生成长度
                sos_token = tf.tile(tf.constant(1, shape=[1, 1], dtype=tf.int32), [batch_size, 1]) #(?,1) all 1
                eos_value = tf.shape(encoder_output)[1]-1
                eos_token = tf.tile(tf.fill([1, 1], eos_value), [batch_size, 1])
                pad_token = tf.tile(tf.constant(0, shape=[1, 1], dtype=tf.int32), [batch_size, 1]) #(?,1) all 0

                # auto-regressive decoding 
                if decode_method == "greedy":
                    generated_tokens = sos_token  # [batch_size, 1]
                    # 用于记录已使用的token
                    num_tokens = tf.shape(encoder_output)[1] # 33
                    used_tokens = tf.zeros([batch_size, num_tokens], dtype=tf.bool) # (?,33)
                    # 将 SOS token标记为已使用
                    batch_range = tf.range(batch_size) #(?,)
                    used_tokens = tf.tensor_scatter_nd_update(
                        used_tokens,
                        tf.stack([batch_range, tf.squeeze(sos_token, axis=1)], axis=1), # 形成 (?,2) 的二维索引矩阵
                        tf.ones([batch_size], dtype=tf.bool)
                    )
                    used_tokens = tf.tensor_scatter_nd_update(
                        used_tokens,
                        tf.stack([batch_range, tf.squeeze(eos_token, axis=1)], axis=1),
                        tf.ones([batch_size], dtype=tf.bool)
                    )
                    used_tokens = tf.tensor_scatter_nd_update(
                        used_tokens,
                        tf.stack([batch_range, tf.squeeze(pad_token, axis=1)], axis=1),
                        tf.ones([batch_size], dtype=tf.bool)
                    )  
                    # greedy search
                    for step in range(max_length):
                        # 获取当前序列的embedding
                        batch_indices = tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), [1, tf.shape(generated_tokens)[1]]) #[batch_size, step + 1]
                        print("batch_indices shape", batch_indices.shape)
                        gather_indices = tf.stack([batch_indices, generated_tokens], axis=-1) # [batch_size, seq_len]
                        print("gather_indices shape", gather_indices.shape)
                        decoder_input = tf.gather_nd(encoder_output, gather_indices)  # 已生成token的embedding [batch_size, seq_len, dim] (?,1,32)
                        # decoder前向传播
                        decoder_output = model.forward_decoder(encoder_output, decoder_input, training=self.training) #(?,33,32)
                        # 计算下一个token的概率
                        decoder_output_trans = tf.transpose(decoder_output, perm=[0, 2, 1])  # [batch_size, dim, seq_len]
                        logits = tf.matmul(encoder_output, decoder_output_trans)  # [batch_size, , seq_len]
                        # 只关注最后一个位置的预测
                        next_token_logits = logits[:, :, -1]  # [batch_size, ]
                        # 将已使用的token的logits设为负无穷
                        next_token_logits = tf.where(
                            used_tokens,
                            tf.fill(tf.shape(next_token_logits), float('-inf')),
                            next_token_logits
                        )
                        # 采样或解码
                        temperature = 1.0
                        # 推理时使用贪婪解码
                        next_token = tf.cast(tf.expand_dims(tf.argmax(next_token_logits, axis=-1), axis=1), dtype=tf.int32)  # [batch_size, 1]
                        
                        # 更新已使用的token记录
                        used_tokens = tf.tensor_scatter_nd_update(
                            used_tokens,
                            tf.stack([batch_range, tf.squeeze(next_token, axis=1)], axis=1),
                            tf.ones([batch_size], dtype=tf.bool)
                        )
                        
                        # 拼接新token
                        generated_tokens = tf.concat([generated_tokens, next_token], axis=1)
                    # 移除SOS token
                    generated_sequence = generated_tokens[:, 1:]  # [batch_size, max_length]
                
                elif decode_method == "beam_search":
                    beam_size = 1
                    logits, generated_sequence, preward, best_sequences, probs = beam_search(
                        model,
                        encoder_output,
                        sos_token,
                        eos_token,
                        pad_token,
                        beam_size,
                        max_length,
                    )
                    print("logits shape",logits.shape)
                    print("generated_sequence shape",generated_sequence.shape)
                return logits, generated_sequence