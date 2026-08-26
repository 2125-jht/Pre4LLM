import sys
from numpy import dtype
import tensorflow as tf
from feature_attr_extract import user_fea_names,explore_profile_fea_names,photo_fea_names,source_fea_names,fountain_seq_pid_names,fountain_seq_aid_names
from rerank.generator.splash_gen_model_prm.v0.modules_ import *


class FountainDeepLtrMultiTaskModel:
    def __init__(self, parameters_dict, label_value_dict, loss_names, print_ops, list_size, candidates_size, dim=32, extra_param_dict= None, training=True):
        self._label_value_dict = label_value_dict
        self._parameters_dict = parameters_dict
        self.loss_names = loss_names
        self.list_size = list_size
        self.candidates_size = candidates_size
        self.dim = dim
        self.training = training
        
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
            print("user_id embeding shape", input_dicts['user_id'].shape)
            user_embs = tf.concat([input_dicts[k] for k in input_dicts if k in user_fea_names], axis=-1)
            user_embs = tf.tile(tf.expand_dims(user_embs, axis=1),[1,list_dim,1]) if self.training else user_embs
            print("user_embs shape ", user_embs.shape) # train: (?, 30, 120), infer: (?, 120)
            photo_embs    = tf.concat([input_dicts[k] for k in input_dicts if k in photo_fea_names], axis=-1)
            print("photo_embs shape ", photo_embs.shape) # (?, 30, 380)
            source_embs    = tf.concat([input_dicts[k] for k in input_dicts if k in source_fea_names], axis=-1)
            source_embs = tf.tile(tf.expand_dims(source_embs, axis=1),[1,list_dim,1]) if self.training else source_embs
            print("source_embs shape ", source_embs.shape) # 
            rt_seq_mha    = tf.concat([input_dicts[k] for k in input_dicts if k in explore_profile_fea_names], axis=-1)
            rt_seq_mha  = tf.reduce_mean(rt_seq_mha, axis=1)
            rt_seq_mha = tf.tile(tf.expand_dims(rt_seq_mha, axis=1),[1,list_dim,1]) if self.training else rt_seq_mha
            print("rt_seq_mha shape ", rt_seq_mha.shape) # 
            fountain_seq_pid_mha    = tf.concat([input_dicts[k] for k in input_dicts if k in fountain_seq_pid_names], axis=-1)
            fountain_seq_pid_mha  = tf.reduce_mean(fountain_seq_pid_mha, axis=1)
            fountain_seq_pid_mha = tf.tile(tf.expand_dims(fountain_seq_pid_mha, axis=1),[1,list_dim,1]) if self.training else fountain_seq_pid_mha
            print("fountain_seq_pid_mha shape ", fountain_seq_pid_mha.shape) # 
            fountain_seq_aid_mha    = tf.concat([input_dicts[k] for k in input_dicts if k in fountain_seq_aid_names], axis=-1)
            fountain_seq_aid_mha  = tf.reduce_mean(fountain_seq_aid_mha, axis=1)
            fountain_seq_aid_mha = tf.tile(tf.expand_dims(fountain_seq_aid_mha, axis=1),[1,list_dim,1]) if self.training else fountain_seq_aid_mha
            print("fountain_seq_aid_mha shape ", fountain_seq_aid_mha.shape) # 
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

            common_embs = tf.concat([user_embs, source_embs, photo_embs, rt_seq_mha, fountain_seq_pid_mha, fountain_seq_aid_mha], axis=-1)
            # common_embs = tf.concat([user_embs, source_embs, photo_embs, rt_seq_mha], axis=-1)
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
          
        with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
            def greedy_decode(hidden_states, item_embeddings):
                item_embedding = model.forward_decoder(hidden_states, item_embeddings, training=self.training)
                # 或者选择拼接方式
                # a = tf.tile(tf.expand_dims(tf.range(1, self.list_size), 0), [self.list_size, 1]) # list_size * (list_size - 1)
                # b = tf.tile(tf.expand_dims(tf.range(self.list_size - 1, -1, -1), -1), [1, self.list_size - 1]) # list_size * (list_size - 1)
                # prefix = tf.cast(tf.where(a - b >= 0, a - b, tf.zeros_like(a)), dtype=tf.int32)
                # prefix = tf.tile(tf.expand_dims(prefix, 0), [batch_size, 1, 1]) # (?, list_size, list_size - 1)
                # item_indices = tf.tile(tf.reshape(tf.range(1, self.list_size + 1), [1, self.list_size, 1]), [batch_size, 1, 1]) # (?, lsit_size, 1)
                # item_indices = tf.reshape(tf.concat([prefix, item_indices], axis=-1), [-1, self.list_size * self.list_size]) # (?, list_size * list_size)
                # self.print_ops.append(tf.print("prefix item_indices ", item_indices[2], summarize=8, output_stream=sys.stdout))
                # batch_indices = tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), [1, self.list_size * self.list_size]) # (?, self.list_size * list_size)
                # gather_indices = tf.stack([batch_indices, item_indices], axis=-1) # (?, self.list_size * list_size, 2)
                # item_embedding = tf.gather_nd(hidden_states, gather_indices) # (?, list_size * list_size, 32)
                # item_embedding = tf.reshape(item_embedding, [-1, self.list_size, self.list_size * 32])
                print("item_embedding shape ", item_embedding.shape) # (?, self.list_size, 32)
                item_embedding = self._mlp_layer("decoder_output", item_embedding, [64, 32], activation=None) # (?,list_size,32)
                output = self._mlp_layer("mlp_layer_out", item_embedding, [1], activation=tf.nn.sigmoid) # (?,list_size,1)
                return output

            input_dicts = self._parameters_dict
            common_embs = self._get_shared_features(input_dicts, self.candidates_size) # (?,30,532)
            batch_size = tf.shape(common_embs)[0] # 1
            hidden_states = self._mlp_layer("mlp_layer_1", common_embs, [64, 32]) # (?,candidates_size,32)
            print("hidden_states shape", hidden_states.shape)
            # 初始化transformer模型
            model = StackedTransformerModel(num_layers=1, dim=32, num_heads=4, hidden_dim=128, dropout_rate=0.1, k=6)
            hidden_states = model.forward(hidden_states, training=self.training)
            encoder_output = hidden_states
            print("encoder output shape ",encoder_output.shape) # (?,candidates_size+3,32)
            if self.training:
                label_dicts = self._label_value_dict
                rerank_label = label_dicts['context_info__real_show_list']
                rerank_label = tf.reshape(rerank_label, [-1, self.candidates_size])
                rerank_label = rerank_label[:,:self.list_size] # (?,list_size)，截断为 list_size 个
                indices_shape = tf.shape(rerank_label)
                rerank_label = tf.cast(rerank_label,dtype=tf.int32) #(?,list_size)

                col_indices = tf.tile(tf.expand_dims(tf.range(indices_shape[1]),0),[indices_shape[0],1]) # (?,list_size) 从第0个起
                print("rerank label shape ",rerank_label.shape)
                print("col indices shape ",col_indices.shape)
                rank_indices = tf.cast(col_indices * rerank_label,dtype=tf.int32) # label为0 1，过滤了未曝光的index
                # 从hidden_states查对应emb表示
                batch_indices = tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), [1, self.list_size]) # (?, self.list_size)
                gather_indices = tf.stack([batch_indices, rank_indices], axis=-1) # (?, self.list_size, 2)
                item_embeddings = tf.gather_nd(encoder_output, gather_indices) # (?,candidates_size+1,32) 中查找对应 list idx 的 emb
                print("item_embeddings shape", item_embeddings.shape) # (?,list_size,32)

                # decoder 出的结果直接当做该位置的表征，接单点后验loss，infer时第一步将所有item送入得到单点预测值，贪心选择最大，然后拼上
                output = greedy_decode(hidden_states=encoder_output, item_embeddings=item_embeddings) # (?,list_size,1)
                output_dict = {"ctr": tf.reshape(output, [-1, self.list_size])}

                return output_dict

            else:
                # 初始化解码过程
                max_length = 4  # 目标生成长度
                # auto-regressive decoding 
                if decode_method == "greedy":
                    encoder_dim = tf.shape(encoder_output)[-1]
                    scores = tf.zeros([batch_size, self.candidates_size])
                    used_mask = tf.zeros([batch_size, self.candidates_size])
                    for step in range(max_length):
                        if step == 0:
                            item_embeddings = tf.reshape(encoder_output, [batch_size * self.candidates_size, 1, encoder_dim])
                            output = greedy_decode(hidden_states=encoder_output, item_embeddings=item_embeddings) # (?*cand_size, 1, 1)
                            output = tf.reshape(output, [batch_size, self.candidates_size]) # (?, cand_szie)
                            max_idx = tf.math.argmax(output + used_mask * 1e-19, axis=-1) # (?,1)
                            batch_range = tf.range(batch_size) #(?,)
                            scores = tf.tensor_scatter_nd_update(
                                scores,
                                tf.stack([batch_range, tf.squeeze(max_idx, axis=1)], axis=1), # 形成 (?,2) 的二维索引矩阵
                                tf.ones([batch_size], dtype=tf.bool) * (max_length - step + 1)
                            )
                            used_mask = tf.tensor_scatter_nd_update(
                                used_mask,
                                tf.stack([batch_range, tf.squeeze(max_idx, axis=1)], axis=1), # 形成 (?,2) 的二维索引矩阵
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
                        beam_size,
                        max_length,
                    )
                    print("logits shape",logits.shape)
                    print("generated_sequence shape",generated_sequence.shape)
                return logits, generated_sequence