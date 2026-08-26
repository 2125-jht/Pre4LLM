from math import tau
from numpy import dtype
import sys
import tensorflow as tf
from modules_ import *
from feature_attr_extract import user_fea_names,photo_fea_names,source_fea_names,explore_profile_fea_names,fountain_seq_pid_names,fountain_seq_aid_names

    
class StackedTransformerModel():
    def __init__(self, num_layers, dim, num_heads, dk, dropout_rate, k, training=False):
        '''
        dim: query 的维度
        dk: key 投影矩阵的维度
        '''
        super(StackedTransformerModel, self).__init__()
        self.num_layers = num_layers
        self.k = k
        self.dim = dim
        self.layers = [EncoderLayer(f"transformer_layer_{i}", dim, num_heads, dk, dropout_rate, training=training) for i in range(num_layers)]
        self.decoder_layers = [DecoderLayer(f"position_layer_{i}", dim, num_heads, dk, dropout_rate, training=training) for i in range(num_layers)]
        
    def forward(self, hidden_states, training):
        for i in range(self.num_layers):
            hidden_states = self.layers[i].forward(hidden_states, training=training)
        return hidden_states
    
    def forward_decoder(self, hidden_states, item_embedding, training):
        for i in range(self.num_layers):
            item_embedding = self.decoder_layers[i].forward(item_embedding, hidden_states, training=training)
        return item_embedding

class Evaluator():
    def __init__(self, num_layers, dim, num_heads, dk, dropout_rate, k, training=False):
        super(Evaluator, self).__init__()
        self.num_layers = num_layers
        self.k = k
        self.dim = dim
        self.layers = [EncoderLayer(f"transformer_layer_{i}", dim, num_heads, dk, dropout_rate, training=training) for i in range(num_layers)]
        
    def forward(self, hidden_states, training):
        for i in range(self.num_layers):
            hidden_states = self.layers[i].forward(hidden_states, training=training)
        return hidden_states


class FountainDeepLtrMultiTaskModel:
    def __init__(self, parameters_dict, label_value_dict, dense_value_dict, print_ops, list_size, candidates_size, dim=32, extra_param_dict= None, training=True):
        self._pxtr_names = [
            "context_pctr",
            "context_pltr",
            "context_pwtr", # 关注
            "context_pftr", # 分享
            "context_plvtr",
            "context_pvtr",
            "context_pptr",
            "context_pcmtr",
            "context_pepstr",
            "context_pcpr",
            "context_pcltr",
            "context_psvr",
            "context_pwtd",
            "context_fountain_related_score_v2",
        ]
        self._photo_attr_names = [
            "photo_hetu_tag_level1_list",
            "photo_hetu_tag_level2_list",
            "photo_hetu_tag_level3_list",
            "photo_tag",
            "photo_duration_ms",
        ]
        self._parameters_dict = parameters_dict
        self._label_value_dict = label_value_dict
        self._dense_value_dict = dense_value_dict
        self.list_size = list_size
        self.candidates_size = candidates_size
        self.dim = dim
        self.training = training
        self.position_embeddings = tf.get_variable(
            name='position_embeddings', 
            shape=[list_size, 64], 
            initializer=tf.random_normal_initializer()
        )
        # Create [sos] and [eos] embeddings
        self.sos_embedding = tf.get_variable(
            "sos_embedding", shape=[1, 64], initializer=tf.random_uniform_initializer()
        )
        self.eos_embedding = tf.get_variable(
            "eos_embedding", shape=[1, 64], initializer=tf.random_uniform_initializer()
        )
        self.pad_embedding = tf.get_variable(
            "pad_embedding", shape=[1, 64], initializer=tf.random_uniform_initializer()
        )
        self.print_ops = print_ops

    def _z_score(self, x):
        mean, std = tf.reduce_mean(x), tf.math.reduce_std(x)
        x = (x - mean) / (std + 1e-7)
        return x
    def _min_max_score(self, x):
        min, max = tf.reduce_min(x), tf.math.reduce_max(x)
        x = (x - min) / (max - min + 1e-7)
        return x

    def _mlp_layer(self,
                  scope_name,
                  hidden_states: tf.Tensor,
                  hidden_units: list,
                  activation=tf.nn.relu) -> tf.Tensor:
        with tf.variable_scope(f"{scope_name}_mlp_layer", reuse=tf.AUTO_REUSE):
            for i, hidden_unit in enumerate(hidden_units):
                hidden_states = tf.layers.dense(hidden_states, hidden_unit, activation=activation, use_bias=True)
        return hidden_states

    def _get_shared_features(self, input_dicts) -> tuple:
        with tf.variable_scope("share_bottom", reuse=tf.AUTO_REUSE):
            user_embs_origin     = tf.concat([input_dicts[k] for k in input_dicts if k in user_fea_names], axis=-1)
            user_embs     = tf.tile(tf.expand_dims(user_embs_origin, axis=1), [1,self.candidates_size,1]) if self._training else user_embs_origin
            photo_embs    = tf.concat([input_dicts[k] for k in input_dicts if k in photo_fea_names], axis=-1)
            source_embs_origin   = tf.concat([input_dicts[k] for k in input_dicts if k in source_fea_names], axis=-1)
            source_embs   = tf.tile(tf.expand_dims(source_embs_origin, axis=1), [1,self.candidates_size,1]) if self._training else source_embs_origin

            common_embs   = tf.concat([user_embs, photo_embs, source_embs], axis=-1)
            if not self._training:
                '''
                    infer 时需要注意实际请求的 batch size = items 长度, 但计算图 batch 的 shape 由 uni_predict_fused 中 executor_batchsizes 决定,
                    如600。为了通过计算图编译, 需要 rashape (1, -1, dim), 实际 -1 为请求端发送 items 长度
                '''
                emb_dim = common_embs.shape[-1]
                common_embs = tf.reshape(common_embs, [1, -1, emb_dim])
            return common_embs, user_embs_origin, source_embs_origin, photo_embs
        
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
            with tf.variable_scope("predict_token_nn", reuse=tf.AUTO_REUSE):
                decoder_emb = tf.expand_dims(decoder_emb, axis=2) # (?,list_size+1,1,32)
                vocab_emb = tf.expand_dims(self.photo_embs, axis=1) # (?,1,candidates_size+3,32)
                concat_emb = tf.concat([decoder_emb, vocab_emb], axis=-1) # (?,list_size+1,candidates_size+3,64)
                predict = tf.layers.dense(concat_emb, 128, activation=tf.nn.relu)
                predict = tf.layers.dense(predict, 64, activation=tf.nn.relu)
                predict = tf.layers.dense(predict, 1, activation=tf.nn.sigmoid, name="prob_layer") # (?,list_size+1,candidates_size+3,1)
                predict = tf.squeeze(predict, axis=-1) # (?,list_size+1,candidates_size+3)
        elif method == 1:
            # cosine 选取
            with tf.variable_scope("predict_cosine", reuse=tf.AUTO_REUSE):
                predict = tf.matmul(decoder_emb, tf.transpose(self.photo_embs,  perm=[0, 2, 1])) # (?, list_size+1, candidates_size+3)
                # predict = tf.nn.softmax(predict, axis=-1)
        if use_gumbel_softmax:
            predict = self.gumbel_softmax(predict, tau=tau, hard=hard)
        else:
            predict = tf.nn.softmax(predict, axis=-1) # (?,list_size+1,candidates_size+3)
        return predict
    
    def gen_neg_seq(self, decoder_emb, method=0, use_gumbel_softmax=False, tau=1.0, hard=True):
    
    def get_view_label(self, playing_time, duration_ms):
        playing_time = tf.cast(playing_time, dtype=tf.int32)
        duration_ms = tf.cast(duration_ms, dtype=tf.int32)
        ones = tf.ones_like(duration_ms)
        eff_threshold = ones * 12700
        long_threshold = ones * 79700

        long_threshold = tf.where(duration_ms <= 195000, ones * 92500, long_threshold)
        eff_threshold = tf.where(duration_ms <= 195000, ones * 17600, eff_threshold)
        
        long_threshold = tf.where(duration_ms <= 118200, ones * 74900, long_threshold)
        eff_threshold = tf.where(duration_ms <= 118200, ones * 18300, eff_threshold)
        
        long_threshold = tf.where(duration_ms <= 71800, ones * 46600, long_threshold)
        eff_threshold = tf.where(duration_ms <= 71800, ones * 13100, eff_threshold)
        
        long_threshold = tf.where(duration_ms <= 38800, ones * 28800, long_threshold)
        eff_threshold = tf.where(duration_ms <= 38800, ones * 11400, eff_threshold)
        
        long_threshold = tf.where(duration_ms <= 20300, ones * 18400, long_threshold)
        eff_threshold = tf.where(duration_ms <= 20300, ones * 9900, eff_threshold)
        
        long_threshold = tf.where(duration_ms <= 12700, ones * 13600, long_threshold)
        eff_threshold = tf.where(duration_ms <= 12700, ones * 8700, eff_threshold)
        
        long_threshold = tf.where(duration_ms <= 8700, ones * 12000, long_threshold)
        eff_threshold = tf.where(duration_ms <= 8700, ones * 7200, eff_threshold)

        long_threshold = tf.where(tf.equal(duration_ms, 0), ones * 13100, long_threshold)
        eff_threshold = tf.where(tf.equal(duration_ms, 0), ones * 4500, eff_threshold)

        effective_view = playing_time >= eff_threshold
        effective_view = tf.cast(effective_view, dtype=tf.float32)
        long_view = playing_time >= long_threshold
        long_view = tf.cast(long_view, dtype=tf.float32)

        return effective_view, long_view

    def mha_layer_4d(name, query, key, dim_in=64, num_heads=4, dropout_rate=0.0, training=False, causal_mask=False):
        '''
        query: (?, cand_size, dim1)
        key: (?, cand_size, key_len, dim2)
        '''
        batch_size, cand_size, key_len = tf.shape(key)[0], tf.shape(key)[1], tf.shape(key)[2]
        query = tf.expand_dims(query, axis=2) # (?, cand_size, 1, dim1)
        query = tf.reshape(query, [batch_size * cand_size, 1, -1])
        key = tf.reshape(key, [batch_size * cand_size, key_len, 1, -1])
        attn_out = multi_head_attention(name, query, key, key, dim_in=dim_in, num_heads=num_heads,
                                        dropout_rate=dropout_rate, training=training, causal_mask=causal_mask) # (?*cand_size,1,dim)
        attn_out = tf.reshape(attn_out, [batch_size, cand_size, -1]) # (?, cand_size, dim1)
        return attn_out

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
                # logits = tf.matmul(encoder_output, tf.transpose(decoder_output, perm=[0, 1, 3, 2])) # [batch_size, beam_size, vocab_size, seq_length]
                logits = tf.matmul(self.photo_embs, tf.transpose(decoder_output, perm=[0, 1, 3, 2])) # [batch_size, beam_size, vocab_size, seq_length]
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
            common_embs, user_embs, source_embs, photo_embs = self._get_shared_features(self._parameters_dict, self.candidates_size) # (?,30,d)
            photo_attr_embs    = tf.concat([self._parameters_dict[k] for k in self._photo_attr_names], axis=-1) # (?,cand_size,dim)
            pxtr_embs    = tf.concat([tf.expand_dims(self._parameters_dict[k], axis=2) for k in self._parameters_dict if k in self._pxtr_names], axis=1) # (?,cand_size,n,dim)
            pxtr_mha_0 = self.mha_layer_4d("pxtr_mha_0", photo_attr_embs, pxtr_embs, dim_in=64, num_heads=4, dropout_rate=0.1) # (?,cand_size,d)
            query_emb = tf.concat([user_embs, source_embs], axis=-1) # (?,cand_size,d)
            pxtr_mha_1 = multi_head_attention("pxtr_mha_1", query_emb, pxtr_embs, dim_in=128, num_heads=4, dropout_rate=0.1) # (?,cand_size,d)
            photo_embs = tf.layers.dense(tf.concat([photo_embs, pxtr_mha_0, pxtr_mha_1]), 64, activation=tf.nn.relu)
            common_embs = tf.layers.dense(tf.concat([common_embs, pxtr_mha_0, pxtr_mha_1]), 128, activation=tf.nn.relu)
            batch_size = tf.shape(common_embs)[0]
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
            self.photo_embs = tf.concat([pad_embedding, sos_embedding, photo_embs, eos_embedding], axis=1) # (?,candidates_size+3,32)

            if self.training:
                label_dicts = self._label_value_dict
                realshow_label = label_dicts['context_info__real_show_list']
                realshow_label = realshow_label[:,:self.list_size] # (?,list_size)，截断为 list_size 个
                click_label = tf.cast(label_dicts["fountain_click_label_list"],dtype=tf.float32)
                click_label = click_label[:, :self.list_size] # (?,list_size)，截断为 list_size 个
                realshow_weight = label_dicts['context_info__playing_time_list']
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

        with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
            
            # 初始化transformer模型
            model = StackedTransformerModel(num_layers=1, dim=128, num_heads=4, hidden_dim=128, dropout_rate=0.1, k=6)
            hidden_states = model.forward(common_embs, training=True)
            encoder_output = hidden_states
            print("encoder output shape ",encoder_output.shape) # (?,candidates_size+3,32)
            if self.training:
                # 从hidden_states查对应emb表示
                # item_embeddings = tf.gather_nd(hidden_states, gather_indices) # (?,candidates_size+3,32) 中查找对应 list idx 的 emb
                item_embeddings = tf.gather_nd(self.photo_embs, gather_indices) # (?,candidates_size+3,32) 中查找对应 list idx 的 emb
                print("item_embeddings shape", item_embeddings.shape) # (?,list_size+1,32)

                # decoder 出的表征应当与实际结果相近，因此与双向 transformer 编码后的表示做选择
                item_embedding = model.forward_decoder(hidden_states, item_embeddings, training=True) # (?,list_size+1,32)
                print("item_embedding shape ", item_embedding.shape)

                # 选取 item, 0: nn, 1: cosine; 是否进行采样
                predict = self.choose_item(item_embedding, method=0, use_gumbel_softmax=True, tau=0.1, hard=True) # (?,list_size+1,candidates_size+3)
                output_indices = tf.expand_dims(outputs, axis=2) # (?,list_size+1,1)
                pos_output = tf.batch_gather(predict, output_indices) # (?,list_size+1,1) 拿到真实index对应的score, 非全局emb Matrix 需要使用batch_gather
                vector = tf.zeros([batch_size, 1], dtype=tf.int32) # mask EOS token
                realshow_label = tf.concat([realshow_label, vector], axis=1)
                print("pos_output shape", pos_output.shape)
                print("outputs shape", outputs.shape)

                pos_output = tf.squeeze(pos_output, axis=-1) #(?,list_size+1)
                realshow_label = tf.cast(realshow_label,dtype=tf.float32)
                valid_pos_output = -tf.log(pos_output+1e-9)*realshow_label
                position_weight = tf.cast(tf.tile(tf.expand_dims(tf.constant([1.5, 1.3, 1]), axis=0), [batch_size, 1]), dtype=tf.float32) # (?, 3)
                position_weight = tf.concat([position_weight, tf.ones([batch_size, self.list_size - 2], dtype=tf.float32)], axis=-1) # (?, list_size)
                print(f"position_weight shape: {position_weight.shape}")
                self.print_ops.append(tf.print("realshow_label ", realshow_label[2], summarize=8, output_stream=sys.stdout))
                self.print_ops.append(tf.print("pos_output ", pos_output[2], summarize=8, output_stream=sys.stdout))

                valid_counts = tf.reduce_sum(realshow_label, axis=-1)+1e-9
                item_weight = tf.clip_by_value(realshow_weight, 0, 600) # (?,cand_size)
                vector = tf.ones([batch_size, 1], dtype=tf.float32)
                item_weight = tf.concat([item_weight, vector], axis=1) # (?, list_size + 1) 单点reward
                seq_weight = tf.reduce_sum(item_weight * realshow_label,axis=-1) / valid_counts # (?,)
                seq_weight = tf.where(seq_weight > 7, tf.log(seq_weight) - 0.9, tf.ones_like(seq_weight, dtype=tf.float32))
                # item_weight = tf.where(item_weight > 7, tf.log(item_weight) / tf.math.log(1.4) - 4.6, tf.ones_like(item_weight, dtype=tf.float32))
                # item_weight *= realshow_label
                self.print_ops.append(tf.print("seq_weight ", seq_weight[2], summarize=8, output_stream=sys.stdout))
                gen_loss = tf.reduce_sum(valid_pos_output, axis=-1)*seq_weight/valid_counts
                print("gen_loss shape", gen_loss.shape)
                gen_loss = tf.reduce_mean(gen_loss)
            
            else:
                # 初始化解码过程
                max_length = 4  # 目标生成长度
                # beam_size = 10
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
                return logits, generated_sequence, preward, best_sequences, probs

        with tf.variable_scope("evaluator", reuse=tf.AUTO_REUSE): # 仅训练使用
            def get_eval_logit(item_emb):
                position_ids = tf.range(self.list_size, dtype=tf.int32)
                position_ids = tf.expand_dims(position_ids, 0)
                position_embeddings = tf.nn.embedding_lookup(self.position_embeddings, position_ids)
                position_embeddings = tf.tile(position_embeddings, [tf.shape(hidden_states)[0], 1, 1])
                print("position_embeddings ", position_embeddings)
                hidden_states_in = item_emb + position_embeddings
                eval_model = Evaluator(num_layers=2, dim=32, num_heads=4, hidden_dim=128, dropout_rate=0.0,
                                       k=self.list_size, training=self.training)
                eval_hidden = eval_model.forward(hidden_states=hidden_states_in, training=self.training) # (?, list_size, 128)
                eval_logits = self._mlp_layer("mlp_layer_2", eval_hidden, [64])
                eval_logits = self._mlp_layer("mlp_layer_out", eval_hidden, [1], activation=tf.nn.sigmoid)
                eval_logits = tf.squeeze(eval_logits, axis=-1) # (?, list_size)

                Q = tf.layers.dense(eval_hidden, 64, use_bias=False) # (?, list_size, 128)
                K = tf.layers.dense(eval_hidden, 64, use_bias=False)
                V = tf.layers.dense(eval_hidden, 64, use_bias=False)
                att_out, att_weight = scaled_dot_product_attention(Q, K, V, mask=None) # (?, list_size, 64)
                list_eval_logits = tf.concat(tf.split(att_out, self.list_size, axis=1), axis=-1) # (?, 64*list_size)
                list_eval_logits = self._mlp_layer("mlp_layer_list_1", list_eval_logits, [64, 32])
                list_eval_logits = self._mlp_layer("mlp_layer_out", list_eval_logits, [1], activation=tf.nn.sigmoid)
                list_eval_logits = tf.squeeze(list_eval_logits, axis=-1) # (?, 1)
                return list_eval_logits, eval_logits

            if self.training:
                # 训练 Evaluator
                common_embs = tf.concat([pad_embedding, sos_embedding, common_embs, eos_embedding], axis=1) # (?,candidates_size+3,32)
                item_embeddings = tf.gather_nd(common_embs, gather_indices) # (?,list+1,32) 查找对应 list idx 的 emb
                list_eval_logits, eval_logits = get_eval_logit(item_emb=item_embeddings[:, :-1,]) # (?, 1) (?, list_size)
                click_label = click_label[:, :self.list_size]
                realshow_label = realshow_label[:, :self.list_size]
                realshow_weight = realshow_weight[:, :self.list_size]
                valid_counts = tf.reduce_sum(realshow_label, axis=-1, keep_dims=True) + 1e-15 # (?, 1)
                item_weight = tf.clip_by_value(realshow_weight, 0, 600) # (?,list_size)
                item_weight = tf.where(item_weight > 7, tf.log(item_weight) / tf.math.log(1.4) - 4.6, tf.ones_like(item_weight, dtype=tf.float32))
                eval_weight = item_weight * realshow_label # (?,list_size)
                eval_label = realshow_label
                eval_loss = tf.losses.log_loss(labels=eval_label, predictions=eval_logits, weights=eval_weight, reduction=tf.losses.Reduction.NONE)
                print("eval_loss ", eval_loss)
                eval_loss = tf.reduce_mean(tf.reduce_sum(eval_loss, axis=-1, keep_dims=True) / valid_counts)
                # 设计 list label
                list_click_total = tf.reduce_sum(click_label[:, :2], axis=-1, keep_dims=True) # (?, 1)
                list_time_total = tf.reduce_sum(realshow_weight, axis=-1, keep_dims=True)
                # list_label = tf.where(list_click_total >= 2.0, tf.ones([batch_size, 1], dtype=tf.float32),
                #                       tf.zeros([batch_size, 1], dtype=tf.float32))
                list_label = tf.where(list_time_total >= 20.0, tf.ones([batch_size, 1], dtype=tf.float32), tf.zeros([batch_size, 1], dtype=tf.float32))
                self.print_ops.append(tf.print("list_time_total ", list_time_total[2], summarize=8, output_stream=sys.stdout))
                self.print_ops.append(tf.print("list_label ", list_label[2], summarize=8, output_stream=sys.stdout))
                self.print_ops.append(tf.print("list_eval_logits ", list_eval_logits[2], summarize=8, output_stream=sys.stdout))

                # gen-eval 不更新Evaluator
                gen_emb = tf.matmul(predict[:, :-1, :], common_embs) # (?,list_size,32) generator选择evaluator阶段对应item的emb，前向每个位置只选择一个emb
                gen_eval_logits, _ = get_eval_logit(item_emb=gen_emb) # (?, 1)
                gen_eval_logits = tf.tile(tf.expand_dims(gen_eval_logits, axis=-1), [1, self.list_size, self.candidates_size + 3])
                gen_eval_logits = tf.stop_gradient(gen_eval_logits - predict[:, :-1, :]) + predict[:, :-1, :] # gen-eval loss只更新predict
                gen_eval_logits = tf.reshape(gen_eval_logits[:, 0, 0], [-1, 1])
                gen_eval_labels = tf.ones_like(gen_eval_logits, dtype=tf.float32)
                
                return predict[:, :-1, 2:-1], gen_loss, eval_loss, eval_logits, eval_weight, eval_label, list_eval_logits, list_label, gen_eval_logits, gen_eval_labels