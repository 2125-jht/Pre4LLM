import sys
import tensorflow as tf
from feature_attr_extract import user_fea_names,explore_profile_fea_names,photo_fea_names,source_fea_names

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

def layer_norm(x, epsilon=1e-6):
    with tf.variable_scope("layer_norm", reuse=tf.AUTO_REUSE):
        gamma = tf.get_variable("gamma", [x.get_shape()[-1]], initializer=tf.ones_initializer())
        beta = tf.get_variable("beta", [x.get_shape()[-1]], initializer=tf.zeros_initializer())

        mean, variance = tf.nn.moments(x, axes=[-1], keep_dims=True)
        normalized = (x - mean) / tf.sqrt(variance + epsilon)
        output = gamma * normalized + beta
    return output

def multi_head_attention(queries, keys, values, num_heads, dropout_rate, training=False,
                         causal_mask=False, use_seq_mask=False, seq_mask=None):
        def split_heads(x, num_heads):
            # 判断输入维度
            input_shape = tf.shape(x)
            is_4d = len(x.get_shape().as_list()) == 4
            
            if is_4d:
                # 四维输入 [batch_size, beam_size, seq_len, dim]
                batch_size = input_shape[0]
                beam_size = input_shape[1]
                depth = x.get_shape().as_list()[-1] // num_heads
                
                # 重塑为 [batch_size * beam_size, seq_len, num_heads, depth]
                reshaped = tf.reshape(x, [batch_size * beam_size, -1, num_heads, depth])
            else:
                # 三维输入 [batch_size, seq_len, dim]
                batch_size = input_shape[0]
                depth = x.get_shape().as_list()[-1] // num_heads
                reshaped = tf.reshape(x, [batch_size, -1, num_heads, depth])
                
            # 转置为 [batch_size(*beam_size), num_heads, seq_len, depth]
            return tf.transpose(reshaped, [0, 2, 1, 3]), is_4d

        def scaled_dot_product_attention(Q, K, V, mask=None):
            # [batch_size(*beam_size), num_heads, seq_len, depth]
            matmul_qk = tf.matmul(Q, K, transpose_b=True) # [batch_size(*beam_size), num_heads, q_len, k_len]
            dk = tf.cast(tf.shape(K)[-1], tf.float32)
            scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
            if mask is not None:
                scaled_attention_logits = scaled_attention_logits + (mask * -1e9)
            attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
            output = tf.matmul(attention_weights, V)
            return output, attention_weights
        
        # def create_causal_mask(batch_size, num_heads, seq_len):
        def create_causal_mask(batch_size, seq_len, num_heads, is_4d=False, beam_size=None):
            """
            Creates a causal mask to prevent attention to future tokens.
            Args:
                seq_len: Length of the sequence.
            Returns:
                A causal mask of shape (seq_len, seq_len).
            """
            # mask = 1-tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)  # Lower triangular 
            mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
            mask = tf.expand_dims(tf.expand_dims(mask, 0), 0)
            # causal_mask = tf.expand_dims(mask, 0)  # Add batch dimension
            # causal_mask = tf.expand_dims(causal_mask, 1)  # Add head dimension
            # causal_mask = tf.tile(causal_mask, [batch_size, num_heads, 1, 1])
            # return causal_mask
            if is_4d:
                # 四维情况下复制到所有batch和beam
                mask = tf.tile(mask, [batch_size * beam_size, num_heads, 1, 1])
            else:
                # 三维情况下只复制到batch
                mask = tf.tile(mask, [batch_size, num_heads, 1, 1])
            return mask
        
        training = tf.constant(training, dtype=tf.bool)
        with tf.variable_scope(f"multi_head_attention", reuse=tf.AUTO_REUSE):
            # 判断输入维度
            is_4d = len(queries.get_shape().as_list()) == 4
            input_shape = tf.shape(queries)
            
            if is_4d:
                batch_size = input_shape[0]
                beam_size = input_shape[1]
                seq_len = input_shape[2]
                depth = queries.get_shape().as_list()[-1]
            else:
                batch_size = input_shape[0]
                seq_len = input_shape[1]
                depth = queries.get_shape().as_list()[-1]
                beam_size = None
            # depth = queries.get_shape().as_list()[-1]
            # batch_size = tf.shape(queries)[0]
            # seq_len = tf.shape(queries)[1]
            Q = tf.layers.dense(queries, depth, use_bias=False)
            K = tf.layers.dense(keys, depth, use_bias=False)
            V = tf.layers.dense(values, depth, use_bias=False)

            # Q = split_heads(Q, num_heads)
            # K = split_heads(K, num_heads)
            # V = split_heads(V, num_heads)
            
            # 分离头并获取维度信息
            Q, is_4d = split_heads(Q, num_heads)
            K, _ = split_heads(K, num_heads)
            V, _ = split_heads(V, num_heads)
            
            mask = None
            if causal_mask:
                # mask = create_causal_mask(batch_size, num_heads, seq_len)
                mask = create_causal_mask(batch_size, seq_len, num_heads, is_4d, beam_size)
                print(f"mask shape: {mask.shape}")
            elif use_seq_mask: # 输入为 (?,T) 值为0 1，代表有效位置，T=keys长度
                mask = tf.expand_dims(tf.expand_dims(seq_mask, 1), 1) # (?,1,1,T)
                query_len = queries.shape[-2]
                if is_4d:
                    mask = tf.tile(mask, [beam_size, num_heads, query_len, 1]) # (?*beam_size,num_heads,query_len,T)
                else:
                    mask = tf.tile(mask, [1, num_heads, query_len, 1]) # (?,num_heads,query_len,T)
                print(f"seq_mask shape: {mask.shape}")

            scaled_attention, attention_weights = scaled_dot_product_attention(Q, K, V, mask=mask)
            scaled_attention = tf.transpose(scaled_attention, [0, 2, 1, 3])

            # concat_attention = tf.reshape(scaled_attention, [tf.shape(queries)[0], -1, depth])
            if is_4d:
                concat_attention = tf.reshape(scaled_attention, 
                                       [batch_size, beam_size, seq_len, depth])
            else:
                concat_attention = tf.reshape(scaled_attention, 
                                        [batch_size, seq_len, depth])
                
            output = tf.layers.dense(concat_attention, depth)
            output = tf.cond(training, lambda: tf.nn.dropout(output, rate=dropout_rate), lambda: output)
        return output
    
def feed_forward_network(dim, hidden_dim, dropout_rate, training=False):
    def ffn(x, training=training):
        training = tf.constant(training, dtype=tf.bool)
        with tf.variable_scope(f"feed_forward_network", reuse=tf.AUTO_REUSE):
            x = tf.layers.dense(x, hidden_dim, activation=tf.nn.relu)
            x = tf.layers.dense(x, dim)
            # x = tf.nn.dropout(x, rate=dropout_rate)
            x = tf.cond(training, lambda: tf.nn.dropout(x, rate=dropout_rate), lambda: x)
            return x
    return ffn
    
class TransformerLayer:
    def __init__(self, name, dim, num_heads, hidden_dim, dropout_rate, training=False):
        super(TransformerLayer, self).__init__()
        self.name = name
        self.dim = dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
        self.mha = multi_head_attention
        self.ffn = feed_forward_network(dim, hidden_dim, dropout_rate)
        
    def forward(self, x, training, causal_mask=False):
        with tf.variable_scope(f"{self.name}", reuse=tf.AUTO_REUSE):
            # attn_output = self.mha(x, x, x, self.num_heads, self.dropout_rate, training=training, causal_mask=True)
            attn_output = self.mha(x, x, x, self.num_heads, self.dropout_rate, training=training, causal_mask=causal_mask)
            out1 = layer_norm(x + attn_output)
            
            ffn_output = self.ffn(out1, training=training)
            out2 = layer_norm(out1 + ffn_output)
        
        return out2

class DecoderLayer:
    def __init__(self, name, dim, num_heads, hidden_dim, dropout_rate, training=False):
        super(DecoderLayer, self).__init__()
        self.name = name
        self.dim = dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
        self.self_attention = multi_head_attention
        self.cross_attention = multi_head_attention
        self.ffn = feed_forward_network(dim, hidden_dim, dropout_rate)

    def forward(self, x, enc_output, training, causal_mask=False):
        with tf.variable_scope(f"{self.name}", reuse=tf.AUTO_REUSE):
            # 1. Self Attention
            self_attn_output = self.self_attention(x, x, x, 
                                                 self.num_heads, 
                                                 self.dropout_rate, 
                                                 training=training, 
                                                 causal_mask=True)
            out1 = layer_norm(x + self_attn_output)
            
            # 2. Cross Attention
            cross_attn_output = self.cross_attention(out1,
                                                   enc_output,
                                                   enc_output,
                                                   self.num_heads,
                                                   self.dropout_rate,
                                                   training=training)
            out2 = layer_norm(out1 + cross_attn_output)
            
            # 3. Feed Forward
            ffn_output = self.ffn(out2, training=training)
            out3 = layer_norm(out2 + ffn_output)
            
            return out3
    
class StackedTransformerModel():
    def __init__(self, num_layers, dim, num_heads, hidden_dim, dropout_rate, k, training=False):
        super(StackedTransformerModel, self).__init__()
        self.num_layers = num_layers
        self.k = k
        self.dim = dim
        self.layers = [TransformerLayer(f"transformer_layer_{i}", dim, num_heads, hidden_dim, dropout_rate, training=training) for i in range(num_layers)]
        self.decoder_layers = [DecoderLayer(f"position_layer_{i}", dim, num_heads, hidden_dim, dropout_rate, training=training) for i in range(num_layers)]
        
    def forward(self, hidden_states, training):
        for i in range(self.num_layers):
            hidden_states = self.layers[i].forward(hidden_states, training=training)
        return hidden_states
    
    def forward_decoder(self, hidden_states, item_embedding, training):
        for i in range(self.num_layers):
            item_embedding = self.decoder_layers[i].forward(item_embedding, hidden_states, training=training)
        return item_embedding

class Evaluator():
    def __init__(self, num_layers, dim, num_heads, hidden_dim, dropout_rate, k, training=False):
        super(Evaluator, self).__init__()
        self.num_layers = num_layers
        self.k = k
        self.dim = dim
        self.position = tf.get_variable('s', shape=[self.k, self.dim], initializer=tf.random_normal_initializer())
        self.layers = [TransformerLayer(f"transformer_layer_{i}", dim, num_heads, hidden_dim, dropout_rate, training=training) for i in range(num_layers)]
        
    def forward(self, hidden_states, training):
        pos_embedding = tf.reshape(self.position, [1, self.k, self.dim])
        pos_embedding = tf.tile(pos_embedding, [tf.shape(hidden_states)[0], 1, 1])
        for i in range(self.num_layers):
            hidden_states = self.layers[i].forward(hidden_states, training=training)
        return hidden_states


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

    # def _calc_pointwise_reward(self):
    #     # 没有dense值，暂时反离散化
    #     pctr = tf.cast(self._dense_value_dict["context_pctr"], dtype=tf.float32) / tf.constant(500.0)
    #     pvtr = tf.cast(self._dense_value_dict["context_pvtr"], dtype=tf.float32) / tf.constant(500.0)
    #     pwtd = tf.cast(self._dense_value_dict["context_pwtd"], dtype=tf.float32) / tf.constant(100000.0 / 300.0)
    #     pwtr = tf.cast(self._dense_value_dict["context_pwtr"], dtype=tf.float32) / tf.constant(500.0 * 0.01)
    #     pltr = tf.cast(self._dense_value_dict["context_pltr"], dtype=tf.float32) / tf.constant(500.0 * 0.01)
    #     print(f"pctr shape: {pctr.shape}, pvtr shape: {pvtr.shape}, pwtd shape: {pwtd.shape}")

    def _mlp_layer(self,
                  scope_name,
                  hidden_states: tf.Tensor,
                  hidden_units: list,
                  activation=tf.nn.relu) -> tf.Tensor:
        with tf.variable_scope(f"{scope_name}_mlp_layer", reuse=tf.AUTO_REUSE):
            for i, hidden_unit in enumerate(hidden_units):
                hidden_states = tf.layers.dense(hidden_states, hidden_unit, activation=activation, use_bias=True)
        return hidden_states

    def _fea_seq_attn(self, query, input_dicts, fea_names, merge_type):
        # 计算序列特征attn, query shape: (?,list_len,d)
        rt_seq_mha_list = []
        for fea in fea_names:
            # 这里假设最后一维全0是padding
            mask = tf.logical_not(tf.reduce_all(tf.equal(input_dicts[fea], 0.0), axis=-1)) # (?,T)
            mask = tf.cast(mask, dtype=tf.float32)
            # print(f"{fea} mask shape: {mask.shape}")
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
            explore_embs  = tf.concat([input_dicts[k] for k in input_dicts if k in explore_profile_fea_names], axis=2)
            explore_embs  = tf.reduce_mean(explore_embs, axis=1)
            explore_embs = tf.tile(tf.expand_dims(explore_embs, axis=1),[1,list_dim,1]) if self.training else explore_embs
            print("explore_embs shape ", photo_embs.shape)
            source_embs    = tf.concat([input_dicts[k] for k in input_dicts if k in source_fea_names], axis=-1)
            source_embs = tf.tile(tf.expand_dims(source_embs, axis=1),[1,list_dim,1]) if self.training else source_embs
            print("source_embs shape ", source_embs.shape) # 

            common_embs = tf.concat([source_embs, user_embs, explore_embs, photo_embs], axis=-1)
            if not self.training:
                '''
                    infer 时需要注意实际请求的 batch size = items 长度, 但计算图 batch 的 shape 由 uni_predict_fused 中 executor_batchsizes 决定,
                    如600。为了通过计算图编译, 需要 rashape (1, -1, dim), 实际 -1 为请求端发送 items 长度, user 特征猜测被已经框架 tile 到 items 长度
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
            print(f"sequences shape: {sequences.shape}, scores shape: {scores.shape}")
            
            # repeat encoder_output for each beam 
            encoder_output = tf.tile(tf.expand_dims(encoder_output, axis=1), [1, beam_size, 1, 1])  # [batch_size, beam_size, vocab_size, dim])
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
                print(f"sequences shape: {sequences.shape}") # (?, beam_size, seq_length)
                # 提取已生成序列的embedding
                gather_indices = tf.stack([batch_indices, beam_indices, sequences], axis=-1) # [batch_size, beam_size, seq_length, 3]
                decoder_input = tf.gather_nd(encoder_output, gather_indices)  # [batch_size, beam_size, seq_length, dim]
                # decoder forward
                decoder_output = model.forward_decoder(encoder_output, decoder_input, training=self.training)  # [batch_size, beam_size, seq_length, dim]
                decoder_output = self._mlp_layer("decoder_output", decoder_output, [32], activation=None) # (?,candidates_size,32)
                print("xxx")
                # 计算 logits
                logits = tf.matmul(encoder_output, tf.transpose(decoder_output, perm=[0, 1, 3, 2])) # [batch_size, beam_size, vocab_size, seq_length]
                next_token_logits = logits[:, :, :, -1]  # [batch_size, beam_size, vocab_size]
                # 选择下一个token

                tau = 10 # tau 控制的随机性
                next_token_probs = tf.nn.softmax(next_token_logits/tau, axis=-1)  # [batch_size, beam_size, vocab_size]
                log_probs = tf.math.log(next_token_probs+1e-9)  # 转换为 log 概率

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
                )  # [batch_size, beam_size, vocab_size]
                print("used_token shape",used_token.shape)

                one_hot = tf.one_hot(sequences, vocab_size, on_value=True, off_value=False)  # shape: [batch_size, beam_size, seq_len, vocab_size]
                used_token_tmp = tf.reduce_any(one_hot, axis=2)  # shape: [batch_size, beam_size, vocab_size] 出现过的词都为true
                used_token = tf.logical_or(used_token_tmp, used_token) # 合并使用过的词
                
                # 根据used_token将已生成tokn的分数设为-inf
                log_probs = tf.where(used_token, tf.fill(tf.shape(log_probs), float('-inf')), log_probs)  # [batch_size, beam_size, vocab_size]
                # 计算总分数  (当前路径分数 + 新 token 的分数)
                scores = tf.expand_dims(scores, axis=-1) + log_probs  # [batch_size, beam_size, vocab_size]
                
                # topk最高分数 
                scores_flat = tf.reshape(scores, [batch_size, -1]) # [batch_size, beam_size * vocab_size]
                top_k_scores, top_k_indices = tf.math.top_k(scores_flat, k=beam_size, sorted=True)  # [batch_size,beam_size)
                # 更新序列和分数
                beam_indices = top_k_indices // vocab_size  # [batch_size, beam_size]
                token_indices = top_k_indices % vocab_size  # [batch_size, beam_size]

                # 更新生成的序列的beam索引
                batch_indices = tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), [1, beam_size])  # [batch_size, beam_size]
                gather_indices = tf.stack([batch_indices, beam_indices], axis=-1)  # [batch_size, beam_size, 2]
                # 获取要更新的beam
                selected_sequences = tf.gather_nd(sequences, gather_indices)  # [batch_size, beam_size, seq_length]
                # 将新token添加到beam序列末尾
                sequences = tf.concat([selected_sequences, tf.expand_dims(token_indices, axis=-1)], axis=-1) # [batch_size, beam_size, seq_length + 1]
                scores = top_k_scores
                    
            # 从 beam_size 条路径中选择分数最高的路径
            # best_sequence_indices = tf.expand_dims(tf.argmax(scores, axis=1), axis=-1)  # [batch_size,1]
            # best_sequences = tf.gather_nd(sequences, tf.concat([tf.expand_dims(tf.cast(tf.range(batch_size),dtype=tf.int64), axis=-1), best_sequence_indices], axis=-1)) # [batch_size, max_length]
            # generated_sequence = best_sequences[:,1:]
            generated_sequence = sequences[:,:,1:]
            # print("beam search end!")
            # self.print_ops.append(tf.print("used_token ", used_token, summarize=10, output_stream=sys.stdout))
            return logits, generated_sequence
          
        with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
            input_dicts = self._parameters_dict
            common_embs = self._get_shared_features(input_dicts, self.candidates_size) # (?,30,532)
            batch_size = tf.shape(common_embs)[0] # 1
            hidden_states = self._mlp_layer("mlp_layer_1", common_embs, [64, 32]) # (?,candidates_size,32)
            pt_modle_input = hidden_states
            # 添加特殊token的embedding
            pad_embedding = tf.tile(tf.expand_dims(self.pad_embedding, axis=0), #(?,1,32)
                                [batch_size, 1, 1])
            sos_embedding = tf.tile(tf.expand_dims(self.sos_embedding, axis=0),  #(?,1,32)
                                [batch_size, 1, 1])
            eos_embedding = tf.tile(tf.expand_dims(self.eos_embedding, axis=0), #(?,1,32)
                                [batch_size, 1, 1])

            hidden_states = tf.concat([pad_embedding, sos_embedding, hidden_states, eos_embedding], axis=1) # (?,candidates_size+3,32)
            print("hidden_states shape", hidden_states.shape)
            # 初始化transformer模型
            model = StackedTransformerModel(num_layers=1, dim=32, num_heads=4, hidden_dim=128, dropout_rate=0.1, k=6)
            hidden_states = model.forward(hidden_states, training=self.training)
            encoder_output = hidden_states
            print("encoder output shape ",encoder_output.shape) # (?,candidates_size+3,32)
            # 单点 mlp
            # pt_output = self._mlp_layer("pt", pt_modle_input, [64, 32]) # (?,candidates_size,32)
            # pt_output = self._mlp_layer("pt_output", pt_output, [1], activation=None) # (?,candidates_size,1)
            # pt_output = tf.squeeze(pt_output, -1) # (?,candidates_size)
            # pt_logit = tf.nn.sigmoid(pt_output) # (?,candidates_size)
            if self.training:
                label_dicts = self._label_value_dict
                rerank_label = label_dicts['context_info__real_show_list']
                rerank_label = tf.reshape(rerank_label, [-1, self.candidates_size])
                rerank_label = rerank_label[:,:self.list_size] # (?,list_size)，截断为 list_size 个
                indices_shape = tf.shape(rerank_label)
                rerank_label = tf.cast(rerank_label,dtype=tf.int32) #(?,list_size)
                
                col_indices = tf.tile(tf.expand_dims(tf.range(indices_shape[1]),0),[indices_shape[0],1])+2 # (?,list_size) 从第2个起
                # self.print_ops.append(tf.print("col_indices shape ", tf.shape(col_indices), summarize=10, output_stream=sys.stdout))
                # self.print_ops.append(tf.print("rerank shape ", tf.shape(rerank_label), summarize=10, output_stream=sys.stdout))
                print("rerank label shape ",rerank_label.shape)
                print("col indices shape ",col_indices.shape)
                rank_indices = tf.cast(col_indices * rerank_label,dtype=tf.int32) # label为0 1，过滤了未曝光的index

                realshow_weight = label_dicts['fountain_fulllink_rerank_realshow_label_weight_list']
                wtd_weight = tf.cast(label_dicts['fountain_wtd_label_list'], dtype=tf.float32)
                ltr_label = tf.cast(label_dicts['fountain_ltr_label_list'], dtype=tf.float32)
                ltr_weight = tf.cast(label_dicts['fountain_ltr_weight_list'], dtype=tf.float32) * ltr_label
                realshow_weight = tf.reshape(realshow_weight, [-1, self.candidates_size])
                all_weight = tf.clip_by_value(realshow_weight, 0, 100)/10.0
                realshow_weight = realshow_weight[:,:self.list_size] # (?,list_size)
                item_weight = tf.clip_by_value(realshow_weight, 0, 100)/10.0
                # item_weight = wtd_weight * 3.0 + ltr_weight + 1
                # item_weight = item_weight[:, : self.list_size]
                # item_weight = tf.clip_by_value(item_weight, 0, 10)
                item_weight = tf.clip_by_value(item_weight, 0, 1) # 不使用weight
                # self.print_ops.append(tf.print("item_weight: ", item_weight, summarize=-1, output_stream=sys.stdout))

                # 单点 weighted log loss
                # realshow_label = label_dicts['context_info__real_show_list'] # 形如 [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                # realshow_label = tf.reshape(realshow_label, [-1, self.candidates_size])
                # pt_mask = tf.cast(tf.greater(realshow_label, 0), tf.float32) # [?, candidates_size] 暂时只用曝光样本
                # pt_label = tf.cast(label_dicts["fountain_click_label_list"],dtype=tf.int32)
                # pt_label = tf.reshape(pt_label, [-1, self.candidates_size])
                # pt_loss = tf.losses.log_loss(labels=pt_label, predictions=pt_logit, weights=all_weight * pt_mask)
                # pt_loss = tf.reduce_mean(pt_loss)

                # 筛选正样本序列
                # total_play_time = tf.reduce_sum(realshow_weight, axis=-1, keepdims=True) - self.candidates_size
                # is_seq_play = tf.math.greater_equal(total_play_time, 20) # (?, 1) 60%分位数
                # fountain_click_sum = tf.reduce_sum(pt_label, axis=-1, keepdims=True) # (?, 1)
                # is_seq_click = tf.math.greater_equal(fountain_click_sum, 2) # (?, 1)
                # seq_reward_mask = tf.cast(tf.logical_and(is_seq_play, is_seq_click), dtype=tf.float32) # (?, 1)

                sos_token = tf.constant(1, shape=[1, 1], dtype=tf.int32)
                sos_token = tf.tile(sos_token, [batch_size,1]) #(?,1)
                eos_token = tf.constant(self.list_size + 2, shape=[1, 1], dtype=tf.int32)
                eos_token = tf.tile(eos_token, [batch_size,1])
                pad_token = tf.tile(tf.constant(0, shape=[1, 1], dtype=tf.int32), [batch_size, 1]) #(?,1) all 0
                inputs = tf.concat([sos_token, rank_indices], axis=1) # (?,list_size+1)
                outputs = tf.concat([rank_indices, eos_token], axis=1) # (?,list_size+1)

                print("inputs shape", inputs.shape)
                print("outputs shape", outputs.shape)

                # 从hidden_states查对应emb表示
                batch_indices = tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), [1, self.list_size+1]) # (?, self.list_size+1)
                gather_indices = tf.stack([batch_indices, inputs], axis=-1) # (?, self.list_size+1, 2)
                item_embeddings = tf.gather_nd(hidden_states, gather_indices) # (?,candidates_size+3,32) 中查找对应 list idx 的 emb
                print("item_embeddings shape", item_embeddings.shape) # (?,list_size+1,32)

                # decoder 出的表征应当与实际结果相近，因此与双向 transformer 编码后的表示做选择
                # todo: 这里也可以做其他的匹配
                item_embedding = model.forward_decoder(hidden_states, item_embeddings, training=True)
                item_embedding = self._mlp_layer("decoder_output", item_embedding, [32], activation=None) # (?,candidates_size,32)
                print("hidden_states shape ", hidden_states.shape) # (?, 33, 32)
                print("item_embedding shape ", item_embedding.shape) # (?, self.list_size+1, 32)

                item_embedding_trans = tf.transpose(item_embedding,  perm=[0, 2, 1])
                predict = tf.matmul(hidden_states, item_embedding_trans) # (?, candidates_size+3, 7)
                print("predict shape", predict.shape)
                tau = 1
                predict = tf.nn.softmax(predict/tau, axis=1)

                norm_outputs = hidden_states / tf.norm(hidden_states, axis=2, keepdims=True)
                cosine_scores_outputs = tf.matmul(norm_outputs, tf.transpose(norm_outputs, perm=[0, 2, 1]))
                print("cosine_scores_outputs shape", cosine_scores_outputs.shape)
                cl_loss_outputs = self._contrastive_loss(cosine_scores_outputs, seqlen=self.candidates_size+3)
                cl_loss = cl_loss_outputs

                predict = tf.transpose(predict,  perm=[0, 2, 1]) # (?, list_size+1, candidates_size+3)
                print("predict shape", predict.shape)
                output_indices = tf.expand_dims(outputs, axis=2) # (?,list_size+1,1)
                pos_output = tf.batch_gather(predict, output_indices) # (?,list_size+1,1) 拿到真实index对应的score, 非全局emb Matrix 需要使用batch_gather
                vector = tf.zeros([batch_size, 1], dtype=tf.int32) # mask EOS token
                rerank_label = tf.concat([rerank_label, vector], axis=1)
                print("pos_output shape", pos_output.shape)
                print("outputs shape", outputs.shape)
                vector = tf.ones([batch_size, 1], dtype=tf.float32)
                item_weight = tf.concat([item_weight, vector], axis=1)
                item_weight = tf.reduce_sum(item_weight,axis=-1)
                # print("rerank_label shape", rerank_label.shape)

                pos_output = tf.squeeze(pos_output, axis=-1) #(?,6)
                rerank_label = tf.cast(rerank_label,dtype=tf.float32)
                valid_pos_output = -tf.log(pos_output+1e-9)*rerank_label

                valid_counts = tf.reduce_sum(rerank_label, axis=-1)+1e-9
                gen_loss = tf.reduce_sum(valid_pos_output, axis=-1)*item_weight/valid_counts
                # gen_loss = gen_loss * seq_reward_mask
                print("gen_loss shape", gen_loss.shape)
                gen_loss = tf.reduce_mean(gen_loss)
                
                return None, gen_loss, cl_loss, pos_output, predict
            
            else:
                # 初始化解码过程
                max_length = 2  # 目标生成长度
                sos_token = tf.tile(tf.constant(1, shape=[1, 1], dtype=tf.int32), [batch_size, 1]) #(?,1) all 1
                eos_value = tf.shape(hidden_states)[1]-1
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
                        decoder_output = model.forward_decoder(encoder_output, decoder_input, training=True) #(?,33,32)
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
                    beam_size = 10
                    logits, generated_sequence = beam_search(
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