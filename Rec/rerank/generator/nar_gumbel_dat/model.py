import tensorflow as tf
from feature_attr_extract import user_fea_names, photo_fea_names, source_fea_names
from modules_ import *
import sys


def layer_norm(x, scope=None, epsilon=1e-6):
    with tf.variable_scope(scope or "layer_norm", reuse=tf.AUTO_REUSE):
        gamma = tf.get_variable("gamma", [x.get_shape()[-1]], initializer=tf.ones_initializer())
        beta = tf.get_variable("beta", [x.get_shape()[-1]], initializer=tf.zeros_initializer())
        mean, variance = tf.nn.moments(x, axes=[-1], keep_dims=True)
        normalized = (x - mean) / tf.sqrt(variance + epsilon)
        output = gamma * normalized + beta
    return output


def multi_head_attention(name, queries, keys, values, num_heads, dropout_rate, training=False):
        def split_heads(x, num_heads):
            batch_size = tf.shape(x)[0]
            depth = x.get_shape().as_list()[-1] // num_heads
            reshaped = tf.reshape(x, [batch_size, -1, num_heads, depth])
            return tf.transpose(reshaped, [0, 2, 1, 3])

        def scaled_dot_product_attention(Q, K, V):
            matmul_qk = tf.matmul(Q, K, transpose_b=True)
            dk = tf.cast(tf.shape(K)[-1], tf.float32)
            scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
            attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
            output = tf.matmul(attention_weights, V)
            return output, attention_weights

        training = tf.constant(training, dtype=tf.bool)
        with tf.variable_scope(f"{name}", reuse=tf.AUTO_REUSE):
            depth = queries.get_shape().as_list()[-1]
            Q = tf.layers.dense(queries, depth, use_bias=False)
            K = tf.layers.dense(keys, depth, use_bias=False)
            V = tf.layers.dense(values, depth, use_bias=False)

            Q = split_heads(Q, num_heads)
            K = split_heads(K, num_heads)
            V = split_heads(V, num_heads)

            scaled_attention, attention_weights = scaled_dot_product_attention(Q, K, V)
            scaled_attention = tf.transpose(scaled_attention, [0, 2, 1, 3])

            concat_attention = tf.reshape(scaled_attention, [tf.shape(queries)[0], -1, depth])
            output = tf.layers.dense(concat_attention, depth)
            output = tf.cond(training, lambda: tf.nn.dropout(output, rate=dropout_rate), lambda: output)
        return output

def feed_forward_network(dim, hidden_dim, dropout_rate, training=False):
    def ffn(x, training=training):
        training = tf.constant(training, dtype=tf.bool)
        with tf.variable_scope(f"feed_forward_network", reuse=tf.AUTO_REUSE):
            x = tf.layers.dense(x, hidden_dim, activation=tf.nn.relu)
            x = tf.layers.dense(x, dim)
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

    def forward(self, x, training):
        with tf.variable_scope(f"{self.name}", reuse=tf.AUTO_REUSE):
            attn_output = self.mha("self_atten", x, x, x, self.num_heads, self.dropout_rate, training=training)
            out1 = layer_norm(x + attn_output)
            ffn_output = self.ffn(out1, training=training)
            out2 = layer_norm(out1 + ffn_output)
        return out2

class PositionLayer:
    def __init__(self, name, dim, num_heads, hidden_dim, dropout_rate, training=False):
        super(PositionLayer, self).__init__()
        self.name = name
        self.dim = dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.self_attention = multi_head_attention
        self.cross_attention = multi_head_attention
        self.ffn = feed_forward_network(dim, hidden_dim, dropout_rate)

    def forward(self, x, enc_output, training):
        with tf.variable_scope(f"{self.name}", reuse=tf.AUTO_REUSE):
            attn_output = self.self_attention(f"self_atten", x, x, x, self.num_heads, self.dropout_rate, training=training)
            out1 = layer_norm(x + attn_output)
            cross_attn_output = self.cross_attention(f"cross_attn", out1, enc_output, enc_output, self.num_heads, self.dropout_rate, training=training)
            out2 = layer_norm(out1 + cross_attn_output)
            ffn_output = self.ffn(out2, training=training)
            out3 = layer_norm(out2 + ffn_output)
        return out3


class StackedTransformerModel():
    def __init__(self, num_layers, dim, num_heads, hidden_dim, dropout_rate, k, training=False):
        super(StackedTransformerModel, self).__init__()
        self.num_layers = num_layers
        self.k = 6
        self.dim = dim
        self.position = tf.get_variable('s', shape=[self.k, self.dim], initializer=tf.random_normal_initializer())
        self.layers = [TransformerLayer(f"transformer_layer_{i}", dim, num_heads, hidden_dim, dropout_rate, training=training) for i in range(num_layers)]
        self.position_layers = [PositionLayer(f"position_layer_{i}", dim, num_heads, hidden_dim, dropout_rate, training=training) for i in range(num_layers)]

    def forward(self, hidden_states, training):
        pos_embedding = tf.reshape(self.position, [1, self.k, self.dim])
        pos_embedding = tf.tile(pos_embedding, [tf.shape(hidden_states)[0], 1, 1])
        for i in range(self.num_layers):
            hidden_states = self.layers[i].forward(hidden_states, training=training)
            pos_embedding = self.position_layers[i].forward(pos_embedding, hidden_states, training=training)
        return hidden_states, pos_embedding


class Evaluator():
    def __init__(self, num_layers, dim, num_heads, hidden_dim, dropout_rate, k, training=False):
        super(Evaluator, self).__init__()
        self.num_layers = num_layers
        self.k = 6
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
    def __init__(self, parameters_dict, label_value_dict, loss_names, print_ops, dim=32, extra_param_dict=None):
        self._parameters_dict = parameters_dict
        self._label_value_dict = label_value_dict
        self.dim = dim
        self.print_ops = print_ops
        self.loss_names = loss_names
        self.position_embeddings = tf.get_variable(
            name='position_embeddings',
            shape=[6, 32],
            initializer=tf.random_normal_initializer()
        )
        self.num_vertex = 60
        self.vertex_position_embeddings = tf.get_variable(
            name='vertex_position_embeddings',
            shape=[self.num_vertex, 32],
            initializer=tf.random_normal_initializer()
        )
        self.Wq = tf.get_variable(name='wq', shape=[32, 32], initializer=tf.random_normal_initializer())
        self.Wk = tf.get_variable(name='wk', shape=[32, 32], initializer=tf.random_normal_initializer())

    def _mlp_layer(self,
                  scope_name,
                  hidden_states: Tensor,
                  hidden_units: list,
                  activation=tf.nn.relu) -> Tensor:
        with tf.variable_scope(f"{scope_name}_mlp_layer", reuse=tf.AUTO_REUSE):
            for i, hidden_unit in enumerate(hidden_units):
                hidden_states = tf.layers.dense(hidden_states, hidden_unit, activation=activation, use_bias=True)
        return hidden_states

    def _get_shared_features(self, input_dicts) -> tuple:
        with tf.variable_scope("share_bottom", reuse=tf.AUTO_REUSE):
            list_dim = 60
            user_embs = tf.concat([input_dicts[k] for k in input_dicts if k in user_fea_names], axis=-1)
            user_embs = tf.tile(tf.expand_dims(user_embs, axis=1), [1, list_dim, 1])
            photo_embs = tf.concat([input_dicts[k] for k in input_dicts if k in photo_fea_names], axis=-1)
            source_embs = tf.concat([input_dicts[k] for k in input_dicts if k in source_fea_names], axis=-1)
            source_embs = tf.tile(tf.expand_dims(source_embs, axis=1), [1, list_dim, 1])
            common_embs = tf.concat([user_embs, photo_embs, source_embs], axis=-1)
            return common_embs

    def _contrastive_loss(self, score_matrix, margin=0.8, seqlen=6):
       gold_score = tf.linalg.diag_part(score_matrix)
       gold_score = tf.expand_dims(gold_score, axis=2)
       difference_matrix = gold_score - score_matrix
       loss_matrix = margin - difference_matrix
       loss_matrix = tf.nn.relu(loss_matrix)
       base_mask = tf.ones((seqlen, seqlen)) - tf.linalg.diag(tf.ones(seqlen))
       base_mask = tf.expand_dims(base_mask, axis=0)
       base_mask = tf.tile(base_mask, [tf.shape(score_matrix)[0], 1, 1])
       cl_loss = tf.reduce_mean(loss_matrix * base_mask)
       return cl_loss

    def _similarity_matrix(self, score_matrix, margin=0.0, seqlen=6):
       base_mask = tf.ones((seqlen, seqlen)) - tf.linalg.diag(tf.ones(seqlen))
       base_mask = tf.expand_dims(base_mask, axis=0)
       base_mask = tf.tile(base_mask, [tf.shape(score_matrix)[0], 1, 1])
       score_matrix = score_matrix * base_mask
       return score_matrix

    def transition_matrix(self, vertex_states, num_vertex):
        Q = tf.matmul(vertex_states, self.Wq)
        K = tf.matmul(vertex_states, self.Wk)
        d = tf.cast(tf.shape(vertex_states)[-1], tf.float32)
        logits = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(d)
        log_E = tf.nn.log_softmax(logits, axis=-1)
        return log_E

    def directed_acyclic_decoder(self, hidden_states):
        with tf.variable_scope("dat_decoder", reuse=tf.AUTO_REUSE):
            vertex_position_embeddings = tf.tile(tf.expand_dims(self.vertex_position_embeddings, axis=0), [tf.shape(hidden_states)[0], 1, 1])
            n_layers = 3
            num_vertex = self.num_vertex
            for i in range(n_layers):
                pe_self_atten = multi_head_attention('self_attention', vertex_position_embeddings, vertex_position_embeddings, vertex_position_embeddings, 4, 0.1, training=True)
                pe_cross_atten = multi_head_attention('cross_attention', pe_self_atten, hidden_states, hidden_states, 4, 0.1, training=True)
                pe_ffn = feed_forward_network(32, 128, 0.1, training=True)(pe_cross_atten, training=True)
                vertex_position_embeddings = pe_ffn
            vertex_states = vertex_position_embeddings
            transition_matrix = self.transition_matrix(vertex_states, num_vertex)
            vertex_states = tf.transpose(vertex_states, perm=[0, 2, 1])
            token_distribution = tf.matmul(hidden_states, vertex_states)
            token_distribution = tf.transpose(token_distribution, perm=[0, 2, 1])
            return transition_matrix, token_distribution

    def gumbel_softmax(self, logits, tau=1.0, hard=False, dim=-1):
        def sample_gumbel(shape):
            uniform_samples = tf.random_uniform(shape, minval=0, maxval=1)
            return -tf.log(-tf.log(uniform_samples + 1e-20) + 1e-20)
        gumbels = sample_gumbel(tf.shape(logits))
        gumbels = (logits + gumbels) / tau
        y_soft = tf.nn.softmax(gumbels, axis=dim)
        if hard:
            index = tf.argmax(y_soft, axis=dim)
            y_hard = tf.one_hot(index, depth=tf.shape(logits)[dim], dtype=logits.dtype)
            y_hard = tf.reshape(y_hard, tf.shape(logits))
            ret = tf.stop_gradient(y_hard - y_soft) + y_soft
        else:
            ret = y_soft
        return ret

    def weighted_log_loss(self, y_true, y_pred, weights):
        epsilon = 1e-15
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        log_loss = - weights * (y_true * tf.log(y_pred) + (1 - y_true) * tf.log(1 - y_pred))
        weighted_log_loss = log_loss
        return tf.reduce_mean(weighted_log_loss)

    def cal_gen_loss(self, predict, gen_model_weight, pos_indices, neg_indices, realshow_label):
        gen_model_weight = tf.batch_gather(gen_model_weight, pos_indices)
        pos_indices = tf.expand_dims(pos_indices, axis=2)
        pos_output = tf.squeeze(tf.batch_gather(predict, pos_indices), axis=-1)
        eps = 1e-9
        valid_counts = tf.cast(tf.reduce_sum(realshow_label, axis=-1), tf.float32)
        gen_loss = tf.reduce_sum(tf.log(pos_output + eps), axis=-1) / valid_counts
        gen_loss_mean = -tf.reduce_mean(gen_loss)
        return gen_loss_mean

    def logsumexp_keepdim(self, x, dim):
        m = tf.reduce_max(x, axis=dim, keepdims=True)
        mask = tf.equal(m, float('-inf'))
        m = tf.where(mask, tf.zeros_like(m), m)
        s = tf.reduce_sum(tf.exp(x - m), axis=dim, keepdims=True)
        s = tf.where(mask, tf.ones_like(s), s)
        return tf.log(s) + tf.where(mask, tf.fill(tf.shape(m), float('-inf')), m)

    def loop_function_noempty(self, last_f, links, match, real_show_mask_step):
        summed = last_f + links
        f_next = self.logsumexp_keepdim(summed, 1)
        f_next = tf.transpose(f_next, perm=[0, 2, 1])
        match = tf.where(real_show_mask_step, match, tf.fill(tf.shape(match), float('-inf')))
        f_next = f_next + match
        return f_next

    def loop_function_noempty_max(self, last_f, links, match):
        f_next = tf.reduce_max(last_f + links, axis=1)
        f_next = tf.expand_dims(f_next, -1) + match
        return f_next

    def dag_logsoftmax_gather_inplace(self, token_distribution, target_index, real_show_mask):
        logits = tf.nn.log_softmax(token_distribution, axis=-1)
        select_token_distribution = tf.gather(logits, target_index, batch_dims=2)
        select_token_distribution = tf.where(
            real_show_mask,
            select_token_distribution,
            tf.fill(tf.shape(select_token_distribution), float('-inf'))
        )
        token_distribution = logits
        return token_distribution, select_token_distribution

    def restore_valid_links(self, links):
        batch_size = tf.shape(links)[0]
        prelen = tf.shape(links)[1]
        translen = tf.shape(links)[2]
        row_indices = tf.range(prelen)[:, tf.newaxis]
        col_indices = tf.range(translen)[tf.newaxis, :]
        valid_links_idx = row_indices + col_indices + 1
        invalid_idx_mask = tf.greater_equal(valid_links_idx, prelen)
        valid_links_idx = tf.where(invalid_idx_mask,
                                tf.ones_like(valid_links_idx) * prelen,
                                valid_links_idx)
        batch_indices = tf.range(batch_size)[:, tf.newaxis, tf.newaxis]
        row_indices = tf.range(prelen)[tf.newaxis, :, tf.newaxis]
        batch_indices = tf.tile(batch_indices, [1, prelen, translen])
        row_indices = tf.tile(row_indices, [batch_size, 1, translen])
        valid_links_idx = tf.tile(valid_links_idx[tf.newaxis], [batch_size, 1, 1])
        indices = tf.stack([batch_indices, row_indices, valid_links_idx], axis=-1)
        indices = tf.reshape(indices, [-1, 3])
        res = tf.fill([batch_size, prelen, prelen + 1],
                    tf.cast(float('-inf'), links.dtype))
        res = tf.tensor_scatter_nd_update(
            res,
            indices,
            tf.reshape(links, [-1])
        )
        return res[:, :, :prelen]

    def tf_dag_loss(self, match_all, links, output_length, target_length, real_show_mask):
        batch_size = tf.shape(match_all)[0]
        prelen = tf.shape(match_all)[1]
        tarlen = 6
        links_shape = tf.shape(links)
        with tf.control_dependencies([tf.assert_equal(links_shape[1], links_shape[2])]):
            links = tf.identity(links)
        f_arr = []
        f_init = tf.fill([batch_size, prelen, 1], float('-inf'))
        first_match = tf.where(
            real_show_mask[:, 0:1, 0:1],
            match_all[:, 0:1, 0:1],
            tf.zeros_like(match_all[:, 0:1, 0:1])
        )
        f_init = tf.tensor_scatter_nd_update(
            f_init,
            tf.stack([tf.range(batch_size), tf.zeros([batch_size], dtype=tf.int32), tf.zeros([batch_size], dtype=tf.int32)], axis=1),
            first_match[:, 0, 0]
        )
        f_arr.append(f_init)
        match_chunks = tf.split(match_all, tarlen, axis=-1)
        for k in range(1, tarlen):
            real_show_mask_step = real_show_mask[:, :, k:k+1]
            f_now = self.loop_function_noempty(f_arr[-1], links, match_chunks[k], real_show_mask_step)
            f_arr.append(f_now)
        f_all = tf.concat(f_arr, axis=-1)
        batch_indices = tf.range(batch_size)
        output_length = tf.fill([batch_size], output_length - 1)
        target_length = tf.fill([batch_size], target_length - 1)
        loss_result = tf.gather_nd(
            f_all,
            tf.stack([
                batch_indices,
                output_length,
                target_length
            ], axis=1)
        )
        invalid_mask = tf.logical_or(tf.is_nan(loss_result), tf.is_inf(loss_result))
        loss_result_valid = tf.where(invalid_mask, tf.zeros_like(loss_result), loss_result)
        valid_ratio = 1.0 - tf.reduce_mean(tf.cast(invalid_mask, tf.float32))
        mean_loss_valid = -tf.reduce_mean(loss_result_valid)
        return loss_result_valid, mean_loss_valid

    def model(self):
        with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
            input_dicts = self._parameters_dict
            common_embs = self._get_shared_features(input_dicts)
            hidden_states = self._mlp_layer("mlp_layer_1", common_embs, [64, 32])
            dim = 32
            num_heads = 4
            hidden_dim = 128
            dropout_rate = 0.1
            num_layers = 3
            k = 6
            model = StackedTransformerModel(num_layers, dim, num_heads, hidden_dim, dropout_rate, k)
            hidden_states, pos_embedding = model.forward(hidden_states, training=True)
            pos_embedding_trans = tf.transpose(pos_embedding, perm=[0, 2, 1])
            predict_ori = tf.matmul(hidden_states, pos_embedding_trans)

            tau = 0.05
            predict = self.gumbel_softmax(predict_ori, tau, hard=True, dim=1)
            predict_ori = tf.nn.softmax(predict_ori / tau, axis=1)
            norm_rep = pos_embedding / tf.norm(pos_embedding, axis=2, keepdims=True)
            cosine_scores_rep = tf.matmul(norm_rep, tf.transpose(norm_rep, perm=[0, 2, 1]))
            cl_loss_pad = self._contrastive_loss(cosine_scores_rep)

            norm_outputs = hidden_states / tf.norm(hidden_states, axis=2, keepdims=True)
            cosine_scores_outputs = tf.matmul(norm_outputs, tf.transpose(norm_outputs, perm=[0, 2, 1]))
            cl_loss_outputs = self._contrastive_loss(cosine_scores_outputs, seqlen=60)
            cl_loss = cl_loss_pad + cl_loss_outputs

            transition_matrix, token_distribution = self.directed_acyclic_decoder(hidden_states)

        predict = tf.transpose(predict, perm=[0, 2, 1])
        predict_ori = tf.transpose(predict_ori, perm=[0, 2, 1])
        generator_embeding = tf.matmul(predict, common_embs)
        item_embedding_gen = hidden_states

        num_vertex = self.num_vertex
        target_len = 6
        label_dicts = self._label_value_dict
        batch_size = tf.shape(item_embedding_gen)[0]
        batch_indices = tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), [1, 6])
        rerank_wtd = label_dicts["fountain_wtd_label_list"]
        rerank_wtd_label = tf.reshape(rerank_wtd, [-1, 60])
        rerank_wtd = rerank_wtd_label[:, :6]
        wtd_label = tf.cast(tf.math.greater(rerank_wtd, 0.05), tf.int32)

        rerank_ltr = label_dicts["fountain_ltr_label_list"]
        rerank_ltr = tf.reshape(rerank_ltr, [-1, 60])
        rerank_ltr = rerank_ltr[:, :6]
        ltr_label = tf.cast(rerank_ltr, tf.int32)

        rerank_weight = label_dicts["fountain_fulllink_rerank_realshow_label_weight_list"]
        rerank_weight = tf.reshape(rerank_weight, [-1, 60])

        realshow_label = label_dicts['context_info__real_show_list']
        realshow_label = tf.reshape(realshow_label, [-1, 60])
        realshow_label = realshow_label[:, :6]

        rerank_label = tf.math.logical_or(
            tf.math.equal(wtd_label, 1),
            tf.math.equal(ltr_label, 1)
        )
        rerank_label = tf.cast(rerank_label, dtype=tf.int32)
        indices_shape = tf.shape(rerank_label)
        col_indices = tf.tile(tf.expand_dims(tf.range(indices_shape[1]), 0), [indices_shape[0], 1])
        rank_indices = tf.cast(col_indices * rerank_label, dtype=tf.int32)
        rank_indices = tf.expand_dims(rank_indices, axis=1)
        target_index = tf.tile(rank_indices, [1, num_vertex, 1])

        real_show_mask = realshow_label
        real_show_mask = tf.cast(real_show_mask, dtype=tf.bool)
        real_show_mask = tf.expand_dims(real_show_mask, axis=1)
        real_show_mask = tf.tile(real_show_mask, [1, num_vertex, 1])
        token_distribution, select_token_distribution = self.dag_logsoftmax_gather_inplace(token_distribution, target_index, real_show_mask)
        valid_links = self.restore_valid_links(transition_matrix)
        _, dat_loss = self.tf_dag_loss(select_token_distribution, valid_links, num_vertex, target_len, real_show_mask)

        with tf.variable_scope("evaluator", reuse=tf.AUTO_REUSE):
            input_dicts = self._parameters_dict
            common_embs = self._get_shared_features(input_dicts)
            batch_size = tf.shape(common_embs)[0]
            label_dicts = self._label_value_dict
            batch_indices = tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), [1, 6])
            rerank_wtd = label_dicts["fountain_wtd_label_list"]
            rerank_wtd_label = tf.reshape(rerank_wtd, [-1, 60])
            rerank_wtd = rerank_wtd_label[:, :6]
            wtd_label = tf.cast(tf.math.greater(rerank_wtd, 0.05), tf.int32)

            rerank_ltr = label_dicts["fountain_ltr_label_list"]
            rerank_ltr = tf.reshape(rerank_ltr, [-1, 60])
            rerank_ltr = rerank_ltr[:, :6]
            ltr_label = tf.cast(rerank_ltr, tf.int32)

            rerank_weight = label_dicts["fountain_fulllink_rerank_realshow_label_weight_list"]
            rerank_weight = tf.reshape(rerank_weight, [-1, 60])

            realshow_label = label_dicts['context_info__real_show_list']
            realshow_label = tf.reshape(realshow_label, [-1, 60])
            realshow_label = realshow_label[:, :6]

            rerank_label = tf.math.logical_or(
                tf.math.equal(wtd_label, 1),
                tf.math.equal(ltr_label, 1)
            )
            rerank_label = tf.cast(rerank_label, dtype=tf.int32)
            indices_shape = tf.shape(rerank_label)
            col_indices = tf.tile(tf.expand_dims(tf.range(indices_shape[1]), 0), [indices_shape[0], 1])
            rank_indices = tf.cast(col_indices * rerank_label, dtype=tf.int32)
            neg_indices = tf.cast(col_indices * (1 - rerank_label), dtype=tf.int32)

            gather_indices = tf.stack([batch_indices, rank_indices], axis=-1)
            item_embeddings = tf.gather_nd(common_embs, gather_indices)
            hidden_states = self._mlp_layer("mlp_layer_1", item_embeddings, [64, 32])
            position_ids = tf.range(6, dtype=tf.int32)
            position_ids = tf.expand_dims(position_ids, 0)
            position_embeddings = tf.nn.embedding_lookup(self.position_embeddings, position_ids)
            position_embeddings = tf.tile(position_embeddings, [tf.shape(hidden_states)[0], 1, 1])
            eval_embedding = hidden_states + position_embeddings

            ple_layer = PLE(self.loss_names, shared_key="vtr", cgc_layers=1, task_expert_num=1, shared_expert_num=4,
                                expert_tower_dim=[256, 128], gate_tower_dim=[256, 128], print_ops=self.print_ops)
            input_feature_dict = {x: eval_embedding for x in self.loss_names}
            output_fea_dict = ple_layer(input_feature_dict, input_feature_dict)
            output_list = []
            key_output_list = []
            for key in output_fea_dict.keys():
                key_output_list.append(key)
                output_list.append(output_fea_dict[key])
            output_list = tf.stack(output_list, axis=2)
            output_list = tf.split(output_list, len(output_fea_dict), axis=2)
            for j in range(len(output_fea_dict)):
                output_fea_dict[key_output_list[j]] = tf.squeeze(output_list[j], axis=2)
            output_dict = {}
            for loss_name, output in output_fea_dict.items():
                with tf.variable_scope(f"output_mlp_{loss_name}", reuse=tf.AUTO_REUSE):
                    output = tf.layers.dense(output, 128, activation=tf.nn.leaky_relu)
                    output = layer_norm(output, scope="layer_norm_1")
                    output = tf.layers.dense(output, 64, activation=tf.nn.leaky_relu)
                    output = layer_norm(output, scope="layer_norm_2")
                    output = tf.layers.dense(output, 1, activation=tf.nn.sigmoid)
                    output = tf.reshape(output, [-1, 6])
                output_dict[loss_name] = output

            generator_embeding = self._mlp_layer("mlp_layer_1", generator_embeding, [64, 32])
            generator_embeding = generator_embeding + position_embeddings
            input_feature_dict_gen = {x: generator_embeding for x in self.loss_names}
            output_fea_dict_gen = ple_layer(input_feature_dict_gen, input_feature_dict_gen)
            output_list_gen = []
            key_output_list_gen = []
            for key in output_fea_dict_gen.keys():
                key_output_list_gen.append(key)
                output_list_gen.append(output_fea_dict_gen[key])
            output_list_gen = tf.stack(output_list_gen, axis=2)
            output_list_gen = tf.split(output_list_gen, len(output_fea_dict), axis=2)
            for j in range(len(output_fea_dict_gen)):
                output_fea_dict_gen[key_output_list_gen[j]] = tf.squeeze(output_list_gen[j], axis=2)
            output_dict_gen = {}
            for loss_name, output in output_fea_dict_gen.items():
                with tf.variable_scope(f"output_mlp_{loss_name}", reuse=tf.AUTO_REUSE):
                    output = tf.layers.dense(output, 128, activation=tf.nn.leaky_relu)
                    output = layer_norm(output, scope="layer_norm_1")
                    output = tf.layers.dense(output, 64, activation=tf.nn.leaky_relu)
                    output = layer_norm(output, scope="layer_norm_2")
                    output = tf.layers.dense(output, 1, activation=tf.nn.sigmoid)
                    output = tf.reshape(output, [-1, 6])
                output_dict_gen[loss_name] = output

            rerank_weight = tf.clip_by_value(rerank_weight, 0, 100) / 10.0
            gen_loss = self.cal_gen_loss(predict_ori, rerank_weight, rank_indices, neg_indices, realshow_label)

            return output_dict, output_dict_gen, gen_loss, cl_loss, predict_ori, item_embedding_gen, rank_indices, dat_loss, transition_matrix, token_distribution, valid_links
