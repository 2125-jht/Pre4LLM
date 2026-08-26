import tensorflow as tf
import sys
from feature_attr_extract import *
from modulesV2 import *
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
    def __init__(self, dim=16, vocab_sizes=[16, 32, 32, 32], print_ops=None):
        # self._feature_emb_dict = feature_emb_dict
        # self._feature_emb_size_dict = feature_emb_size_dict
        self._print_ops = print_ops
        self._vocab_sizes = vocab_sizes
        self._total_vocab_size = sum(self._vocab_sizes)
        self._embedding = tf.get_variable(shape=[self._total_vocab_size+1, dim], name='embedding', \
                initializer=tf.random_uniform_initializer(minval=-1.0/dim, maxval=1.0/dim), trainable=True)
        # self._embedding = tf.Variable(tf.random.uniform([vocab_sizes[0]+vocab_sizes[1]+vocab_sizes[2]+vocab_sizes[3], dim]), name="embedding")
        # self._start_token = tf.Variable(tf.random.uniform([1, 1, dim]), name="start_token")
        self._dim = dim

    def model(self, user_sid_list, photo_sid, label, photo_semantic_id_int):
        user_sid_emb = tf.nn.embedding_lookup(self._embedding, user_sid_list)
        batch_size = tf.shape(user_sid_emb)[0]
        start_token_indice = tf.tile(tf.constant(self._total_vocab_size, shape=(1, 1), dtype=tf.int32), [batch_size, 1])
        photo_with_start_token = tf.concat([start_token_indice, photo_sid], axis=1)
        decoder_input = tf.nn.embedding_lookup(self._embedding, photo_with_start_token)

        user_sid_emb_reshape = tf.reshape(user_sid_emb, [batch_size, -1])
        print_tensor("user_sid_emb_sim", calc_sim_cos(user_sid_emb_reshape))
        encoder_output = transformer_encoder_layer(user_sid_emb, 3, dim=self._dim) # [batch_size, seq_len, dim]
        encoder_output_sim = tf.reshape(encoder_output, [batch_size, -1])
        print_tensor("encoder_output_sim", calc_sim_cos(encoder_output_sim))
        # self._print_ops.append(tf.print("encoder_output_shape", tf.shape(encoder_output), summarize=-1, output_stream=sys.stdout))
        decoder_output = transformer_decoder_layer(encoder_output, decoder_input, 3, dim=self._dim)
        for i in range(4):
            similarity = calc_sim_cos(decoder_output[:, i, :])
            print_tensor('decoder_sim/decoder_output_%d' % i, similarity)
        # self._print_ops.append(tf.print("decoder_output_shape", tf.shape(decoder_output), summarize=-1, output_stream=sys.stdout))
        losses = []
        for step in range(len(self._vocab_sizes)):
            with tf.variable_scope('proj_%d' % step):
                scope = tf.get_variable_scope()
                # print("train scope:", scope.name)
                pred_logit = tf.layers.dense(decoder_output[:, step, :], self._vocab_sizes[step])
                print_tensor("logits/pred_logit_%d" % step, pred_logit)
                # self._print_ops.append(tf.print("pred_input_shape_%d" % i, tf.shape(decoder_output[:, i+1, :]), summarize=-1, output_stream=sys.stdout))
                # self._print_ops.append(tf.print("pred_logit_shape_%d" % i, tf.shape(pred_logit), summarize=-1, output_stream=sys.stdout))
                one_hot_labels = tf.one_hot(label[:, step], self._vocab_sizes[step])
                loss_i = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels, logits=pred_logit / 2.0)
                # self._print_ops.append(tf.print("loss_%d_shape" % i, tf.shape(loss_i), summarize=-1, output_stream=sys.stdout))
                losses.append(loss_i)
                recall_at_k(pred_logit, label[:, step], top_k=[1, 2, 4, 8],  name="predict_recall_%d" % step)

        loss_mask = tf.where(photo_semantic_id_int > 0,  tf.ones_like(photo_semantic_id_int, dtype=tf.float32), 
            tf.zeros_like(photo_semantic_id_int, dtype=tf.float32))
        print_tensor("loss_mask", loss_mask)
        loss = tf.reduce_mean((losses[0] * 10 + losses[1] * 10 + losses[2] * 10 + losses[3] * 5) * loss_mask)
        return loss
      
    def beam_search(self, user_sid_list, beam_sizes=[16, 32, 64]):
        """
        Args:
            encoder_output: 形如[batch_size, seq_len, dim]的张量

        Returns:
            best_sequences: 形如[batch_size, max_length] 表示生成的序列
        """
        user_sid_emb = tf.nn.embedding_lookup(self._embedding, user_sid_list)
        encoder_output = transformer_encoder_layer(user_sid_emb, 3, dim=self._dim)
        batch_size = tf.shape(encoder_output)[0]

        scores = tf.zeros([batch_size, 1])
        #encoder_output = tf.tile(tf.expand_dims(encoder_output, axis=1), [1, beam_size, 1, 1]) # [batch_size, beam_size, vocab_size, dim]
        selected_sequences = tf.tile(tf.constant(self._total_vocab_size, shape=[1, 1, 1]), [batch_size, 1, 1]) # [batch_size, 1, seq_len] 

        for step in range(len(beam_sizes)):
            seq_len = tf.shape(selected_sequences)[2]

            decoder_input = tf.nn.embedding_lookup(self._embedding, selected_sequences)
            decoder_input_beam_size = tf.shape(decoder_input)[1]
            encoder_output_expand = tf.tile(tf.expand_dims(encoder_output, axis=1), [1, decoder_input_beam_size, 1, 1]) # [batch_size, beam_size, vocab_size, dim]
            decoder_output = transformer_decoder_layer(encoder_output_expand, decoder_input, 3, dim=self._dim) # [batch_size, beam_size, seq_length, dim]
            self._print_ops.append(tf.print("decoder_output_shape_%d" % step, tf.shape(decoder_output), summarize=-1, output_stream=sys.stdout))

            with tf.variable_scope('proj_%d' % step, reuse=tf.AUTO_REUSE):
                logits = tf.layers.dense(decoder_output[:, :, step, :], self._vocab_sizes[step]) # [batch_size, beam_size, vocab_size]
            
            # self._print_ops.append(tf.print("logits%d" % step, logits, summarize=-1, output_stream=sys.stdout))
            next_token_probs = tf.nn.softmax(logits/4.0, axis=-1) #[batch_size, beam_size, vocab_size]
            log_probs = tf.math.log(next_token_probs)
            # log_probs = next_token_probs
            # self._print_ops.append(tf.print("log_probs_%d" % step, log_probs, summarize=-1, output_stream=sys.stdout))

            if step == 0:
                append_selected_sequences = tf.tile(tf.expand_dims(tf.range(self._vocab_sizes[0], dtype=tf.int32), axis=0), [batch_size, 1]) # batch_size, vocab_size[0]
                append_selected_sequences = tf.expand_dims(append_selected_sequences, axis=2)
                selected_sequences = tf.tile(selected_sequences, [1, self._vocab_sizes[0], 1])
                selected_sequences = tf.concat([selected_sequences, append_selected_sequences], axis=2) # batch_size, vocab_size[0], 2
                scores = tf.reshape(log_probs, [batch_size, self._vocab_sizes[0]])
            # elif step == 1:
            #     last_beam_size = tf.shape(selected_sequences)[1]
            #     cur_beam_size = beam_sizes[1]
            #     candidate_scores = tf.expand_dims(scores, -1) + log_probs # [batch_size, min(beam_size, vocab_size[0]), vocab_size]
            #     # self._print_ops.append(tf.print("candidate_scores_%d" % step, candidate_scores, summarize=-1, output_stream=sys.stdout))
            #     candidate_scores = tf.reshape(candidate_scores, [-1, last_beam_size * self._vocab_sizes[step]]) # [batch_size, beam_size * vocab_size]
            #     candidate_sequences = tf.expand_dims(selected_sequences, axis=2) # [batch_size, beam_size,  1, last_seq_length]
            #     candidate_sequences = tf.tile(candidate_sequences, [1, 1, self._vocab_sizes[step], 1]) # [batch_size, beam_size, vocab_size, last_seq_length]
            #     add_token = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.range(self._vocab_sizes[step]), axis=1), axis=0), axis=0) # [1, 1, vocab_size, 1]
            #     add_token = tf.tile(add_token, [batch_size, last_beam_size, 1, 1])
            #     candidate_sequences = tf.concat([candidate_sequences, add_token], axis=-1) # [batch_size, beam_size, vocab_size, seq_length]
            #     candidate_sequences = tf.reshape(candidate_sequences, [batch_size, last_beam_size*self._vocab_sizes[step], seq_len + 1]) # [batch_size, beam_size*vocab_size, seq_length]
            #     top_k_scores, top_k_indices = tf.math.top_k(candidate_scores, k=cur_beam_size, sorted=True) #[batch_size, k]
            #     batch_indices = tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), [1, cur_beam_size]) # [batch_size, beam_size]
            #     gather_indices = tf.stack([batch_indices, top_k_indices], axis=2) 
            #     new_sequence = tf.gather_nd(candidate_sequences, gather_indices)

            #     # 将新token添加到beam序列末尾
            #     scores = top_k_scores # [batch_size, beam_size]
            #     selected_sequences = new_sequence 
            else:
                last_beam_size = tf.shape(selected_sequences)[1]
                cur_beam_size = beam_sizes[step]
                cur_num = tf.cast(cur_beam_size / last_beam_size, dtype=tf.int32)
                candidate_scores = tf.expand_dims(scores, -1) + log_probs # [batch_size, last_beam_size, vocab_size]
                # self._print_ops.append(tf.print("candidate_scores_shape_%d" % step, tf.shape(candidate_scores), summarize=-1, output_stream=sys.stdout))
                candidate_sequences = tf.expand_dims(selected_sequences, axis=2) # [batch_size, beam_size,  1, last_seq_length]
                candidate_sequences = tf.tile(candidate_sequences, [1, 1, self._vocab_sizes[step], 1]) # [batch_size, last_beam_size, vocab_size, last_seq_length]

                add_token = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.range(self._vocab_sizes[step]), axis=1), axis=0), axis=0) # [1, 1, vocab_size, 1]
                add_token = tf.tile(add_token, [batch_size, last_beam_size, 1, 1])
                candidate_sequences = tf.concat([candidate_sequences, add_token], axis=-1) # [batch_size, beam_size, vocab_size, seq_length]
                # self._print_ops.append(tf.print("candidate_sequences_shape_%d" % step, tf.shape(candidate_sequences), summarize=-1, output_stream=sys.stdout))
                top_k_scores, top_k_indices = tf.math.top_k(candidate_scores, k=cur_num, sorted=True) #[batch_size, last_beam_size, cur_num]
    
                batch_idx = tf.reshape(tf.range(batch_size), [batch_size, 1, 1])
                batch_idx = tf.tile(batch_idx, [1, last_beam_size, cur_num])
                beam_idx = tf.reshape(tf.range(last_beam_size), [1, last_beam_size, 1])
                beam_idx = tf.tile(beam_idx, [batch_size, 1, cur_num])
                gather_indices = tf.stack([batch_idx, beam_idx, top_k_indices], axis=-1)

                # 使用 gather_nd 获取对应的切片: [batch_size, beam_size, k, seq]
                new_sequence = tf.gather_nd(candidate_sequences, gather_indices)
                # self._print_ops.append(tf.print("new_sequence_shape_%d" % step, tf.shape(new_sequence), summarize=-1, output_stream=sys.stdout))

                # 将新token添加到beam序列末尾
                scores = tf.reshape(top_k_scores, [batch_size, cur_beam_size]) # [batch_size, cur_beam_size]] 
                selected_sequences = tf.reshape(new_sequence, [batch_size, cur_beam_size, -1]) # [batch_size, cur_beam_size, seq_length + 1]
        for step in range(len(beam_sizes), 4):
            with tf.variable_scope('proj_%d' % step, reuse=tf.AUTO_REUSE):
                logits = tf.layers.dense(tf.random_uniform([batch_size, self._dim]), self._vocab_sizes[step]) # [batch_size, beam_size, vocab_size]
        # self._print_ops.append(tf.print("selected_sequences_shape", tf.shape(selected_sequences), summarize=-1, output_stream=sys.stdout))
        return selected_sequences
            
