import tensorflow as tf
import sys
from feature_attr_extract import *
from modulesV2 import *
from modules_ import *

user_static_fea_names = [
    "user_id", 
    "user_gender",
    "user_age_segment",
    "user_level"
]

user_click_fea_names = [
    "user_profile_v1_click_pid_list",
    "user_profile_v1_click_aid_list"
]

photo_fea_names = [
    "photo_id",
    "photo_author_id"
]

class MultiInterestModel(object):
    def __init__(self, feature_emb_dict, feature_emb_size_dict,  dim=128, vocab_sizes=[8192, 8192, 8192], print_ops=None):
        self._feature_emb_dict = feature_emb_dict
        self._feature_emb_size_dict = feature_emb_size_dict
        self._print_ops = print_ops
        self._vocab_sizes = vocab_sizes
        self._total_vocab_size = sum(self._vocab_sizes)
        self._embedding = tf.get_variable(shape=[self._total_vocab_size+1, dim], name='embedding', \
                initializer=tf.random_uniform_initializer(minval=-1.0/dim, maxval=1.0/dim), trainable=True)
        self._dim = dim

    def model(self, photo_sid, label, photo_semantic_id_int):
        user_static_fea = tf.concat([self._feature_emb_dict[fea] for fea in user_static_fea_names], axis=1)
        user_static_emb = mlp('user_static_emb', user_static_fea, [2*self._dim], self._dim, activation=tf.nn.leaky_relu)
        batch_size = tf.shape(user_static_emb)[0]
        user_static_emb = tf.reshape(user_static_emb, [batch_size, 1, self._dim])

        user_click_fea = tf.concat([self._feature_emb_dict[fea] for fea in user_click_fea_names], axis=2)
        # self._print_ops.append(tf.print("user_click_fea_shape_1", tf.shape(user_click_fea), summarize=-1, output_stream=sys.stdout))
        user_click_emb = mlp('user_click_emb', user_click_fea, [4*self._dim], self._dim, activation=tf.nn.leaky_relu)
        #self._print_ops.append(tf.print("user_click_emb_shape", tf.shape(user_click_emb), summarize=-1, output_stream=sys.stdout))

        start_token_indice = tf.tile(tf.constant(self._total_vocab_size, shape=(1, 1), dtype=tf.int32), [batch_size, 1])
        token_emb = tf.nn.embedding_lookup(self._embedding, start_token_indice) #[batch, 1, dim]
        encoder_input = tf.concat([user_static_emb, user_click_emb, token_emb], axis=1)
        
        # photo_with_start_token = tf.concat([start_token_indice, photo_sid], axis=1)
        # decoder_input = tf.nn.embedding_lookup(self._embedding, start_token_indice) #[batch, 1, dim]
 
        # user_click_list_reshape = tf.reshape(user_click_list, [batch_size, -1])
        # print_tensor("user_sid_emb_sim", calc_sim_cos(user_click_list_reshape))

        transformer = StackedTransformerModel(num_layers=4, dim=self._dim, num_heads=8, dropout_rate=0.1, hidden_dim=self._dim*2)
        encoder_output = transformer.forward(encoder_input, training=True) # [batch_size, seq_len, dim]
        
        encoder_output_sim = tf.reshape(encoder_output, [batch_size, -1])
        print_tensor("encoder_output_sim", calc_sim_cos(encoder_output_sim))
        # self._print_ops.append(tf.print("encoder_output_shape", tf.shape(encoder_output), summarize=-1, output_stream=sys.stdout))
        # decoder_output = transformer_decoder_layer(encoder_output, decoder_input, 4, dim=self._dim)
        
        user_top_embedding = tf.squeeze(encoder_output[:, -1:, ], axis=1)

        photo_static_fea = tf.concat([self._feature_emb_dict[fea] for fea in photo_fea_names], axis=1)
        photo_top_embedding = mlp('photo_emb', photo_static_fea, [2*self._dim], self._dim, activation=tf.nn.leaky_relu)

        return user_top_embedding, photo_top_embedding

