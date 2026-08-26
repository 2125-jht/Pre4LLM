# !/user/local/bin/python
# coding=utf-8


# -*- coding: UTF8 -*-
from __future__ import absolute_import, division, print_function
import kuiba_utils
from tensorflow.python.ops.losses.losses_impl import Reduction
from manager.model.encode_base import Encoder
from manager.model.model_base import ModelOut
import tensorflow as tf
import random

import numpy as np
from manager.model.model_base import ModelBase

pos_loss = 'listwise_loss'
ctr_loss = 'ctr_loss'
# wtd_loss = "wtd_loss"
# slide_loss = 'slide_loss'
like_loss = 'like_loss'
follow_loss = 'follow_loss'
profile_loss = 'profile_loss'
longview_loss = 'longview_loss'
evtr_loss = 'evtr_loss'
interact_loss = 'interact_loss'

class ListModel(ModelBase):
    def __init__(self, loss_function_conf):
        super().__init__()
        self.use_pos_embedding = True
        self.use_trainable_pos_embedding = True
        self.pos_embedding_size = 64
        self.loss_function_conf = loss_function_conf
        self.temp_name_list = [pos_loss] + [ctr_loss] + [like_loss] + [follow_loss] + [profile_loss] + [longview_loss] + [interact_loss] + [evtr_loss] 
        self.hot_loss_name_list = ['hot_' + self.temp_name_list[i] for i in range(6)]
        self.slide_loss_name_list = ['slide_' + self.temp_name_list[i] for i in [0, 1, 5, 6, 7]]
        self.loss_name_list = self.hot_loss_name_list + self.slide_loss_name_list
        self.seq_len = 10
        self.slide_len = 6
        self.prefix_extra_name = list(self.loss_function_conf.feature_conf_pool.set_extra_feature().keys())
        self.prefix_slide_name = list(self.loss_function_conf.feature_conf_pool.set_slide_feature().keys())
        # self.prefix_listwise_extra_name = list(self.loss_function_conf.feature_conf_pool.set_extra_feature_listwise().keys())
        # self.context_len = 16
        self.user_name = list(self.loss_function_conf.feature_conf_pool.set_user_feature().keys())
        self.photo_name = list(self.loss_function_conf.feature_conf_pool.set_photo_feature().keys())
        # self.context_name = list(self.loss_function_conf.feature_conf_pool.set_context_features().keys())
        
        # 不作为 特征， 作为 reward label 等
        print("{}={}".format("prefix_extra_name", ",".join(self.prefix_extra_name)))
        print("{}={}".format("user_name", ",".join(self.user_name)))
        self.prefix_extra_feature_list = [prefix + "_idx" + str(pos) for pos in range(self.seq_len) for prefix in self.prefix_extra_name] + [prefix + "_idx" + str(pos) for pos in range(6) for prefix in self.prefix_slide_name]
        print("prefix_extra_feature_list={}".format(",".join(self.prefix_extra_feature_list)))
        

    def init_tmodel(self, d_model=64, d_inner_hid=128, n_head=1, d_k=64, d_v=64, layers=1,
                    dropout=0.1):
        self.encoder = Encoder(d_model=d_model,
                               d_inner_hid=d_inner_hid,
                               n_head=n_head,
                               d_k=d_k, d_v=d_v,
                               layers=layers,
                               dropout=dropout)

    def hot_split_inputs_feature(self, inputs_dict):
        if (random.uniform(1, 1000) > 998):
            print("{}={}".format("input key name:", ",".join(inputs_dict.keys())))
        user_fea = []
        iv_fea = [[ None for _ in range(len(self.photo_name))] for _ in range(self.seq_len)]
        for fa, fv in inputs_dict.items():
            if fa in self.prefix_extra_feature_list:
                # 跳过label和weight
                continue
            if fa in self.user_name: 
                # user 特征
                user_fea.append(fv)
            else:
                if '_' in fa and fa.startswith('param.') and ('_'.join(fa[6:].split('_')[:-1]) in self.photo_name):
                    pos = int(fa[-1])
                    index = self.photo_name.index('_'.join(fa[6:].split('_')[:-1]))
                    iv_fea[pos][index] = fv
                elif '_' in fa and '_'.join(fa.split('_')[:-1]) in self.photo_name:
                    pos = int(fa[-1])
                    index = self.photo_name.index('_'.join(fa.split('_')[:-1]))
                    iv_fea[pos][index] = fv
                        
        output_list = []
        pred_ctr_list = []
        pred_ltr_list = []
        pred_wtr_list = []
        pred_profile_list = []
        pred_longview_list = []
        for pos in range(self.seq_len):
            # 进入trf之前的网路部分
                # 设置pv dim=16, 辅助训练塔128-64-16-1
            with tf.variable_scope("pv_layers_ctr", reuse=kuiba_utils.reuse_variables()):
                name_pv = "pv_layer_ctr"
                input_pv = tf.concat(iv_fea[pos] + user_fea, axis=1)
                input_pv = self.dense_with_prelu(input_pv, 128, name = name_pv + "_1")
                input_pv = self.dense_with_prelu(input_pv, 64, name = name_pv + "_2")
                output_pre_ctr = self.dense_with_prelu(input_pv, 16, name = name_pv + "_3")
                pred_ctr = tf.layers.dense(output_pre_ctr, 1, name=name_pv + '_output', activation=tf.nn.sigmoid)
                pred_ctr_list.append(pred_ctr)
            with tf.variable_scope("pv_layers_ltr", reuse=kuiba_utils.reuse_variables()):
                name_pv = "pv_layer_ltr"
                input_pv = tf.concat(iv_fea[pos] + user_fea, axis=1)
                input_pv = self.dense_with_prelu(input_pv, 128, name = name_pv + "_1")
                input_pv = self.dense_with_prelu(input_pv, 64, name = name_pv + "_2")
                output_pre_ltr = self.dense_with_prelu(input_pv, 16, name = name_pv + "_3")
                pred_ltr = tf.layers.dense(output_pre_ltr, 1, name=name_pv + '_output', activation=tf.nn.sigmoid)
                pred_ltr_list.append(pred_ltr)
            with tf.variable_scope("pv_layers_wtr", reuse=kuiba_utils.reuse_variables()):
                name_pv = "pv_layer_wtr"
                input_pv = tf.concat(iv_fea[pos] + user_fea, axis=1)
                input_pv = self.dense_with_prelu(input_pv, 128, name = name_pv + "_1")
                input_pv = self.dense_with_prelu(input_pv, 64, name = name_pv + "_2")
                output_pre_wtr = self.dense_with_prelu(input_pv, 16, name = name_pv + "_3")
                pred_wtr = tf.layers.dense(output_pre_ctr, 1, name=name_pv + '_output', activation=tf.nn.sigmoid)
                pred_wtr_list.append(pred_wtr)
            with tf.variable_scope("pv_layers_profile", reuse=kuiba_utils.reuse_variables()):
                name_pv = "pv_layer_profile"
                input_pv = tf.concat(iv_fea[pos] + user_fea, axis=1)
                input_pv = self.dense_with_prelu(input_pv, 128, name = name_pv + "_1")
                input_pv = self.dense_with_prelu(input_pv, 64, name = name_pv + "_2")
                output_pre_profile = self.dense_with_prelu(input_pv, 16, name = name_pv + "_3")
                pred_profile = tf.layers.dense(output_pre_ctr, 1, name=name_pv + '_output', activation=tf.nn.sigmoid)
                pred_profile_list.append(pred_profile)
            with tf.variable_scope("pv_layers_longview", reuse=kuiba_utils.reuse_variables()):
                name_pv = "pv_layer_longview"
                input_pv = tf.concat(iv_fea[pos] + user_fea, axis=1)
                input_pv = self.dense_with_prelu(input_pv, 128, name = name_pv + "_1")
                input_pv = self.dense_with_prelu(input_pv, 64, name = name_pv + "_2")
                output_pre_longview = self.dense_with_prelu(input_pv, 16, name = name_pv + "_3")
                pred_longview = tf.layers.dense(output_pre_longview, 1, name=name_pv + '_output', activation=tf.nn.sigmoid)
                pred_longview_list.append(pred_longview)
            # 设置iv dim=48, iv+pv+pe
            with tf.variable_scope("iv_layers", reuse=kuiba_utils.reuse_variables()):
                name_iv = "idx_" + str(pos) + "iv_layer"
                input_iv = tf.concat(iv_fea[pos], axis=1)
                input_iv = tf.stop_gradient(input_iv)
                input_iv = self.dense_with_prelu(input_iv, 128, name = name_iv + "_1")
                iv = self.dense_with_prelu(input_iv, 48, name = name_iv + "_2")
                pv_ctr = tf.stop_gradient(output_pre_ctr) #pv不会往主模型回传梯度
                pv_ltr = tf.stop_gradient(output_pre_ltr)
                pv_wtr = tf.stop_gradient(output_pre_wtr)
                pv_profile = tf.stop_gradient(output_pre_profile)
                pv_longview = tf.stop_gradient(output_pre_longview)
                pv_ctr = tf.multiply(pv_ctr, self.dense_with_prelu(pv_ctr, 1, name='pv_ctr'))
                pv_ltr = tf.multiply(pv_ltr, self.dense_with_prelu(pv_ltr, 1, name='pv_ltr'))
                pv_wtr = tf.multiply(pv_wtr, self.dense_with_prelu(pv_wtr, 1, name='pv_wtr'))
                pv_profile = tf.multiply(pv_profile, self.dense_with_prelu(pv_profile, 1, name='pv_profile'))
                pv_longview = tf.multiply(pv_longview, self.dense_with_prelu(pv_longview, 1, name='pv_longview'))
                pv = tf.add_n([pv_ctr, pv_ltr, pv_wtr, pv_profile, pv_longview])
                if not self.use_pos_embedding:
                    output_list.append(tf.concat([pv, iv], axis=1))
                else:
                    batch_size = tf.shape(iv)[0]
                    pos_embedding = self.hot_get_pos_embedding(pos, batch_size)
                    output_list.append(tf.concat([pv, iv], axis=1) + pos_embedding)
        return output_list, pred_ctr_list, pred_ltr_list, pred_wtr_list, pred_profile_list, pred_longview_list

    def slide_split_inputs_feature(self, inputs_dict):
        if (random.uniform(1, 1000) > 998):
            print("{}={}".format("input key name:", ",".join(inputs_dict.keys())))
        user_fea = []
        iv_fea = [[ None for _ in range(len(self.photo_name))] for _ in range(self.seq_len)]
        for fa, fv in inputs_dict.items():
            if fa in self.prefix_extra_feature_list:
                # 跳过label和weight
                continue
            if fa in self.user_name: 
                # user 特征
                user_fea.append(fv)
            else:
                if '_' in fa and fa.startswith('param.') and ('_'.join(fa[6:].split('_')[:-1]) in self.photo_name):
                    pos = int(fa[-1])
                    index = self.photo_name.index('_'.join(fa[6:].split('_')[:-1]))
                    iv_fea[pos][index] = fv
                elif '_' in fa and '_'.join(fa.split('_')[:-1]) in self.photo_name:
                    pos = int(fa[-1])
                    index = self.photo_name.index('_'.join(fa.split('_')[:-1]))
                    iv_fea[pos][index] = fv
                        
        output_list = []
        pred_ctr_list = []
        pred_interact_list = []
        pred_longview_list = []
        pred_evtr_list = []
        for pos in range(6):
            # 进入trf之前的网路部分
                # 设置pv dim=16, 辅助训练塔128-64-16-1
            with tf.variable_scope("slide_pv_layers_ctr", reuse=kuiba_utils.reuse_variables()):
                name_pv = "pv_layer_ctr"
                input_pv = tf.concat(iv_fea[pos] + user_fea, axis=1)
                input_pv = self.dense_with_prelu(input_pv, 128, name = name_pv + "_1")
                input_pv = self.dense_with_prelu(input_pv, 64, name = name_pv + "_2")
                output_pre_ctr = self.dense_with_prelu(input_pv, 16, name = name_pv + "_3")
                pred_ctr = tf.layers.dense(output_pre_ctr, 1, name=name_pv + '_output', activation=tf.nn.sigmoid)
                pred_ctr_list.append(pred_ctr)
            with tf.variable_scope("slide_pv_layers_interact", reuse=kuiba_utils.reuse_variables()):
                name_pv = "pv_layer_interact"
                input_pv = tf.concat(iv_fea[pos] + user_fea, axis=1)
                input_pv = self.dense_with_prelu(input_pv, 128, name = name_pv + "_1")
                input_pv = self.dense_with_prelu(input_pv, 64, name = name_pv + "_2")
                output_pre_interact = self.dense_with_prelu(input_pv, 16, name = name_pv + "_3")
                pred_interact = tf.layers.dense(output_pre_interact, 1, name=name_pv + '_output', activation=tf.nn.sigmoid)
                pred_interact_list.append(pred_interact)
            
            with tf.variable_scope("slide_pv_layers_longview", reuse=kuiba_utils.reuse_variables()):
                name_pv = "pv_layer_longview"
                input_pv = tf.concat(iv_fea[pos] + user_fea, axis=1)
                input_pv = self.dense_with_prelu(input_pv, 128, name = name_pv + "_1")
                input_pv = self.dense_with_prelu(input_pv, 64, name = name_pv + "_2")
                output_pre_longview = self.dense_with_prelu(input_pv, 16, name = name_pv + "_3")
                pred_longview = tf.layers.dense(output_pre_longview, 1, name=name_pv + '_output', activation=tf.nn.sigmoid)
                pred_longview_list.append(pred_longview)
            with tf.variable_scope("slide_pv_layers_evtr", reuse=kuiba_utils.reuse_variables()):
                name_pv = "pv_layer_evtr"
                input_pv = tf.concat(iv_fea[pos] + user_fea, axis=1)
                input_pv = self.dense_with_prelu(input_pv, 128, name = name_pv + "_1")
                input_pv = self.dense_with_prelu(input_pv, 64, name = name_pv + "_2")
                output_pre_evtr = self.dense_with_prelu(input_pv, 16, name = name_pv + "_3")
                pred_evtr = tf.layers.dense(output_pre_evtr, 1, name=name_pv + '_output', activation=tf.nn.sigmoid)
                pred_evtr_list.append(pred_evtr)
            # 设置iv dim=48, iv+pv+pe
            with tf.variable_scope("slide_iv_layers", reuse=kuiba_utils.reuse_variables()):
                name_iv = "idx_" + str(pos) + "iv_layer"
                input_iv = tf.concat(iv_fea[pos], axis=1)
                input_iv = tf.stop_gradient(input_iv)
                input_iv = self.dense_with_prelu(input_iv, 128, name = name_iv + "_1")
                iv = self.dense_with_prelu(input_iv, 48, name = name_iv + "_2")
                pv_ctr = tf.stop_gradient(output_pre_ctr) #pv不会往主模型回传梯度
                pv_interact = tf.stop_gradient(output_pre_interact)
                pv_longview = tf.stop_gradient(output_pre_longview)
                pv_evtr = tf.stop_gradient(output_pre_evtr)
                pv_ctr = tf.multiply(pv_ctr, self.dense_with_prelu(pv_ctr, 1, name='pv_ctr'))
                pv_interact = tf.multiply(pv_interact, self.dense_with_prelu(pv_interact, 1, name='pv_ltr'))
                pv_longview = tf.multiply(pv_longview, self.dense_with_prelu(pv_longview, 1, name='pv_longview'))
                pv_evtr = tf.multiply(pv_evtr, self.dense_with_prelu(pv_evtr, 1, name='pv_evtr'))
                pv = tf.add_n([pv_ctr, pv_interact, pv_longview, pv_evtr])
                if not self.use_pos_embedding:
                    output_list.append(tf.concat([pv, iv], axis=1))
                else:
                    batch_size = tf.shape(iv)[0]
                    pos_embedding = self.slide_get_pos_embedding(pos, batch_size)
                    output_list.append(tf.concat([pv, iv], axis=1) + pos_embedding)
        return output_list, pred_ctr_list, pred_interact_list, pred_longview_list, pred_evtr_list


    def hot_get_pos_embedding(self, pos_index, batch_size):
        with tf.variable_scope("hot_pos_embedding", reuse=kuiba_utils.reuse_variables()):
            embedding_param = tf.get_variable(name="hot_pos_embedding", shape=[10, self.pos_embedding_size])
            ids = tf.ones([batch_size, 1], name="id_{}".format(pos_index), dtype=tf.int32) * pos_index
            embedding = tf.nn.embedding_lookup(ids=ids, params=embedding_param)
            embedding = tf.reshape(embedding, shape=[-1, self.pos_embedding_size])
        return embedding
    
    def slide_get_pos_embedding(self, pos_index, batch_size):
        with tf.variable_scope("slide_pos_embedding", reuse=kuiba_utils.reuse_variables()):
            embedding_param = tf.get_variable(name="slide_pos_embedding", shape=[6, self.pos_embedding_size])
            ids = tf.ones([batch_size, 1], name="id_{}".format(pos_index), dtype=tf.int32) * pos_index
            embedding = tf.nn.embedding_lookup(ids=ids, params=embedding_param)
            embedding = tf.reshape(embedding, shape=[-1, self.pos_embedding_size])
        return embedding
    
    def prelu(self, _x, name=""):
        """
        Parametric ReLU
        """
        with tf.variable_scope(name + "prelu", reuse=kuiba_utils.reuse_variables()):
            alphas = tf.get_variable(name, _x.get_shape()[-1],
                            initializer=tf.constant_initializer(0.1),
                                dtype=tf.float32, trainable=True)
            pos = tf.nn.relu(_x)
            neg = alphas * (_x - abs(_x)) * 0.5

        return pos + neg
    
    def dense_with_prelu(self, _x, units, name):
        with tf.variable_scope(name + "dense_prelu", reuse=kuiba_utils.reuse_variables()):
            output = tf.layers.dense(_x, units, name=name)
            output = self.prelu(output, name)
        return output
    
    def dice(self, _x, axis=-1, name=''):
        with tf.variable_scope(name_or_scope='', reuse=kuiba_utils.reuse_variables()):
            alphas = tf.get_variable('alpha' + name, _x.get_shape()[-1],
                                    initializer=tf.constant_initializer(0.0),
                                    dtype=tf.float32)
            beta = tf.get_variable('beta' + name, _x.get_shape()[-1],
                                initializer=tf.constant_initializer(0.0),
                                dtype=tf.float32)
            input_shape = list(_x.get_shape())

            reduction_axes = list(range(len(input_shape)))
            del reduction_axes[axis]
            broadcast_shape = [1] * len(input_shape)
            broadcast_shape[axis] = input_shape[axis]
            x_normed = tf.layers.batch_normalization(_x, center=False, scale=False, name=name, reuse=kuiba_utils.reuse_variables())
            x_p = tf.sigmoid(beta * x_normed)
        return alphas * (1.0 - x_p) * _x + x_p * _x
    
    def dense_with_dice(self, _x, units, name=''):
        with tf.variable_scope(name + "dense_dice", reuse=kuiba_utils.reuse_variables()):
            output = tf.layers.dense(_x, units, name=name)
            output = self.dice(output)
        return output
    
    def build_tModel_layers(self, inputs, seq_len):
        """
    :param inputs:  [-1, 4*64]
    :return:
    """
        with tf.variable_scope("tModel_layers", reuse=kuiba_utils.reuse_variables()):
            self.init_tmodel()
            X = tf.reshape(inputs, shape=[-1, seq_len, 64])
            enc_output = self.encoder(X, mask=True)  # (batch_size, seq_len, d_model)
            print("build_tModel_layers::enc_output {}".format(str(enc_output)))
            return enc_output
    
    def slide_head_layers(self, input, name="slide_common"):
        with tf.variable_scope("multi_head_layers", reuse=kuiba_utils.reuse_variables()):
            name = "idx_" + name
            if kuiba_utils.train_mode():
                tf.summary.histogram(name + "_head", input)
            output = self.dense_with_prelu(input, 128, name=name + '_head_1')
            output = self.dense_with_prelu(input, 64, name=name + '_head_2')
            output = tf.layers.dense(output, 1, name=name + '_head_2', activation=tf.nn.sigmoid)
            if kuiba_utils.train_mode():
                tf.summary.histogram(name + "_output", output)
        return output
    
    # 双目标
    def multi_level_extraction_network2(self, hidden_layer, num_level, experts_units, experts_num, name="ple_common"):
        with tf.variable_scope("multi_level_extraction_network", reuse=kuiba_utils.reuse_variables()):
            gate_output_task1_final = hidden_layer
            gate_output_task2_final = hidden_layer
            gate_output_shared_final = hidden_layer
            for i in range(num_level):
                experts_weight = tf.get_variable(name="experts_weight{}".format(i), dtype=tf.float32,
                                                shape=(gate_output_shared_final.get_shape()[-1], experts_units, experts_num),
                                                initializer=tf.glorot_uniform_initializer())
                experts_bias = tf.get_variable(name="experts_bias{}".format(i), dtype=tf.float32,
                                                shape=(experts_units, experts_num),
                                                initializer=tf.glorot_uniform_initializer())
                experts_weight_task1 = tf.get_variable(name="experts_weight_task1{}".format(i), dtype=tf.float32,
                                                shape=(gate_output_task1_final.get_shape()[-1], experts_units, experts_num),
                                                initializer=tf.glorot_uniform_initializer())
                experts_bias_task1 = tf.get_variable(name="experts_bias_task1{}".format(i), dtype=tf.float32,
                                                shape=(experts_units, experts_num),
                                                initializer=tf.glorot_uniform_initializer())
                experts_weight_task2 = tf.get_variable(name="experts_weight_task2{}".format(i), dtype=tf.float32,
                                                shape=(gate_output_task2_final.get_shape()[-1], experts_units, experts_num),
                                                initializer=tf.glorot_uniform_initializer())
                experts_bias_task2 = tf.get_variable(name="experts_bias_task2{}".format(i), dtype=tf.float32,
                                                shape=(experts_units, experts_num),
                                                initializer=tf.glorot_uniform_initializer())
                gate_weight_t1 = tf.get_variable(name="gate_weight_t1{}".format(i), dtype=tf.float32,
                                        shape=(gate_output_task1_final.get_shape()[-1], experts_num * 2),
                                        initializer=tf.glorot_uniform_initializer())
                gate_bias_t1 = tf.get_variable(name="gate_bias_t1{}".format(i), dtype=tf.float32,
                                        shape=(experts_num * 2),
                                        initializer=tf.glorot_uniform_initializer())
                gate_weight_t2 = tf.get_variable(name="gate_weight_t2{}".format(i), dtype=tf.float32,
                                        shape=(gate_output_task2_final.get_shape()[-1], experts_num * 2),
                                        initializer=tf.glorot_uniform_initializer())
                gate_bias_t2 = tf.get_variable(name="gate_bias_t2{}".format(i), dtype=tf.float32,
                                        shape=(experts_num * 2),
                                        initializer=tf.glorot_uniform_initializer())
                gate_shared_weight = tf.get_variable(name="gate_shared_weights{}".format(i),dtype=tf.float32,
                                                    shape=(gate_output_shared_final.get_shape()[-1], experts_num * 3),
                                                    initializer=tf.glorot_uniform_initializer())
                gate_shared_bias = tf.get_variable(name="gate_shared_bias{}".format(i), dtype=tf.float32,
                                                   shape=(experts_num * 3),
                                                   initializer=tf.glorot_uniform_initializer())
                experts_out_s = self.prelu(tf.add(tf.tensordot(gate_output_shared_final, experts_weight, axes=1), experts_bias), name="experts_out_s{}".format(i))
                experts_out_t1 = self.prelu(tf.add(tf.tensordot(gate_output_task1_final, experts_weight_task1, axes=1), experts_bias_task1), name="experts_out_t1{}".format(i))
                experts_out_t2 = self.prelu(tf.add(tf.tensordot(gate_output_task2_final, experts_weight_task2, axes=1), experts_bias_task2), name="experts_out_t2{}".format(i))
                # gate_out1
                gate_output_task1 = tf.matmul(gate_output_task1_final, gate_weight_t1)
                gate_output_task1 = tf.add(gate_output_task1, gate_bias_t1)
                gate_output_task1 = tf.nn.softmax(gate_output_task1)
                gate_output_task1 = tf.multiply(tf.concat([experts_out_t1, experts_out_s], axis=2), tf.expand_dims(gate_output_task1, axis=1))
                gate_output_task1 = tf.reduce_sum(gate_output_task1, axis=2)
                gate_output_task1_final = tf.reshape(gate_output_task1, [-1, experts_units])
                # gate_out2
                gate_output_task2 = tf.matmul(gate_output_task2_final, gate_weight_t2)
                gate_output_task2 = tf.add(gate_output_task2, gate_bias_t2)
                gate_output_task2 = tf.nn.softmax(gate_output_task2)
                gate_output_task2 = tf.multiply(tf.concat([experts_out_t2, experts_out_s], axis=2), tf.expand_dims(gate_output_task2, axis=1))
                gate_output_task2 = tf.reduce_sum(gate_output_task2, axis=2)
                gate_output_task2_final = tf.reshape(gate_output_task2, [-1, experts_units])
                # gate_shared
                gate_output_shared = tf.matmul(gate_output_shared_final, gate_shared_weight)
                gate_output_shared = tf.add(gate_output_shared, gate_shared_bias)
                gate_output_shared = tf.nn.softmax(gate_output_shared)
                gate_output_shared = tf.multiply(tf.concat([experts_out_s, experts_out_t1, experts_out_t2], axis=2), tf.expand_dims(gate_output_shared, axis=1))
                gate_output_shared = tf.reduce_sum(gate_output_shared, axis=2)
                gate_output_shared_final = tf.reshape(gate_output_shared, [-1, experts_units])
        return gate_output_task1_final, gate_output_task2_final
    
    # 多目标PLE框架
    def multi_level_mtl_extraction_network(self, inputs, num_level, num_labels, experts_units, experts_num, name="mtl_common"):
        with tf.variable_scope("multi_level_extraction_network", reuse=kuiba_utils.reuse_variables()):
            gate_output_tasks = [None] * num_labels
            for i in range(num_labels):
                if i == 0: #ltr是否反传梯度
                    gate_output_tasks[i] = tf.stop_gradient(inputs)
                gate_output_tasks[i] = inputs
            gate_output_shared_final = inputs
            for i in range(num_level):
                experts_weight = tf.get_variable(name="experts_weight{}".format(i), dtype=tf.float32,
                                                shape=(gate_output_shared_final.get_shape()[-1], experts_units, experts_num),
                                                initializer=tf.glorot_uniform_initializer())
                experts_bias = tf.get_variable(name="experts_bias{}".format(i), dtype=tf.float32,
                                                shape=(experts_units, experts_num),
                                                initializer=tf.glorot_uniform_initializer())
                gate_shared_weight = tf.get_variable(name="gate_shared_weights{}".format(i),dtype=tf.float32,
                                                    shape=(gate_output_shared_final.get_shape()[-1], experts_num * (num_labels+1)),
                                                    initializer=tf.glorot_uniform_initializer())
                gate_shared_bias = tf.get_variable(name="gate_shared_bias{}".format(i), dtype=tf.float32,
                                                   shape=(experts_num * (num_labels+1)),
                                                   initializer=tf.glorot_uniform_initializer())
                experts_out_shared = self.prelu(tf.add(tf.tensordot(gate_output_shared_final, experts_weight, axes=1), experts_bias), name="experts_out_shared{}".format(i))
                experts_out_tasks = []
                for j in range(num_labels):
                    experts_weight_task = tf.get_variable(name="experts_weight_task{}_{}".format(j, i), dtype=tf.float32,
                                                shape=(gate_output_tasks[i].get_shape()[-1], experts_units, experts_num),
                                                initializer=tf.glorot_uniform_initializer())
                    experts_bias_task = tf.get_variable(name="experts_bias_task{}_{}".format(j, i), dtype=tf.float32,
                                                shape=(experts_units, experts_num),
                                                initializer=tf.glorot_uniform_initializer())
                    gate_weight_task = tf.get_variable(name="gate_weight_task{}_{}".format(j, i), dtype=tf.float32,
                                        shape=(gate_output_tasks[i].get_shape()[-1], experts_num * 2),
                                        initializer=tf.glorot_uniform_initializer())
                    gate_bias_task = tf.get_variable(name="gate_bias_task{}_{}".format(j, i), dtype=tf.float32,
                                        shape=(experts_num * 2),
                                        initializer=tf.glorot_uniform_initializer())
                    experts_out_task = self.prelu(tf.add(tf.tensordot(gate_output_tasks[i], experts_weight_task, axes=1), experts_bias_task), name="experts_out_task{}_{}".format(j, i))
                    experts_out_tasks.append(experts_out_task)
                    gate_output_task = tf.matmul(gate_output_tasks[i], gate_weight_task)
                    gate_output_task = tf.add(gate_output_task, gate_bias_task)
                    gate_output_task = tf.nn.softmax(gate_output_task)
                    gate_output_task = tf.multiply(tf.concat([experts_out_task, experts_out_shared], axis=2), tf.expand_dims(gate_output_task, axis=1))
                    gate_output_task = tf.reduce_sum(gate_output_task, axis=2)
                    gate_output_tasks[i] = tf.reshape(gate_output_task, [-1, experts_units])
                    # experts_out_shared = tf.concat([experts_out_shared, experts_out_task], axis=2)
                gate_output_shared = tf.matmul(gate_output_shared_final, gate_shared_weight)
                gate_output_shared = tf.add(gate_output_shared, gate_shared_bias)
                gate_output_shared = tf.nn.softmax(gate_output_shared)
                for num in range(num_labels):
                    experts_out_shared = tf.concat([experts_out_shared, experts_out_tasks[num]], axis=2)
                gate_output_shared = tf.multiply(experts_out_shared, tf.expand_dims(gate_output_shared, axis=1))
                gate_output_shared = tf.reduce_sum(gate_output_shared, axis=2)
                gate_output_shared_final = tf.reshape(gate_output_shared, [-1, experts_units])
            return gate_output_tasks
    
    def multi_head_layers(self, input, name=""):
        with tf.variable_scope("multi_head_layers" + name, reuse=kuiba_utils.reuse_variables()):
            name = "idx_" + name
            if kuiba_utils.train_mode():
                tf.summary.histogram(name + "_head", input)
            output = self.dense_with_prelu(input, 128, name=name + '_head_1')
            output = self.dense_with_prelu(input, 128, name=name + '_head_2')
            output = tf.layers.dense(output, 1, name=name + '_head_2', activation=tf.nn.sigmoid)
            if kuiba_utils.train_mode():
                tf.summary.histogram(name + "_output", output)
        return output


    # 三目标
    def multi_level_extraction_network3(self, hidden_layer, num_level, experts_units, experts_num, name="ple_common"):
        with tf.variable_scope("multi_level_extraction_network", reuse=kuiba_utils.reuse_variables()):
            gate_output_task1_final = tf.stop_gradient(hidden_layer)
            gate_output_task2_final = hidden_layer
            gate_output_task3_final = hidden_layer
            gate_output_shared_final = hidden_layer
            for i in range(num_level):
                experts_weight = tf.get_variable(name="experts_weight{}".format(i), dtype=tf.float32,
                                                shape=(gate_output_shared_final.get_shape()[-1], experts_units, experts_num),
                                                initializer=tf.glorot_uniform_initializer())
                experts_bias = tf.get_variable(name="experts_bias{}".format(i), dtype=tf.float32,
                                                shape=(experts_units, experts_num),
                                                initializer=tf.glorot_uniform_initializer())
                experts_weight_task1 = tf.get_variable(name="experts_weight_task1{}".format(i), dtype=tf.float32,
                                                shape=(gate_output_task1_final.get_shape()[-1], experts_units, experts_num),
                                                initializer=tf.glorot_uniform_initializer())
                experts_bias_task1 = tf.get_variable(name="experts_bias_task1{}".format(i), dtype=tf.float32,
                                                shape=(experts_units, experts_num),
                                                initializer=tf.glorot_uniform_initializer())
                experts_weight_task2 = tf.get_variable(name="experts_weight_task2{}".format(i), dtype=tf.float32,
                                                shape=(gate_output_task2_final.get_shape()[-1], experts_units, experts_num),
                                                initializer=tf.glorot_uniform_initializer())
                experts_bias_task2 = tf.get_variable(name="experts_bias_task2{}".format(i), dtype=tf.float32,
                                                shape=(experts_units, experts_num),
                                                initializer=tf.glorot_uniform_initializer())
                experts_weight_task3 = tf.get_variable(name="experts_weight_task3{}".format(i), dtype=tf.float32,
                                                shape=(gate_output_task3_final.get_shape()[-1], experts_units, experts_num),
                                                initializer=tf.glorot_uniform_initializer())
                experts_bias_task3 = tf.get_variable(name="experts_bias_task3{}".format(i), dtype=tf.float32,
                                                shape=(experts_units, experts_num),
                                                initializer=tf.glorot_uniform_initializer())
                gate_weight_t1 = tf.get_variable(name="gate_weight_t1{}".format(i), dtype=tf.float32,
                                        shape=(gate_output_task1_final.get_shape()[-1], experts_num * 2),
                                        initializer=tf.glorot_uniform_initializer())
                gate_bias_t1 = tf.get_variable(name="gate_bias_t1{}".format(i), dtype=tf.float32,
                                        shape=(experts_num * 2),
                                        initializer=tf.glorot_uniform_initializer())
                gate_weight_t2 = tf.get_variable(name="gate_weight_t2{}".format(i), dtype=tf.float32,
                                        shape=(gate_output_task2_final.get_shape()[-1], experts_num * 2),
                                        initializer=tf.glorot_uniform_initializer())
                gate_bias_t2 = tf.get_variable(name="gate_bias_t2{}".format(i), dtype=tf.float32,
                                        shape=(experts_num * 2),
                                        initializer=tf.glorot_uniform_initializer())
                gate_weight_t3 = tf.get_variable(name="gate_weight_t3{}".format(i), dtype=tf.float32,
                                        shape=(gate_output_task3_final.get_shape()[-1], experts_num * 2),
                                        initializer=tf.glorot_uniform_initializer())
                gate_bias_t3 = tf.get_variable(name="gate_bias_t3{}".format(i), dtype=tf.float32,
                                        shape=(experts_num * 2),
                                        initializer=tf.glorot_uniform_initializer())
                gate_shared_weight = tf.get_variable(name="gate_shared_weights{}".format(i),dtype=tf.float32,
                                                    shape=(gate_output_shared_final.get_shape()[-1], experts_num * 4),
                                                    initializer=tf.glorot_uniform_initializer())
                gate_shared_bias = tf.get_variable(name="gate_shared_bias{}".format(i), dtype=tf.float32,
                                                   shape=(experts_num * 4),
                                                   initializer=tf.glorot_uniform_initializer())
                experts_out_s = self.prelu(tf.add(tf.tensordot(gate_output_shared_final, experts_weight, axes=1), experts_bias), name="experts_out_s{}".format(i))
                experts_out_t1 = self.prelu(tf.add(tf.tensordot(gate_output_task1_final, experts_weight_task1, axes=1), experts_bias_task1), name="experts_out_t1{}".format(i))
                experts_out_t2 = self.prelu(tf.add(tf.tensordot(gate_output_task2_final, experts_weight_task2, axes=1), experts_bias_task2), name="experts_out_t2{}".format(i))
                experts_out_t3 = self.prelu(tf.add(tf.tensordot(gate_output_task3_final, experts_weight_task3, axes=1), experts_bias_task3), name="experts_out_t3{}".format(i))
                # gate_out1
                gate_output_task1 = tf.matmul(gate_output_task1_final, gate_weight_t1)
                gate_output_task1 = tf.add(gate_output_task1, gate_bias_t1)
                gate_output_task1 = tf.nn.softmax(gate_output_task1)
                gate_output_task1 = tf.multiply(tf.concat([experts_out_t1, experts_out_s], axis=2), tf.expand_dims(gate_output_task1, axis=1))
                gate_output_task1 = tf.reduce_sum(gate_output_task1, axis=2)
                gate_output_task1_final = tf.reshape(gate_output_task1, [-1, experts_units])
                # gate_out2
                gate_output_task2 = tf.matmul(gate_output_task2_final, gate_weight_t2)
                gate_output_task2 = tf.add(gate_output_task2, gate_bias_t2)
                gate_output_task2 = tf.nn.softmax(gate_output_task2)
                gate_output_task2 = tf.multiply(tf.concat([experts_out_t2, experts_out_s], axis=2), tf.expand_dims(gate_output_task2, axis=1))
                gate_output_task2 = tf.reduce_sum(gate_output_task2, axis=2)
                gate_output_task2_final = tf.reshape(gate_output_task2, [-1, experts_units])
                # gate_out3
                gate_output_task3 = tf.matmul(gate_output_task3_final, gate_weight_t3)
                gate_output_task3 = tf.add(gate_output_task3, gate_bias_t3)
                gate_output_task3 = tf.nn.softmax(gate_output_task3)
                gate_output_task3 = tf.multiply(tf.concat([experts_out_t3, experts_out_s], axis=2), tf.expand_dims(gate_output_task3, axis=1))
                gate_output_task3 = tf.reduce_sum(gate_output_task3, axis=2)
                gate_output_task3_final = tf.reshape(gate_output_task3, [-1, experts_units])
                # gate_shared
                gate_output_shared = tf.matmul(gate_output_shared_final, gate_shared_weight)
                gate_output_shared = tf.add(gate_output_shared, gate_shared_bias)
                gate_output_shared = tf.nn.softmax(gate_output_shared)
                gate_output_shared = tf.multiply(tf.concat([experts_out_s, experts_out_t1, experts_out_t2, experts_out_t3], axis=2), tf.expand_dims(gate_output_shared, axis=1))
                gate_output_shared = tf.reduce_sum(gate_output_shared, axis=2)
                gate_output_shared_final = tf.reshape(gate_output_shared, [-1, experts_units])
        return gate_output_task1_final, gate_output_task2_final, gate_output_task3_final

    def model(self, hot_l2r_dict, hot_ctr_dict, hot_ltr_dict, hot_wtr_dict, hot_profile_dict, hot_longview_dict,
              slide_l2r_dict, slide_ctr_dict, slide_interact_dict, slide_longview_dict, slide_evtr_dict):
        with tf.variable_scope("hot_model", reuse=kuiba_utils.reuse_variables()):
            input_list, _, _, _, _, _ = self.hot_split_inputs_feature(hot_l2r_dict)
            trf_inputs = tf.concat(input_list, axis=1)
            trf_output = self.build_tModel_layers(trf_inputs, 10)  # (batch_size, seq_len, d_model)
            trf_output = tf.reshape(trf_output, shape=(-1, self.seq_len * 64))
            hot_scores_list = []
            for pos in range(self.seq_len):
                trf_output_pos = trf_output[:, pos * 64:(pos + 1) * 64]
                # 共用一个输出
                score_pos = self.multi_head_layers(trf_output_pos)
                # print('pos_{} score = {}'.format(str(pos), str(score_pos)))
                hot_scores_list.append(score_pos)
            hot_model_out = self.get_model_out(hot_scores_list, hot_l2r_dict)

        with tf.variable_scope("hot_aux_model", reuse=kuiba_utils.reuse_variables()):
            _, ctr_scores_list, ltr_scores_list, wtr_scores_list, profile_socres_list, longview_socres_list = self.hot_split_inputs_feature(hot_ctr_dict)
            hot_ctr_model_out = self.get_aux_model_out(ctr_scores_list, hot_ctr_dict, name="valid_click_pos_label", index=1)
            hot_ltr_model_out = self.get_aux_model_out(ltr_scores_list, hot_ltr_dict, name="valid_like_pos_label", index=2)
            hot_wtr_model_out = self.get_aux_model_out(wtr_scores_list, hot_wtr_dict, name="valid_follow_pos_label", index=3)
            hot_profile_model_out = self.get_aux_model_out(profile_socres_list, hot_profile_dict, name="valid_profile_pos_label", index=4)
            hot_longview_model_out = self.get_aux_model_out(longview_socres_list, hot_longview_dict, name="valid_longview_pos_label", index=5)


        with tf.variable_scope("slide_model", reuse=kuiba_utils.reuse_variables()):
            input_list, _, _, _, _ = self.slide_split_inputs_feature(slide_l2r_dict)
            trf_inputs = tf.concat(input_list, axis=1)
            trf_output = self.build_tModel_layers(trf_inputs, 6)  # (batch_size, seq_len, d_model)
            trf_output = tf.reshape(trf_output, shape=(-1, 6 * 64))
            slide_scores_list = []
            for pos in range(6):
                trf_output_pos = trf_output[:, pos * 64:(pos + 1) * 64]
                # 共用一个输出
                score_pos = self.slide_head_layers(trf_output_pos)
                # print('pos_{} score = {}'.format(str(pos), str(score_pos)))
                slide_scores_list.append(score_pos)
            slide_model_out = self.get_slide_model_out(slide_scores_list, slide_l2r_dict)

        with tf.variable_scope("slide_aux_model", reuse=kuiba_utils.reuse_variables()):
            _, ctr_scores_list, interact_scores_list, longview_socres_list, evtr_socres_list = self.slide_split_inputs_feature(slide_ctr_dict)
            slide_ctr_model_out = self.get_slide_aux_model_out(ctr_scores_list, slide_ctr_dict, name="slide_click_pos_label", index=1)
            slide_interact_model_out = self.get_slide_aux_model_out(interact_scores_list, slide_interact_dict, name="slide_interact_pos_label", index=4)
            slide_longview_model_out = self.get_slide_aux_model_out(longview_socres_list, slide_longview_dict, name="slide_longview_pos_label", index=2)
            slide_evtr_model_out = self.get_slide_aux_model_out(evtr_socres_list, slide_evtr_dict, name="slide_evtr_pos_label", index=3)
        if kuiba_utils.predict_mode():
            hot_scores_list = tf.identity(tf.concat(hot_scores_list, axis=1), "scores_n")
            slide_scores_list = tf.identity(tf.concat(slide_scores_list, axis=1), "scores_n_slide")

        return hot_model_out, hot_ctr_model_out, hot_ltr_model_out, hot_wtr_model_out, hot_profile_model_out, hot_longview_model_out, slide_model_out, slide_ctr_model_out, slide_interact_model_out, slide_longview_model_out, slide_evtr_model_out


    def get_reward(self, inputs_dict):
        rewards_list = []
        label_list = []
        for pos in range(self.seq_len):
            rewards_list.append(tf.cast(inputs_dict["pvalid_click_weight" + "_idx" + str(pos)], tf.float32))
            label_list.append(tf.cast(inputs_dict["valid_click_pos_label" + "_idx" + str(pos)], tf.float32))
        return label_list,rewards_list

    def get_aux_reward(self, inputs_dict, name=""):
        rewards_list = []
        label_list = []
        for pos in range(self.seq_len):
            rewards_list.append(tf.cast(1.0, tf.float32))
            label_list.append(tf.cast(inputs_dict[name + "_idx" + str(pos)], tf.float32))
        return label_list,rewards_list


    def get_slide_reward(self, inputs_dict):
        rewards_list = []
        label_list = []
        for pos in range(6):
            rewards_list.append(tf.cast(inputs_dict["pvalid_slide_click_weight" + "_idx" + str(pos)], tf.float32))
            label_list.append(tf.cast(inputs_dict["slide_evtr_pos_label" + "_idx" + str(pos)], tf.float32))
        return label_list,rewards_list

    def get_slide_aux_reward(self, inputs_dict, name=""):
        rewards_list = []
        label_list = []
        for pos in range(6):
            rewards_list.append(tf.cast(1.0, tf.float32))
            label_list.append(tf.cast(inputs_dict[name + "_idx" + str(pos)], tf.float32))
        return label_list,rewards_list

    def get_model_out(self, scores_list, inputs_dict):
        model_out = ModelOut(scores_list[0])
        label_list, rewards_list = self.get_reward(inputs_dict)

        def loss_function_(logits, predictions, weights, labels_tensor):
            # top k
            calLoss = CalLoss(self.seq_len)
            loss_sum = calLoss.calLoss(scores_list, label_list, rewards_list)

            def _metric_aplha_element(loss_name, group_key, item_key, label, loss):
                return [ModelOut.push_metric(
                    tag="pos_{}".format(i), dim=1, item_key=item_key,
                    group_key=group_key, pred=scores_list[i], label=label_list[i], loss=calLoss.loss_list[i]) for i in range(self.seq_len)]

            model_out.set_group_name("user_id")
            model_out.set_item_name("user_id")
            model_out.set_loss_name(self.hot_loss_name_list[0])
            model_out.set_metric_function_list(_metric_aplha_element)
            return loss_sum

        model_out.set_loss_function(loss_function_, True)
        return model_out
    
    def get_slide_model_out(self, scores_list, inputs_dict):
        model_out = ModelOut(scores_list[0])
        label_list, rewards_list = self.get_slide_reward(inputs_dict)

        def loss_function_(logits, predictions, weights, labels_tensor):
            # top k
            calLoss = CalLoss(6)
            loss_sum = calLoss.calLoss(scores_list, label_list, rewards_list)

            def _metric_aplha_element(loss_name, group_key, item_key, label, loss):
                return [ModelOut.push_metric(
                    tag="slide_{}".format(i), dim=1, item_key=item_key,
                    group_key=group_key, pred=scores_list[i], label=label_list[i], loss=calLoss.loss_list[i]) for i in range(6)]

            model_out.set_group_name("user_id")
            model_out.set_item_name("user_id")
            model_out.set_loss_name(self.slide_loss_name_list[0])
            model_out.set_metric_function_list(_metric_aplha_element)
            return loss_sum

        model_out.set_loss_function(loss_function_, True)
        return model_out

    def get_aux_model_out(self, scores_list, inputs_dict, name, index):
        model_out = ModelOut(scores_list[0])
        label_list, rewards_list = self.get_aux_reward(inputs_dict, name=name)
        name_str = ['hot_ctr', 'hot_like', 'hot_follow', 'hot_profile', 'hot_longview']
        def loss_function_(logits, predictions, weights, labels_tensor):
            # top k
            calLoss = CalLoss(self.seq_len)
            loss_sum = calLoss.calLoss(scores_list, label_list, rewards_list)

            def _metric_aplha_element(loss_name, group_key, item_key, label, loss):
                return [ModelOut.push_metric(
                    tag=name_str[index-1] +"_{}".format(i), dim=1, item_key=item_key,
                    group_key=group_key, pred=scores_list[i], label=label_list[i], loss=calLoss.loss_list[i]) for i in range(self.seq_len)]

            model_out.set_group_name("user_id")
            model_out.set_item_name("user_id")
            model_out.set_loss_name(self.hot_loss_name_list[index])
            model_out.set_metric_function_list(_metric_aplha_element)
            return loss_sum

        model_out.set_loss_function(loss_function_, True)
        return model_out
    
    def get_slide_aux_model_out(self, scores_list, inputs_dict, name, index):
        model_out = ModelOut(scores_list[0])
        label_list, rewards_list = self.get_slide_aux_reward(inputs_dict, name=name)
        name_str = ["slide_ctr", 'slide_longview', 'slide_evtr', 'slide_interact']
        def loss_function_(logits, predictions, weights, labels_tensor):
            # top k
            calLoss = CalLoss(6)
            loss_sum = calLoss.calLoss(scores_list, label_list, rewards_list)

            def _metric_aplha_element(loss_name, group_key, item_key, label, loss):
                return [ModelOut.push_metric(
                    tag=name_str[index-1]+"_{}".format(i), dim=1, item_key=item_key,
                    group_key=group_key, pred=scores_list[i], label=label_list[i], loss=calLoss.loss_list[i]) for i in range(6)]

            model_out.set_group_name("user_id")
            model_out.set_item_name("user_id")
            model_out.set_loss_name(self.slide_loss_name_list[index])
            model_out.set_metric_function_list(_metric_aplha_element)
            return loss_sum

        model_out.set_loss_function(loss_function_, True)
        return model_out
class CalLoss():
    def __init__(self, dim):
        self.dim = dim
        self.loss_list = []

    def calLoss(self, scores_list, label_list, rewards_list):
        for i in range(self.dim):
            label = label_list[i]
            weight = rewards_list[i]
            score = scores_list[i]
            # print('pos_{}, label={}, weight={}, score={}'.format(str(i), str(label), str(weight), str(score)))

            single_loss = tf.losses.log_loss(labels=label, predictions=score, weights=weight,
                                          reduction=Reduction.NONE)
            self.loss_list.append(single_loss)
        return tf.reduce_sum(tf.concat(self.loss_list,axis=1))
    
