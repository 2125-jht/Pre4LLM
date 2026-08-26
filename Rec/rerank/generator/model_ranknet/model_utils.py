import os
import sys
import json
import logging
import argparse
import functools
import tensorflow as tf
import numpy


def gen_autodis_net(feas, feas_num, head=3, units=[8], scope="default"):
    # 默认feas中特征都是numric，feas.shape[-1] 和feas_num一样
    def gen_meta_emb(feas, feas_num, head, units):
        emb_output = []
        for k in range(head):
            emb_net = feas
            for idx, unit in enumerate(units):
                emb_net = tf.layers.dense(
                    emb_net,
                    unit*feas_num,
                    activation=tf.nn.leaky_relu,
                    kernel_initializer=tf.keras.initializers.he_normal(seed=1),
                    name='autodis_meta_emb{}_layer{}'.format(k, idx))
            emb_output.append(tf.reshape(emb_net, [-1,feas_num,units[-1]]))
        emb_output = tf.stack(emb_output, axis=-2)
        print("cying_emb_output：", emb_output)
        return emb_output #b * fea_num * head * dim

    def gen_auto_dis(feas, feas_num, head, t=1.0): 
        feas = tf.expand_dims(feas, -1) #batch * feas_num * 1
        auto_kernels = tf.get_variable(name="auto_kernel", shape=[feas_num, head], initializer=tf.keras.initializers.he_normal(seed=1)) 
        auto_kernels = tf.expand_dims(auto_kernels,0)# 1 * feas_num * head
        feas = tf.transpose(feas, (1,0,2)) 
        auto_kernels = tf.transpose(auto_kernels, (1,0,2))
        auto_output = tf.matmul(feas, auto_kernels) # feas_num * batch * head
        auto_output = tf.nn.softmax(auto_output/t)
        auto_output = tf.transpose(auto_output, (1,0,2))
        print("cying_auto_emb:", auto_output)
        return auto_output  #batch * feas_num * head

    emb = gen_meta_emb(feas, feas_num, head, units)
    wt = gen_auto_dis(feas, feas_num, head)
    autodis_emb = tf.multiply(tf.reshape(wt,[-1,feas_num,head,1]), emb) #feas_num * batch * embdim ,其中autodis_embedding维度为bacth * feas_num * head * embdim
    autodis_emb = tf.reduce_sum(autodis_emb, axis=-2)
    return autodis_emb
   
def multi_head_attention(query, key, head_num=1, mask=True, scope = "name"):
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    dim = key.get_shape().as_list()[-1]
    q = tf.layers.dense(query, head_num * dim, activation=tf.nn.leaky_relu)
    k = tf.layers.dense(key, head_num * dim, activation=tf.nn.leaky_relu)
    v = tf.layers.dense(key, head_num * dim, activation=tf.nn.leaky_relu)
    
    q = tf.concat(tf.split(q,head_num,axis=-1),axis=0)
    k = tf.concat(tf.split(k,head_num,axis=-1),axis=0)
    v = tf.concat(tf.split(v,head_num,axis=-1),axis=0)

    output = tf.matmul(q, tf.transpose(k,[0,2,1]))
    output = output/(dim**0.5)

    #去掉padding值
    key_mask = tf.sign(tf.abs(tf.reduce_sum(key, axis=-1)))
    key_mask = tf.tile(key_mask, [head_num,1])
    key_mask = tf.tile(tf.expand_dims(key_mask, axis=1),[1,query.get_shape().as_list()[1],1])
    padding = tf.ones_like(output)*(-2**32+1)
    output = tf.where(tf.equal(key_mask,0),padding,output)
    if mask :
      diag_val = tf.ones_like(output[0::])
      mask = tf.linalg.LinearOperatorLowerTriangular(diag_val).to_dense() # 下三角，队首为最近看的，队尾是很久以前看的
      padding = tf.ones_like(output)*(-2**32+1)
      output = tf.where(tf.equal(mask,0),padding,output)

    output = tf.nn.softmax(output)
    query_mask = tf.sign(tf.abs(tf.reduce_sum(query,axis=-1)))
    query_mask = tf.tile(query_mask,[head_num,1])
    query_mask = tf.tile(tf.expand_dims(query_mask,-1),[1,1,key.get_shape().as_list()[1]])

    output = output * query_mask
    output = tf.matmul(output,v)
    output = tf.concat(tf.split(output,head_num,axis=0),axis=-1)
  return output
      