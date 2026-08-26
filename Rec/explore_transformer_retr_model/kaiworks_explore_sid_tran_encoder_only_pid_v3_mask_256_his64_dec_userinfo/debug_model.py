#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MultiInterestModel 调试脚本

用于测试和调试 MultiInterestModel.model 方法的功能，
包括模型的前向传播、损失计算和束搜索等功能。
"""

import tensorflow as tf
import numpy as np
import sys
import os

# 导入模型相关模块
from model import SIDRecModel
from modules_ import *
from modulesV2 import *
from util import processInput, processLabel, processOutputV2
def create_mock_feature_dict(batch_size=4, seq_len=10, dim=64):
    """
    创建模拟的特征字典和大小字典
    
    Args:
        batch_size: 批次大小
        seq_len: 序列长度
        dim: 特征维度
        
    Returns:
        feature_emb_dict: 模拟特征嵌入字典
        feature_emb_size_dict: 模拟特征大小字典
    """
    print("=== 创建模拟特征字典 ===")
    
    # 用户静态特征（单值特征）
    feature_emb_dict = {
        "user_id": tf.random.normal([batch_size, dim], dtype=tf.float32),
        "user_gender": tf.random.normal([batch_size, dim], dtype=tf.float32),
        "user_age_segment": tf.random.normal([batch_size, dim], dtype=tf.float32),
        "user_level": tf.random.normal([batch_size, dim], dtype=tf.float32),
    }
    
    # 用户点击行为特征（序列特征，3D张量）
    feature_emb_dict.update({
        "user_profile_v1_click_pid_list": tf.random.normal([batch_size, seq_len, dim], dtype=tf.float32),
        "user_profile_v1_click_aid_list": tf.random.normal([batch_size, seq_len, dim], dtype=tf.float32),
    })
    
    # 特征大小字典（用于序列长度信息）
    feature_emb_size_dict = {
        "user_id": tf.ones([batch_size], dtype=tf.int32),
        "user_gender": tf.ones([batch_size], dtype=tf.int32),
        "user_age_segment": tf.ones([batch_size], dtype=tf.int32),
        "user_level": tf.ones([batch_size], dtype=tf.int32),
        "user_profile_v1_click_pid_list": tf.random.uniform([batch_size], minval=5, maxval=seq_len+1, dtype=tf.int32),
        "user_profile_v1_click_aid_list": tf.random.uniform([batch_size], minval=5, maxval=seq_len+1, dtype=tf.int32),
    }
    
    print(f"✓ 创建了 {len(feature_emb_dict)} 个特征")
    for name, tensor in feature_emb_dict.items():
        print(f"  {name}: {tensor.shape}")
    
    return feature_emb_dict, feature_emb_size_dict

def create_mock_input_data(batch_size=4, seq_len=3):
    """
    创建模拟的输入数据
    
    Args:
        batch_size: 批次大小
        seq_len: 序列长度
        
    Returns:
        photo_semantic_id_int: 原始语义ID
        photo_sid: 处理后的语义ID
        label: 标签数据
    """
    print("=== 创建模拟输入数据 ===")
    
    # 创建模拟的语义ID数据
    # 模拟压缩的语义ID：3个15位的值打包成一个int64
    vocab_sizes = [8192, 8192, 8192]  # 三个层级的词汇表大小
    
    # 随机生成三个层级的语义ID
    a_values = np.random.randint(0, vocab_sizes[0], (batch_size, seq_len))
    b_values = np.random.randint(0, vocab_sizes[1], (batch_size, seq_len))
    c_values = np.random.randint(0, vocab_sizes[2], (batch_size, seq_len))
    
    # 打包成压缩格式：a占高15位，b占中15位，c占低15位
    photo_semantic_id_int_np = (
        (a_values.astype(np.int64) << 30) + 
        (b_values.astype(np.int64) << 15) + 
        c_values.astype(np.int64)
    )
    
    photo_semantic_id_int = tf.constant(photo_semantic_id_int_np, dtype=tf.int64)
    
    # 处理输入数据
    photo_sid = processInput(photo_semantic_id_int)  # [batch_size, seq_len*3]
    label = processLabel(photo_semantic_id_int)      # [batch_size, seq_len*3]
    
    print(f"✓ photo_semantic_id_int: {photo_semantic_id_int.shape}")
    print(f"✓ photo_sid: {photo_sid.shape}")
    print(f"✓ label: {label.shape}")
    
    return photo_semantic_id_int, photo_sid, label

def test_model_forward():
    """
    测试模型的前向传播
    """
    print("\n" + "="*50)
    print("开始测试 MultiInterestModel.model 方法")
    print("="*50)
    
    # 设置参数
    batch_size = 4
    seq_len = 3
    dim = 64
    vocab_sizes = [8192, 8192, 8192]
    
    with tf.Session() as sess:
        # 1. 创建模拟数据
        feature_emb_dict, feature_emb_size_dict = create_mock_feature_dict(batch_size, seq_len, dim)
        photo_semantic_id_int, photo_sid, label = create_mock_input_data(batch_size, seq_len)
        
        # 2. 创建模型实例
        print("\n=== 创建模型实例 ===")
        print_ops = []
        model = SIDRecModel(
            feature_emb_dict=feature_emb_dict,
            feature_emb_size_dict=feature_emb_size_dict,
            dim=dim,
            vocab_sizes=vocab_sizes,
            print_ops=print_ops
        )
        print("✓ 模型创建成功")
        # 3. 前向传播
        print("\n=== 模型前向传播 ===")
        try:
            loss = model.model(photo_sid, label, photo_semantic_id_int)
            print(f"✓ 前向传播成功，损失张量形状: {loss.shape}")
            
            # 初始化变量
            sess.run(tf.global_variables_initializer())
            
            # 运行模型
            loss_value = sess.run(loss)
            print(f"✓ 模型执行成功")
            print(f"  损失值: {loss_value:.6f}")
            print(f"  损失类型: {type(loss_value)}")
            
        except Exception as e:
            print(f"✗ 前向传播失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 4. 测试束搜索
        print("\n=== 测试束搜索 ===")
        try:
            beam_size = 8
            beam_result = model.beam_search(beam_size=beam_size)
            print(f"✓ 束搜索定义成功，结果形状: {beam_result.shape}")
            
            # 重新初始化变量（因为可能有新的变量）
            sess.run(tf.global_variables_initializer())
            
            # 运行束搜索
            beam_sequences = sess.run(beam_result)
            print(f"✓ 束搜索执行成功")
            print(f"  输出形状: {beam_sequences.shape}")
            print(f"  输出数据类型: {beam_sequences.dtype}")
            print(f"  输出值范围: [{beam_sequences.min()}, {beam_sequences.max()}]")
            
            # 显示一些样本结果
            print(f"  第一个样本的前3个beam:")
            for i in range(min(3, beam_sequences.shape[1])):
                print(f"    Beam {i}: {beam_sequences[0, i, :]}")
                
        except Exception as e:
            print(f"✗ 束搜索失败: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        return True

def test_data_processing():
    """
    测试数据处理函数
    """
    print("\n" + "="*50)
    print("测试数据处理函数")
    print("="*50)
    
    # 创建测试数据
    batch_size = 2
    seq_len = 4
    
    # 模拟原始语义ID
    test_data = np.array([
        [123456789, 987654321, 555666777, 111222333],
        [444555666, 777888999, 123123123, 456456456]
    ], dtype=np.int64)
    
    input_tensor = tf.constant(test_data)
    
    with tf.Session() as sess:
        # 测试processInput
        print("=== 测试 processInput ===")
        processed_input = processInput(input_tensor)
        result_input = sess.run(processed_input)
        print(f"原始输入形状: {input_tensor.shape}")
        print(f"处理后形状: {processed_input.shape}")
        print(f"处理结果:")
        print(result_input)
        
        # 测试processLabel
        print("\n=== 测试 processLabel ===")
        processed_label = processLabel(input_tensor)
        result_label = sess.run(processed_label)
        print(f"标签处理后形状: {processed_label.shape}")
        print(f"标签结果:")
        print(result_label)
        
        # 测试processOutputV2
        print("\n=== 测试 processOutputV2 ===")
        # 创建3D输入用于测试processOutputV2
        test_3d = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=tf.int32)
        processed_output = processOutputV2(test_3d)
        result_output = sess.run(processed_output)
        print(f"3D输入形状: {test_3d.shape}")
        print(f"输出处理后形状: {processed_output.shape}")
        print(f"输出结果:")
        print(result_output)

def test_model_components():
    """
    测试模型组件
    """
    print("\n" + "="*50)
    print("测试模型组件")
    print("="*50)
    
    batch_size = 2
    seq_len = 5
    dim = 32
    
    with tf.Session() as sess:
        # 测试MLP
        print("=== 测试 MLP ===")
        input_data = tf.random.normal([batch_size, dim])
        mlp_output = mlp("test_mlp", input_data, [64, 32], 16)
        print(f"MLP输入形状: {input_data.shape}")
        print(f"MLP输出形状: {mlp_output.shape}")
        
        # 测试similarity
        print("\n=== 测试 similarity ===")
        emb_data = tf.random.normal([batch_size, dim])
        similarity(emb_data, name="test_sim")
        print("✓ 相似度计算成功")
        
        # 测试recall_at_k
        print("\n=== 测试 recall_at_k ===")
        predictions = tf.random.normal([batch_size, 100])  # 100个类别的预测
        labels = tf.random.uniform([batch_size], maxval=100, dtype=tf.int32)
        mask = tf.ones([batch_size])
        recall_at_k(predictions, labels, mask, [], top_k=[1, 5, 10], name="test_recall")
        print("✓ recall_at_k 计算成功")
        
        # 初始化并运行
        sess.run(tf.global_variables_initializer())
        mlp_result = sess.run(mlp_output)
        print(f"✓ MLP执行成功，输出形状: {mlp_result.shape}")

if __name__ == "__main__":
    print("MultiInterestModel 调试脚本")
    print("TensorFlow 版本:", tf.__version__)
    
    # 设置TensorFlow日志级别
    tf.logging.set_verbosity(tf.logging.ERROR)
    
    try:
        # 1. 测试数据处理函数
        test_data_processing()
        
        # 2. 测试模型组件
        test_model_components()
        
        # 3. 测试完整模型
        success = test_model_forward()
        
        if success:
            print("\n" + "="*50)
            print("🎉 所有测试完成！模型运行正常！")
            print("="*50)
        else:
            print("\n" + "="*50)
            print("❌ 测试失败！请检查错误信息")
            print("="*50)
            
    except Exception as e:
        print(f"\n❌ 调试脚本执行失败: {e}")
        import traceback
        traceback.print_exc()
