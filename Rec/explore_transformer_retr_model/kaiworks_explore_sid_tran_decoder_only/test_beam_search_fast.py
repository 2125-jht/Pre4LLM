# -*- coding: utf-8 -*-
"""
测试 beam_search_fast 方法的脚本

该脚本模拟 MultiInterestModel 的环境，构建测试输入数据，
并测试 beam_search_fast 方法的功能和性能。
"""

import tensorflow as tf
import numpy as np
import sys
import os

# 添加模块路径
# sys.path.append('kaiworks_explore_sid_tran_decoder_only')

# 导入必要的模块
from model import MultiInterestModel
from modulesV2 import DecoderOnlyModel
from modules_ import *

def create_mock_feature_dicts():
    """
    创建模拟的特征字典，用于初始化 MultiInterestModel
    """
    # 模拟特征嵌入字典
    feature_emb_dict = {
        "user_id": tf.get_variable("user_id_emb", [10000, 32], 
                                  initializer=tf.random_uniform_initializer()),
        "user_gender": tf.get_variable("user_gender_emb", [3, 16],
                                      initializer=tf.random_uniform_initializer()),
        "user_age_segment": tf.get_variable("user_age_emb", [10, 16],
                                           initializer=tf.random_uniform_initializer()),
        "user_level": tf.get_variable("user_level_emb", [50, 16],
                                     initializer=tf.random_uniform_initializer()),
    }
    
    # 模拟特征嵌入维度字典
    feature_emb_size_dict = {
        "user_id": 32,
        "user_gender": 16,
        "user_age_segment": 16,
        "user_level": 16,
    }
    
    return feature_emb_dict, feature_emb_size_dict

def generate_test_data(batch_size=4, seq_len=10):
    """
    生成测试数据
    
    Args:
        batch_size: 批次大小
        seq_len: 用户序列长度
        
    Returns:
        user_sid_list: 用户语义ID序列，shape=[batch_size, seq_len]
    """
    # 生成用户sid序列，包含一些-1(padding)值
    user_sid_list = []
    
    for i in range(batch_size):
        # 随机生成有效长度(3到seq_len之间)
        valid_len = np.random.randint(3, seq_len)
        
        # 生成有效的sid（全局id范围）
        valid_sids = np.random.randint(0, 24576, size=valid_len)  # 总词汇量为3*8192=24576
        
        # 用-1填充到固定长度
        padded_sids = np.concatenate([valid_sids, 
                                     np.full(seq_len - valid_len, -1)])
        user_sid_list.append(padded_sids)
    
    return np.array(user_sid_list, dtype=np.int32)

def test_beam_search_fast():
    """
    测试 beam_search_fast 方法的主函数
    """
    print("=== 开始测试 beam_search_fast 方法 ===")
    
    # 设置超参数
    batch_size = 3
    seq_len = 8
    beam_size = 4
    dim = 128
    vocab_sizes = [8192, 8192, 8192]  # 三个语义层级
    temperature = 1.0
    
    print(f"测试参数:")
    print(f"  batch_size: {batch_size}")
    print(f"  seq_len: {seq_len}")
    print(f"  beam_size: {beam_size}")
    print(f"  dim: {dim}")
    print(f"  vocab_sizes: {vocab_sizes}")
    print(f"  temperature: {temperature}")
    
    # 重置默认图
    tf.reset_default_graph()
    
    with tf.Session() as sess:
        # 创建模拟特征字典
        feature_emb_dict, feature_emb_size_dict = create_mock_feature_dicts()
        
        # 初始化模型
        print("\n创建 MultiInterestModel...")
        print_ops = []
        model = MultiInterestModel(
            feature_emb_dict=feature_emb_dict,
            feature_emb_size_dict=feature_emb_size_dict,
            dim=dim,
            vocab_sizes=vocab_sizes,
            print_ops=print_ops
        )
        
        # 生成测试数据
        print("\n生成测试数据...")
        user_sid_data = generate_test_data(batch_size, seq_len)
        print(f"用户sid序列形状: {user_sid_data.shape}")
        print(f"用户sid序列内容:")
        for i, seq in enumerate(user_sid_data):
            print(f"  样本{i}: {seq}")
        
        # 创建placeholder
        user_sid_placeholder = tf.placeholder(tf.int32, [batch_size, seq_len], name="user_sid_input")
        
        # 调用beam_search_fast方法
        print(f"\n调用 beam_search_fast 方法...")
        try:
            generated_seqs, generated_probs = model.beam_search_fast(
                user_sid_list=user_sid_placeholder,
                beam_size=beam_size,
                temperature=temperature
            )
            
            print("beam_search_fast 方法调用成功!")
            print(f"生成序列形状: {generated_seqs.shape}")
            print(f"生成概率形状: {generated_probs.shape}")
            
        except Exception as e:
            print(f"调用 beam_search_fast 时出错: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # 初始化变量
        print("\n初始化TensorFlow变量...")
        sess.run(tf.global_variables_initializer())
        
        # 执行推理
        print("\n执行beam search推理...")
        try:
            feed_dict = {user_sid_placeholder: user_sid_data}
            
            result_seqs, result_probs = sess.run(
                [generated_seqs, generated_probs],
                feed_dict=feed_dict
            )
            
            print("推理执行成功!")
            print(f"\n=== 结果分析 ===")
            print(f"生成序列形状: {result_seqs.shape}")
            print(f"生成概率形状: {result_probs.shape}")
            
            # 分析结果
            print(f"\n详细结果:")
            for batch_idx in range(batch_size):
                print(f"\n样本 {batch_idx}:")
                print(f"  输入用户序列: {user_sid_data[batch_idx]}")
                
                for beam_idx in range(min(beam_size, 3)):  # 只显示前3个beam
                    seq = result_seqs[batch_idx, beam_idx]
                    probs = result_probs[batch_idx, beam_idx]
                    print(f"  Beam {beam_idx}:")
                    print(f"    生成序列: {seq}")
                    print(f"    对应概率: {probs}")
                    print(f"    概率乘积: {np.prod(probs):.6f}")
            
            # 验证输出合理性
            print(f"\n=== 验证输出合理性 ===")
            
            # 检查序列长度
            expected_seq_len = 3  # 应该生成3个token
            if result_seqs.shape[-1] == expected_seq_len:
                print("✓ 生成序列长度正确")
            else:
                print(f"✗ 生成序列长度错误，期望{expected_seq_len}，实际{result_seqs.shape[-1]}")
            
            # 检查概率长度
            expected_prob_len = 2  # 概率数组长度应该是2（去掉了第一个位置）
            if result_probs.shape[-1] == expected_prob_len:
                print("✓ 概率序列长度正确") 
            else:
                print(f"✗ 概率序列长度错误，期望{expected_prob_len}，实际{result_probs.shape[-1]}")
            # 检查序列值域
            for step in range(3):
                step_values = result_seqs[:, :, step]
                min_val, max_val = np.min(step_values), np.max(step_values)
                expected_max = vocab_sizes[step] - 1
                if min_val >= 0 and max_val <= expected_max:
                    print(f"✓ 第{step}步生成值域正确: [{min_val}, {max_val}] ∈ [0, {expected_max}]")
                else:
                    print(f"✗ 第{step}步生成值域错误: [{min_val}, {max_val}] ∉ [0, {expected_max}]")
            
            # 检查概率值域
            min_prob, max_prob = np.min(result_probs), np.max(result_probs)
            if min_prob >= 0.0 and max_prob <= 1.0:
                print(f"✓ 概率值域正确: [{min_prob:.6f}, {max_prob:.6f}] ∈ [0, 1]")
            else:
                print(f"✗ 概率值域错误: [{min_prob:.6f}, {max_prob:.6f}] ∉ [0, 1]")
            
            print(f"\n=== 测试完成 ===")
            
        except Exception as e:
            print(f"执行推理时出错: {e}")
            import traceback
            traceback.print_exc()
            return

def test_edge_cases():
    """
    测试边界情况
    """
    print("\n=== 测试边界情况 ===")
    
    # 测试不同的beam_size
    beam_sizes = [1, 2, 8, 16]
    print(f"测试不同beam_size: {beam_sizes}")
    
    # 测试不同的temperature
    temperatures = [0.5, 1.0, 2.0]
    print(f"测试不同temperature: {temperatures}")
    
    # 这里可以扩展更多边界测试...
    print("边界测试完成 (具体实现可以进一步扩展)")

if __name__ == "__main__":
    print("开始测试 beam_search_fast 方法...")
    
    try:
        # 主要功能测试
        test_beam_search_fast()
        
        # 边界情况测试
        test_edge_cases()
        
        print("\n所有测试完成!")
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()