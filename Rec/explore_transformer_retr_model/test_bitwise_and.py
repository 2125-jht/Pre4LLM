#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

# 兼容 TF2
if tf.__version__.startswith("2"):
    tf.compat.v1.disable_eager_execution()
    tf = tf.compat.v1

def test_bitwise_and():
    """测试 tf.bitwise.bitwise_and(a, 0x7FFF) 的效果"""
    
    # 创建测试数据
    test_values = [
        100,        # 小数
        32767,      # 0x7FFF (15位全1)
        32768,      # 0x8000 (第16位为1)
        65535,      # 0xFFFF (16位全1)
        100000,     # 大数
        -1,         # 负数
        -100,       # 负数
    ]
    
    a = tf.constant(test_values, dtype=tf.int32)
    result = tf.bitwise.bitwise_and(a, 0x7FFF)
    
    with tf.Session() as sess:
        original, masked = sess.run([a, result])
        
    print("tf.bitwise.bitwise_and(a, 0x7FFF) 示例:")
    print("=" * 60)
    print(f"{'原值':<8} {'十六进制':<10} {'二进制':<18} {'结果':<8} {'结果(hex)':<10}")
    print("-" * 60)
    
    for i, (orig, res) in enumerate(zip(original, masked)):
        # 处理负数的显示
        if orig >= 0:
            orig_hex = f"0x{orig:X}"
            orig_bin = f"{orig:016b}"
        else:
            # 负数用补码表示
            orig_hex = f"-0x{abs(orig):X}"
            orig_bin = f"{orig & 0xFFFFFFFF:032b}"[-16:]  # 只显示低16位
            
        res_hex = f"0x{res:X}"
        
        print(f"{orig:<8} {orig_hex:<10} {orig_bin:<18} {res:<8} {res_hex:<10}")

def explain_operation():
    """解释操作原理"""
    print("\n操作原理解释:")
    print("=" * 60)
    print("0x7FFF = 32767 = 0111111111111111 (二进制)")
    print("按位与运算 (&) 的作用:")
    print("- 保留原数的低15位")
    print("- 清除第16位及以上的所有位")
    print("- 相当于对 32768 取模运算")
    
    print("\n具体例子:")
    examples = [
        (100, "小于32768的数保持不变"),
        (32768, "0x8000 → 0x0000，第16位被清除"),
        (65535, "0xFFFF → 0x7FFF，第16位被清除"),
        (100000, "大数被截断到低15位"),
    ]
    
    for val, desc in examples:
        result = val & 0x7FFF
        print(f"  {val:6d} (0x{val:04X}) & 0x7FFF = {result:5d} (0x{result:04X}) - {desc}")

if __name__ == "__main__":
    test_bitwise_and()
    explain_operation()