import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def get_position_encoding(seq_len, dim, dtype=tf.float32):
    """
    Sin-Cos positional encoding.  返回形状 [1, seq_len, dim]，
    方便直接与 batch 张量相加（广播）。
    """
    # [seq_len, 1]
    position = tf.cast(tf.range(seq_len), dtype)[:, tf.newaxis]         

    # [dim/2]：偶数维的角频率
    div_term = tf.exp(
        tf.cast(tf.range(0, dim, 2), dtype) *
        -(tf.math.log(tf.constant(10000.0, dtype=dtype)) / tf.cast(dim, dtype))
    )                                            

    # (seq_len, dim/2)
    angles = position * div_term                 

    # 交替填充 sin / cos
    sin_part = tf.sin(angles)
    cos_part = tf.cos(angles)

    # interleave: [seq_len, dim]
    pos_encoding = tf.reshape(
        tf.stack([sin_part, cos_part], axis=-1),  # (seq_len, dim/2, 2)
        [seq_len, dim]
    )

    # 添加 batch 维，便于后续广播到 [batch, seq_len, dim]
    return pos_encoding[tf.newaxis, ...]          # [1, seq_len, dim]

def get_position_encoding_reference(seq_len, dim):
    """
    参考实现：按照原始Transformer论文的公式
    """
    pos_encoding = np.zeros((seq_len, dim))
    
    for pos in range(seq_len):
        for i in range(0, dim, 2):
            angle = pos / np.power(10000, 2 * i / dim)
            pos_encoding[pos, i] = np.sin(angle)
            if i + 1 < dim:
                pos_encoding[pos, i + 1] = np.cos(angle)
    
    return pos_encoding

def test_position_encoding():
    print("=== 测试Position Encoding实现 ===")
    
    seq_len = 10
    dim = 8
    
    with tf.Session() as sess:
        # 测试我们的实现
        pe_tf = get_position_encoding(seq_len, dim)
        pe_result = sess.run(pe_tf)[0]  # 去掉batch维度
        
        # 参考实现
        pe_ref = get_position_encoding_reference(seq_len, dim)
        
        # 检查形状
        print(f"TensorFlow实现形状: {pe_result.shape}")
        print(f"参考实现形状: {pe_ref.shape}")
        
        # 检查数值差异
        diff = np.abs(pe_result - pe_ref)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        print(f"最大差异: {max_diff:.10f}")
        print(f"平均差异: {mean_diff:.10f}")
        
        # 检查sin/cos模式
        print("\n=== 检查Sin/Cos交替模式 ===")
        print("前几个位置的前8维:")
        for pos in range(3):
            print(f"位置{pos}: {pe_result[pos, :8]}")
            
        # 验证sin/cos关系
        print("\n=== 验证Sin/Cos关系 ===")
        for i in range(0, min(4, dim), 2):
            sin_val = pe_result[0, i]
            cos_val = pe_result[0, i+1] if i+1 < dim else 0
            sin_cos_sum = sin_val**2 + cos_val**2
            print(f"维度{i},{i+1}: sin²+cos² = {sin_cos_sum:.6f}")
            
        # 检查是否通过
        if max_diff < 1e-6:
            print("\n✅ Position Encoding实现正确!")
        else:
            print(f"\n❌ Position Encoding实现有问题，最大差异: {max_diff}")
            
        return pe_result, pe_ref, max_diff < 1e-6

if __name__ == "__main__":
    test_position_encoding()
