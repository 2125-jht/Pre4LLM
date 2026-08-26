import tensorflow as tf
import numpy as np
import time

def test_bucketize_performance():
    """测试 tf.raw_ops.Bucketize 的性能"""
    
    # 测试数据准备
    data_sizes = [1000, 10000, 100000, 1000000]
    boundary_counts = [10, 100, 1000]
    
    print("tf.raw_ops.Bucketize 性能测试")
    print("=" * 50)
    
    for data_size in data_sizes:
        for boundary_count in boundary_counts:
            # 生成测试数据
            input_data = tf.random.uniform([data_size], 0, 100, dtype=tf.float32)
            boundaries = np.linspace(1, 99, boundary_count).tolist()
            
            # 测试 tf.raw_ops.Bucketize
            start_time = time.time()
            for _ in range(100):  # 重复100次取平均
                result = tf.raw_ops.Bucketize(input=input_data, boundaries=boundaries)
                _ = result.numpy()  # 强制执行
            bucketize_time = (time.time() - start_time) / 100
            
            # 测试等价的searchsorted方法
            boundaries_tensor = tf.constant(boundaries, dtype=tf.float32)
            start_time = time.time()
            for _ in range(100):
                result = tf.searchsorted(boundaries_tensor, input_data, side='right')
                _ = result.numpy()
            searchsorted_time = (time.time() - start_time) / 100
            
            print(f"数据量: {data_size:>7}, 边界数: {boundary_count:>4}")
            print(f"  Bucketize:   {bucketize_time*1000:.3f} ms")
            print(f"  SearchSorted: {searchsorted_time*1000:.3f} ms")
            print(f"  速度比: {searchsorted_time/bucketize_time:.2f}x")
            print()

def test_memory_efficiency():
    """测试内存使用效率"""
    print("内存效率测试")
    print("=" * 30)
    
    # 大数据集测试
    large_data = tf.random.uniform([10000000], 0, 1000, dtype=tf.float32)
    boundaries = list(range(0, 1000, 10))  # 100个边界
    
    # 监控内存使用
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    
    # 执行前内存
    mem_before = process.memory_info().rss / 1024 / 1024
    
    # 执行bucketize
    start_time = time.time()
    result = tf.raw_ops.Bucketize(input=large_data, boundaries=boundaries)
    _ = result.numpy()
    execution_time = time.time() - start_time
    
    # 执行后内存
    mem_after = process.memory_info().rss / 1024 / 1024
    
    print(f"数据量: 10M 元素")
    print(f"边界数: {len(boundaries)}")
    print(f"执行时间: {execution_time*1000:.2f} ms")
    print(f"内存增量: {mem_after - mem_before:.2f} MB")
    print(f"吞吐量: {len(large_data)/execution_time/1000000:.2f} M元素/秒")

def compare_with_manual_implementation():
    """与手动实现对比"""
    print("与手动实现对比")
    print("=" * 30)
    
    def manual_bucketize(values, boundaries):
        """手动实现的分桶函数"""
        result = []
        for val in values:
            bucket = 0
            for i, boundary in enumerate(boundaries):
                if val >= boundary:
                    bucket = i + 1
                else:
                    break
            result.append(bucket)
        return result
    
    # 测试数据
    test_data = np.random.uniform(0, 100, 10000)
    boundaries = [10, 25, 50, 75, 90]
    
    # TensorFlow实现
    tf_data = tf.constant(test_data, dtype=tf.float32)
    start_time = time.time()
    tf_result = tf.raw_ops.Bucketize(input=tf_data, boundaries=boundaries)
    tf_result = tf_result.numpy()
    tf_time = time.time() - start_time
    
    # 手动实现
    start_time = time.time()
    manual_result = manual_bucketize(test_data, boundaries)
    manual_time = time.time() - start_time
    
    print(f"TensorFlow: {tf_time*1000:.2f} ms")
    print(f"手动实现: {manual_time*1000:.2f} ms")
    print(f"加速比: {manual_time/tf_time:.1f}x")
    
    # 验证结果一致性
    print(f"结果一致性: {np.array_equal(tf_result, manual_result)}")

if __name__ == "__main__":
    test_bucketize_performance()
    test_memory_efficiency()
    compare_with_manual_implementation()
