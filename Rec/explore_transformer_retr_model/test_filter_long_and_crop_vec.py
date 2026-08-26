#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 filter_long_and_crop_vec 函数
验证长播放过滤和序列截断的逻辑
"""

import tensorflow as tf
import numpy as np

def _const_like(x, val):
    """返回与 x 同形、值全为 val 的张量（保持 dtype）"""
    return tf.fill(tf.shape(x), tf.cast(val, x.dtype))

def filter_long_and_crop_vec(seq_emb,
                     play_time, duration, channel,
                     valid_len,
                     max_len,
                     pad_value=0.0,
                     name="filter_long_and_crop_vec"):
    """
    Args
    ----
    seq_emb   : [B, L, D]  float32/16   点击序列 embedding
    play_time : [B, L]     int64        播放时长 (ms)
    duration  : [B, L]     int64        视频总时长 (ms)
    channel   : [B, L]     int64        频道 id
    valid_len : [B]        int32        每条序列真实长度 (≤ L)
    max_len   : int                       只保留最近 max_len 条
    pad_value : float                     左侧 padding 常量

    Returns
    -------
    out_emb      : [B, max_len, D]   截断 + 左 PAD
    longview_len : [B] int32         满足长播条件的条数 (未截断)
    used_len     : [B] int32         min(longview_len, max_len)
    """
    with tf.name_scope(name):
        seq_emb   = tf.convert_to_tensor(seq_emb)
        play_time = tf.convert_to_tensor(play_time)
        duration  = tf.convert_to_tensor(duration)
        channel   = tf.convert_to_tensor(channel)
        valid_len = tf.reshape(tf.cast(valid_len, tf.int32), [-1])      # [B]

        B = tf.shape(seq_emb)[0]
        L = tf.shape(seq_emb)[1]
        D = tf.shape(seq_emb)[2]

        # ---------- ① 计算阈值 thr ----------
        one_like_d = tf.ones_like(duration, dtype=duration.dtype)  # shape [B,L]
        thr_base = _const_like(duration, 79700)                # 默认 79.7s

        thr_base = tf.where(duration <=     0, _const_like(one_like_d, 13100), thr_base)
        thr_base = tf.where(duration <=  8700, _const_like(one_like_d, 12000), thr_base)
        thr_base = tf.where(duration <= 12700, _const_like(one_like_d, 13600), thr_base)
        thr_base = tf.where(duration <= 20300, _const_like(one_like_d, 18400), thr_base)
        thr_base = tf.where(duration <= 38800, _const_like(one_like_d, 28800), thr_base)
        thr_base = tf.where(duration <= 71800, _const_like(one_like_d, 46600), thr_base)
        thr_base = tf.where(duration <=118200, _const_like(one_like_d, 74900), thr_base)
        thr_base = tf.where(duration <=195000, _const_like(one_like_d, 92500), thr_base)

        thr_chan_1     = thr_base // 6
        thr_chan_other = thr_base // 2
        thr = tf.where(tf.equal(channel, 1), thr_chan_1, thr_chan_other)  # shape [B,L]

        # ---------- ② 长播掩码 + 有效长度掩码 ----------
        long_mask  = tf.greater_equal(play_time, thr)                      # [B, L] bool
        rng        = tf.range(L, dtype=tf.int32)[tf.newaxis, :]            # [1, L]
        valid_mask = tf.less(rng, valid_len[:, tf.newaxis])                # [B, L] bool
        keep_mask  = tf.logical_and(long_mask, valid_mask)                 # [B, L] bool

        longview_len = tf.reduce_sum(tf.cast(keep_mask, tf.int32), axis=1) # [B]

        # ---------- ③ 取最近 max_len 条 ----------
        # recency_idx: 越大越新（适配"最新在最前"）
        idx_full    = tf.range(L, dtype=tf.int32)[tf.newaxis, :]   # [1,L] 0,1,2,...
        idx_full    = tf.tile(idx_full, [B, 1])                    # [B,L]
        recency_idx = (L - 1) - idx_full                           # [B,L] 反向

        # 只在 keep_mask 位置保留 recency_idx，其余设 -1
        masked_ridx = tf.where(keep_mask, recency_idx,
                            tf.fill(tf.shape(recency_idx), -1))

        # 取每行 recency_idx 最大的 k 个（=最近的 k 条）
        topk_ridx, _ = tf.nn.top_k(masked_ridx, k=max_len)         # [B,max_len] desc
        ordered_ridx = tf.reverse(topk_ridx, axis=[1])             # asc → 左 PAD
        keep_topk = tf.cast(ordered_ridx >= 0, seq_emb.dtype)      # [B,max_len]
        used_len  = tf.reduce_sum(tf.cast(keep_topk, tf.int32), 1) # [B]

        # 把 recency_idx 反映射回原始列号
        ordered_idx = (L - 1) - ordered_ridx                       # [B,max_len]
        safe_idx    = tf.maximum(ordered_idx, 0)                   # pad 位先占 0

        batch_ids   = tf.tile(tf.range(B, dtype=tf.int32)[:, tf.newaxis],
                            [1, max_len])                        # [B,max_len]
        gather_nd_idx = tf.stack([batch_ids, safe_idx], axis=-1)   # [B,max_len,2]

        gathered = tf.gather_nd(seq_emb, gather_nd_idx)            # [B,max_len,D]
        pad_tensor = tf.fill([B, max_len, D], tf.cast(pad_value, seq_emb.dtype))
        out_emb = keep_topk[:, :, tf.newaxis] * gathered + \
                (1.0 - keep_topk[:, :, tf.newaxis]) * pad_tensor # [B,max_len,D]

        return out_emb, longview_len, used_len

def create_test_data():
    """创建测试数据"""
    batch_size = 2
    seq_len = 6  # 较小的序列长度便于观察
    emb_dim = 4
    max_len = 3
    
    # 创建测试数据
    # batch 0: 5个视频，其中3个长播放
    # batch 1: 4个视频，其中2个长播放
    
    # 视频embedding (用不同值便于区分)
    seq_emb = np.array([
        # batch 0: 6个位置，只有前5个有效
        [[1.0, 0.1, 0.1, 0.1],  # 视频0
         [2.0, 0.2, 0.2, 0.2],  # 视频1  
         [3.0, 0.3, 0.3, 0.3],  # 视频2
         [4.0, 0.4, 0.4, 0.4],  # 视频3
         [5.0, 0.5, 0.5, 0.5],  # 视频4
         [0.0, 0.0, 0.0, 0.0]], # padding
        
        # batch 1: 6个位置，只有前4个有效  
        [[1.1, 0.11, 0.11, 0.11],  # 视频0
         [2.1, 0.21, 0.21, 0.21],  # 视频1
         [3.1, 0.31, 0.31, 0.31],  # 视频2  
         [4.1, 0.41, 0.41, 0.41],  # 视频3
         [0.0, 0.0, 0.0, 0.0],     # padding
         [0.0, 0.0, 0.0, 0.0]]     # padding
    ], dtype=np.float32)
    
    # 播放时长 (ms)
    play_time = np.array([
        [8000, 25000, 5000, 90000, 35000, 0],   # batch 0
        [15000, 3000, 40000, 20000, 0, 0]       # batch 1  
    ], dtype=np.int64)
    
    # 视频总时长 (ms)  
    duration = np.array([
        [10000, 30000, 15000, 120000, 45000, 0],  # batch 0
        [20000, 15000, 50000, 25000, 0, 0]        # batch 1
    ], dtype=np.int64)
    
    # 频道ID
    channel = np.array([
        [2, 1, 2, 2, 1, 0],    # batch 0
        [1, 2, 2, 1, 0, 0]     # batch 1
    ], dtype=np.int64)
    
    # 有效长度
    valid_len = np.array([5, 4], dtype=np.int32)
    
    return seq_emb, play_time, duration, channel, valid_len, max_len

def print_test_analysis():
    """打印测试数据分析"""
    print("=== 测试数据分析 ===")
    print("\nBatch 0 (5个有效视频):")
    print("视频0: 10s视频播放8s,  频道2 → 阈值=6s  → 8s≥6s  ✓长播放")
    print("视频1: 30s视频播放25s, 频道1 → 阈值=4.8s → 25s≥4.8s ✓长播放") 
    print("视频2: 15s视频播放5s,  频道2 → 阈值=6.8s → 5s<6.8s  ✗短播放")
    print("视频3: 120s视频播放90s, 频道2 → 阈值=37.5s → 90s≥37.5s ✓长播放")
    print("视频4: 45s视频播放35s, 频道1 → 阈值=4.8s → 35s≥4.8s ✓长播放")
    print("预期: 4条长播放，取最近3条 → 视频1,3,4")
    
    print("\nBatch 1 (4个有效视频):")
    print("视频0: 20s视频播放15s, 频道1 → 阈值=3.1s → 15s≥3.1s ✓长播放")
    print("视频1: 15s视频播放3s,  频道2 → 阈值=6.8s → 3s<6.8s  ✗短播放") 
    print("视频2: 50s视频播放40s, 频道2 → 阈값=14.4s → 40s≥14.4s ✓长播放")
    print("视频3: 25s视频播放20s, 频道1 → 阈값=3.1s → 20s≥3.1s ✓长播放")
    print("预期: 3条长播放，取最近3条 → 视频0,2,3")

def run_test():
    """运行测试"""
    print("=== filter_long_and_crop_vec 函数测试 ===\n")
    
    # 创建测试数据
    seq_emb, play_time, duration, channel, valid_len, max_len = create_test_data()
    
    # 打印分析
    print_test_analysis()
    
    # 构建计算图
    with tf.Session() as sess:
        # 创建placeholder
        seq_emb_ph = tf.placeholder(tf.float32, shape=[None, None, None])
        play_time_ph = tf.placeholder(tf.int64, shape=[None, None])
        duration_ph = tf.placeholder(tf.int64, shape=[None, None])
        channel_ph = tf.placeholder(tf.int64, shape=[None, None])
        valid_len_ph = tf.placeholder(tf.int32, shape=[None])
        
        # 调用函数
        out_emb, longview_len, used_len = filter_long_and_crop_vec(
            seq_emb=seq_emb_ph,
            play_time=play_time_ph,
            duration=duration_ph,
            channel=channel_ph,
            valid_len=valid_len_ph,
            max_len=max_len,
            pad_value=-999.0  # 使用特殊值便于观察padding
        )
        # 执行计算
        feed_dict = {
            seq_emb_ph: seq_emb,
            play_time_ph: play_time,
            duration_ph: duration,
            channel_ph: channel,
            valid_len_ph: valid_len
        }
        
        result_emb, result_longview_len, result_used_len = sess.run(
            [out_emb, longview_len, used_len], feed_dict=feed_dict)
        
        # 打印结果
        print("\n=== 测试结果 ===")
        print(f"longview_len: {result_longview_len}")
        print(f"used_len: {result_used_len}")
        
        print(f"\nout_emb shape: {result_emb.shape}")
        print("\nBatch 0 输出 embedding (前3列):")
        for i in range(max_len):
            print(f"  位置{i}: {result_emb[0, i, :3]}")
            
        print("\nBatch 1 输出 embedding (前3列):")
        for i in range(max_len):
            print(f"  位置{i}: {result_emb[1, i, :3]}")
            
        # 验证结果
        print("\n=== 结果验证 ===")
        
        # 检查longview_len
        expected_longview = [4, 3]  # batch0有4条长播放, batch1有3条长播放
        if np.array_equal(result_longview_len, expected_longview):
            print("✓ longview_len 正确")
        else:
            print(f"✗ longview_len 错误，期望{expected_longview}，实际{result_longview_len}")
        
        # 检查used_len  
        expected_used = [3, 3]  # 都取最近3条
        if np.array_equal(result_used_len, expected_used):
            print("✓ used_len 正确")
        else:
            print(f"✗ used_len 错误，期望{expected_used}，实际{result_used_len}")
            
        # 检查embedding内容
        # Batch 0: 应该包含视频1,3,4的embedding
        # 视频1: [2.0, 0.2, 0.2, 0.2]
        # 视频3: [4.0, 0.4, 0.4, 0.4] 
        # 视频4: [5.0, 0.5, 0.5, 0.5]
        print("\nBatch 0 详细验证:")
        if abs(result_emb[0, 0, 0] - 2.0) < 1e-5:  # 第0位置应该是视频1
            print("✓ 位置0包含视频1的embedding")
        else:
            print(f"✗ 位置0错误，期望2.0，实际{result_emb[0, 0, 0]}")
            
        if abs(result_emb[0, 1, 0] - 4.0) < 1e-5:  # 第1位置应该是视频3
            print("✓ 位置1包含视频3的embedding")
        else:
            print(f"✗ 位置1错误，期望4.0，实际{result_emb[0, 1, 0]}")
        if abs(result_emb[0, 2, 0] - 5.0) < 1e-5:  # 第2位置应该是视频4
            print("✓ 位置2包含视频4的embedding")
        else:
            print(f"✗ 位置2错误，期望5.0，实际{result_emb[0, 2, 0]}")

if __name__ == "__main__":
    # 禁用TF的警告信息
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.logging.set_verbosity(tf.logging.ERROR)
    
    run_test()
