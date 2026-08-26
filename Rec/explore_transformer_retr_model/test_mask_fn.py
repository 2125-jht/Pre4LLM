import tensorflow as tf
import numpy as np

# 构造示例输入数据
def create_sample_batch():
    batch_size = 5
    
    # 用户交互行为标签 (0或1)
    batch = {
        'context_info__like': [1, 0, 1, 0, 1],
        'context_info__follow': [0, 1, 0, 1, 0],
        'context_info__comment': [1, 0, 0, 1, 1],
        'context_info__collect': [0, 0, 1, 0, 0],
        'context_info__download': [0, 1, 0, 0, 1],
        'context_info__profile_enter': [1, 0, 1, 1, 0],
        
        # 播放时长 (毫秒)
        'context_info__playing_time': [15000, 5000, 25000, 8000, 45000],
        
        # 视频总时长 (毫秒)
        'photo_info__duration_ms': [30000, 12000, 40000, 15000, 60000]
    }
    
    # 转换为 Tensor
    for key, value in batch.items():
        batch[key] = tf.constant(value, dtype=tf.int32)
    
    return batch

# 定义 mask_fn 函数（从您的代码中复制）
def mask_fn(batch):
    # ===== 1. 原有标签 =====
    label_like          = tf.cast(batch['context_info__like'],          tf.float32)
    label_follow        = tf.cast(batch['context_info__follow'],        tf.float32)
    label_comment       = tf.cast(batch['context_info__comment'],       tf.float32)
    label_collect       = tf.cast(batch['context_info__collect'],       tf.float32)
    label_download      = tf.cast(batch['context_info__download'],      tf.float32)
    label_profile_enter = tf.cast(batch['context_info__profile_enter'], tf.float32)

    playing_time = tf.cast(batch['context_info__playing_time'], tf.float32)   # ms
    duration_ms  = tf.cast(batch['photo_info__duration_ms'],   tf.float32)   # ms
    
    # ===== 2. 基础播放标签 =====
    label_finish = tf.where(tf.greater(playing_time, duration_ms), tf.ones_like(playing_time), tf.zeros_like(playing_time))
    label_play_over_3s = tf.where(tf.greater(playing_time, 3000.), tf.ones_like(playing_time), tf.zeros_like(playing_time))
    
    # ===== 3. 计算 “长播” 阈值并生成 label_long =====
    # —— 3.1 duration 分桶
    boundaries = [0., 8700., 12700., 20300., 38800., 71800., 118200., 195000.]                  # 升序
    thr_table  = [13100., 12000., 13600., 18400., 28800., 46600., 74900., 92500., 79700.]        # 对应阈值
    bin_id     = tf.raw_ops.Bucketize(input=duration_ms, boundaries=boundaries)          # [B] float32
    base_thr   = tf.gather(thr_table, bin_id)                                            # [B] float32

    # —— 3.3 是否满足长播
    label_long = tf.where(tf.greater(playing_time, base_thr), tf.ones_like(playing_time), tf.zeros_like(playing_time))       # [B] float32

    # ===== 4. 汇总互动行为 =====
    interact_cnt = (label_like + label_finish + label_follow + label_comment +
                    label_collect + label_download + label_profile_enter)

    action_cnt = label_play_over_3s * interact_cnt + label_long
    mask = tf.less(action_cnt, 1)
    return mask

# 执行示例
def run_example():
    # 创建示例数据
    sample_batch = create_sample_batch()
    
    # print("=== 输入数据 ===")
    # for key, value in sample_batch.items():
    #     print(f"{key}: {value}")
    
    # 调用 mask_fn
    mask_result = mask_fn(sample_batch)
    
    # print("\n=== 中间计算结果 ===")
    # # 为了展示中间过程，重新计算一遍
    # playing_time = tf.cast(sample_batch['context_info__playing_time'], tf.float32)
    # duration_ms = tf.cast(sample_batch['photo_info__duration_ms'], tf.float32)
    
    # # 分桶结果
    # boundaries = [0, 8700, 12700, 20300, 38800, 71800, 118200, 195000]
    # bin_ids = tf.raw_ops.Bucketize(input=duration_ms, boundaries=boundaries)
    # print(f"视频时长分桶ID: {bin_ids.numpy()}")
    
    # # 长播阈值
    # thr_table = [13100, 12000, 13600, 18400, 28800, 46600, 74900, 92500, 79700]
    # base_thresholds = tf.gather(thr_table, bin_ids)
    # print(f"对应长播阈值: {base_thresholds.numpy()}")
    
    # # 是否长播
    # label_long = tf.where(tf.greater(playing_time, base_thresholds), 1., 0.)
    # print(f"是否长播: {label_long.numpy()}")
    
    # print(f"\n=== 最终结果 ===")
    # print(f"掩码结果: {mask_result.numpy()}")
    # print(f"解释: True表示该样本需要被过滤掉，False表示保留")

# 运行示例
if __name__ == "__main__":
    run_example()