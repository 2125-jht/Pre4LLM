# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf

# 兼容 TF2 直接用 1.x 图执行
if tf.__version__.startswith("2"):
    tf.compat.v1.disable_eager_execution()
    tf = tf.compat.v1

# ====== 你自己的实现应已在同进程中 ======
from model import SIDRecModel
from util import processInput  # 若你的函数不在这里，改成正确来源

def build_toy_inputs(B=3, L_hist=12, max_sid=500):
    """
    造一批用户历史/时长/动作：
      - user_sid: [B, L_hist]，0 表示无效
      - user_time: [B, L_hist]，单位随便取，这里 0..10
      - user_act: [B, L_hist]，0/1
    让靠后的若干位满足筛选条件（>=7 或 >=3 且 act=1），确保有有效样本。
    """
    rng = np.random.RandomState(42)
    # user_sid = rng.randint(1, max_sid, size=(B, L_hist)).astype("int64")
    user_sid = np.zeros((B, L_hist), dtype="int64")
    tgt_sid = rng.randint(1, max_sid, size=(B, 1)).astype("int64")

    # # 前半段置 0 当作空位
    # for b in range(B):
    #     pad_len = rng.randint(0, L_hist // 3)  # 随机左侧 pad
    #     user_sid[b, :pad_len] = 0

    # 行为与时长
    user_time = rng.randint(0, 11, size=(B, L_hist)).astype("int64")
    user_act  = rng.randint(0, 2,  size=(B, L_hist)).astype("int64")

    # 保证末尾至少 4 条有效（便于 select_size<=L）
    user_time[:, -4:] = rng.randint(7, 11, size=(B, 4)).astype("int64")
    user_act[:,  -4:] = 1

    return user_sid, user_time, user_act, tgt_sid

def run_beam_search_test():
    # --------- 超参（小尺寸更容易跑通） ----------
    dim = 128
    select_size = 64            # 要 <= 历史长度
    vocab_sizes = [8192, 8192, 8192] # 三层

    # --------- 构造模型 ----------
    feature_emb_dict = {}
    feature_emb_size_dict = {}
    model = SIDRecModel(feature_emb_dict, feature_emb_size_dict,
                        dim=dim, select_size=select_size,
                        vocab_sizes=vocab_sizes, print_ops=[])

    # --------- 构造输入 ----------
    B, L_hist = 128, 1000
    user_sid_np, user_time_np, user_act_np, tgt_sid = build_toy_inputs(B=B, L_hist=L_hist, max_sid=1000)

    user_sid = tf.constant(user_sid_np, dtype=tf.int64)   # [B,L]
    user_time = tf.constant(user_time_np, dtype=tf.int64) # [B,L]
    user_act = tf.constant(user_act_np, dtype=tf.int64)   # [B,L]
    tgt_sid = tf.constant(tgt_sid, dtype=tf.int64)
    
    # --------- 调用 beam_search_fast ----------
    loss = model.model(user_sid, user_time, user_act, tgt_sid)
    
    print("loss (tensor):", loss)

    # --------- 会话执行 ----------
    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    with tf.Session(config=cfg) as sess:
        sess.run(tf.global_variables_initializer())
        loss = sess.run([loss])[0]
        # loc, prob = sess.run([user_sid, flats])

    # --------- 打印与基本检查 ----------
    print("loss (tensor):", loss)
    print("loss shape :", loss.shape)

if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    run_beam_search_test()
