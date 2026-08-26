# -*- coding: utf-8 -*-
# test_sidrec_model.py
from __future__ import print_function
import os
import numpy as np
import traceback

import tensorflow as tf
tf.compat.v1.disable_eager_execution()  # 用 TF1.x 图执行

# 如果你的源码不在同一目录，请按需修改 PYTHONPATH
# import sys; sys.path.append('/path/to/your/repo')

from model import SIDRecModel  # 确保 model.py 里定义了该类

def build_toy_inputs(B=3, L_hist=12, max_sid=500):
    """
    造一批用户历史/时长/动作：
      - user_sid: [B, L_hist]，0 表示无效
      - user_time: [B, L_hist]，单位随便取，这里 0..10
      - user_act: [B, L_hist]，0/1
    让靠后的若干位满足筛选条件（>=7 或 >=3 且 act=1），确保有有效样本。
    """
    rng = np.random.RandomState(42)
    user_sid = rng.randint(1, max_sid, size=(B, L_hist)).astype("int64")
    tgt_sid = rng.randint(1, max_sid, size=(B, 1)).astype("int64")

    # 前半段置 0 当作空位
    for b in range(B):
        pad_len = rng.randint(0, L_hist // 3)  # 随机左侧 pad
        user_sid[b, :pad_len] = 0
        
    # user_sid = np.zeros((B, L_hist), dtype="int64")

    # 行为与时长
    user_time = rng.randint(0, 11, size=(B, L_hist)).astype("int64")
    user_act  = rng.randint(0, 2,  size=(B, L_hist)).astype("int64")

    # 保证末尾至少 4 条有效（便于 select_size<=L）
    user_time[:, -4:] = rng.randint(7, 11, size=(B, 4)).astype("int64")
    user_act[:,  -4:] = 1

    return user_sid, user_time, user_act, tgt_sid

def make_batch(B=8, L=1000, vocab_sizes=(8192, 8192, 8192), seed=42):
    """构造一批随机样本，含 padding=0、时间/行为规则满足一定比例。"""
    rng = np.random.RandomState(seed)
    total_vocab = sum(vocab_sizes)

    # 随机产生 sid，60% 概率为 0（padding），40% 有效
    user_sid = rng.choice([0, 1], size=(B, L), p=[1, 0]).astype(np.int32)
    # nonzero_idx = user_sid.nonzero()
    # user_sid[nonzero_idx] = rng.randint(0, total_vocab, size=len(nonzero_idx[0]), dtype=np.int32)
    # user_sid[nonzero_idx] = rng.randint(0, 1000, size=len(nonzero_idx[0]), dtype=np.int32)

    # 播放时长：让一部分满足 >=7，另一部分满足 [3,6] 且 action=1
    user_time = rng.randint(0, 10, size=(B, L)).astype(np.int32)
    user_act  = rng.randint(0, 2,  size=(B, L)).astype(np.int32)

    # 让非零位置里有 50% 满足 >=7
    mask_nz = (user_sid != 0)
    pick_hi = rng.rand(B, L) < 0.5
    user_time[np.where(mask_nz & pick_hi)] = rng.randint(7, 10, size=np.count_nonzero(mask_nz & pick_hi))
    # 让剩余一部分满足 >=3 且 action=1
    rest = mask_nz & (~pick_hi)
    user_time[np.where(rest)] = rng.randint(0, 7, size=np.count_nonzero(rest))  # 可能<3
    # 把其中一半强制满足 (time>=3 & action=1)
    sub = np.where(rest)
    take = len(sub[0]) // 2
    if take > 0:
        idx = rng.choice(np.arange(len(sub[0])), size=take, replace=False)
        rr, cc = sub[0][idx], sub[1][idx]
        user_time[rr, cc] = rng.randint(3, 7, size=take)
        user_act[rr, cc]  = 1

    # 当前目标 sid（训练里会拼到历史右侧）
    # tgt_sid = rng.randint(0, total_vocab, size=(B, 1), dtype=np.int32)
    tgt_sid = rng.randint(0, 1000, size=(B, 1), dtype=np.int32)

    return user_sid, user_time, user_act, tgt_sid

def main():
    B = 8
    L = 200
    dim = 128
    select_size = 64
    vocab_sizes = (8192, 8192, 8192)

    # 1) 构图：占位符
    with tf.compat.v1.Graph().as_default():
        user_sid_ph = tf.compat.v1.placeholder(tf.int32, shape=[None, L],   name="user_sid")
        user_tim_ph = tf.compat.v1.placeholder(tf.int32, shape=[None, L],   name="user_time")
        user_act_ph = tf.compat.v1.placeholder(tf.int32, shape=[None, L],   name="user_act")
        tgt_sid_ph  = tf.compat.v1.placeholder(tf.int32, shape=[None, 1],   name="tgt_sid")

        # 2) 实例化模型（feature_emb_dict/size_dict 用不到，给空即可）
        model = SIDRecModel(feature_emb_dict={}, feature_emb_size_dict={},
                            dim=dim, select_size=select_size,
                            vocab_sizes=list(vocab_sizes), print_ops=[])

        # 3) 前向 + loss
        loss = model.model(user_sid_ph, tgt_sid_ph)

        # 数值检查（若有 NaN/Inf，会在此处抛出）
        checks = tf.compat.v1.add_check_numerics_ops()

        # 4) 一个最简单的优化器（跑几步，看会不会 NaN）
        opt = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4)
        train_op = opt.minimize(loss)

        # 5) Beam Search 跑一遍
        gen_loc, gen_prob = model.beam_search_fast(
            user_sid_ph, beam_size=4, temperature=1.0
        )

        # 6) 会话
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=config)

        sess.run(tf.compat.v1.global_variables_initializer())

        # 7) 造一批数据并喂入
        # batch = make_batch(B=B, L=L, vocab_sizes=vocab_sizes, seed=123)
        batch = build_toy_inputs(B=B, L_hist=L, max_sid=1000)
        feed = {
            user_sid_ph: batch[0],
            user_tim_ph: batch[1],
            user_act_ph: batch[2],
            tgt_sid_ph:  batch[3],
        }

        print(">>> Run forward once")
        loss_val, _ = sess.run([loss, checks], feed_dict=feed)
        print("Loss:", float(loss_val))

        print(">>> Train 3 steps")
        for step in range(3):
            loss_val, _ = sess.run([loss, train_op], feed_dict=feed)
            if not np.isfinite(loss_val):
                print("[WARN] loss is not finite at step", step, "->", loss_val)
                break
            print("Step %d | loss=%.6f" % (step, loss_val))

        print(">>> Run beam_search_fast")
        gl, gp = sess.run([gen_loc, gen_prob], feed_dict=feed)
        print("beam gen_part_loc shape:", gl.shape, " sample[0,0]:", gl[0,0])
        print("beam probs shape:", gp.shape, " sample[0,0]:", gp[0,0])

        print("DONE.")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n===== EXCEPTION CAUGHT =====")
        traceback.print_exc()
        print("============================\n")
