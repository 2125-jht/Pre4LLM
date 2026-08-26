# -*- coding: utf-8 -*-
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from time import perf_counter  # ← 新增：高精度计时

# === 你的工程内模块（确保 PYTHONPATH 正确） ===
from feature_attr_extract import *   # 需要 mlp/print_tensor/recall_at_k 等实现
from modulesV2 import *
from modules_ import *
from model import SIDRecModel   # ← 把文件名替换成你定义 MultiInterestModel 的源码文件名


def build_dummy_inputs(batch_size=4, max_len=200, dim=256,
                       vocab_sizes=(8192, 8192, 8192),
                       d_user_each=32, d_click_pid=64, d_click_aid=64):
    """构建占位符与随机伪数据"""
    # 1) 占位符（与模型代码中读取的 key 一一对应）
    ph = {}
    # 用户静态（拼 axis=1 → 2D）
    ph["user_id"]          = tf.placeholder(tf.float32, [None, d_user_each], name="ph_user_id")
    ph["user_gender"]      = tf.placeholder(tf.float32, [None, d_user_each], name="ph_user_gender")
    ph["user_age_segment"] = tf.placeholder(tf.float32, [None, d_user_each], name="ph_user_age_segment")
    ph["user_level"]       = tf.placeholder(tf.float32, [None, d_user_each], name="ph_user_level")
    # 用户点击（拼 axis=2 → 3D）
    ph["user_profile_v1_click_pid_list"] = tf.placeholder(tf.float32, [None, max_len, d_click_pid], name="ph_click_pid")
    ph["user_profile_v1_click_aid_list"] = tf.placeholder(tf.float32, [None, max_len, d_click_aid], name="ph_click_aid")
    # 点击有效长度
    ph["click_len"]        = tf.placeholder(tf.int32,   [None], name="ph_click_len")

    # 解码侧：teacher forcing 的输入与标签
    ph["photo_sid"]   = tf.placeholder(tf.int32, [None, 3], name="ph_photo_sid")   # 训练时 decoder 输入（全局ID）
    ph["label"]       = tf.placeholder(tf.int32, [None, 3], name="ph_label")       # 三个层级的标签（局部ID）
    ph["loss_masker"] = tf.placeholder(tf.int32, [None],   name="ph_loss_masker")  # >0 的样本参与 loss

    # 2) 随机伪数据
    rng = np.random.RandomState(2025)
    B = batch_size
    V0, V1, V2 = vocab_sizes
    total_vocab = sum(vocab_sizes)

    fd = {
        ph["user_id"]:          rng.randn(B, d_user_each).astype(np.float32),
        ph["user_gender"]:      rng.randn(B, d_user_each).astype(np.float32),
        ph["user_age_segment"]: rng.randn(B, d_user_each).astype(np.float32),
        ph["user_level"]:       rng.randn(B, d_user_each).astype(np.float32),

        ph["user_profile_v1_click_pid_list"]: rng.randn(B, max_len, d_click_pid).astype(np.float32),
        ph["user_profile_v1_click_aid_list"]: rng.randn(B, max_len, d_click_aid).astype(np.float32),

        # 每条样本的有效长度（1..max_len）
        ph["click_len"]:        rng.randint(1, max_len+1, size=(B,), dtype=np.int32),

        # 解码输入：用任意合法全局ID（[0, total_vocab)），训练里会在前面加 <START>=total_vocab
        ph["photo_sid"]:        rng.randint(0, total_vocab, size=(B, 3), dtype=np.int32),

        # 标签：分层局部ID
        ph["label"]: np.stack([
            rng.randint(0, V0, size=(B,), dtype=np.int32),
            rng.randint(0, V1, size=(B,), dtype=np.int32),
            rng.randint(0, V2, size=(B,), dtype=np.int32),
        ], axis=1),

        # 所有样本都参与 loss
        ph["loss_masker"]: np.ones((B,), dtype=np.int32),
    }

    # 3) 组装成模型需要的两个 dict
    feature_emb_dict = {
        "user_id":          ph["user_id"],
        "user_gender":      ph["user_gender"],
        "user_age_segment": ph["user_age_segment"],
        "user_level":       ph["user_level"],
        "user_profile_v1_click_pid_list": ph["user_profile_v1_click_pid_list"],
        "user_profile_v1_click_aid_list": ph["user_profile_v1_click_aid_list"],
    }
    feature_emb_size_dict = {
        # 模型里只读取这个 key，所以把 click_len 挂在这个 key 上
        "user_profile_v1_click_pid_list": ph["click_len"]
    }

    return ph, fd, feature_emb_dict, feature_emb_size_dict


def main():
    dim = 256
    max_len = 200
    vocab_sizes = (8192, 8192, 8192)
    batch_size = 1

    g = tf.Graph()
    with g.as_default():
        print_ops_bucket = []

        # 1) 构建输入
        ph, fd, feature_emb_dict, feature_emb_size_dict = build_dummy_inputs(
            batch_size=batch_size, max_len=max_len, dim=dim, vocab_sizes=vocab_sizes
        )

        # 2) 构建模型（假定 Decoder 已经替换为 MoE SwiGLU + Top-2，并在 router 里 add_to_collection("moe_losses", ...)）
        model = SIDRecModel(
            feature_emb_dict=feature_emb_dict,
            feature_emb_size_dict=feature_emb_size_dict,
            dim=dim,
            vocab_sizes=list(vocab_sizes),
            print_ops=print_ops_bucket
        )

        # 3) 前向与损失
        total_loss, result = model.model(
            photo_sid=ph["photo_sid"],
            label=ph["label"],
            photo_semantic_id_int=ph["loss_masker"]
        )

        # 3-A) lb loss
        lb_losses = tf.get_collection("lb_loss")
        total_lb_loss = tf.add_n(lb_losses) if lb_losses else tf.constant(0.0, tf.float32)

        # 4) 优化器（一次小步）
        global_step = tf.train.get_or_create_global_step()
        train_op = tf.train.AdamOptimizer(1e-3).minimize(total_loss, global_step=global_step)

        # 5) Beam Search（两个版本）
        gen_loc_greedy, probs_greedy = model.beam_search_fast(beam_size=512, temperature=1.0)

        # 6) 运行
        init = tf.global_variables_initializer()

        num_steps  = 100        # 训练总步数
        log_every  = 1        # 打印间隔
        eval_every = 1        # 做一次 beamsearch 间隔

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sess.run(init)

            # === 新增：耗时统计桶 ===
            train_times = []   # 每次 sess.run(train_op, ...) 的耗时（秒）
            infer_times = []   # 每次 sess.run(beam_search, ...) 的耗时（秒）

            for t in range(1, num_steps + 1):
                # === 训练计时 ===
                t0 = perf_counter()
                _, lb_v, total_v, gs = sess.run(
                    [train_op, total_lb_loss, total_loss, global_step],
                    feed_dict=fd   # 复用同一份伪数据
                )
                train_dur = perf_counter() - t0
                train_times.append(train_dur)

                if t % log_every == 0:
                    print("[Train] step=%d  lb=%.6f  total=%.6f  time=%.2f ms" %
                          (gs, lb_v, total_v, train_dur * 1000.0))

                # === 推理/评估计时（beam search） ===
                if t % eval_every == 0:
                    t1 = perf_counter()
                    gl_greedy, pr_greedy = sess.run([gen_loc_greedy, probs_greedy], feed_dict=fd)
                    infer_dur = perf_counter() - t1
                    infer_times.append(infer_dur)

                    print("[Eval]  step=%d  beam(greedy) gen_part_loc[0,0]: %s  time=%.2f ms" %
                          (gs, str(gl_greedy[0,0]), infer_dur * 1000.0))

            # 训练结束后再做一次完整评估（计时）
            t2 = perf_counter()
            gl_greedy, pr_greedy = sess.run([gen_loc_greedy, probs_greedy], feed_dict=fd)
            final_infer_dur = perf_counter() - t2
            infer_times.append(final_infer_dur)
            print("[Final Eval] beam(greedy) gen_part_loc[0,0]: %s  time=%.2f ms" %
                  (str(gl_greedy[0,0]), final_infer_dur * 1000.0))

            # === 汇总统计（平均/95分位/最大） ===
            def _fmt_stats(ts):
                if not ts:
                    return "N/A"
                ts_ms = np.asarray(ts) * 1000.0  # 转毫秒
                return "avg=%.2f ms  p95=%.2f ms  max=%.2f ms  (N=%d)" % (
                    ts_ms.mean(), np.percentile(ts_ms, 95), ts_ms.max(), ts_ms.size
                )

            print("[Summary] Train: %s" % _fmt_stats(train_times))
            print("[Summary] Infer : %s" % _fmt_stats(infer_times))


if __name__ == "__main__":
    main()
