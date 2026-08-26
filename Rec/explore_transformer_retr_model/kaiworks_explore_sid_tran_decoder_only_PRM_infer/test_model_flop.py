# test_mirec_transformer_moe_profile.py
# -*- coding: utf-8 -*-
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

# === 你的工程内模块（确保 PYTHONPATH 正确） ===
from feature_attr_extract import *   # 需要 mlp/print_tensor/recall_at_k 等实现
from modulesV2 import *
from modules_ import *
from model import MultiInterestModel   # ← 把文件名替换成你定义 MultiInterestModel 的源码文件名

# -------------------- 工具函数：参数量 & FLOPs 统计 -------------------- #
def count_params(trainable_only=True):
    """统计参数量（trainable_only=True 只统计可训练参数，否则统计所有变量）"""
    vars_ = tf.compat.v1.trainable_variables() if trainable_only else tf.compat.v1.global_variables()
    total = 0
    detail = []
    for v in vars_:
        shape = v.shape.as_list()
        n = 1
        for s in shape:
            if s is None:
                n = None
                break
            n *= s
        if n is None:
            continue
        total += n
        detail.append((v.name, n, shape))
    return total, detail

def profile_flops(graph, sess, fetches, feed_dict, note=""):
    """
    用 RunMetadata 捕获一次实际运行的 FLOPs。
    返回 (total_float_ops, total_time_ms)
    """
    run_meta = tf.compat.v1.RunMetadata()
    run_opts = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)

    # 实际跑一次
    _ = sess.run(fetches, feed_dict=feed_dict, options=run_opts, run_metadata=run_meta)

    # 统计 FLOPs
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    opts['output'] = 'none'     # 不直接打印，将结果对象返回
    prof = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, options=opts)
    total_flops = prof.total_float_ops if prof is not None else 0

    # 统计时长（可选）
    # end_to_end 时间可以从 step_stats 拿；这里只做个粗略合计
    ms = 0.0
    if run_meta and run_meta.step_stats:
        for dev in run_meta.step_stats.dev_stats:
            for node in dev.node_stats:
                ms += node.all_start_micros + node.all_end_rel_micros  # 不是精准 wall time，仅供参考
        ms = ms / 1000.0

    if note:
        print("[Profile] %-18s FLOPs = %.4f GFLOPs (per run)\n" % (note, total_flops / 1e9))
    return total_flops, ms

def human(n):
    if n >= 1e9: return f"{n/1e9:.3f}B"
    if n >= 1e6: return f"{n/1e6:.3f}M"
    if n >= 1e3: return f"{n/1e3:.3f}K"
    return str(n)

# -------------------- 构造伪输入 -------------------- #
def build_dummy_inputs(batch_size=4, max_len=200, dim=256,
                       vocab_sizes=(8192, 8192, 8192),
                       d_user_each=32, d_click_pid=64, d_click_aid=64):
    """构建占位符与随机伪数据"""
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

        ph["click_len"]:        rng.randint(1, max_len+1, size=(B,), dtype=np.int32),

        ph["photo_sid"]:        rng.randint(0, total_vocab, size=(B, 3), dtype=np.int32),

        ph["label"]: np.stack([
            rng.randint(0, V0, size=(B,), dtype=np.int32),
            rng.randint(0, V1, size=(B,), dtype=np.int32),
            rng.randint(0, V2, size=(B,), dtype=np.int32),
        ], axis=1),

        ph["loss_masker"]: np.ones((B,), dtype=np.int32),
    }

    feature_emb_dict = {
        "user_id":          ph["user_id"],
        "user_gender":      ph["user_gender"],
        "user_age_segment": ph["user_age_segment"],
        "user_level":       ph["user_level"],
        "user_profile_v1_click_pid_list": ph["user_profile_v1_click_pid_list"],
        "user_profile_v1_click_aid_list": ph["user_profile_v1_click_aid_list"],
    }
    feature_emb_size_dict = {
        "user_profile_v1_click_pid_list": ph["click_len"]
    }

    return ph, fd, feature_emb_dict, feature_emb_size_dict

# -------------------- 主逻辑：构图 + 训练/推理 + 统计 -------------------- #
def main():
    dim = 256
    max_len = 200
    vocab_sizes = (8192, 8192, 8192)
    batch_size = 1
    beam_size = 512

    # ========== 训练图（用于参数 & 训练 FLOPs） ==========
    g_train = tf.Graph()
    with g_train.as_default():
        print_ops_bucket = []
        ph, fd, feature_emb_dict, feature_emb_size_dict = build_dummy_inputs(
            batch_size=batch_size, max_len=max_len, dim=dim, vocab_sizes=vocab_sizes
        )

        model = MultiInterestModel(
            feature_emb_dict=feature_emb_dict,
            feature_emb_size_dict=feature_emb_size_dict,
            dim=dim,
            vocab_sizes=list(vocab_sizes),
            print_ops=print_ops_bucket
        )

        total_loss, result = model.model(
            photo_sid=ph["photo_sid"],
            label=ph["label"],
            photo_semantic_id_int=ph["loss_masker"]
        )

        # lb loss（若存在）
        lb_losses = tf.get_collection("lb_loss")
        total_lb_loss = tf.add_n(lb_losses) if lb_losses else tf.constant(0.0, tf.float32)

        # 优化器
        global_step = tf.train.get_or_create_global_step()
        train_op = tf.train.AdamOptimizer(1e-3).minimize(total_loss, global_step=global_step)

        # 统计参数（trainable + all）
        trainable_params, _ = count_params(trainable_only=True)
        all_params, _ = count_params(trainable_only=False)

        init = tf.global_variables_initializer()
        cfg = tf.ConfigProto(allow_soft_placement=True)

        with tf.Session(graph=g_train, config=cfg) as sess:
            sess.run(init)

            # 先跑一次 loss（forward-only）用于「训练前向 FLOPs」
            fwd_flops, _ = profile_flops(g_train, sess, fetches=[total_loss], feed_dict=fd, note="Forward(loss)")

            # 再跑一次 train_op（含前向+反向+优化器）
            train_flops, _ = profile_flops(g_train, sess, fetches=[train_op, total_lb_loss, total_loss], feed_dict=fd, note="Train step")

            print("===== Params =====")
            print(f"Trainable params: {human(trainable_params)}")
            print(f"All variables  : {human(all_params)}")
            print("===== FLOPs (per step, batch_size=%d) =====" % batch_size)
            print(f"Forward-only (loss) : {fwd_flops/1e9:.4f} GFLOPs")
            print(f"Train step (F+B+Opt): {train_flops/1e9:.4f} GFLOPs")

    # ========== 推理图（用于推理/beamsearch FLOPs） ==========
    g_infer = tf.Graph()
    with g_infer.as_default():
        print_ops_bucket = []
        ph, fd, feature_emb_dict, feature_emb_size_dict = build_dummy_inputs(
            batch_size=batch_size, max_len=max_len, dim=dim, vocab_sizes=vocab_sizes
        )

        model = MultiInterestModel(
            feature_emb_dict=feature_emb_dict,
            feature_emb_size_dict=feature_emb_size_dict,
            dim=dim,
            vocab_sizes=list(vocab_sizes),
            print_ops=print_ops_bucket
        )

        # 前向 logits / loss（可选）
        total_loss, result = model.model(
            photo_sid=ph["photo_sid"],
            label=ph["label"],
            photo_semantic_id_int=ph["loss_masker"]
        )

        # Beam Search 推理（你已有的快速版本）
        gen_loc_greedy, probs_greedy = model.beam_search_fast(beam_size=beam_size, temperature=1.0)

        init = tf.global_variables_initializer()
        cfg = tf.ConfigProto(allow_soft_placement=True)

        with tf.Session(graph=g_infer, config=cfg) as sess:
            sess.run(init)

            # 只跑一次「推理前向」（loss） FLOPs（不含反向）
            infer_fwd_flops, _ = profile_flops(g_infer, sess, fetches=[total_loss], feed_dict=fd, note="Infer forward(loss)")

            # 跑一次 BeamSearch FLOPs（真正的解码路径，受 beam_size / 步数影响）
            beam_flops, _ = profile_flops(g_infer, sess, fetches=[gen_loc_greedy, probs_greedy], feed_dict=fd, note=f"BeamSearch (beam={beam_size})")

            print("===== Inference FLOPs (per run, batch_size=%d) =====" % batch_size)
            print(f"Infer forward(loss): {infer_fwd_flops/1e9:.4f} GFLOPs")
            print(f"BeamSearch        : {beam_flops/1e9:.4f} GFLOPs")

if __name__ == "__main__":
    main()
