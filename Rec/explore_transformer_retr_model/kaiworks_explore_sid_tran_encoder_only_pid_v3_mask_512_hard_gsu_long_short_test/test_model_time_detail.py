# -*- coding: utf-8 -*-
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from time import perf_counter  # 高精度计时

# === 你的工程内模块（确保 PYTHONPATH 正确） ===
from feature_attr_extract import *   # 需要 mlp/print_tensor/recall_at_k 等实现
from modulesV2 import *
from modules_ import *
from model import SIDRecModel   # ← 把文件名替换成你定义 MultiInterestModel 的源码文件名

# ========== 新增：逐 op profiling 工具 ==========
from tensorflow.python.client import timeline

def profile_run(sess, fetches, feed_dict, tag, topk_min_us=10, max_depth=100):
    """
    跑一次 sess.run，并打印逐 op 耗时/内存榜单，并保存 Chrome Timeline JSON。
    - tag: 用于文件名区分，如 'train_step1' / 'infer_step5'
    """
    run_options  = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    outputs = sess.run(fetches, feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)

    # ---- 组装 Profile 选项（兼容返回 builder 或 dict 的两种实现）----
    try:
        # 有的 TF 版本需要把初始 options 传入构造函数再链式设置
        Builder = tf.profiler.ProfileOptionBuilder
        opts = (Builder(Builder.time_and_memory())
                .with_max_depth(max_depth)
                .with_min_micros(topk_min_us)
                .build())
    except Exception:
        # 有的 TF 版本 time_and_memory() 直接返回 dict
        opts = tf.profiler.ProfileOptionBuilder.time_and_memory()
        opts['min_micros'] = topk_min_us
        opts['max_depth']  = max_depth

    print("\n===== [TF Profiler][op] {} =====".format(tag))
    tf.profiler.profile(sess.graph, run_meta=run_metadata, cmd='op', options=opts)

    print("\n===== [TF Profiler][scope] {} =====".format(tag))
    tf.profiler.profile(sess.graph, run_meta=run_metadata, cmd='scope', options=opts)

    # ---- 保存 Chrome Timeline ----
    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    trace_path = "trace_{}.json".format(tag)
    with open(trace_path, "w") as f:
        f.write(ctf)
    print("[Timeline] Chrome trace saved -> {}".format(trace_path))

    return outputs


def build_dummy_inputs(batch_size=4, click_max_len=200, colossus_max_len=1000,
                       vocab_sizes=(8192, 8192, 8192),
                       d_user_each=512,
                       d_click_pid=512, d_click_aid=512,
                       d_colossus_pid=512, d_colossus_aid=512):
    """构建占位符与随机伪数据"""
    # 1) 占位符（与模型代码中读取的 key 一一对应）
    ph = {}
    # 用户静态（拼 axis=1 → 2D）
    ph["user_id"]          = tf.placeholder(tf.float32, [None, d_user_each], name="ph_user_id")
    ph["user_gender"]      = tf.placeholder(tf.float32, [None, d_user_each], name="ph_user_gender")
    ph["user_age_segment"] = tf.placeholder(tf.float32, [None, d_user_each], name="ph_user_age_segment")
    ph["user_level"]       = tf.placeholder(tf.float32, [None, d_user_each], name="ph_user_level")
    
    # 用户点击（拼 axis=2 → 3D）
    ph["user_profile_v1_click_pid_list"] = tf.placeholder(tf.float32, [None, click_max_len, d_click_pid], name="ph_click_pid")
    ph["user_profile_v1_click_aid_list"] = tf.placeholder(tf.float32, [None, click_max_len, d_click_aid], name="ph_click_aid")
    ph["user_colossus_pid_list"] = tf.placeholder(tf.float32, [None, colossus_max_len, d_colossus_pid], name="ph_colossus_pid")
    ph["user_colossus_aid_list"] = tf.placeholder(tf.float32, [None, colossus_max_len, d_colossus_aid], name="ph_colossus_aid")
    
    ph["user_click_sid"]    = tf.placeholder(tf.int64, [None, click_max_len], name="ph_click_sid")
    ph["user_colossus_sid"] = tf.placeholder(tf.int64, [None, colossus_max_len], name="ph_profile_sid")
    
    # 点击有效长度
    ph["click_len"]        = tf.placeholder(tf.int32,   [None], name="ph_click_len")
    ph["colossus_len"]     = tf.placeholder(tf.int32,   [None], name="ph_colossus_len")

    # 解码侧：teacher forcing 的输入与标签
    ph["photo_sid"]   = tf.placeholder(tf.int32, [None, 3], name="ph_photo_sid")   # 训练时 decoder 输入（全局ID）
    ph["label"]       = tf.placeholder(tf.int32, [None, 3], name="ph_label")       # 三个层级的标签（局部ID）
    ph["loss_masker"] = tf.placeholder(tf.int32, [None],    name="ph_loss_masker")  # >0 的样本参与 loss

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

        ph["user_profile_v1_click_pid_list"]: rng.randn(B, click_max_len, d_click_pid).astype(np.float32),
        ph["user_profile_v1_click_aid_list"]: rng.randn(B, click_max_len, d_click_aid).astype(np.float32),
        
        ph["user_colossus_pid_list"]: rng.randn(B, colossus_max_len, d_colossus_pid).astype(np.float32),
        ph["user_colossus_aid_list"]: rng.randn(B, colossus_max_len, d_colossus_aid).astype(np.float32),
       
        ph["user_click_sid"]: rng.randint(1, 500, size=(B, click_max_len)).astype("int64"),
        ph["user_colossus_sid"]: rng.randint(1, 500, size=(B, colossus_max_len)).astype("int64"),

        # 每条样本的有效长度（1..max_len）
        ph["click_len"]:        rng.randint(1, click_max_len+1, size=(B,), dtype=np.int32),
        ph["colossus_len"]:     rng.randint(1, colossus_max_len+1, size=(B,), dtype=np.int32),

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
        
        "user_colossus_pid_list": ph["user_colossus_pid_list"],
        "user_colossus_aid_list": ph["user_colossus_aid_list"],
    }
    feature_emb_size_dict = {
        "user_profile_v1_click_pid_list": ph["click_len"],
        "user_colossus_pid_list": ph["colossus_len"]
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
            batch_size=batch_size, click_max_len=max_len, vocab_sizes=vocab_sizes
        )

        # 2) 构建模型
        model = SIDRecModel(
            feature_emb_dict=feature_emb_dict,
            feature_emb_size_dict=feature_emb_size_dict,
            dim=dim,
            vocab_sizes=list(vocab_sizes),
            print_ops=print_ops_bucket
        )

        # 3) 前向与损失
        total_loss = model.model(
            ph["user_click_sid"],
            ph["user_colossus_sid"],
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

        # 5) Beam Search（你的新 infer 版本）
        gen_loc_greedy, probs_greedy = model.beam_search_fast(
            ph["user_click_sid"], ph["user_colossus_sid"], beam_size=512, temperature=1.0
        )

        # 6) 运行
        init = tf.global_variables_initializer()

        num_steps  = 10        # 训练总步数
        log_every  = 1         # 打印间隔
        eval_every = 1         # 做一次 beamsearch 间隔

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sess.run(init)

            # === 耗时统计桶 ===
            train_times = []   # 每次训练 step 耗时（秒）
            infer_times = []   # 每次推理 step 耗时（秒）

            for t in range(1, num_steps + 1):
                # === 训练计时 ===
                t0 = perf_counter()

                if t == 1:
                    # 首步做一次详细 profile
                    (_, lb_v, total_v, gs) = profile_run(
                        sess,
                        fetches=[train_op, total_lb_loss, total_loss, global_step],
                        feed_dict=fd,
                        tag="train_step{}".format(t),
                        topk_min_us=50,     # 过滤掉更小的op，日志更干净
                        max_depth=100
                    )
                else:
                    _, lb_v, total_v, gs = sess.run(
                        [train_op, total_lb_loss, total_loss, global_step],
                        feed_dict=fd
                    )

                train_dur = perf_counter() - t0
                train_times.append(train_dur)

                if t % log_every == 0:
                    print("[Train] step=%d  lb=%.6f  total=%.6f  time=%.2f ms" %
                          (gs, lb_v, total_v, train_dur * 1000.0))

                # === 推理/评估计时（beam search） ===
                if t % eval_every == 0:
                    t1 = perf_counter()
                    (gl_greedy, pr_greedy) = profile_run(
                        sess,
                        fetches=[gen_loc_greedy, probs_greedy],
                        feed_dict=fd,
                        tag="infer_step{}".format(t),
                        topk_min_us=50,
                        max_depth=100
                    )
                    infer_dur = perf_counter() - t1
                    infer_times.append(infer_dur)

                    print("[Eval]  step=%d  beam(greedy) gen_part_loc[0,0]: %s  time=%.2f ms" %
                          (gs, str(gl_greedy[0,0]), infer_dur * 1000.0))

            # 训练结束后再做一次完整评估（计时）
            t2 = perf_counter()
            (gl_greedy, pr_greedy) = profile_run(
                sess,
                fetches=[gen_loc_greedy, probs_greedy],
                feed_dict=fd,
                tag="infer_final",
                topk_min_us=50,
                max_depth=100
            )
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
