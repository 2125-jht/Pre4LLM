#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
benchmark_beam.py
-------------------------------------------------
比较“手写 concat 版 beam search” 与
“gather_tree 回溯版 beam search”的执行时间。
仅依赖 tensorflow-1.x，随机伪造输入即可运行。
"""

import time
import numpy as np
import tensorflow as tf
tf.disable_eager_execution()          # graph-mode

# ---------------- 超参数 / 随机输入 ----------------
BATCH = 4
DIM   = 32
BEAM  = 64            # 尽量调大可放大差距
VOCAB_SIZES = [512, 512, 512]   # 三段语义 token
SEQ_T = len(VOCAB_SIZES)        # =3
ENC_LEN = 10                    # encoder 序列长度
WARM, RUN = 2, 10               # 预热轮与计时轮

rng = np.random.RandomState(42)

# ---------------- 伪造一个“模型” ------------------
def mlp(x, out_dim, name):
    return tf.layers.dense(x, out_dim, activation=tf.nn.relu,
                           name=name, reuse=tf.AUTO_REUSE)

def dummy_encoder(x):
    # x: [B,L,D] -> same  (只是占位)
    return tf.identity(x)

def dummy_decoder(enc, dec):
    # enc: [B*beam,L,D]  dec: [B*beam,T,D]
    # 直接 return dec 作为“decoder 输出”
    return tf.identity(dec)

# --------------- 构建计算图 -----------------------
graph = tf.Graph()
with graph.as_default():
    # ---- 随机 encoder 输入 ----
    enc_input = tf.constant(rng.randn(BATCH, ENC_LEN, DIM), dtype=tf.float32)
    enc_out   = dummy_encoder(enc_input)                   # [B,L,D]

    # ---- 共享 embedding ----
    TOTAL_VOCAB = sum(VOCAB_SIZES)
    embedding = tf.get_variable('embed',
                    shape=[TOTAL_VOCAB+1, DIM],
                    initializer=tf.random_uniform_initializer(-0.1,0.1))

    # --------------------------------------------------
    # 1) 旧版：每步 tile + concat 序列
    # --------------------------------------------------
    def old_beam_search():
        scores = tf.zeros([BATCH, BEAM])
        seqs   = tf.tile(
            tf.fill([1,1,1], TOTAL_VOCAB), [BATCH, BEAM, 1])   # [B,K,1]

        for step, vsize in enumerate(VOCAB_SIZES):
            dec_in = tf.nn.embedding_lookup(embedding, seqs)   # [B,K,T,D]
            enc_tile = tf.tile(enc_out[:,None,:,:], [1,BEAM,1,1])
            dec_out = dummy_decoder(enc_tile, dec_in)          # same shape
            logits  = mlp(dec_out[:,:,step,:], vsize, f'proj{step}')
            logp    = tf.nn.log_softmax(logits/2.0)            # [B,K,V]

            cand = tf.expand_dims(scores,-1)+logp
            flat = tf.reshape(cand, [BATCH, -1])               # [B,K*V]
            topk, idx = tf.nn.top_k(flat, k=BEAM)

            parent = idx // vsize
            token  = idx %  vsize

            # 更新序列 / 分数
            gather = tf.stack([tf.range(BATCH)[:,None], parent], -1)
            seqs = tf.gather_nd(seqs, gather)                  # [B,K,T]
            seqs = tf.concat([seqs, tf.expand_dims(token,-1)], -1)
            scores = topk
        return seqs[:,0,:]     # 取 best beam

    # --------------------------------------------------
    # 2) 新版：记录 (step_ids,parent_ids) + gather_tree
    # --------------------------------------------------
    from tensorflow.contrib.seq2seq import beam_search_ops
    def fast_beam_search():
        enc_tile = tf.contrib.seq2seq.tile_batch(enc_out, BEAM)  # [B*K,L,D]
        seqs = tf.fill([BATCH*BEAM, 1], TOTAL_VOCAB)             # [B*K,1]
        scores = tf.zeros([BATCH, BEAM])

        step_arr   = tf.TensorArray(tf.int32, size=SEQ_T)
        parent_arr = tf.TensorArray(tf.int32, size=SEQ_T)

        for t, vsize in enumerate(VOCAB_SIZES):
            dec_in  = tf.nn.embedding_lookup(embedding, seqs)    # [B*K,T,D]
            dec_out = dummy_decoder(enc_tile, dec_in)
            logits  = mlp(dec_out[:, -1, :], vsize, f'proj{t}')
            logp = tf.nn.log_softmax(logits/2.0)
            logp = tf.reshape(logp, [BATCH, BEAM, vsize])

            cand = tf.expand_dims(scores,-1)+logp
            flat = tf.reshape(cand, [BATCH, -1])
            topk, idx = tf.nn.top_k(flat, k=BEAM)

            parent = tf.cast(idx // vsize, tf.int32)
            token  = tf.cast(idx %  vsize, tf.int32)
            step_arr   = step_arr.write(t, token)
            parent_arr = parent_arr.write(t, parent)
            scores = topk

            gather = parent + tf.expand_dims(
                     tf.range(BATCH)*BEAM, 1)
            flat_beam = tf.reshape(gather, [-1])
            seqs  = tf.gather(seqs, flat_beam)
            seqs  = tf.concat([seqs,
                     tf.reshape(token, [-1,1])], 1)
            enc_tile = tf.gather(enc_tile, flat_beam)

        step_ids   = step_arr.stack()        # [T,B,K]
        parent_ids = parent_arr.stack()
        seq_len    = tf.fill([BATCH, BEAM], SEQ_T)
        final_ids  = beam_search_ops.gather_tree(step_ids,
                         parent_ids, seq_len)            # [T,B,K]
        return tf.transpose(final_ids,[1,2,0])[:,0,:]   # best beam

    out_old = old_beam_search()
    out_new = fast_beam_search()

# ------------------ Session & Benchmark ------------------
def bench(fetch, sess, warm=WARM, run=RUN):
    for _ in range(warm):
        sess.run(fetch)
    t0 = time.time()
    for _ in range(run):
        sess.run(fetch)
    return (time.time() - t0)/run

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    t_old = bench(out_old, sess)
    t_new = bench(out_new, sess)

    print(f"[Beam={BEAM}] old  avg  {t_old*1e3:6.2f} ms / batch")
    print(f"[Beam={BEAM}] fast avg  {t_new*1e3:6.2f} ms / batch")
