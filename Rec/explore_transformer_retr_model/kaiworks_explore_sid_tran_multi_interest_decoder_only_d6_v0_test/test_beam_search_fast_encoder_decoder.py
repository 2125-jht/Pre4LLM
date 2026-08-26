# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import sys
import os
import random
# 固定随机种子，确保结果可重复
RANDOM_SEED = 42

# 设置Python随机种子
random.seed(RANDOM_SEED)

# 设置NumPy随机种子
np.random.seed(RANDOM_SEED)

# 设置TensorFlow随机种子
tf.compat.v1.set_random_seed(RANDOM_SEED)

from modulesV2 import *

# 禁用 TF2 行为
tf.compat.v1.disable_v2_behavior()
# 全局特征名称（模拟）
user_static_fea_names = ['age', 'gender', 'location']
user_click_fea_names = ['item_1', 'item_2', 'item_3']

class TestBeamSearchModel:
    def __init__(self, vocab_sizes=[8192, 8192, 8192], dim=128):
        self._vocab_sizes = vocab_sizes
        self._dim = dim
        self._total_vocab_size = sum(vocab_sizes)
        self._selected_size = 5
        
        # 创建 embedding 表，增加初始化随机性
        self._embedding = tf.get_variable(
            'embedding_table',
            shape=[self._total_vocab_size + 1, self._dim],  # +1 for start token
            initializer=tf.random_normal_initializer(0, 0.1)  # 增加随机性
        )
        
        # 模拟特征 embedding 字典
        self._feature_emb_dict = {}
        
        # 为用户静态特征创建 embedding
        for feat in user_static_fea_names:
            self._feature_emb_dict[feat] = tf.placeholder(
                tf.float32, [None, self._dim], name=f'{feat}_emb'
            )
        
        # 为用户点击特征创建 embedding
        for feat in user_click_fea_names:
            self._feature_emb_dict[feat] = tf.placeholder(
                tf.float32, [None, None, self._dim], name=f'{feat}_emb'
            )

    def beam_search_fast(self, beam_size=512, temperature=1):
        """
        O(batch·beam·logV) 近似复杂度的束搜索（显存与 beam_size 线性）

        改进版本：
        * **step=0** 仅用 1 条 beam，从 |V_0| 里直接选 top‑k 形成不同路径，
        避免所有 beam 被同一起点锁死。
        * step>0 时保持固定 beam_size。

        返回：
            gen_part_loc  – shape [B, beam_size, seq_len] 的推荐 sid 局部 id 序列
            probs         – 同形状，逐 token 的 softmax 概率（便于做温度/多样性分析）
        """
            # ---------------- 0. 一次性建全图（warm-up） -----------------
        with tf.variable_scope("warmup", reuse=tf.AUTO_REUSE):
            dummy_B, dummy_beam = 1, 1
            dummy_seq_len = 3          # 随便 ≥1
            dummy_enc_len = 4

            # ① 造假的 encoder / decoder 输入
            dummy_enc = tf.zeros([dummy_B, dummy_enc_len, self._dim])
            dummy_dec = tf.zeros([dummy_B*dummy_beam, dummy_seq_len, self._dim])

            encoder_model = EncoderModel(num_layers=4, dim=self._dim, num_heads=8,
                                        dropout_rate=0.1, hidden_dim=self._dim * 2)
            decoder_model = DecoderModel(num_layers=4, dim=self._dim, num_heads=8,
                                        dropout_rate=0.1, hidden_dim=self._dim * 2)

            enc_out = encoder_model.forward(dummy_enc, training=False)

            # ——任选其一即可；二者都跑更保险——
            _ = decoder_model.forward(dummy_dec, enc_out, training=False)      # full-forward
            _ , _ = decoder_model.step(dummy_dec[:, :1, :], 1, enc_out, {})    # single-step
        
        
        # ------------- 常量 & 子模型 -------------
        offsets = [0,
                self._vocab_sizes[0],
                self._vocab_sizes[0] + self._vocab_sizes[1]]       # 局部→全局 id 偏移

        encoder_model = EncoderModel(num_layers=4, dim=self._dim, num_heads=8,
                                    dropout_rate=0.1, hidden_dim=self._dim * 2)
        decoder_model = DecoderModel(num_layers=4, dim=self._dim, num_heads=8,
                                    dropout_rate=0.1, hidden_dim=self._dim * 2)

        # ---------- ① 预处理：编码用户 ----------
        user_static = tf.concat([self._feature_emb_dict[f] for f in user_static_fea_names], axis=1)
        user_static = mlp('user_static_emb', user_static, [2 * self._dim], self._dim,
                        activation=tf.nn.leaky_relu)

        B = tf.shape(user_static)[0]
        user_static = tf.reshape(user_static, [B, 1, self._dim])

        user_click = tf.concat([self._feature_emb_dict[f] for f in user_click_fea_names], axis=2)
        user_click = mlp('user_click_emb', user_click, [4 * self._dim], self._dim,
                        activation=tf.nn.leaky_relu)

        enc_in_base = tf.concat([user_static, user_click], axis=1)            # [B, L_enc, C]
        enc_out_base = encoder_model.forward(enc_in_base, training=False)     # [B, L_enc, C]

        # ---------- ② Beam 状态初始化 ----------
        start_tok = tf.fill([B, 1], self._total_vocab_size)   # global id of <START>
        seqs   = tf.expand_dims(start_tok, 1)                 # [B, 1, 1]
        probs  = tf.ones_like(seqs, dtype=tf.float32)         # [B, 1, 1]
        scores = tf.zeros([B, 1], dtype=tf.float32)           # [B, 1]

        cur_beam = 1  # 当前 beam 数
        cache = {}                    # 全层 KV

        # ---------- ③ 逐层解码 ----------
        for step, V in enumerate(self._vocab_sizes):
            # 只 embed 当前 token
            dec_in = tf.nn.embedding_lookup(self._embedding, seqs[:, :, -1])  # [B,beam,1,C]
            dec_in = tf.reshape(dec_in, [B*cur_beam, 1, self._dim])

            dec_out, cache = decoder_model.step(
                dec_in, cur_beam, enc_out_base, cache)            # 只算一步

            last_h = tf.reshape(dec_out, [B, cur_beam, self._dim])
            # # 仅取最后一个 time‑step（上一 token）输出做投影
            # dec_out = tf.reshape(dec_out, [B, cur_beam, -1, self._dim])
            # last_h = dec_out[:, :, -1, :]                                      # [B, cur_beam, C]
            
            with tf.variable_scope('proj_%d' % step):
                logits = tf.layers.dense(last_h, V, name=f'pred', reuse=tf.AUTO_REUSE)

            logp = tf.nn.log_softmax(logits / temperature)                     # [B, cur_beam, V]

            # --- 本轮候选：parent_beam × top‑V → (cur_beam*V)
            k = beam_size if step == 0 else beam_size                          # 第 0 步从 |V| 里挑 beam_size
            topk_logp, topk_tok = tf.nn.top_k(logp, k=k)                       # [B, cur_beam, k]
            topk_prob = tf.exp(topk_logp)

            # 累积得分
            cand_scores = tf.expand_dims(scores, -1) + topk_logp               # [B, cur_beam, k]

            # --- 选全局 top‑beam_size ---
            flat_scores = tf.reshape(cand_scores, [B, -1])                     # [B, cur_beam*k]
            best_scores, best_idx = tf.nn.top_k(flat_scores, k=beam_size)      # 取新的 beam

            parent_beam = best_idx // k                                        # index in 0..cur_beam‑1
            tok_rank    = best_idx %  k                                        # index in 0..k‑1

            batch_idx = tf.tile(tf.expand_dims(tf.range(B), 1), [1, beam_size])

            # gather 父路径
            gather_parent = tf.stack([batch_idx, parent_beam], axis=2)         # [B, beam, 2]
            parent_seq   = tf.gather_nd(seqs,  gather_parent)                  # [B, beam, T]
            parent_prob  = tf.gather_nd(probs, gather_parent)

            def gather_cache(old_cache, gp):
                new_cache = {}
                for k, v in old_cache.items():
                    if k.startswith(("k_self_", "v_self_")):
                        new_cache[k] = tf.gather_nd(v, gp)   # [B, beam, H, T, Dh] → 重新排序
                    else:
                        new_cache[k] = v                     # k_enc / v_enc 原样保留
                return new_cache
            cache = gather_cache(cache, gather_parent)
            
            # gather 新 token
            tok_gather = tf.stack([batch_idx, parent_beam, tok_rank], axis=2)
            next_tok   = tf.gather_nd(topk_tok,  tok_gather)                   # [B, beam]
            next_prob  = tf.gather_nd(topk_prob, tok_gather)                   # [B, beam]
            
            # map 到全局 id
            next_tok_glb = next_tok + offsets[step]

            # 更新序列
            seqs  = tf.concat([parent_seq, tf.expand_dims(next_tok_glb, -1)], axis=-1)  # [B, beam, T+1]
            probs = tf.concat([parent_prob, tf.expand_dims(next_prob,  -1)], axis=-1)
            scores = best_scores                                                # [B, beam]

            cur_beam = beam_size            # 以后固定

        # 去掉 <START>
        seqs  = seqs[:, :, 1:]
        probs = probs[:, :, 1:]

        # 转回局部 id
        offsets_t = tf.constant(offsets, dtype=seqs.dtype)          # [3]
        gen_part_loc = seqs - offsets_t

        return gen_part_loc, probs




def test_beam_search_fast():
    """测试 beam_search_fast 函数"""
    print("=== 测试 Beam Search Fast (改进版) ===")
    
    import time
    start_time = time.time()
    
    # 构建计算图
    beam_size = 256
    batch_size = 8
    seq_len = 256
    dim = 64
    
    print(f"测试参数: beam_size={beam_size}, batch_size={batch_size}, seq_len={seq_len}, dim={dim}")
    
    # 创建模型
    print("开始构建计算图...")
    graph_start_time = time.time()
    
    model = TestBeamSearchModel(vocab_sizes=[8192, 8192, 8192], dim=dim)
    gen_sequences, gen_probs = model.beam_search_fast(beam_size=beam_size, temperature=1)  # 调整温度
    
    graph_end_time = time.time()
    print(f"✓ 计算图构建完成，耗时: {graph_end_time - graph_start_time:.3f}秒")
    
    feed_dict = {}
    # 用户静态特征 (增加更多随机性)
    # np.random.seed(20)  # 不固定随机种子
    for feat in user_static_fea_names:
        # 增加特征的差异性
        feat_data = np.random.randn(batch_size, dim).astype(np.float32) * 0.5
        # 为不同batch添加不同的偏置
        for b in range(batch_size):
            feat_data[b] += np.random.randn(dim) * 0.2
        feed_dict[model._feature_emb_dict[feat]] = feat_data
    
    # 用户点击特征 (增加更多随机性)
    for feat in user_click_fea_names:
        feat_data = np.random.randn(batch_size, seq_len, dim).astype(np.float32) * 0.5
        # 为不同batch添加不同的模式
        for b in range(batch_size):
            feat_data[b] += np.random.randn(seq_len, dim) * 0.3
        feed_dict[model._feature_emb_dict[feat]] = feat_data
    
    # 运行测试
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    
    with tf.Session(config=config) as sess:
        # 初始化变量
        print("开始初始化变量...")
        init_start_time = time.time()
        sess.run(tf.global_variables_initializer())
        init_end_time = time.time()
        print(f"✓ 变量初始化完成，耗时: {init_end_time - init_start_time:.3f}秒")
        
        print("开始运行 beam search...")
        try:
            # 第一次运行（包含图编译时间）
            inference_start_time = time.time()
            sequences, probs = sess.run([gen_sequences, gen_probs], feed_dict=feed_dict)
            first_inference_end_time = time.time()
            first_inference_time = first_inference_end_time - inference_start_time
            print(f"✓ 第一次推理完成（包含图编译），耗时: {first_inference_time:.3f}秒")
            
            # 第二次运行（纯推理时间）
            inference_start_time = time.time()
            sequences, probs = sess.run([gen_sequences, gen_probs], feed_dict=feed_dict)
            second_inference_end_time = time.time()
            pure_inference_time = second_inference_end_time - inference_start_time
            print(f"✓ 第二次推理完成（纯推理时间），耗时: {pure_inference_time:.3f}秒")
            
            print(f"生成序列形状: {sequences.shape}")
            print(f"概率形状: {probs.shape}")
            
            # 计算性能指标
            total_sequences = batch_size * beam_size
            sequences_per_second = total_sequences / pure_inference_time
            print(f"\n性能指标:")
            print(f"总序列数: {total_sequences}")
            print(f"推理速度: {sequences_per_second:.1f} 序列/秒")
            print(f"每个序列平均耗时: {pure_inference_time * 1000 / total_sequences:.2f} 毫秒")
            
            print("\n生成的序列 (局部ID) - 仅显示前2个batch:")
            for b in range(min(2, batch_size)):  # 只显示前2个batch以节省输出
                print(f"Batch {b}:")
                unique_sequences = set()
                for beam in range(min(5, beam_size)):  # 只显示前5个beam
                    seq = sequences[b, beam]
                    prob = probs[b, beam]
                    seq_tuple = tuple(seq)
                    unique_sequences.add(seq_tuple)
                    print(f"  Beam {beam}: 序列={seq}, 概率={prob[:3]}...")  # 只显示前3个概率
                print(f"  当前batch唯一序列数量: {len(unique_sequences)}")
                print()
            
            # 分析多样性
            total_unique = 0
            for b in range(batch_size):
                unique_seqs = set()
                for beam in range(beam_size):
                    seq_tuple = tuple(sequences[b, beam])
                    unique_seqs.add(seq_tuple)
                total_unique += len(unique_seqs)
            
            avg_diversity = total_unique / batch_size
            print(f"平均每个batch的唯一序列数: {avg_diversity:.2f} / {beam_size}")
            
            # 验证输出范围
            print("验证输出范围:")
            print(f"序列ID范围: [{sequences.min()}, {sequences.max()}]")
            print(f"概率范围: [{probs.min():.6f}, {probs.max():.6f}]")
            # 检查每个位置的ID是否在正确范围内
            vocab_sizes = [8192, 8192, 8192]
            for pos in range(3):
                pos_values = sequences[:, :, pos]
                expected_range = vocab_sizes[pos]
                print(f"位置 {pos}: ID范围 [0, {expected_range-1}], 实际范围 [{pos_values.min()}, {pos_values.max()}]")
            
            if avg_diversity > 1.0:
                print("✓ Beam Search 生成了多样化的结果!")
            else:
                print("⚠ Beam Search 结果多样性不足，可能需要进一步调整参数")
            
            # 总体时间统计
            total_time = time.time() - start_time
            print(f"\n时间统计总结:")
            print(f"总耗时: {total_time:.3f}秒")
            print(f"├─ 计算图构建: {graph_end_time - graph_start_time:.3f}秒 ({(graph_end_time - graph_start_time)/total_time*100:.1f}%)")
            print(f"├─ 变量初始化: {init_end_time - init_start_time:.3f}秒 ({(init_end_time - init_start_time)/total_time*100:.1f}%)")
            print(f"├─ 首次推理(含编译): {first_inference_time:.3f}秒 ({first_inference_time/total_time*100:.1f}%)")
            print(f"└─ 纯推理时间: {pure_inference_time:.3f}秒 ({pure_inference_time/total_time*100:.1f}%)")
            
            print("✓ Beam Search 测试完成!")
            
        except Exception as e:
            print(f"✗ 测试失败: {e}")
            import traceback
            traceback.print_exc()
if __name__ == "__main__":
    test_beam_search_fast()
