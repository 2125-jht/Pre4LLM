# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import sys
import os

# 添加模型路径
sys.path.append('kaiworks_explore_sid_tran_encoder_only_pid_v2')

# 禁用 TF2 行为
tf.compat.v1.disable_v2_behavior()

# 模拟必要的模块和函数
def mlp(name, net, hidden_units, output_unit=None, activation=tf.nn.relu):
    """改进的 MLP 实现，增加随机性"""
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        for i, units in enumerate(hidden_units):
            net = tf.layers.dense(net, units, activation=activation, name=f'layer_{i}',
                                # 增加权重初始化的随机性
                                kernel_initializer=tf.random_normal_initializer(0, 0.1),
                                bias_initializer=tf.random_normal_initializer(0, 0.01))
        if output_unit is not None:
            net = tf.layers.dense(net, output_unit, activation=None, name='output',
                                kernel_initializer=tf.random_normal_initializer(0, 0.1),
                                bias_initializer=tf.random_normal_initializer(0, 0.01))
    return net

class EncoderModel:
    def __init__(self, num_layers=4, dim=128, num_heads=8, dropout_rate=0.1, hidden_dim=256):
        self.num_layers = num_layers
        self.dim = dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.hidden_dim = hidden_dim
    
    def forward(self, inputs, training=False):
        # 改进的 encoder，增加更多变化
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            output = inputs
            for i in range(self.num_layers):
                # 添加残差连接和更复杂的变换
                residual = output
                output = tf.layers.dense(output, self.hidden_dim, activation=tf.nn.relu, 
                                       name=f'encoder_expand_{i}',
                                       kernel_initializer=tf.random_normal_initializer(0, 0.1))
                output = tf.layers.dense(output, self.dim, activation=None,
                                       name=f'encoder_project_{i}',
                                       kernel_initializer=tf.random_normal_initializer(0, 0.1))
                # 简单的残差连接
                output = output + residual
                output = tf.nn.relu(output)
        return output

class DecoderModel:
    def __init__(self, num_layers=4, dim=128, num_heads=8, dropout_rate=0.1, hidden_dim=256):
        self.num_layers = num_layers
        self.dim = dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.hidden_dim = hidden_dim
    
    def forward(self, dec_inputs, enc_outputs, training=False):
        # 改进的 decoder，增加 encoder-decoder attention 模拟
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            output = dec_inputs
            for i in range(self.num_layers):
                # Self-attention 模拟
                residual = output
                output = tf.layers.dense(output, self.hidden_dim, activation=tf.nn.relu,
                                       name=f'decoder_self_attn_{i}',
                                       kernel_initializer=tf.random_normal_initializer(0, 0.1))
                output = tf.layers.dense(output, self.dim, activation=None,
                                       name=f'decoder_self_proj_{i}',
                                       kernel_initializer=tf.random_normal_initializer(0, 0.1))
                output = output + residual
                
                # Cross-attention 模拟 (使用 encoder 输出)
                # 简化版本：将 encoder 输出投影后与 decoder 状态结合
                enc_context = tf.reduce_mean(enc_outputs, axis=1, keepdims=True)  # [B*beam, 1, dim]
                enc_context = tf.tile(enc_context, [1, tf.shape(output)[1], 1])   # 广播到序列长度
                
                combined = tf.concat([output, enc_context], axis=-1)  # [B*beam, seq, 2*dim]
                output = tf.layers.dense(combined, self.dim, activation=tf.nn.relu,
                                       name=f'decoder_cross_attn_{i}',
                                       kernel_initializer=tf.random_normal_initializer(0, 0.1))
                
        return output
# 全局特征名称（模拟）
user_static_fea_names = ['age', 'gender', 'location']
user_click_fea_names = ['item_1', 'item_2', 'item_3']

class TestBeamSearchModel:
    def __init__(self, vocab_sizes=[8192, 8192, 8192], dim=128):
        self._vocab_sizes = vocab_sizes
        self._dim = dim
        self._total_vocab_size = sum(vocab_sizes)
        
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

    def beam_search_fast_v2(self, beam_sizes=(16, 128, 1024), temperature=1.0):
        """Tree‑style beam search.

        Each decoding step uses a (possibly) different beam width:
            step 0 -> beam_sizes[0]
            step 1 -> beam_sizes[1]
            step 2 -> beam_sizes[2]
        …and so on.  Therefore the final output contains `beam_sizes[-1]` paths.

        Args:
            beam_sizes: tuple/list with length = len(self._vocab_sizes)
            temperature: softmax temperature.

        Returns:
            gen_part_loc: [B, beam_sizes[-1], seq_len] local‑id sequence
            probs       : same shape, token‑level probabilities.
        """
        # ----------- sanity check -----------
        num_levels = len(self._vocab_sizes)
        beam_sizes = list(beam_sizes)
        assert len(beam_sizes) == num_levels, "beam_sizes length must match number of vocab levels"

        offsets = [0,
                self._vocab_sizes[0],
                self._vocab_sizes[0] + self._vocab_sizes[1]]

        encoder_model = EncoderModel(num_layers=4, dim=self._dim, num_heads=8,
                                    dropout_rate=0.1, hidden_dim=self._dim * 2)
        decoder_model = DecoderModel(num_layers=4, dim=self._dim, num_heads=8,
                                    dropout_rate=0.1, hidden_dim=self._dim * 2)

        # ---------- encode user ----------
        user_static = tf.concat([self._feature_emb_dict[f] for f in user_static_fea_names], axis=1)
        user_static = mlp('user_static_emb', user_static, [2 * self._dim], self._dim,
                        activation=tf.nn.leaky_relu)
        B = tf.shape(user_static)[0]
        user_static = tf.reshape(user_static, [B, 1, self._dim])

        user_click = tf.concat([self._feature_emb_dict[f] for f in user_click_fea_names], axis=2)
        user_click = mlp('user_click_emb', user_click, [4 * self._dim], self._dim,
                        activation=tf.nn.leaky_relu)

        enc_in_base = tf.concat([user_static, user_click], axis=1)
        enc_out_base = encoder_model.forward(enc_in_base, training=False)      # [B, L_enc, C]

        # ---------- init beam ----------
        start_tok = tf.fill([B, 1], self._total_vocab_size)
        seqs   = tf.expand_dims(start_tok, 1)           # [B, 1, 1]
        probs  = tf.ones_like(seqs, tf.float32)
        scores = tf.zeros([B, 1], tf.float32)
        cur_beam = 1

        # ---------- decode levels ----------
        for step, V in enumerate(self._vocab_sizes):
            k_target = beam_sizes[step]                 # desired beam width this level

            # ---- prepare decoder input ----
            dec_in = tf.nn.embedding_lookup(self._embedding, seqs)            # [B, cur_beam, T, C]
            dec_in = tf.reshape(dec_in, [B * cur_beam, -1, self._dim])

            enc_out = tf.tile(tf.expand_dims(enc_out_base, 1), [1, cur_beam, 1, 1])
            enc_out = tf.reshape(enc_out, [B * cur_beam, -1, self._dim])

            dec_out = decoder_model.forward(dec_in, enc_out, training=False)  # [B*cur_beam, T, C]
            dec_out = tf.reshape(dec_out, [B, cur_beam, -1, self._dim])
            last_h  = dec_out[:, :, -1, :]                                    # [B, cur_beam, C]

            logits = tf.layers.dense(last_h, V, name=f'proj_{step}', reuse=tf.AUTO_REUSE)
            logp   = tf.nn.log_softmax(logits / temperature)                  # [B, cur_beam, V]

            # top‑k over vocabulary
            topk_logp, topk_tok = tf.nn.top_k(logp, k=V)  # keep full vocab first
            # reshape for candidate enumeration: [B, cur_beam * V]
            cand_scores = tf.reshape(scores[..., None] + topk_logp, [B, -1])
            best_scores, best_idx = tf.nn.top_k(cand_scores, k=k_target)       # pick target beam

            parent_beam = best_idx // V
            tok_rank    = best_idx %  V
            batch_idx   = tf.tile(tf.expand_dims(tf.range(B), 1), [1, k_target])

            gather_parent = tf.stack([batch_idx, parent_beam], axis=2)
            parent_seq  = tf.gather_nd(seqs,  gather_parent)
            parent_prob = tf.gather_nd(probs, gather_parent)

            tok_gather = tf.stack([batch_idx, parent_beam, tok_rank], axis=2)
            next_tok   = tf.gather_nd(topk_tok,  tok_gather)
            next_prob  = tf.gather_nd(tf.exp(topk_logp), tok_gather)

            # local→global id
            next_tok_glb = next_tok + offsets[step]

            seqs   = tf.concat([parent_seq, tf.expand_dims(next_tok_glb, -1)], axis=-1)
            probs  = tf.concat([parent_prob, tf.expand_dims(next_prob,   -1)], axis=-1)
            scores = best_scores
            cur_beam = k_target

        # strip <START>
        seqs  = seqs[:, :, 1:]
        probs = probs[:, :, 1:]

        offsets_t = tf.constant(offsets, dtype=seqs.dtype)          # [3]
        gen_part_loc = seqs - offsets_t

        return gen_part_loc, probs


def test_beam_search_fast():
    """测试 beam_search_fast 函数"""
    print("=== 测试 Beam Search Fast (改进版) ===")
    
    # 创建模型
    model = TestBeamSearchModel(vocab_sizes=[8192, 8192, 8192], dim=64)
    
    # 构建计算图
    # beam_size = 10
    # gen_sequences, gen_probs = model.beam_search_fast(beam_size=beam_size, temperature=20)  # 调整温度
    gen_sequences, gen_probs = model.beam_search_fast_v2(beam_sizes=(16, 128, 1024), temperature=20)  # 调整温度
    
    # 创建测试数据 - 增加随机性
    batch_size = 3  # 增加 batch size
    seq_len = 5
    dim = 64
    
    feed_dict = {}
    
    # 用户静态特征 (增加更多随机性)
    np.random.seed(None)  # 不固定随机种子
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
        sess.run(tf.global_variables_initializer())
        
        print("运行 beam search...")
        try:
            sequences, probs = sess.run([gen_sequences, gen_probs], feed_dict=feed_dict)
            print(f"生成序列形状: {sequences.shape}")
            print(f"概率形状: {probs.shape}")
            
            print("\n生成的序列 (局部ID):")
            for b in range(batch_size):
                print(f"Batch {b}:")
                unique_sequences = set()
                for beam in range(1024):
                    seq = sequences[b, beam]
                    prob = probs[b, beam]
                    seq_tuple = tuple(seq)
                    unique_sequences.add(seq_tuple)
                    print(f"  Beam {beam}: 序列={seq}, 概率={prob}")
                print(f"  唯一序列数量: {len(unique_sequences)}")
                print()
            
            # 分析多样性
            total_unique = 0
            for b in range(batch_size):
                unique_seqs = set()
                for beam in range(1024):
                    seq_tuple = tuple(sequences[b, beam])
                    unique_seqs.add(seq_tuple)
                total_unique += len(unique_seqs)
            
            avg_diversity = total_unique / batch_size
            print(f"平均每个batch的唯一序列数: {avg_diversity:.2f} / {1024}")
            
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
            
            print("✓ Beam Search 测试完成!")
            
        except Exception as e:
            print(f"✗ 测试失败: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_beam_search_fast()
