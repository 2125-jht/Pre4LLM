import tensorflow as tf
import numpy as np
from modules_ import *

def layer_norm(x, scope, eps=1e-6):
    with tf.variable_scope(f"{scope}/layer_norm", reuse=tf.AUTO_REUSE):
        gamma = tf.get_variable("gamma", [x.get_shape()[-1]], initializer=tf.ones_initializer())
        beta = tf.get_variable("beta", [x.get_shape()[-1]], initializer=tf.zeros_initializer())

        mean, variance = tf.nn.moments(x, axes=[-1], keep_dims=True)
        normalized = (x - mean) / tf.sqrt(variance + eps)
        output = gamma * normalized + beta
    return output

def swish(x):
    return x * tf.nn.sigmoid(x)

def moe_swiglu_feed_forward_network(dim,
                                    hidden_dim,          # 建议 d_ff = (2/3)*4*d_model
                                    num_experts=24,
                                    top_k=2,
                                    dropout_rate=0.1,
                                    training=False):
    """
    稀疏 MoE FFN（Top-2 路由）+ SwiGLU Expert
    - 只对被选中的 token 计算对应 expert 前向，避免 24 个 expert 全部计算
    - 使用 scatter 累加把各 expert 的加权输出合回原序列
    形状：
      x: [B, L, D] -> out: [B, L, D]
    """
    assert top_k == 2, "本实现固定为 Top-2（Act=2）"

    def expert_forward(x_sel, expert_id):
        with tf.variable_scope(f"expert_{expert_id}", reuse=tf.AUTO_REUSE):
            # SwiGLU 上升两支
            a = tf.layers.dense(x_sel, hidden_dim, use_bias=True, name="w_up_a")  # [M, H]
            b = tf.layers.dense(x_sel, hidden_dim, use_bias=True, name="w_up_b")  # [M, H]
            h = a * swish(b)                                                      # SwiGLU
            h = tf.layers.dropout(h, rate=dropout_rate,
                                   training=tf.convert_to_tensor(training, tf.bool))
            out = tf.layers.dense(h, dim, use_bias=True, name="w_down")           # [M, D]
        return out

    def ffn(x, training=training, token_mask=None):
        # training = tf.convert_to_tensor(training, dtype=tf.bool)
        B = tf.shape(x)[0]; L = tf.shape(x)[1]
        D = x.get_shape().as_list()[-1]
        N = B * L

        with tf.variable_scope("moe_ffn", reuse=tf.AUTO_REUSE):
            # ---------- Router ----------
            gate_logits = tf.layers.dense(x, num_experts, use_bias=True, name="router")  # [B,L,E]
            gate_probs  = tf.nn.softmax(gate_logits, axis=-1)                            # [B,L,E]

            # ---------- token mask 处理 ----------
            if token_mask is not None:
                # [B,L] -> [N], bool / float 皆可
                mask_f = tf.cast(token_mask, gate_probs.dtype)       # [B,L]
                mask_b = tf.cast(mask_f > 0.5, tf.bool)              # [B,L]
                valid_flat_f = tf.reshape(mask_f, [N])               # [N]
                valid_flat_b = tf.reshape(mask_b, [N])               # [N]
                valid_count = tf.maximum(tf.reduce_sum(valid_flat_f), 1e-9)
            else:
                # 全部有效
                valid_flat_f = tf.ones([N], dtype=gate_probs.dtype)
                valid_flat_b = tf.ones([N], dtype=tf.bool)
                valid_count  = tf.cast(N, gate_probs.dtype)

            probs_flat = tf.reshape(gate_probs, [N, num_experts])    # [N,E]

            if training:
                # ---------- Switch-style Load Balance Loss ----------
                # p_e: 有效 token 上 softmax 概率的均值
                p_e = tf.reduce_sum(probs_flat * tf.expand_dims(valid_flat_f, 1), axis=0) / valid_count  # [E]

                # f_e: 有效 token 的 top-1 分配占比
                assigned = tf.argmax(probs_flat, axis=-1, output_type=tf.int32)                          # [N]
                one_hot  = tf.one_hot(assigned, depth=num_experts, dtype=probs_flat.dtype)               # [N,E]
                one_hot_valid = one_hot * tf.expand_dims(valid_flat_f, 1)                                # [N,E]
                f_e = tf.reduce_sum(one_hot_valid, axis=0) / valid_count                                 # [E]
                
                lb_loss = num_experts * tf.reduce_mean(p_e * f_e)
                tf.add_to_collection("lb_loss", lb_loss)
            else:
                tf.add_to_collection("lb_loss", 0.0)

            # ---------- Top-2 路由 + 稀疏执行 ----------
            topk_probs, topk_indices = tf.nn.top_k(gate_probs, k=top_k)       # [B,L,2], [B,L,2]
            topk_probs_f   = tf.reshape(topk_probs,   [N, top_k])              # [N,2]
            topk_indices_f = tf.reshape(topk_indices, [N, top_k])              # [N,2]

            # 仅对有效 token 执行：用于筛 index 与权重
            valid2_b = tf.stack([valid_flat_b, valid_flat_b], axis=1)          # [N,2]
            valid2_f = tf.cast(valid2_b, topk_probs_f.dtype)                    # [N,2]

            out_flat = tf.zeros([N, D], dtype=x.dtype)

            for e in range(num_experts):
                mask2 = tf.equal(topk_indices_f, e)                            # [N,2] bool
                # 只有同时满足(被该 expert 选中) 且 (是有效 token) 才参与
                mask2_and_valid = tf.logical_and(mask2, valid2_b)              # [N,2]
                any_mask = tf.reduce_any(mask2_and_valid, axis=1)              # [N] bool
                idx = tf.squeeze(tf.where(any_mask), axis=1)                   # [M]

                def branch_non_empty():
                    x_flat = tf.reshape(x, [N, D])                              # [N,D]
                    x_sel = tf.gather(x_flat, idx)                              # [M,D]

                    out_sel = expert_forward(x_sel, expert_id=e)                # [M,D]

                    probs_rows = tf.gather(topk_probs_f, idx)                   # [M,2]
                    mask_rows  = tf.gather(mask2,        idx)                   # [M,2] bool
                    valid_rows = tf.gather(valid2_f,     idx)                   # [M,2] float

                    w_sel = tf.reduce_sum(tf.cast(mask_rows, probs_rows.dtype) * probs_rows * valid_rows,
                                          axis=1, keepdims=True)                # [M,1]

                    contrib = out_sel * w_sel                                   # [M,D]
                    return tf.scatter_nd(tf.expand_dims(idx, 1), contrib, [N, D])

                def branch_empty():
                    return tf.zeros([N, D], dtype=x.dtype)

                out_flat += tf.cond(tf.size(idx) > 0, branch_non_empty, branch_empty)

            out = tf.reshape(out_flat, [B, L, D])

            # 把 PAD 位置输出清零，行为更可控
            if token_mask is not None:
                out = out * tf.expand_dims(mask_f, -1)  # [B,L,1]

            return out

    return ffn

def apply_mask_add(logits, src_mask, neg_large=-1e9):
    """
    logits: [B,H,Lq,Lk]
    src_mask: [B,1,1,Lk]  (0=pad, 1=valid; any dtype)
    """
    src_mask_f = tf.cast(src_mask, logits.dtype)
    neg = tf.constant(neg_large, dtype=logits.dtype)
    return logits + (1. - src_mask_f) * neg
    
def multi_head_attention(queries, keys, values, num_heads, src_mask, dropout_rate, training=False):
    
    def split_heads(x, num_heads):
        batch_size = tf.shape(x)[0]
        depth = x.get_shape().as_list()[-1] // num_heads
        reshaped = tf.reshape(x, [batch_size, -1, num_heads, depth])
        return tf.transpose(reshaped, [0, 2, 1, 3])

    training = tf.constant(training, dtype=tf.bool)

    def scaled_dot_product_attention(Q, K, V, src_mask):
        matmul_qk = tf.matmul(Q, K, transpose_b=True)
        dk = tf.cast(tf.shape(K)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        B = tf.shape(scaled_attention_logits)[0]
        H = tf.shape(scaled_attention_logits)[1]
        T_q = tf.shape(scaled_attention_logits)[-2]
        T_k = tf.shape(scaled_attention_logits)[-1]
        
        # 扩展到 [B, H, T_q, T_k] 用 [1,1,T_q,T_k] 让广播去做
        src_mask = tf.broadcast_to(src_mask, [B, H, T_q, T_k])
        scaled_attention_logits = apply_mask_add(scaled_attention_logits, src_mask)
        
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        attention_weights = tf.layers.dropout(attention_weights, rate=dropout_rate,
                                training=training, name="attn_dropout")
        
        output = tf.matmul(attention_weights, V)
        return output, attention_weights
    
    with tf.variable_scope("multi_head_attention", reuse=tf.AUTO_REUSE):
        depth = queries.get_shape().as_list()[-1]
        Q = tf.layers.dense(queries, depth, use_bias=False, name="w_q")
        K = tf.layers.dense(keys,    depth, use_bias=False, name="w_k")
        V = tf.layers.dense(values,  depth, use_bias=False, name="w_v")

        Q = split_heads(Q, num_heads)
        K = split_heads(K, num_heads)
        V = split_heads(V, num_heads)

        scaled_attention, attention_weights = scaled_dot_product_attention(Q, K, V, src_mask)
        scaled_attention = tf.transpose(scaled_attention, [0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention, [tf.shape(queries)[0], -1, depth])
        output = tf.layers.dense(concat_attention, depth, name="w_o")

    return output

def multi_head_attention_causality(queries, keys, values, num_heads, dropout_rate, training=False):
    
    def apply_masks(atten_scores):
        """
        atten_scores: [B, H, T_q, T_k]
        """
        B = tf.shape(atten_scores)[0]
        T_q = tf.shape(atten_scores)[2]
        T_k = tf.shape(atten_scores)[3]

        # 1) 因果下三角掩码，先做 [T_q, T_k]
        causal_mask = tf.linalg.band_part(tf.ones([T_q, T_k]), -1, 0)

        # 2) 扩展到 [B, H, T_q, T_k] 用 [1,1,T_q,T_k] 让广播去做
        causal_mask = tf.reshape(causal_mask, [1, 1, T_q, T_k])          # 4 维
        causal_mask = tf.broadcast_to(causal_mask, [B, num_heads, T_q, T_k])

        # atten_scores  = tf.where(tf.equal(causal_mask, 0), paddings, atten_scores)
        atten_scores = apply_mask_add(atten_scores, causal_mask)

        return atten_scores
    
    def split_heads(x, num_heads):
        batch_size = tf.shape(x)[0]
        depth = x.get_shape().as_list()[-1] // num_heads
        reshaped = tf.reshape(x, [batch_size, -1, num_heads, depth])
        return tf.transpose(reshaped, [0, 2, 1, 3])
    
    training = tf.constant(training, dtype=tf.bool)
    
    def scaled_dot_product_attention(Q, K, V):
        matmul_qk = tf.matmul(Q, K, transpose_b=True)
        dk = tf.cast(tf.shape(K)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        scaled_attention_logits = apply_masks(scaled_attention_logits)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        
        attention_weights = tf.layers.dropout(attention_weights, rate=dropout_rate,
                                training=training, name="attn_dropout")
        
        output = tf.matmul(attention_weights, V)
        return output, attention_weights
    
    with tf.variable_scope("multi_head_attention_causality", reuse=tf.AUTO_REUSE):
        depth = queries.get_shape().as_list()[-1]
        Q = tf.layers.dense(queries, depth, use_bias=False, name="w_q")
        K = tf.layers.dense(keys,    depth, use_bias=False, name="w_k")
        V = tf.layers.dense(values,  depth, use_bias=False, name="w_v")

        Q = split_heads(Q, num_heads)
        K = split_heads(K, num_heads)
        V = split_heads(V, num_heads)

        scaled_attention, attention_weights = scaled_dot_product_attention(Q, K, V)
        scaled_attention = tf.transpose(scaled_attention, [0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention, [tf.shape(queries)[0], -1, depth])
        output = tf.layers.dense(concat_attention, depth, name="w_o")

    return output
    
    
def feed_forward_network(dim, hidden_dim, dropout_rate, training=False):
    def ffn(x, training=training):
        training = tf.constant(training, dtype=tf.bool)
        with tf.variable_scope(f"feed_forward_network", reuse=tf.AUTO_REUSE):
            x = tf.layers.dense(x, hidden_dim, activation=tf.nn.relu, name="w_up")
            x = tf.layers.dropout(x, rate=dropout_rate,
                                    training=training, name="ffn_dropout")
            x = tf.layers.dense(x, dim, name="w_down")
            # x = tf.nn.dropout(x, rate=dropout_rate)
            # x = tf.cond(training, lambda: tf.nn.dropout(x, rate=dropout_rate), lambda: x)
            return x
    return ffn

def scaled_attention(q,            # [B*beam, H, T_q, Dh]  (单步)
                     k,            # [B*beam, H, T_k, Dh] (已缓存)
                     v,            # [B*beam, H, T_k, Dh]
                     cur_beam,
                     causal=False,
                     src_mask=None,        # [B, 1, 1, T_k]
                     dropout_rate=0.0,
                     training=False):
    """
    单步 / 小批量通用的缩放点积注意力
    --------------------------------------------------------
        q : 当前 step 的 query (B,H,1,Dh)
        k : 历史 K cache        (B,H,Tk,Dh)
        v : 历史 V cache        (B,H,Tk,Dh)
    返回：
        context : (B,1,D_model) 已经合并 H 头
    """
    training = tf.convert_to_tensor(training, dtype=tf.bool)
    
    B = tf.shape(q)[0] // cur_beam
    H = tf.shape(q)[1]
    T_q = tf.shape(q)[2]
    T_k = tf.shape(k)[2]

    # ---------- 1. 点积并缩放 ----------
    attn_scores = tf.matmul(q, k, transpose_b=True)        # [B*beam,H,Tq,Tk]
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    attn_scores = attn_scores / tf.sqrt(dk)                # 缩放

    # ---------- 2. 掩码 ----------
    if causal:                     # 自回归用下三角
        # Tk = tf.shape(k)[-2]；这里 q 只有长度1，所以只需屏蔽「未来」
        # 亦可省略，因为 q 在末尾
        pass                       # 保留以示可扩展

    if src_mask is not None:       # Padding mask / Cross-attn mask
        
        src_mask = tf.repeat(src_mask, cur_beam, axis=0) # [B*beam, 1, 1, Tk]
        src_mask = tf.broadcast_to(src_mask, [B*cur_beam, H, T_q, T_k])
        attn_scores = apply_mask_add(attn_scores, src_mask)

    # ---------- 3. softmax & dropout ----------
    attn_weights = tf.nn.softmax(attn_scores, axis=-1)      # [B*beam,H,Tq,Tk]
    attn_weights = tf.layers.dropout(attn_weights,
                                        rate=dropout_rate,
                                        training=training)

    # ---------- 4. 加权求和 ----------
    context = tf.matmul(attn_weights, v)        # [B*beam,H,Tq,Dh]
    context = tf.transpose(context, [0, 2, 1, 3])   # → [B*beam,Tq,H,Dh]
    _, _, H, Dh = context.shape.as_list()
    context = tf.reshape(context, [tf.shape(q)[0], T_q, H*Dh]) # [B*beam,Tq,D_model]

    return context          # 返回给上层 (仍是 3D，后续可接线性层)

class EncoderLayer:
    def __init__(self, name, dim, num_heads, hidden_dim, dropout_rate, training=False):
        super(EncoderLayer, self).__init__()
        self.name = name
        self.dim = dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.mha = multi_head_attention
        self.ffn = feed_forward_network(dim, hidden_dim, dropout_rate)
        
    def forward(self, x, src_mask, training):
        with tf.variable_scope(f"{self.name}", reuse=tf.AUTO_REUSE):
            attn_output = self.mha(x, x, x, self.num_heads, src_mask, self.dropout_rate, training=training)
            out1 = layer_norm(x + attn_output, scope="atten_ln")
            ffn_output = self.ffn(out1, training=training)
            out2 = layer_norm(out1 + ffn_output, scope="ffn_ln")
        return out2

class DecoderLayer:
    def __init__(self, name, dim, num_heads, hidden_dim, dropout_rate, training=False):
        super(DecoderLayer, self).__init__()
        self.name = name
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = self.dim // self.num_heads
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.self_attention = multi_head_attention_causality
        self.cross_attention = multi_head_attention
        # self.ffn = feed_forward_network(dim, hidden_dim, dropout_rate) 
        self.ffn = moe_swiglu_feed_forward_network(
            dim=dim,
            hidden_dim=hidden_dim,      # 建议用 (2/3)*4*d_model 的数值
            num_experts=24,
            top_k=2,                    # Act = 2
            dropout_rate=dropout_rate
        )   
        
    def forward(self, x, enc_output, src_mask, training=False):
        with tf.variable_scope(f"{self.name}", reuse=tf.AUTO_REUSE):
            attn_output = self.self_attention(x, x, x, self.num_heads, self.dropout_rate, training=training)
            out1 = layer_norm(x + attn_output, scope="self_atten_ln")
            cross_attn_output = self.cross_attention(out1, enc_output, enc_output, self.num_heads, src_mask, self.dropout_rate, training=training)
            out2 = layer_norm(out1 + cross_attn_output, scope="cross_atten_ln")
            ffn_output = self.ffn(out2, training=training)
            out3 = layer_norm(out2 + ffn_output, scope="ffn_ln")
        
        return out3
    
    def step(self, x_t, cur_beam,         # [B,1,D] 当步输入
             layer_id,
             enc_output, src_mask,
             cache, training=False):
        """
        cache: dict 存各层的 {k_self, v_self, ffn_out, k_enc, v_enc}
        """
        def split_heads(x, num_heads):
            batch_size = tf.shape(x)[0]
            T_x = tf.shape(x)[1]
            depth = x.get_shape().as_list()[-1] // num_heads
            reshaped = tf.reshape(x, [batch_size, T_x, num_heads, depth])
            return tf.transpose(reshaped, [0, 2, 1, 3])
                    
        with tf.variable_scope(f"{self.name}", reuse=tf.AUTO_REUSE):
            with tf.variable_scope("multi_head_attention_causality", reuse=tf.AUTO_REUSE):
                # # ---------- ① Self-Attention ----------
                # ---------- ① 当前 token 的 q/k/v ----------
                depth = x_t.shape[-1]
                q_t = tf.layers.dense(x_t, depth, use_bias=False, name="w_q")
                k_t = tf.layers.dense(x_t, depth, use_bias=False, name="w_k")
                v_t = tf.layers.dense(x_t, depth, use_bias=False, name="w_v")

                q_t = split_heads(q_t, self.num_heads)   # [B*beam,H,1,Dh]
                k_t = split_heads(k_t, self.num_heads)
                v_t = split_heads(v_t, self.num_heads)

                # ---------- ② 取 / 初始化 Self-KV 缓存 ----------
                k_key = f"k_self_{layer_id}"
                v_key = f"v_self_{layer_id}"

                B_beam = tf.shape(q_t)[0]                # == B*beam
                B      = B_beam // cur_beam
                H      = self.num_heads
                Dh     = self.head_dim                   # dim // H

                # reshape 到 5-D 以便 concat & gather：[B,beam,H,1,Dh]
                k_t_5d = tf.reshape(k_t, [B, cur_beam, H, 1, Dh])
                v_t_5d = tf.reshape(v_t, [B, cur_beam, H, 1, Dh])

                if k_key in cache:
                    k_prev = cache[k_key]                # [B, beam, H, T-1, Dh]
                    v_prev = cache[v_key]
                    k_cat  = tf.concat([k_prev, k_t_5d], axis=3)   # T += 1
                    v_cat  = tf.concat([v_prev, v_t_5d], axis=3)
                else:
                    k_cat, v_cat = k_t_5d, v_t_5d        # 第一次，历史为空

                cache[k_key] = k_cat                    # 形状始终 [B, beam, H, T, Dh]
                cache[v_key] = v_cat

                # ---------- ③ 缩放点积注意力 ----------
                # 展平成 [B*beam, H, T, Dh] 供 matmul 使用
                k_cat_4d = tf.reshape(k_cat, [B_beam, H, -1, Dh])
                v_cat_4d = tf.reshape(v_cat, [B_beam, H, -1, Dh])

                self_out = scaled_attention(
                    q_t, k_cat_4d, v_cat_4d,
                    cur_beam,
                    dropout_rate=self.dropout_rate,
                    training=training,      # causal Mask 已在 scaled_attention 内部做
                )                           # [B*beam, 1, D]

                self_out = tf.layers.dense(self_out, depth, name="w_o")  # [B*beam,1,D]
            
            out1 = layer_norm(x_t + self_out, scope="self_atten_ln")
                        
            with tf.variable_scope("multi_head_attention", reuse=tf.AUTO_REUSE):
                # ---------- ② Cross-Attn ----------
                q_cur = tf.layers.dense(out1, self.dim, use_bias=False, name="w_q")
                q_cur = split_heads(q_cur, self.num_heads)
                
                # -- 1. 只在第一步算 enc KV，一份缓存，shape [B,1,H,L_enc,Dh] --
                if f"k_enc_{layer_id}" not in cache:
                    k_enc = split_heads(tf.layers.dense(enc_output, self.dim, use_bias=False,
                                                        name="w_k"), self.num_heads)
                    v_enc = split_heads(tf.layers.dense(enc_output, self.dim, use_bias=False,
                                                        name="w_v"), self.num_heads)
                    cache[f"k_enc_{layer_id}"] = k_enc[:, None]   # [B,1,H,L_enc,Dh]
                    cache[f"v_enc_{layer_id}"] = v_enc[:, None]
                
                k_enc = cache[f"k_enc_{layer_id}"]
                v_enc = cache[f"v_enc_{layer_id}"]
                
                B = tf.shape(k_enc)[0]
                H = tf.shape(k_enc)[2]
                L_enc = tf.shape(k_enc)[3]

                # -- 2. 广播到 beam 维；如果显存紧张可在 scaled_attention 内处理 --
                k_enc = tf.broadcast_to(k_enc, [B, cur_beam, H, L_enc, self.head_dim])
                v_enc = tf.broadcast_to(v_enc, [B, cur_beam, H, L_enc, self.head_dim])
                k_enc = tf.reshape(k_enc, [B*cur_beam, H, L_enc, self.head_dim])
                v_enc = tf.reshape(v_enc, [B*cur_beam, H, L_enc, self.head_dim])
                            
                cross_out = scaled_attention(q_cur, k_enc, v_enc,
                                            cur_beam, src_mask=src_mask,
                                            dropout_rate=self.dropout_rate,
                                            training=training)         # [B,L_dec,D]
                
                cross_out = tf.layers.dense(cross_out, self.dim, name="w_o") # [B,L_dec,D]

            out2 = layer_norm(out1 + cross_out, scope="cross_atten_ln")
        
            # ---------- ③ FFN ----------
            ffn_out = self.ffn(out2, training=False)   # 仅算当前 token
            y_t     = layer_norm(out2 + ffn_out, scope="ffn_ln")

        return y_t, cache


class EncoderModel():
    def __init__(self, num_layers, dim, num_heads, hidden_dim, dropout_rate, training=False):
        super(EncoderModel, self).__init__()
        self.num_layers = num_layers
        self.dim = dim
        self.layers = [EncoderLayer(f"encoder_layer_{i}", dim, num_heads, hidden_dim, dropout_rate, training=training) for i in range(num_layers)]
        
    def forward(self, encoder_embedding, src_mask, training):
        for i in range(self.num_layers):
            encoder_embedding = self.layers[i].forward(encoder_embedding, src_mask, training=training)
        return encoder_embedding


class DecoderModel():
    def __init__(self, num_layers, dim, num_heads, hidden_dim, dropout_rate, training=False):
        super(DecoderModel, self).__init__()
        self.num_layers = num_layers
        self.dim = dim
        self.layers = [DecoderLayer(f"decoder_layer_{i}", dim, num_heads, hidden_dim, dropout_rate, training=training) for i in range(num_layers)]
        
    def forward(self, decoder_embedding, hidden_states, src_mask, training):
        for i in range(self.num_layers):
            decoder_embedding = self.layers[i].forward(decoder_embedding, hidden_states, src_mask, training=training)
        return decoder_embedding
    
    def step(self, x_t, cur_beam, enc_out, src_mask, cache):
        for i, layer in enumerate(self.layers):
            x_t, cache = layer.step(x_t, cur_beam, i, enc_out, src_mask, cache)
        return x_t, cache
