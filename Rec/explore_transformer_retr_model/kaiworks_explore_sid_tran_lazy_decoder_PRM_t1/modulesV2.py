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
        
        from tensorflow.keras import backend as K
        src_mask = K.repeat_elements(src_mask, cur_beam, axis=0) # [B*beam, 1, 1, Tk]
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
        self.ffn = feed_forward_network(dim, hidden_dim, dropout_rate)    
        
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

class PRMModel():
    def __init__(self, dim, num_heads, dropout_rate, training=False):
        super(PRMModel, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.score_hidden_dim = max(dim // 2, 1)
        self.dropout_rate = dropout_rate
        self.cross_attention = multi_head_attention

    def build_variables(self):
        def dense_vars(scope, input_dim, output_dim, use_bias=True):
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                tf.get_variable("kernel", [input_dim, output_dim], initializer=tf.glorot_uniform_initializer())
                if use_bias:
                    tf.get_variable("bias", [output_dim], initializer=tf.zeros_initializer())

        with tf.variable_scope("prm_model", reuse=tf.AUTO_REUSE):
            with tf.variable_scope("multi_head_attention", reuse=tf.AUTO_REUSE):
                dense_vars("w_q", self.dim, self.dim, use_bias=False)
                dense_vars("w_k", self.dim, self.dim, use_bias=False)
                dense_vars("w_v", self.dim, self.dim, use_bias=False)
                dense_vars("w_o", self.dim, self.dim)

            score_hidden_dim_2 = max(self.score_hidden_dim // 4, 1)
            with tf.variable_scope("target_score_mlp", reuse=tf.AUTO_REUSE):
                dense_vars("target_score_mlp_0", self.dim, self.dim)
                dense_vars("target_score_mlp_1", self.dim, self.score_hidden_dim)
                dense_vars("target_score_mlp_2", self.score_hidden_dim, score_hidden_dim_2)
                dense_vars("target_score_mlp_final", score_hidden_dim_2, 1)

    def forward(self, target_embedding, hidden_states, src_mask, training):
        with tf.variable_scope("prm_model", reuse=tf.AUTO_REUSE):
            target_attn = self.cross_attention(
                target_embedding,
                hidden_states,
                hidden_states,
                self.num_heads,
                src_mask,
                self.dropout_rate,
                training=training
            )
            score = mlp(
                "target_score",
                target_attn,
                [self.dim, self.score_hidden_dim,max(self.score_hidden_dim // 4, 1)],
                1,
                activation=tf.nn.leaky_relu
            )
        return tf.squeeze(score, axis=[1, 2])

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


# =====================================================================
# Lazy Decoder 模块 (OneRec-V2)
# =====================================================================
# 核心差异：Cross-Attention 不使用 w_k/w_v 投影，
# 而是直接使用 Context Processor 预计算的 K/V（所有层共享）。
# 操作顺序、归一化方式与现有 DecoderLayer 保持一致，便于 A/B 对比。
# =====================================================================

class LazyDecoderLayer:
    """
    Lazy Decoder Layer — OneRec-V2 风格

    与 DecoderLayer 的差异仅在于 Cross-Attention：
    - 不做 w_k/w_v 投影（删除这两个 dense 层）
    - 直接使用预计算的 context K/V（所有层共享同一组）
    - 其余（Self-Attn、FFN、LayerNorm、操作顺序）与 DecoderLayer 完全一致
    """

    def __init__(self, name, dim, num_heads, hidden_dim, dropout_rate, training=False):
        self.name = name
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = self.dim // self.num_heads
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.ffn = feed_forward_network(dim, hidden_dim, dropout_rate)

    # ------------------------------------------------------------------
    # 训练模式前向传播（并行处理所有 token）
    # ------------------------------------------------------------------
    def forward(self, x, context_k, context_v, context_mask, training=False):
        """
        Args:
            x:            [B, T_dec, dim]      decoder 输入（含 BOS + SID tokens）
            context_k:    [B, H, ctx_len, Dh]  预计算 context key（全层共享）
            context_v:    [B, H, ctx_len, Dh]  预计算 context value
            context_mask: [B, 1, 1, ctx_len]   padding mask
        Returns:
            out3: [B, T_dec, dim]
        """
        def split_heads_train(x, num_heads):
            B = tf.shape(x)[0]
            L = tf.shape(x)[1]
            d = x.get_shape().as_list()[-1] // num_heads
            return tf.transpose(tf.reshape(x, [B, L, num_heads, d]), [0, 2, 1, 3])

        def merge_heads(x):
            # 用 self.dim 替代 H*d_h — matmul 结果的 shape[-1] 在 TF1 下可能是 None
            x = tf.transpose(x, [0, 2, 1, 3])          # [B, T_q, H, d_h]
            return tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[1], self.dim])

        with tf.variable_scope(f"{self.name}", reuse=tf.AUTO_REUSE):
            # ---- ① Self-Attention (与 DecoderLayer.forward 完全一致) ----
            with tf.variable_scope("multi_head_attention_causality", reuse=tf.AUTO_REUSE):
                depth = x.get_shape().as_list()[-1]
                Q = tf.layers.dense(x, depth, use_bias=False, name="w_q")
                K = tf.layers.dense(x, depth, use_bias=False, name="w_k")
                V = tf.layers.dense(x, depth, use_bias=False, name="w_v")

                Q = split_heads_train(Q, self.num_heads)
                K = split_heads_train(K, self.num_heads)
                V = split_heads_train(V, self.num_heads)

                # 因果下三角 mask
                T_q = tf.shape(Q)[2]
                T_k = tf.shape(K)[2]
                B_sz = tf.shape(Q)[0]
                causal_mask = tf.linalg.band_part(tf.ones([T_q, T_k]), -1, 0)
                causal_mask = tf.reshape(causal_mask, [1, 1, T_q, T_k])
                causal_mask = tf.broadcast_to(causal_mask, [B_sz, self.num_heads, T_q, T_k])

                attn_scores = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(tf.cast(self.head_dim, tf.float32))
                attn_scores = apply_mask_add(attn_scores, causal_mask)
                attn_weights = tf.nn.softmax(attn_scores, axis=-1)

                training_const = tf.constant(training, dtype=tf.bool)
                attn_weights = tf.layers.dropout(attn_weights, rate=self.dropout_rate,
                                                 training=training_const, name="attn_dropout")
                self_out = tf.matmul(attn_weights, V)
                self_out = merge_heads(self_out)
                self_out = tf.layers.dense(self_out, self.dim, name="w_o")

            out1 = layer_norm(x + self_out, scope="self_atten_ln")

            # ---- ② Cross-Attention (Lazy: 无 w_k/w_v，直接使用预计算 context_k/v) ----
            with tf.variable_scope("multi_head_attention", reuse=tf.AUTO_REUSE):
                q_cur = tf.layers.dense(out1, self.dim, use_bias=False, name="w_q")
                q_cur = split_heads_train(q_cur, self.num_heads)  # [B, H, T_dec, Dh]

                # context_k/v 已经是 [B, H, ctx_len, Dh]，无需再做 dense 投影
                # 直接使用外部预计算好的 K/V

                # padding mask
                B_sz = tf.shape(q_cur)[0]
                T_dec = tf.shape(q_cur)[2]
                ctx_len = tf.shape(context_k)[2]
                context_mask_bc = tf.broadcast_to(context_mask, [B_sz, self.num_heads, T_dec, ctx_len])

                cross_scores = tf.matmul(q_cur, context_k, transpose_b=True) / tf.sqrt(tf.cast(self.head_dim, tf.float32))
                cross_scores = apply_mask_add(cross_scores, context_mask_bc)
                cross_weights = tf.nn.softmax(cross_scores, axis=-1)
                cross_weights = tf.layers.dropout(cross_weights, rate=self.dropout_rate,
                                                  training=training_const, name="cross_attn_dropout")
                cross_out = tf.matmul(cross_weights, context_v)  # [B, H, T_dec, Dh]
                cross_out = merge_heads(cross_out)
                cross_out = tf.layers.dense(cross_out, self.dim, name="w_o")  # 仅有 w_o，无 w_k/w_v

            out2 = layer_norm(out1 + cross_out, scope="cross_atten_ln")

            # ---- ③ FFN (与 DecoderLayer 完全一致) ----
            ffn_output = self.ffn(out2, training=training)
            out3 = layer_norm(out2 + ffn_output, scope="ffn_ln")

        return out3

    # ------------------------------------------------------------------
    # 推理模式逐步解码
    # ------------------------------------------------------------------
    def step(self, x_t, cur_beam, layer_id,
             context_k, context_v, context_mask,
             cache, training=False):
        """
        Args:
            x_t:          [B*beam, 1, dim]     当前步输入
            cur_beam:      int                  当前 beam 数
            layer_id:      int                  层 ID
            context_k:    [B, 1, H, ctx_len, Dh]  预计算 K（全局共享，含 beam 占位 dim）
            context_v:    [B, 1, H, ctx_len, Dh]  预计算 V
            context_mask: [B, 1, 1, ctx_len]     padding mask
            cache:        dict                  self-attention KV 缓存
        Returns:
            (y_t, cache)
        """
        def split_heads(x, num_heads):
            batch_size = tf.shape(x)[0]
            T_x = tf.shape(x)[1]
            depth = x.get_shape().as_list()[-1] // num_heads
            return tf.transpose(tf.reshape(x, [batch_size, T_x, num_heads, depth]), [0, 2, 1, 3])

        with tf.variable_scope(f"{self.name}", reuse=tf.AUTO_REUSE):
            # ---- ① Self-Attention (与 DecoderLayer.step 完全一致) ----
            with tf.variable_scope("multi_head_attention_causality", reuse=tf.AUTO_REUSE):
                depth = x_t.shape[-1]
                q_t = tf.layers.dense(x_t, depth, use_bias=False, name="w_q")
                k_t = tf.layers.dense(x_t, depth, use_bias=False, name="w_k")
                v_t = tf.layers.dense(x_t, depth, use_bias=False, name="w_v")

                q_t = split_heads(q_t, self.num_heads)
                k_t = split_heads(k_t, self.num_heads)
                v_t = split_heads(v_t, self.num_heads)

                k_key = f"k_self_{layer_id}"
                v_key = f"v_self_{layer_id}"

                B_beam = tf.shape(q_t)[0]
                B = B_beam // cur_beam
                H = self.num_heads
                Dh = self.head_dim

                k_t_5d = tf.reshape(k_t, [B, cur_beam, H, 1, Dh])
                v_t_5d = tf.reshape(v_t, [B, cur_beam, H, 1, Dh])

                if k_key in cache:
                    k_prev = cache[k_key]
                    v_prev = cache[v_key]
                    k_cat = tf.concat([k_prev, k_t_5d], axis=3)
                    v_cat = tf.concat([v_prev, v_t_5d], axis=3)
                else:
                    k_cat, v_cat = k_t_5d, v_t_5d

                cache[k_key] = k_cat
                cache[v_key] = v_cat

                k_cat_4d = tf.reshape(k_cat, [B_beam, H, -1, Dh])
                v_cat_4d = tf.reshape(v_cat, [B_beam, H, -1, Dh])

                self_out = scaled_attention(
                    q_t, k_cat_4d, v_cat_4d,
                    cur_beam,
                    dropout_rate=self.dropout_rate,
                    training=training,
                )
                self_out = tf.layers.dense(self_out, depth, name="w_o")

            out1 = layer_norm(x_t + self_out, scope="self_atten_ln")

            # ---- ② Cross-Attention (Lazy: 无 w_k/w_v，使用预计算 context_k/v) ----
            with tf.variable_scope("multi_head_attention", reuse=tf.AUTO_REUSE):
                q_cur = tf.layers.dense(out1, self.dim, use_bias=False, name="w_q")
                q_cur = split_heads(q_cur, self.num_heads)  # [B*beam, H, 1, Dh]

                # context_k/v: [B, 1, H, ctx_len, Dh] → broadcast 到 beam → [B*beam, H, ctx_len, Dh]
                B_val = tf.shape(context_k)[0]
                H_val = tf.shape(context_k)[2]
                L_enc = tf.shape(context_k)[3]

                k_ctx = tf.broadcast_to(context_k, [B_val, cur_beam, H_val, L_enc, self.head_dim])
                v_ctx = tf.broadcast_to(context_v, [B_val, cur_beam, H_val, L_enc, self.head_dim])
                k_ctx = tf.reshape(k_ctx, [B_val * cur_beam, H_val, L_enc, self.head_dim])
                v_ctx = tf.reshape(v_ctx, [B_val * cur_beam, H_val, L_enc, self.head_dim])

                cross_out = scaled_attention(q_cur, k_ctx, v_ctx,
                                            cur_beam, src_mask=context_mask,
                                            dropout_rate=self.dropout_rate,
                                            training=training)

                cross_out = tf.layers.dense(cross_out, self.dim, name="w_o")  # 仅有 w_o

            out2 = layer_norm(out1 + cross_out, scope="cross_atten_ln")

            # ---- ③ FFN (与 DecoderLayer 完全一致) ----
            ffn_out = self.ffn(out2, training=False)
            y_t = layer_norm(out2 + ffn_out, scope="ffn_ln")

        return y_t, cache


class LazyDecoderModel():
    """
    Lazy Decoder Model — 多层 LazyDecoderLayer 堆叠

    与 DecoderModel 的差异：
    - forward/step 接收 context_k/context_v/context_mask 而非 enc_output/src_mask
    - 内部使用 LazyDecoderLayer 而非 DecoderLayer
    """

    def __init__(self, num_layers, dim, num_heads, hidden_dim, dropout_rate, training=False):
        self.num_layers = num_layers
        self.dim = dim
        self.layers = [LazyDecoderLayer(f"decoder_layer_{i}", dim, num_heads, hidden_dim, dropout_rate, training=training)
                       for i in range(num_layers)]

    def forward(self, decoder_embedding, context_k, context_v, context_mask, training):
        """
        训练模式前向传播

        Args:
            decoder_embedding: [B, T_dec, dim]
            context_k:         [B, H, ctx_len, Dh]  预计算 key
            context_v:         [B, H, ctx_len, Dh]  预计算 value
            context_mask:      [B, 1, 1, ctx_len]   padding mask
        Returns:
            output: [B, T_dec, dim]
        """
        for i in range(self.num_layers):
            decoder_embedding = self.layers[i].forward(
                decoder_embedding, context_k, context_v, context_mask, training=training)
        return decoder_embedding

    def step(self, x_t, cur_beam, context_k, context_v, context_mask, cache):
        """
        推理模式逐步解码

        Args:
            x_t:          [B*beam, 1, dim]
            cur_beam:      int
            context_k:    [B, 1, H, ctx_len, Dh]  全局共享（含 beam 占位 dim）
            context_v:    [B, 1, H, ctx_len, Dh]
            context_mask: [B, 1, 1, ctx_len]
            cache:        dict
        Returns:
            (x_t, cache)
        """
        for i, layer in enumerate(self.layers):
            x_t, cache = layer.step(x_t, cur_beam, i,
                                     context_k, context_v, context_mask,
                                     cache)
        return x_t, cache
