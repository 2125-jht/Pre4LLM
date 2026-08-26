import tensorflow as tf
import numpy as np
from modules_ import *

# def get_position_encoding(seq_len, dim, dtype=tf.float32):
#     """
#     Sin-Cos positional encoding.  返回形状 [1, seq_len, dim]，
#     方便直接与 batch 张量相加（广播）。
#     """
#     # [seq_len, 1]
#     position = tf.cast(tf.range(seq_len), dtype)[:, tf.newaxis]         

#     # [dim/2]：偶数维的角频率
#     div_term = tf.exp(
#         tf.cast(tf.range(0, dim, 2), dtype) *
#         -(tf.math.log(tf.constant(10000.0, dtype=dtype)) / tf.cast(dim, dtype))
#     )                                            

#     # (seq_len, dim/2)
#     angles = position * div_term                 

#     # 交替填充 sin / cos
#     sin_part = tf.sin(angles)
#     cos_part = tf.cos(angles)

#     # interleave: [seq_len, dim]
#     pos_encoding = tf.reshape(
#         tf.stack([sin_part, cos_part], axis=-1),  # (seq_len, dim/2, 2)
#         [seq_len, dim]
#     )

#     # 添加 batch 维，便于后续广播到 [batch, seq_len, dim]
#     return pos_encoding[tf.newaxis, ...]          # [1, seq_len, dim]

# def get_position_encoding(seq_len, dim, dtype=tf.float32):
#     """
#     Learnable (BERT-style) positional embedding.
#     与原函数保持 **完全相同** 的调用方式 & 返回形状：
#         输入:  seq_len, dim
#         输出:  [1, seq_len, dim]   —— 方便直接与 batch 张量相加（广播）

#     Tips
#     ----
#     1. 需要在 **图构建阶段** 只调用一次；否则请确保 variable_scope 重用。
#     2. 若推理长度 < seq_len，可切片：pos_emb[:, :L, :].
#     3. 如果后续要迁移到不同长度，再 `tf.get_variable` 一个新的表或插值即可。
#     """
#     assert dim % 2 == 0, "dim 必须为偶数（保持与原实现一致的假设）"

#     with tf.variable_scope("learned_positional_emb", reuse=tf.AUTO_REUSE):
#         pos_table = tf.get_variable(
#             "pos_table",                       # 名称
#             shape=[seq_len, dim],              # [seq_len, dim]
#             dtype=dtype,
#             initializer=tf.random_uniform_initializer(-0.02, 0.02)
#         )
#     # 添加 batch 维，-> [1, seq_len, dim]
#     return pos_table[tf.newaxis, ...]

# def get_encoder_position_encoding(seq_len, dim, max_len=512, dtype=tf.float32):
#     """
#     Learnable positional embedding 兼容动态 seq_len.
#     返回 [1, seq_len, dim]，可广播到 [batch, seq_len, dim].
#     """
#     with tf.variable_scope("encoder_learned_positional_emb", reuse=tf.AUTO_REUSE):
#         pos_table = tf.get_variable(
#             "pos_table",
#             shape=[max_len, dim],
#             dtype=dtype,
#             initializer=tf.random_uniform_initializer(-0.02, 0.02)
#         )
        
#     # 如果 seq_len 是 Tensor，这里仍支持切片（TensorFlow 会在图里生成 slice op）
#     pos_emb = pos_table[:seq_len,:]
#     return tf.expand_dims(pos_emb, axis=0)

# def get_encoder_position_encoding(seq_len, dim, max_len=512, dtype=tf.float32):
#     return tf.zeros([1,1,1])

# def get_decoder_position_encoding(seq_len, dim, max_len=16, dtype=tf.float32):
#     """
#     Learnable positional embedding 兼容动态 seq_len.
#     返回 [1, seq_len, dim]，可广播到 [batch, seq_len, dim].
#     """
#     with tf.variable_scope("decoder_learned_positional_emb", reuse=tf.AUTO_REUSE):
#         pos_table = tf.get_variable(
#             "pos_table",
#             shape=[max_len, dim],
#             dtype=dtype,
#             initializer=tf.random_uniform_initializer(-0.02, 0.02)
#         )

#     # 如果 seq_len 是 Tensor，这里仍支持切片（TensorFlow 会在图里生成 slice op）
#     pos_emb = pos_table[:seq_len,:]
#     return tf.expand_dims(pos_emb, axis=0)

# def get_decoder_position_encoding(seq_len, dim, max_len=16, dtype=tf.float32):
#     return tf.zeros([1,1,1])



def apply_masks(outputs, mask, causality, num_heads):
    """
    outputs: (h*N, T_q, T_k)
    mask   : (N, T_k) or None   1=keep, 0=pad
    """
    attn_shape   = tf.shape(outputs)
    bs_heads     = attn_shape[0]          # h * N
    T_q, T_k     = attn_shape[1], attn_shape[2]

    # 1) 因果掩码  (T_q, T_k)
    if causality:
        causal_mask = tf.linalg.band_part(tf.ones([T_q, T_k]), -1, 0)   # 下三角
        causal_mask = tf.tile(causal_mask[None, ...], [bs_heads, 1, 1]) # (h*N, T_q, T_k)
    else:
        causal_mask = None

    # 2) padding 掩码 (h*N, 1, T_k) 先扩到 key 维，后面再 broadcast 到 query 维
    if mask is not None:
        # mask: [N, T_k]  →  [N, 1, T_k]
        mask_3d = tf.expand_dims(mask, 1)
        # 重复到多个 head：先 reshape，再 tile
        mask_3d = tf.reshape(tf.tile(mask_3d, [1, num_heads, 1]), 
                             [bs_heads, 1, T_k])           # (h*N, 1, T_k)
        # broadcast 到 T_q 维自动完成 (利用 tf.where 的广播)
    else:
        mask_3d = None

    # 3) 合并掩码：逻辑 AND
    if causal_mask is not None and mask_3d is not None:
        final_mask = causal_mask * mask_3d                  # 1=keep,0=pad
    elif causal_mask is not None:
        final_mask = causal_mask
    elif mask_3d is not None:
        final_mask = tf.tile(mask_3d, [1, T_q, 1])          # 因果关掉时补全维度
    else:
        final_mask = None
    # 4) 一次性贴 -∞
    if final_mask is not None:
        paddings = tf.ones_like(outputs) * (-2**32 + 1)
        outputs  = tf.where(tf.equal(final_mask, 0), paddings, outputs)

    return outputs


def multi_head_attention(queries, keys, values, is_training=False, causality=False, mask=None, 
                         num_heads=8, dropout_rate=0.1, dim=16, return_weights=False):
    with tf.variable_scope("multi_head_attention", reuse=tf.AUTO_REUSE):
        # Set the fall back option for num_units
        input_shape = tf.shape(queries)
        is_4d = len(queries.get_shape().as_list()) == 4
        batch_size, beam_size = None, None
        if is_4d:
            # [batch_size, beam_size, seq_len, dim]
            batch_size = input_shape[0]
            beam_size = input_shape[1]
            query_len = tf.shape(queries)[2]
            kv_len = tf.shape(keys)[2]
            # dim = input_shape[3]
            queries = tf.reshape(queries, [batch_size*beam_size, query_len, dim])
            keys = tf.reshape(keys, [batch_size*beam_size, kv_len, dim])
            values = tf.reshape(values, [batch_size*beam_size, kv_len, dim])
        else:
            batch_size = input_shape[0]
            # dim = input_shape[2]
        # Linear projections
        Q = tf.layers.dense(queries, dim, activation=None, name="dense_q") # (N, T_k, C)
        K = tf.layers.dense(keys,    dim, activation=None, name="dense_k")
        V = tf.layers.dense(values,  dim, activation=None, name="dense_v")
        
        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, C/h) 
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 

        # Multiplication
        atten_scores = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)
        atten_scores = atten_scores / (K_.get_shape().as_list()[-1] ** 0.5)
        
        atten_scores = apply_masks(atten_scores, mask, causality, num_heads)

        # Activation
        atten_weights = tf.nn.softmax(atten_scores) # (h*N, T_q, T_k)
        atten_weights = tf.layers.dropout(atten_weights, rate=dropout_rate, training=tf.convert_to_tensor(is_training), name="drop_atten")
        
        # # 在 multi_head_attention 最后加
        # attn_var = tf.reduce_mean(tf.math.reduce_std(atten_weights, axis=-1))
        # tf.print("attn var", attn_var)
        
        # Weighted sum
        outputs = tf.matmul(atten_weights, V_) # ( h*N, T_q, C/h)
        
        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2) # (N, T_q, C)

        if is_4d:
            outputs = tf.reshape(outputs, [batch_size, beam_size, query_len, dim])
            
        # 输出投影层
        outputs = tf.layers.dense(outputs, dim, activation=None, name="dense_out")
        
        if return_weights:
            return outputs, atten_weights     # 多返回一个
        
        return outputs
    

def feed_forward(x, dim=16, is_training=False, dropout_rate=0.1):
    with tf.variable_scope(f"feed_forward_network", reuse=tf.AUTO_REUSE):
        out = tf.layers.dense(x, 2*dim, activation=tf.nn.relu, name="fc1")
        out = tf.layers.dropout(out, rate=dropout_rate, training=tf.convert_to_tensor(is_training), name="drop_ffn")
        out = tf.layers.dense(out, dim, activation=None, name="fc2")
        return out
    
def layer_norm(x, epsilon=1e-6):
    with tf.variable_scope("layer_norm", reuse=tf.AUTO_REUSE):
        gamma = tf.get_variable("gamma", [x.get_shape()[-1]], initializer=tf.ones_initializer())
        beta = tf.get_variable("beta", [x.get_shape()[-1]], initializer=tf.zeros_initializer())

        mean, variance = tf.nn.moments(x, axes=[-1], keep_dims=True)
        normalized = (x - mean) / tf.sqrt(variance + epsilon)
        output = gamma * normalized + beta
    return output

def transformer_encoder_layer(seq_input_embeddings, num_layer, dropout_rate=0.1, mask=None, is_training=True, dim=16):
    """
    Transformer编码器层
    
    Args:
        seq_input_embeddings: 输入序列嵌入，形状[batch_size, seq_len, dim]
        num_layer: 编码器层数
        dropout_rate: dropout比率
        mask: 注意力掩码
        is_training: 是否为训练模式
        dim: 隐藏维度
        
    Returns:
        编码器输出，形状[batch_size, seq_len, dim]
    """
    batch_size = tf.shape(seq_input_embeddings)[0]
    x = seq_input_embeddings
    
    for i in range(num_layer):
        with tf.variable_scope(f"encoder_layer_{i}", reuse=tf.AUTO_REUSE):
            
            # === Multi-Head Self-Attention ===
            with tf.variable_scope("self_attention", reuse=tf.AUTO_REUSE):
                attn_output, attn_weights = multi_head_attention(
                    queries=x, 
                    keys=x, 
                    values=x,
                    is_training=is_training, 
                    causality=False, 
                    mask=mask, 
                    dim=dim, 
                    dropout_rate=dropout_rate,
                    return_weights=True
                )
                # 残差连接 + 层归一化
                x = layer_norm(attn_output + x)
                print_tensor(f"encoder/self_atten_sim_{i}", 
                           calc_sim_cos(tf.reshape(x, [batch_size, -1])))
                
                # === 写入 TensorBoard ===
                attn_var = tf.reduce_mean(tf.math.reduce_std(attn_weights, axis=-1))  # scalar
                tf.summary.scalar(f"encoder/attn_var_{i}", attn_var)
                tf.summary.histogram(f"encoder/attn_weights_{i}", attn_weights)
                
                # attn_w: (h*N, T_q, T_k)
                head0_map = attn_weights[0]                 # (T_q, T_k)
                img      = tf.expand_dims(head0_map, -1)   # (T_q, T_k, 1)  add channel
                img      = tf.expand_dims(img, 0)          # (1,  T_q, T_k, 1) add batch
                                
                tf.summary.image(
                    f"encoder/atten_map_{i}",
                    img,
                    max_outputs=1
                )
                
            # === Feed Forward Network ===
            with tf.variable_scope("feed_forward", reuse=tf.AUTO_REUSE):
                ff_output = feed_forward(
                    x, 
                    dim=dim, 
                    dropout_rate=dropout_rate, 
                    is_training=is_training
                )
                # 残差连接 + 层归一化
                x = layer_norm(ff_output + x)
                print_tensor(f"encoder/ffn_sim_{i}", 
                           calc_sim_cos(tf.reshape(x, [batch_size, -1])))
                
    return x


def transformer_decoder_layer(encoder_output, decoder_input, num_layer, dropout_rate=0.1, mask=None, is_training=True, dim=16):
    """
    Transformer解码器层
    
    Args:
        encoder_output: 编码器输出，形状[batch_size, encoder_seq_len, dim]
        decoder_input: 解码器输入，形状[batch_size, decoder_seq_len, dim]
        num_layer: 解码器层数
        dropout_rate: dropout比率
        mask: 注意力掩码
        is_training: 是否为训练模式
        dim: 隐藏维度
        
    Returns:
        解码器输出，形状[batch_size, decoder_seq_len, dim]
    """
    x = decoder_input
    batch_size = tf.shape(decoder_input)[0]
    for i in range(num_layer):
        with tf.variable_scope(f"decoder_layer_{i}", reuse=tf.AUTO_REUSE):
            
            # === Masked Multi-Head Self-Attention ===
            with tf.variable_scope("masked_self_attention", reuse=tf.AUTO_REUSE):
                self_attn_output, self_attn_weights = multi_head_attention(
                    queries=x, 
                    keys=x, 
                    values=x,
                    is_training=is_training, 
                    causality=True,  # 使用因果掩码
                    mask=mask, 
                    dim=dim, 
                    dropout_rate=dropout_rate,
                    return_weights=True
                )
                # 残差连接 + 层归一化
                x = layer_norm(self_attn_output + x)
                print_tensor(f"decoder/masked_self_atten_sim_{i}", 
                           calc_sim_cos(tf.reshape(x, [batch_size, -1])))
                
                # === 写入 TensorBoard ===
                attn_var = tf.reduce_mean(tf.math.reduce_std(self_attn_weights, axis=-1))  # scalar
                tf.summary.scalar(f"decoder/self_attn_var_{i}", attn_var)
                tf.summary.histogram(f"decoder/self_attn_weights_{i}", self_attn_weights)
                
                # attn_w: (h*N, T_q, T_k)
                head0_map = self_attn_weights[0]                 # (T_q, T_k)
                img      = tf.expand_dims(head0_map, -1)   # (T_q, T_k, 1)  add channel
                img      = tf.expand_dims(img, 0)          # (1,  T_q, T_k, 1) add batch
                                
                tf.summary.image(
                    f"decoder/self_atten_map_{i}",
                    img,
                    max_outputs=1
                )
                
            # === Multi-Head Cross-Attention ===
            with tf.variable_scope("cross_attention", reuse=tf.AUTO_REUSE):
                cross_attn_output, cross_attn_weights = multi_head_attention(
                    queries=x, 
                    keys=encoder_output, 
                    values=encoder_output,
                    is_training=is_training, 
                    causality=False, 
                    mask=mask, 
                    dim=dim, 
                    dropout_rate=dropout_rate,
                    return_weights=True
                )
                # 残差连接 + 层归一化
                x = layer_norm(cross_attn_output + x)
                print_tensor(f"decoder/cross_atten_sim_{i}", 
                           calc_sim_cos(tf.reshape(x, [batch_size, -1])))
                
                
                # === 写入 TensorBoard ===
                attn_var = tf.reduce_mean(tf.math.reduce_std(cross_attn_weights, axis=-1))  # scalar
                tf.summary.scalar(f"decoder/cross_attn_var_{i}", attn_var)
                tf.summary.histogram(f"decoder/cross_attn_weights_{i}", cross_attn_weights)
                
                # attn_w: (h*N, T_q, T_k)
                head0_map = cross_attn_weights[0]                 # (T_q, T_k)
                img      = tf.expand_dims(head0_map, -1)   # (T_q, T_k, 1)  add channel
                img      = tf.expand_dims(img, 0)          # (1,  T_q, T_k, 1) add batch
                                
                tf.summary.image(
                    f"decoder/cross_atten_map_{i}",
                    img,
                    max_outputs=1
                )
                
            # === Feed Forward Network ===
            with tf.variable_scope("feed_forward", reuse=tf.AUTO_REUSE):
                ff_output = feed_forward(
                    x, 
                    dim=dim, 
                    dropout_rate=dropout_rate, 
                    is_training=is_training
                )
                # 残差连接 + 层归一化
                x = layer_norm(ff_output + x)
                print_tensor(f"decoder/ffn_sim_{i}", 
                           calc_sim_cos(tf.reshape(x, [batch_size, -1])))
                
    return x






### HSTU ###
# ===== 公用工具 ===== #
def _relative_position_bucket(relative_positions,
                              num_buckets=32,
                              max_distance=128):
    """Google T5风格的 bucket，相对距离 → [0, num_buckets)"""
    sign = tf.cast(tf.less(relative_positions, 0), tf.int32)
    n = tf.abs(relative_positions)
    # 小距离：线性，大距离：对数
    max_exact = num_buckets // 2
    is_small = tf.less(n, max_exact)
    val_if_small = n
    val_if_large = max_exact + tf.cast(
        tf.log(tf.cast(n, tf.float32) / max_exact) /
        np.log(max_distance / max_exact) *
        (num_buckets - max_exact), tf.int32)
    buckets = tf.where(is_small, val_if_small, val_if_large)
    return buckets * 2 + sign    # 奇偶区分方向

def _build_relative_bias(L_q, L_k, num_heads,
                         name, num_buckets=32, max_distance=128):

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        rel_pos = tf.expand_dims(tf.range(L_q), 1) - tf.expand_dims(tf.range(L_k), 0)  # [Lq, Lk]
        rp_bucket = _relative_position_bucket(rel_pos, num_buckets, max_distance)
        table = tf.get_variable('rel_bias_table',
                                [num_heads, num_buckets*2],
                                initializer=tf.random_normal_initializer(stddev=0.02))
        values = tf.gather(table, rp_bucket, axis=1)      # [h, Lq, Lk]
        values = tf.expand_dims(values, 0)                # [1, h, Lq, Lk]
    return values

def _silu(x):
    return x * tf.sigmoid(x)

# ===== HSTU layer =====
def hstu_layer(queries, keys, values,
                causality=False,
                num_heads=8,
                dim=16,
                dim_qk=16,
                dim_v=16,
                dropout_rate=0.1,
                is_training=False,
                scope="hstu_layer"):

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        
        input_shape = tf.shape(queries)
        is_4d = len(queries.get_shape().as_list()) == 4
        beam_size = 1
        if is_4d:
            batch_size = input_shape[0]
            beam_size = input_shape[1]
            query_len = tf.shape(queries)[2]
            kv_len = tf.shape(keys)[2]
            queries = tf.reshape(queries, [batch_size*beam_size, query_len, dim_qk])
            keys = tf.reshape(keys, [batch_size*beam_size, kv_len, dim_qk])
            values = tf.reshape(values, [batch_size*beam_size, kv_len, dim_v])
        else:
            batch_size = input_shape[0]
            query_len = tf.shape(queries)[1]
            kv_len = tf.shape(keys)[1]
        new_batch_size = beam_size * batch_size
        
        # === 线性映射 === #
        def _linear_qk(x):
            return tf.layers.dense(x, dim_qk, activation=None)
        
        def _linear_v(x):
            return tf.layers.dense(x, dim_v, activation=None)
        
        Q = _silu(_linear_qk(queries))                       # [B, L, C]
        K = _silu(_linear_qk(keys))
        V = _silu(_linear_v(values))
        U = _silu(_linear_v(queries))
        
        # === & 拆头 === #
        def _split(x, L, dim):
            assert dim % num_heads == 0
            x = tf.reshape(x, [new_batch_size, L, num_heads, dim//num_heads])
            return tf.transpose(x, [0, 2, 1, 3])                # [B, h, L, d]
        
        Qh = _split(Q, query_len, dim_qk)
        Kh = _split(K, kv_len, dim_qk)
        Vh = _split(V, kv_len, dim_v)
        Uh = _split(U, query_len, dim_v)

        # === Dot === #
        logits = tf.matmul(Qh, Kh, transpose_b=True)            # [B,h,L_q,L_k]
        
        # === 多维相对位置偏置 (pos & time，可按需增添 geo) === #
        if query_len == kv_len:          # self-attn
            rel_bias = _build_relative_bias(query_len, query_len, num_heads, "rpbias")
        else:                            # cross-attn
            rel_bias = _build_relative_bias(query_len, kv_len, num_heads, "rpbias")

        logits += rel_bias

        # 生成 2-D mask
        mask = tf.ones([query_len, kv_len], dtype=tf.float32)
        if causality:
            mask *= tf.linalg.band_part(tf.ones_like(mask), -1, 0)   # 下三角
        # 把 mask 升维 & 广播到 logits 形状
        mask = tf.expand_dims(mask, 0)           # [1, Lq, Lk]
        mask = tf.expand_dims(mask, 1)           # [1, 1, Lq, Lk]
        mask = tf.broadcast_to(mask, tf.shape(logits))

        # 用 -1e9 屏蔽
        logits = tf.where(tf.equal(mask, 1.0), logits,
                        tf.fill(tf.shape(logits), -1e9))
        
        # 计算attn权重
        attn = _silu(logits)
        attn = tf.layers.dropout(attn, rate=dropout_rate,
                                 training=tf.convert_to_tensor(is_training))

        # atten加权结果
        out = tf.matmul(attn, Vh)
        
        # 应用门控逐点变换
        out = layer_norm(out) * Uh
        
        # === 还原形状 === #
        out = tf.transpose(out, [0, 2, 1, 3])           # [B,L,h,d]
        out = tf.reshape(out, [new_batch_size, query_len, dim])       # [B,L,C]
        if is_4d:
            out = tf.reshape(out, [batch_size, beam_size, query_len, dim])
            
        # 最后的线性映射
        out = tf.layers.dense(out, dim, activation=None)
        
    return out

# ===== HSTU Encoder ===== #
def hstu_encoder_layer(seq_input_embeddings,
                       num_layer,
                       dropout_rate=0.1,
                       mask=None,           # 仍支持外部padding mask
                       is_training=True,
                       dim=16):
    """
    相同接口：输入/输出形状不变  [B, L, C]
    """
    x = seq_input_embeddings

    for i in range(num_layer):
        with tf.variable_scope(f"hstu_block_{i}", reuse=tf.AUTO_REUSE):

            # hstu #
            x_res = x
            x_hstu = hstu_layer(
                x, x, x,
                causality=False,
                num_heads=8, dim=dim, dim_qk=dim, dim_v=dim,
                dropout_rate=dropout_rate,
                is_training=is_training)
            
            x = layer_norm(x_hstu + x_res)

    return x

# ===== HSTU Decoder ===== #
def hstu_decoder_layer(encoder_output,
                       decoder_input,
                       num_layer,
                       dropout_rate=0.1,
                       mask=None,
                       is_training=True,
                       dim=16):
    """
    同样保持输入/输出 shape: decoder_input [B, L, C] → same
    """
    x   = decoder_input
    enc = encoder_output

    for i in range(num_layer):
        with tf.variable_scope(f"hstu_dec_block_{i}", reuse=tf.AUTO_REUSE):
            # ---- ① Self hstu (带因果) ---- #
            x_res = x
            x_self = hstu_layer(
                x, x, x,
                causality=True,              # 自回归Mask
                num_heads=8, dim=dim, dim_qk=dim, dim_v=dim,
                dropout_rate=dropout_rate,
                is_training=is_training,
                scope="self")
            
            x = layer_norm(x_self + x_res)

            # ---- ③ Cross-Attention (decoder→encoder) ---- #
            c_res = x
            cross = hstu_layer(
                x, enc, enc,
                causality=False,
                num_heads=8, dim=dim, dim_qk=dim, dim_v=dim,
                dropout_rate=dropout_rate,
                is_training=is_training,
                scope="cross")
            
            x = layer_norm(cross + c_res)

    return x

if __name__ == "__main__":
    print("开始测试 HSTU Encoder 和 Decoder...")
    
    # 设置测试参数
    batch_size = 8
    seq_len = 16
    hidden_dim = 16
    num_layers = 2
    
    # 创建测试数据
    with tf.Session() as sess:
        # 1. 创建输入数据
        encoder_input = tf.random.normal([batch_size, 20, 2*seq_len, hidden_dim], dtype=tf.float32)
        decoder_input = tf.random.normal([batch_size, 20, seq_len, hidden_dim], dtype=tf.float32)
        
        print(f"输入形状 - Encoder: {encoder_input.shape}, Decoder: {decoder_input.shape}")

        print("\n=== 测试 HSTU Encoder ===")
        
        encoder_output = hstu_encoder_layer(
            seq_input_embeddings=encoder_input,
            num_layer=num_layers,
            dropout_rate=0.1,
            mask=None,
            is_training=True,
            dim=hidden_dim
        )
        print(f"✓ Encoder 输出形状: {encoder_output.shape}")
        
        # 初始化变量
        sess.run(tf.global_variables_initializer())
        
        # 运行编码器
        enc_result = sess.run(encoder_output)
        print(f"✓ Encoder 执行成功，输出形状: {enc_result.shape}")
        print(f"  输出数值范围: [{enc_result.min():.4f}, {enc_result.max():.4f}]")
        print(f"  输出均值: {enc_result.mean():.4f}")
            
        
        # 3. 测试 HSTU Decoder
        print("\n=== 测试 HSTU Decoder ===")
        try:
            decoder_output = hstu_decoder_layer(
                encoder_output=encoder_output,
                decoder_input=decoder_input,
                num_layer=num_layers,
                dropout_rate=0.1,
                mask=None,
                is_training=True,
                dim=hidden_dim
            )
            print(f"✓ Decoder 输出形状: {decoder_output.shape}")
            
            # 重新初始化变量（因为decoder可能有新变量）
            sess.run(tf.global_variables_initializer())
            
            # 运行解码器
            dec_result = sess.run(decoder_output)
            print(f"✓ Decoder 执行成功，输出形状: {dec_result.shape}")
            print(f"  输出数值范围: [{dec_result.min():.4f}, {dec_result.max():.4f}]")
            print(f"  输出均值: {dec_result.mean():.4f}")
            
        except Exception as e:
            print(f"✗ Decoder 测试失败: {e}")
        
        # 5. 梯度测试
        print("\n=== 梯度测试 ===")
        try:
            # 创建一个简单的损失函数
            target = tf.random.normal([batch_size, 20, seq_len, hidden_dim])
            loss = tf.reduce_mean(tf.square(decoder_output - target))
            
            # 计算梯度
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            train_op = optimizer.minimize(loss)
            
            sess.run(tf.global_variables_initializer())
            
            # 执行一步训练
            _, loss_val = sess.run([train_op, loss])
            print(f"✓ 梯度计算成功，初始损失: {loss_val:.4f}")
            
            # 再执行一步看损失是否变化
            _, loss_val2 = sess.run([train_op, loss])
            print(f"✓ 第二步损失: {loss_val2:.4f}")
            print(f"  损失变化: {loss_val - loss_val2:.6f}")
            
        except Exception as e:
            print(f"✗ 梯度测试失败: {e}")
    
    print("\n=== 测试完成 ===")
    print("所有测试已完成，请检查上述输出结果。")
