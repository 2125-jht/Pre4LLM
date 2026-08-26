# OneRec-V2 Lazy Decoder 分步实施指南

> 本文档严格遵循「每步只改一处，可独立验证效果」的原则。
> 所有原有模块保留，新模块并列添加，通过配置切换。

---

## 现有代码架构概要

```
当前架构（无真正的 Transformer Encoder）:
  encoder_input = concat(user_static_emb, user_click_emb)   # [B, 201, dim]
  encoder_output = layer_norm(encoder_input)                # ← 这就是"encoder"
  DecoderModel.forward(decoder_input, encoder_output, src_mask)
    └─ DecoderLayer.forward:
       Self-Attn → LayerNorm → Cross-Attn(w_q,w_k,w_v,w_o) → LayerNorm → FFN → LayerNorm
    └─ DecoderLayer.step:
       Self-Attn(cache) → LayerNorm → Cross-Attn(w_q, cached_k_enc, cached_v_enc, w_o) → LayerNorm → FFN → LayerNorm

核心改造点：
  Cross-Attn 的 w_k/w_v 投影 → 去掉，改为直接使用 Context 预计算的 KV
```

---

## Step 1：LazyCrossAttention + LazyDecoderLayer + LazyDecoderModel + LazyMultiInterestModel

### 目标

创建 Lazy 版本的 decoder 模块，**核心改动是去掉 cross-attention 的 w_k/w_v 投影**，改为直接使用 context 经过 LayerNorm 后的结果作为 K/V。

### 改动清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `modulesV2.py` | **新增** | `lazy_cross_attention_step()`、`LazyDecoderLayer`、`LazyDecoderModel` |
| `model.py` | **新增** | `LazyMultiInterestModel` 类 |
| `kai_v2_model.py` | **修改** | 添加 `--use_lazy_decoder` 参数切换模型 |

### 详细实现

#### 1.1 modulesV2.py — 新增 `lazy_cross_attention_step`

```python
def lazy_cross_attention_step(queries,       # [B*beam, 1, d_model]
                              k_context,      # [B, 1, H, ctx_len, Dh] — 预计算好的 K，所有层共享
                              v_context,      # [B, 1, H, ctx_len, Dh] — 预计算好的 V，所有层共享
                              context_mask,   # [B, 1, 1, ctx_len]
                              num_heads,
                              head_dim,
                              cur_beam,
                              dropout_rate=0.0,
                              training=False):
    """
    Lazy Cross-Attention Step（推理用）

    与现有 multi_head_attention + scaled_attention 的关键差异：
    1. 无 w_k / w_v 投影 — K/V 直接来自 context 预计算
    2. 仅有 w_q 和 w_o 投影
    3. K/V 在所有 decoder 层间共享（而非每层独立）
    4. K/V 形状为 [B, 1, H, ctx_len, Dh]，其中 dim-1 是给 beam tile 用的占位

    实现逻辑：
    1. Q = w_q(queries), split_heads
    2. 从 cache 中取出 k_context / v_context（shape [B,1,H,ctx,Dh]）
    3. broadcast 到 beam 维：[B, cur_beam, H, ctx_len, Dh] → reshape [B*beam, H, ctx_len, Dh]
    4. scaled_attention(Q, k, v, context_mask)
    5. w_o 投影
    """
```

**关键实现要点**：

- 此函数**不含** `w_q` 和 `w_o` 的定义——它们在 `LazyDecoderLayer.step()` 的 variable_scope 中定义（与现有 DecoderLayer.step 一致）
- K/V 的来源从 `tf.layers.dense(enc_output, name="w_k")` 改为直接从外部传入的预计算结果
- 预计算逻辑在 `LazyMultiInterestModel` 的 `model()` 和 `beam_search_lazy()` 中完成：
  ```python
  # 在 LazyMultiInterestModel 中（不是在每层 decoder 中）
  context_k = layer_norm(encoder_input, scope="context_k_ln")  # 替代 w_k 投影
  context_v = context_k  # S_kv=1 时 K=V 共享
  context_k = split_heads(context_k, num_heads)  # [B, H, ctx_len, Dh]
  context_v = split_heads(context_v, num_heads)
  # 存入 cache 供所有层共享
  cache["k_context"] = context_k[:, None]  # [B, 1, H, ctx_len, Dh]
  cache["v_context"] = context_v[:, None]
  ```

#### 1.2 modulesV2.py — 新增 `LazyDecoderLayer`

```python
class LazyDecoderLayer:
    def __init__(self, name, dim, num_heads, hidden_dim, dropout_rate, training=False):
        """
        与 DecoderLayer 的差异仅在 Cross-Attention：
        - 不做 w_k/w_v 投影
        - 使用预计算的 context K/V（所有层共享同一组）
        - 操作顺序保持 Self-Attn → Cross-Attn → FFN（与现有一致）
        - 归一化保持 LayerNorm（与现有一致）
        """

    def forward(self, x, context_k, context_v, context_mask, training=False):
        """
        训练模式前向传播
        - Self-Attention: 与 DecoderLayer.forward 完全一致
        - Cross-Attention: 使用预计算 context_k/context_v，仅有 w_q/w_o
        """

    def step(self, x_t, cur_beam, layer_id,
             context_k, context_v, context_mask,
             cache, training=False):
        """
        推理模式逐步解码

        与 DecoderLayer.step 的差异：
        1. Cross-Attn 不使用 enc_output + w_k/w_v
        2. Cross-Attn 直接使用 context_k / context_v（从 cache 中取出，所有层共享）
        3. cache 中不再有 "k_enc_{layer_id}" / "v_enc_{layer_id}"（改为全局共享的 "k_context" / "v_context"）
        """
```

**`LazyDecoderLayer.step` 的精确实现差异**（对照 `DecoderLayer.step` L292-324）：

```python
# ---- 现有 DecoderLayer.step (L292-324) ----
with tf.variable_scope("multi_head_attention", reuse=tf.AUTO_REUSE):
    q_cur = tf.layers.dense(out1, self.dim, use_bias=False, name="w_q")
    q_cur = split_heads(q_cur, self.num_heads)

    # ❌ 这两步在 Lazy 版本中被删除：
    if f"k_enc_{layer_id}" not in cache:
        k_enc = split_heads(tf.layers.dense(enc_output, self.dim, use_bias=False, name="w_k"), ...)
        v_enc = split_heads(tf.layers.dense(enc_output, self.dim, use_bias=False, name="w_v"), ...)
        cache[f"k_enc_{layer_id}"] = k_enc[:, None]
        cache[f"v_enc_{layer_id}"] = v_enc[:, None]

    k_enc = cache[f"k_enc_{layer_id}"]
    v_enc = cache[f"v_enc_{layer_id}"]

# ---- Lazy 版本 ----
with tf.variable_scope("multi_head_attention", reuse=tf.AUTO_REUSE):
    q_cur = tf.layers.dense(out1, self.dim, use_bias=False, name="w_q")
    q_cur = split_heads(q_cur, self.num_heads)

    # ✅ 直接使用全局预计算的 context K/V（不再有 w_k/w_v）
    k_context = cache["k_context"]   # [B, 1, H, ctx_len, Dh] — 所有层共享
    v_context = cache["v_context"]   # [B, 1, H, ctx_len, Dh]

    # broadcast 到 beam 维（与现有逻辑一致）
    k_context = tf.broadcast_to(k_context, [B, cur_beam, H, ctx_len, self.head_dim])
    v_context = tf.broadcast_to(v_context, [B, cur_beam, H, ctx_len, self.head_dim])
    k_context = tf.reshape(k_context, [B*cur_beam, H, ctx_len, self.head_dim])
    v_context = tf.reshape(v_context, [B*cur_beam, H, ctx_len, self.head_dim])

    cross_out = scaled_attention(q_cur, k_context, v_context,
                                 cur_beam, src_mask=context_mask,
                                 dropout_rate=self.dropout_rate,
                                 training=training)

    cross_out = tf.layers.dense(cross_out, self.dim, name="w_o")
```

#### 1.3 modulesV2.py — 新增 `LazyDecoderModel`

```python
class LazyDecoderModel:
    def __init__(self, num_layers, dim, num_heads, hidden_dim, dropout_rate, training=False):
        """与 DecoderModel 完全一致，只是 layers 改为 LazyDecoderLayer"""

    def forward(self, decoder_input, context_k, context_v, context_mask, training):
        """训练模式：context_k/v 替代 enc_output"""
        for i in range(self.num_layers):
            decoder_input = self.layers[i].forward(
                decoder_input, context_k, context_v, context_mask, training=training)
        return decoder_input

    def step(self, x_t, cur_beam, context_k, context_v, context_mask, cache):
        """推理模式：context_k/v 替代 enc_output + src_mask"""
        for i, layer in enumerate(self.layers):
            x_t, cache = layer.step(
                x_t, cur_beam, i, context_k, context_v, context_mask, cache)
        return x_t, cache
```

#### 1.4 model.py — 新增 `LazyMultiInterestModel`

```python
class LazyMultiInterestModel(object):
    def __init__(self, feature_emb_dict, feature_emb_size_dict, dim=256,
                 vocab_sizes=[8192, 8192, 8192], print_ops=None):
        """
        与 MultiInterestModel 的差异仅在 forward 路径中 context 的处理方式。

        __init__ 完全一致（共享 vocab_embedding 等参数）。
        """

    def model(self, photo_sid, label, photo_semantic_id_int):
        """
        与 MultiInterestModel.model() 的差异：

        1. encoder_output = layer_norm(encoder_input)  ← 保留
        2. 新增：预计算 context K/V（替代每层 cross-attn 的 w_k/w_v 投影）
           context_k = split_heads(layer_norm(encoder_input, "context_k_ln"), num_heads)
           context_v = context_k  # S_kv=1 共享
        3. decoder_model 改为 LazyDecoderModel
        4. decoder_model.forward(decoder_input, context_k, context_v, context_mask)
        5. PRM loss 中 encoder_output 仍然用 layer_norm(encoder_input, "enc_ln")
           — PRM 不受 lazy decoder 影响
        """
```

**训练前向的精确差异**（对照 `MultiInterestModel.model()` L148-293）：

```python
# ---- 现有代码 L156-168 ----
encoder_output = layer_norm(encoder_input, scope="enc_ln")
decoder_model = DecoderModel(num_layers=2, dim=self._dim, num_heads=8, ...)
decoder_output = decoder_model.forward(decoder_input, encoder_output, src_mask, training=True)

# ---- Lazy 版本 ----
encoder_output = layer_norm(encoder_input, scope="enc_ln")  # PRM 仍需要

# 预计算 context K/V（一次性，所有层共享）
num_heads = 8
head_dim = self._dim // num_heads
def split_heads_ctx(x, num_heads):
    B = tf.shape(x)[0]
    L = tf.shape(x)[1]
    depth = x.get_shape().as_list()[-1] // num_heads
    return tf.transpose(tf.reshape(x, [B, L, num_heads, depth]), [0, 2, 1, 3])

# 核心：用 LayerNorm 替代 w_k/w_v 投影
context_k = split_heads_ctx(layer_norm(encoder_input, scope="context_k_ln"), num_heads)
# [B, H, ctx_len, Dh]
context_v = context_k  # S_kv=1: k=v 共享

decoder_model = LazyDecoderModel(num_layers=2, dim=self._dim, num_heads=8, ...)
# 训练时不需要 cache，直接传 context_k/v
decoder_output = decoder_model.forward(
    decoder_input, context_k, context_v, src_mask, training=True)
# 其余 loss 计算逻辑不变
```

**参数量变化**：

| | 现有 DecoderLayer × 2 | LazyDecoderLayer × 2 |
|---|---|---|
| Cross-Attn w_q | 2 × dim × dim | 2 × dim × dim |
| Cross-Attn w_k | 2 × dim × dim | **0** (全局 context_k_ln 替代) |
| Cross-Attn w_v | 2 × dim × dim | **0** (k=v 共享) |
| Cross-Attn w_o | 2 × dim × dim | 2 × dim × dim |
| context_k_ln | — | gamma: dim, beta: dim |
| **Cross-Attn 合计** | 4 × dim × dim × 2 | 2 × dim × dim × 2 + 2×dim |

当 dim=256: 现有 2×4×256×256 = 524,288 → Lazy 2×2×256×256 + 512 = 262,656 (减少 50%)

#### 1.5 kai_v2_model.py — 添加配置切换

```python
# 新增参数
parser.add_argument('--use_lazy_decoder', default=False, nargs='?', const=True)

# 模型创建处（L411）：
if args.use_lazy_decoder:
    model = LazyMultiInterestModel(all_param_dict, feature_emb_size_dict, print_ops=print_ops)
else:
    model = MultiInterestModel(all_param_dict, feature_emb_size_dict, print_ops=print_ops)
```

### Step 1 验证方法

1. **训练 loss 对比**：`--use_lazy_decoder=False` vs `True`，同一数据，loss 应可比（初始可能略差，因为少了一半 cross-attn 参数）
2. **参数量检查**：打印 trainable variables，验证 Lazy 版本少掉了 w_k/w_v
3. **推理 shape 检查**：beam_search_lazy 输出 `[B, beam_size, 3]` 和 `[B, beam_size]`

---

## Step 2：Beam Search 推理

### 目标

实现 `LazyMultiInterestModel.beam_search_lazy()` 方法，利用 context K/V 共享特性优化推理显存。

### 改动清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `model.py` | **新增** | `LazyMultiInterestModel.beam_search_lazy()` |

### 详细实现

核心差异对照 `MultiInterestModel.beam_search_fast()`：

```python
def beam_search_lazy(self, beam_size=512, temperature=1):
    # ---------- ① 预处理：与现有完全一致 ----------
    user_static_emb = ...  # 同 L322-328
    user_click_emb = ...    # 同 L332-346
    encoder_input = tf.concat([user_static_emb, user_click_emb], axis=1)  # 同 L350
    # src_mask 构建 同 L352-374
    encoder_output = layer_norm(encoder_input, scope="enc_ln")  # PRM 需要

    # ---------- ①.5 预计算 Context K/V（核心新增）----------
    # 替代现有代码中每层独立的 k_enc/v_enc 计算
    num_heads = 8
    head_dim = self._dim // num_heads
    def split_heads_ctx(x, num_heads):
        B = tf.shape(x)[0]
        L = tf.shape(x)[1]
        depth = x.get_shape().as_list()[-1] // num_heads
        return tf.transpose(tf.reshape(x, [B, L, num_heads, depth]), [0, 2, 1, 3])

    context_k = split_heads_ctx(
        layer_norm(encoder_input, scope="context_k_ln"), num_heads)
    # [B, H, ctx_len, Dh]
    context_v = context_k  # S_kv=1 共享

    # 存入 cache 格式：[B, 1, H, ctx_len, Dh]（dim-1 留给 beam broadcast）
    cache["k_context"] = context_k[:, None]
    cache["v_context"] = context_v[:, None]

    # ---------- ② Decoder 初始化 ----------
    decoder_model = LazyDecoderModel(num_layers=2, dim=self._dim, num_heads=8, ...)
    # 其余 seqs, dec_path_log_probs, cur_beam 同 L378-388

    # ---------- ③ 逐层解码 ----------
    for step, V in enumerate(self._vocab_sizes):
        dec_in = tf.nn.embedding_lookup(self._vocab_embedding, seqs[:, :, -1])
        dec_in = tf.reshape(dec_in, [B*cur_beam, 1, self._dim])

        # ✅ 核心差异：传 context_k/v 替代 enc_output
        dec_out, cache = decoder_model.step(
            dec_in, cur_beam,
            context_k=cache["k_context"],   # 全局共享
            context_v=cache["v_context"],
            context_mask=src_mask,
            cache=cache)

        # proj + beam 选择逻辑同 L437-558
        ...

    # PRM 部分仍使用 encoder_output（同现有逻辑）
```

**显存优化关键点**：

| 张量 | 现有 beam_search_fast | beam_search_lazy |
|------|----------------------|------------------|
| encoder_output tile | `[B*1024, 201, 256]` ≈ 53M floats | **不需要** |
| 每层 k_enc/v_enc | `[B, 1, 8, 201, 32]` × 2层 × 2(KV) | **一组** `[B, 1, 8, 201, 32]` 全层共享 |
| src_mask tile | 每层内部 repeat_elements | 每层内部 repeat_elements（同） |

`gather_cache` / `tile_cache_for_first_step` 辅助函数：

```python
def gather_cache_lazy(old_cache, gp):
    """与现有 gather_cache 的差异：
    - k_self_/v_self_ → 重排
    - k_context/v_context → 原样保留（全局共享，不受 beam 选择影响）
    - 不再有 k_enc_{layer_id}/v_enc_{layer_id}
    """
    new_cache = {}
    for ck, v in old_cache.items():
        if ck.startswith(("k_self_", "v_self_")):
            new_cache[ck] = tf.gather_nd(v, gp)
        # k_context / v_context 原样保留
        elif ck in ("k_context", "v_context"):
            new_cache[ck] = v
    return new_cache
```

### Step 2 验证方法

1. **推理结果对比**：`beam_search_fast_no_prm` vs `beam_search_lazy`，beam_size=1 时结果应一致（贪心解码等价）
2. **显存对比**：监控 GPU 显存占用，Lazy 版应减少 encoder_output tile 部分
3. **输出 shape**：`gen_part_loc = [B, beam_size, 3]`，`probs = [B, beam_size]`

---

## Step 3：ContextProcessor（三路特征整合 + 预计算分层 KV）

### 目标

将现有的「user_static + user_click 拼接 → layer_norm → context K/V」升级为论文的三路 Pathway 架构：
- User Static Pathway → Linear
- Short-term Pathway → Linear
- Long-term Pathway → Linear（新增）

并实现分层的 L_kv 组 K/V 对（目前 L_kv=1 就是一组，但架构上支持 L_kv>1）。

### 改动清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `modulesV2.py` | **新增** | `ContextProcessor` 类 |
| `model.py` | **修改** | `LazyMultiInterestModel.model()` 中替换 context 构建逻辑 |
| `feature_attr_extract.py` | **修改** | 新增 long-term 行为特征配置 |

### 详细实现

#### 3.1 ContextProcessor 类

```python
class ContextProcessor:
    def __init__(self, d_model, n_head, l_kv=1, s_kv=1, dropout_rate=0.1):
        """
        Args:
            d_model: 统一输出维度
            n_head: 注意力头数（GQA 时与 g_kv 不同，Step 3 暂不区分）
            l_kv: KV 层数（默认 1，可扩展）
            s_kv: KV 分离系数（1=共享, 2=分离）
        """

    def forward(self,
                user_static_emb,   # [B, N_s, d_input]
                short_term_emb,    # [B, T_short, d_input]
                long_term_emb,     # [B, T_long, d_input]  — Step 5 才真正使用
                short_term_len,    # [B]
                long_term_len,     # [B]
                training=False):
        """
        1. 三路 Linear:
           static_ctx = Linear(user_static_emb, d_model)    # [B, N_s, d_model]
           short_ctx  = Linear(short_term_emb, d_model)     # [B, T_short, d_model]
           long_ctx   = Linear(long_term_emb, d_model)      # [B, T_long, d_model]

        2. 拼接:
           context = concat([static_ctx, short_ctx, long_ctx], axis=1)
           # [B, total_len, d_model]

        3. 生成 KV（L_kv 组）:
           for l in range(self.l_kv):
             # 按 Eq.(2) 从 context 中分片
             chunk = context[:, :, l*self.s_kv*chunk_size : (l+1)*self.s_kv*chunk_size]
             k_l = layer_norm(chunk, scope=f"context_k_ln_{l}")  # Step 3 仍用 LayerNorm
             v_l = k_l  # s_kv=1: k=v 共享
             kv_dict[f"k_{l}"] = split_heads(k_l, n_head)[:, None]  # [B, 1, H, ctx_len, Dh]
             kv_dict[f"v_{l}"] = split_heads(v_l, n_head)[:, None]

        4. 生成 context_mask:
           total_len = N_s + T_short + T_long
           mask = concat([ones([B, N_s]), sequence_mask(short_term_len, T_short),
                          sequence_mask(long_term_len, T_long)], axis=1)
           context_mask = reshape(mask, [B, 1, 1, total_len])

        Returns:
            kv_dict: {k_0, v_0, ..., k_{L_kv-1}, v_{L_kv-1}}
            context_mask: [B, 1, 1, total_len]
        """

    def get_kv_for_layer(self, layer_id, n_layer, kv_dict):
        """
        论文 Eq.(9): l_kv = floor(layer_id * L_kv / N_layer)

        当 L_kv=1 时，所有层都使用 k_0/v_0。
        """
        l_kv_idx = layer_id * self.l_kv // n_layer
        return kv_dict[f"k_{l_kv_idx}"], kv_dict[f"v_{l_kv_idx}"]
```

#### 3.2 LazyMultiInterestModel 中使用 ContextProcessor

```python
# 替换 Step 1 中手动计算 context_k/v 的逻辑：
context_processor = ContextProcessor(d_model=self._dim, n_head=8, l_kv=1, s_kv=1)

# 训练时：
static_input = user_static_emb          # [B, 1, dim] — 已经过 MLP
short_input = user_click_emb            # [B, 200, dim] — 已经过 MLP
long_input = tf.zeros([B, 1, self._dim])  # Step 3 先用占位，Step 5 才真正使用

context_kv, context_mask = context_processor.forward(
    static_input, short_input, long_input,
    used_len, tf.ones([B], dtype=tf.int32),  # long_term_len 暂时全 1
    training=True)

# context_kv["k_0"] shape: [B, 1, H, ctx_len, Dh]
# context_kv["v_0"] shape: [B, 1, H, ctx_len, Dh]
```

### Step 3 验证方法

1. **KV 形状正确**：L_kv=1 时，context_kv 应包含 k_0/v_0 两个 key
2. **Mask 正确**：static 部分全 1，click 部分按 used_len，long 部分暂全 1
3. **效果对比**：与 Step 2（手动 context_k/v）的结果对比，loss 应基本一致（因为 Step 3 仅多了 Linear 投影，功能等价）

---

## Step 4：RMSNorm + GQA

### 目标

1. 将 LayerNorm 替换为 RMSNorm（可选，通过配置切换）
2. 实现 GQA（Grouped Query Attention），G_kv < H_q

### 改动清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `modulesV2.py` | **新增** | `rms_norm()`、`repeat_kv()` |
| `modulesV2.py` | **修改** | `LazyDecoderLayer` 支持 norm_type 和 g_kv 参数 |
| `ContextProcessor` | **修改** | 归一化方式可配置为 RMSNorm |

### 详细实现

#### 4.1 RMSNorm

```python
def rms_norm(x, scope, eps=1e-6):
    """
    与 layer_norm 的差异：
    - 不减均值（仅除 RMS）
    - 无 beta 参数（仅 gamma）
    """
    with tf.variable_scope(f"{scope}/rms_norm", reuse=tf.AUTO_REUSE):
        gamma = tf.get_variable("gamma", [x.get_shape().as_list()[-1]],
                                initializer=tf.ones_initializer())
        rms = tf.sqrt(tf.reduce_mean(x * x, axis=-1, keepdims=True) + eps)
        return gamma * (x / rms)
```

#### 4.2 GQA — `repeat_kv` 辅助函数

```python
def repeat_kv(x, n_rep):
    """
    GQA: 将 G_kv 组 K/V 重复 n_rep 次扩展到 H_q 头

    x: [B, G_kv, seq_len, Dh] 或 [B*beam, G_kv, seq_len, Dh]
    n_rep = H_q // G_kv
    返回: [B, H_q, seq_len, Dh]
    """
    if n_rep == 1:
        return x
    B, G_kv, seq_len, Dh = x.shape  # 或动态 shape
    x = tf.expand_dims(x, axis=2)           # [B, G_kv, 1, seq_len, Dh]
    x = tf.tile(x, [1, 1, n_rep, 1, 1])     # [B, G_kv, n_rep, seq_len, Dh]
    return tf.reshape(x, [B, G_kv * n_rep, seq_len, Dh])  # [B, H_q, seq_len, Dh]
```

#### 4.3 LazyDecoderLayer 增加可配置参数

```python
class LazyDecoderLayer:
    def __init__(self, name, dim, num_heads, hidden_dim, dropout_rate,
                 norm_type="layer_norm",   # 新增：可选 "rms_norm"
                 g_kv=None,                # 新增：默认=None 即 g_kv=num_heads
                 training=False):
```

### Step 4 验证方法

1. **RMSNorm 正确性**：已知输入的手算 RMSNorm 结果对比
2. **GQA 正确性**：g_kv=num_heads 时结果应与无 GQA 完全一致
3. **效果对比**：逐项打开 RMSNorm / GQA，单独验证对 loss 的影响

---

## Step 5：Long-term Pathway + 操作顺序切换

### 目标

1. 新增 long-term 用户行为序列特征
2. 将操作顺序从 Self-Attn → Cross-Attn → FFN 切换为 Cross-Attn → Self-Attn → FFN

### 改动清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `feature_attr_extract.py` | **修改** | 新增 long-term 行为特征 |
| `model.py` | **修改** | `LazyMultiInterestModel._build_context()` 接入 long-term |
| `modulesV2.py` | **修改** | `LazyDecoderLayer` 支持操作顺序配置 |

### 详细实现

#### 5.1 Long-term 行为特征

```python
# feature_attr_extract.py
all_feats = {
    ...
    "user_profile_v1_click_pid_list": {"expand": 200},   # 短期
    "user_profile_v1_click_aid_list": {"expand": 200},    # 短期
    # 新增长期行为
    "user_long_term_pid_list": {"expand": 300},  # 长期行为 pid（线上约 2700）
    "user_long_term_aid_list": {"expand": 300},   # 长期行为 aid
}
```

#### 5.2 LazyDecoderLayer 操作顺序切换

```python
class LazyDecoderLayer:
    def __init__(self, ..., attn_order="self_first"):  # 新增参数
        # attn_order: "self_first" (Self→Cross→FFN) 或 "cross_first" (Cross→Self→FFN)

    def step(self, ...):
        if self.attn_order == "cross_first":
            # 论文 Eq.(6)-(8): Cross → Self → FFN
            h = x_t + lazy_cross_attention(rms_norm(x_t), ...)
            h = h + causal_self_attention(rms_norm(h), ...)
            h = h + ffn(rms_norm(h))
        else:
            # 现有顺序: Self → Cross → FFN
            h = x_t + causal_self_attention(layer_norm(x_t), ...)
            h = h + lazy_cross_attention(layer_norm(h), ...)
            h = h + ffn(layer_norm(h))
```

### Step 5 验证方法

1. **Long-term 特征加载**：验证 long-term embedding 不为空
2. **操作顺序切换**：self_first vs cross_first 的 loss 对比
3. **Context 长度变化**：加入 long-term 后 context_len 从 201 增至 ~501，验证显存增长在可控范围

---

## 全量可配置参数汇总表

### Step 1 即可使用的参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_lazy_decoder` | False | 切换 Lazy/传统 decoder |
| `l_kv` | 1 | Context KV 层数 |
| `s_kv` | 1 | KV 分离系数（1=共享） |

### Step 4 新增参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `norm_type` | "layer_norm" | 归一化类型 |
| `g_kv` | None (=num_heads) | GQA KV 组数 |

### Step 5 新增参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `attn_order` | "self_first" | 操作顺序 |
| `max_long_len` | 300 | 长期行为序列长度 |

### 模型规模预设

| 规模 | d_model | n_layer | n_head | hidden_dim | LR |
|------|---------|---------|--------|------------|----|
| 现有 | 256 | 2 | 8 | 512 | 1e-4 |
| 0.5B | 1408 | 14 | 11 | 5632 | 2.24e-4 |
| 1B | 1792 | 18 | 14 | 7168 | 1.58e-4 |

---

## 单元测试要求（每步都应验证）

### Step 1 测试

1. **LazyDecoderLayer 无 w_k/w_v**：遍历 trainable_variables，确认无 `multi_head_attention/w_k` 和 `multi_head_attention/w_v`
2. **context_k_ln 存在**：确认有 `context_k_ln/layer_norm/gamma` 和 `context_k_ln/layer_norm/beta`
3. **训练 loss 可计算**：输入随机数据，forward + backward 不报错
4. **输出维度**：decoder_output shape = [B, 3, dim]

### Step 2 测试

1. **beam_search_lazy 输出形状**：`[B, beam_size, 3]` + `[B, beam_size]`
2. **beam=1 贪心等价**：beam_size=1 时，结果与逐步贪心解码一致
3. **cache 中 k_context/v_context 共享**：验证所有层使用同一组 context KV
4. **gather_cache_lazy 正确**：beam 选择后 self-attention KV 被正确重排

### Step 3 测试

1. **ContextProcessor 输出形状**：kv_dict 中 k_0/v_0 shape 正确
2. **context_mask 覆盖**：padding 位置为 0
3. **三路 Linear 参数独立**：static/short/long 三路 Linear 的 kernel 不共享

### Step 4 测试

1. **RMSNorm 数学正确性**：手算对比
2. **GQA repeat_kv**：g_kv=2 时 K/V 被 repeat 正确次数
3. **g_kv=num_heads 等价性**：输出与无 GQA 完全一致

### Step 5 测试

1. **Long-term 特征非空**：验证 long_term_emb 不全零
2. **cross_first 顺序**：验证 Cross-Attn 在 Self-Attn 之前执行
3. **Context 总长度**：N_s + T_short + T_long ≈ 501
