# OneRec-V2 Lazy Decoder-Only 架构 — 结构化实施指南

> 本文档是研究论文 OneRec-V2 与 agent 自主实现能力之间的主要接口。
> 包含足够的细节和结构，以实现无需人工干预即可完整复制论文中的方法。

---

## 目录

1. [实施路线图](#1-实施路线图)
2. [模块 0：RMSNorm](#2-模块-0rmsnorm)
3. [模块 1：ContextProcessor](#3-模块-1contextprocessor)
4. [模块 2：LazyCrossAttention](#4-模块-2lazycrossattention)
5. [模块 3：LazyDecoderLayer](#5-模块-3lazydecoderlayer)
6. [模块 4：LazyDecoderModel](#6-模块-4lazydecodermodel)
7. [模块 5：LazyMultiInterestModel（训练）](#7-模块-5laymultiinterestmodel训练)
8. [模块 6：Beam Search（推理）](#8-模块-6beam-search推理)
9. [全量可配置参数汇总](#9-全量可配置参数汇总)
10. [单元测试要求](#10-单元测试要求)

---

## 1. 实施路线图

按照依赖关系从底层到顶层的实施顺序：

```
Phase 1: 基础组件（无外部依赖）
  └─ Module 0: RMSNorm
  └─ Module 0.5: GQA split/merge 辅助函数

Phase 2: Context Processor（依赖 Phase 1）
  └─ Module 1: ContextProcessor
     ├─ 三路 Pathway Linear
     ├─ Context 统一维度变换
     ├─ 分片生成 L_kv 组 KV
     └─ RMSNorm + KV 共享

Phase 3: Lazy Decoder Block（依赖 Phase 1+2）
  └─ Module 2: LazyCrossAttention（GQA + 无 K/V 投影）
  └─ Module 3: LazyDecoderLayer（Cross→Self→FFN, pre-norm）
  └─ Module 4: LazyDecoderModel（N_layer 层堆叠）

Phase 4: 模型集成（依赖 Phase 2+3）
  └─ Module 5: LazyMultiInterestModel 训练前向
  └─ Module 6: Beam Search 推理流程

Phase 5: 特征与配置适配
  └─ feature_attr_extract.py: 新增 long-term pathway
  └─ kai_v2_model.py: 适配新模型接口
```

---

## 2. 模块 0：RMSNorm

### 2.1 精确函数签名

```python
def rms_norm(x: tf.Tensor, scope: str, eps: float = 1e-6) -> tf.Tensor:
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)

    与 LayerNorm 的关键差异：
    - 无均值中心化（不减均值）
    - 无 beta (偏移) 参数
    - 仅使用 gamma (缩放) 参数

    论文 Eq.(3)-(4) 中用于 context processor 的 K/V 归一化
    以及 Eq.(6)-(8) 中 decoder block 的 pre-norm
    """
```

### 2.2 实现逻辑

```
输入 x: [batch, seq_len, d_model] 或 [batch, d_model]
1. 计算 RMS: rms = sqrt(mean(x^2, axis=-1, keepdims=True) + eps)
2. 归一化: x_norm = x / rms
3. 乘以可学习缩放: output = gamma * x_norm（gamma 形状 [d_model]，初始化为 1）
4. 返回 output
```

### 2.3 期望输入/输出

| | 形状 | 说明 |
|---|---|---|
| **输入** x | `[B, L, D]` 或 `[B, D]` | 任意浮点张量，最后一维为归一化维度 |
| **输出** | 同输入形状 | 归一化后的张量 |

**示例**：
```
输入: x = tf.constant([[1.0, 2.0, 3.0, 4.0]])  # [1, 4]
rms = sqrt(mean([1, 4, 9, 16])) = sqrt(7.5) ≈ 2.739
x_norm = [0.366, 0.730, 1.095, 1.461]
output = gamma * x_norm  (gamma 初始为 1)
```

### 2.4 错误条件与处理

| 错误条件 | 处理方式 |
|----------|----------|
| x 最后一维为 0 | raise ValueError("rms_norm: last dimension cannot be 0") |
| eps <= 0 | raise ValueError("rms_norm: eps must be positive") |
| x 含 NaN | RMS 自然会传播 NaN，无需特殊处理（训练时应由 upstream 保证） |
| variable_scope 冲突 | 使用 reuse=tf.AUTO_REUSE |

### 2.5 单元测试要求

1. **正确性测试**：输入已知张量，验证输出与手算 RMSNorm 结果的数值误差 < 1e-5
2. **gamma 可学习**：验证 gamma 初始化为全 1，且梯度可以正常回传
3. **形状不变性**：输入 [2, 3, 4] 输出应为 [2, 3, 4]
4. **与 LayerNorm 对比**：验证 RMSNorm 输出 != LayerNorm 输出（确认去除了均值中心化）
5. **eps 保护**：输入全零张量时，输出不为 NaN（因为 eps > 0）

---

## 3. 模块 1：ContextProcessor

### 3.1 精确函数签名

```python
class ContextProcessor:
    def __init__(self,
                 d_model: int,
                 n_head: int,
                 g_kv: int,
                 l_kv: int,
                 s_kv: int,
                 n_layer: int,
                 n_static_features: int = 1,
                 dropout_rate: float = 0.1):
        """
        Context Processor: 将异构用户特征转换为分层的 KV 对

        论文 Section 2.2.1, Eq.(1)-(4)

        Args:
            d_model: 模型隐藏维度
            n_head: query 头数 (H_q = d_model / d_head)
            g_kv: key-value head 组数（GQA）
            l_kv: context KV 层数
            s_kv: KV 分离系数（1=共享, 2=分离）
            n_layer: decoder 总层数（用于计算 KV 共享映射）
            n_static_features: 用户静态特征 token 数
            dropout_rate: dropout 率
        """

    def forward(self,
                user_static_emb: tf.Tensor,    # [B, N_s, d_context_input]
                short_term_emb: tf.Tensor,      # [B, T_short, d_context_input]
                long_term_emb: tf.Tensor,       # [B, T_long, d_context_input]
                training: bool = False) -> dict:
        """
        处理 context 并返回所有层的 KV 对

        Returns:
            context_kv: dict, keys 为 "k_{l_kv_idx}", "v_{l_kv_idx}"
                        值为 [B, N_s+T_short+T_long, G_kv, d_head]
                        或 [B, N_s+T_short+T_long, d_model] (取决于 GQA 实现)
        """

    def get_kv_for_layer(self, layer_id: int, context_kv: dict) -> tuple:
        """
        根据层 ID 获取对应的 KV 对

        论文 Eq.(9): l_kv = floor(l * L_kv / N_layer)

        Args:
            layer_id: 当前 decoder 层 ID (0-indexed)
            context_kv: forward() 的输出

        Returns:
            (k, v): tuple of tf.Tensor
                    k: [B, context_len, G_kv, d_head]
                    v: [B, context_len, G_kv, d_head]
        """

    def get_context_mask(self,
                         static_len: int,
                         short_term_len: tf.Tensor,  # [B]
                         long_term_len: tf.Tensor,     # [B]
                         max_context_len: int) -> tf.Tensor:
        """
        构建 context padding mask

        Returns:
            context_mask: [B, 1, 1, max_context_len]  (1=valid, 0=pad)
        """
```

### 3.2 实现逻辑

```
ContextProcessor.forward:
1. 三路 Pathway 独立 Linear:
   - user_static_emb → Linear(d_context)  # [B, N_s, d_context]
   - short_term_emb → Linear(d_context)    # [B, T_short, d_context]
   - long_term_emb → Linear(d_context)     # [B, T_long, d_context]

2. 拼接: context = concat([static, short, long], axis=1)
   # [B, N_s+T_short+T_long, d_context]

3. 按 Eq.(2) 分片:
   d_context = S_kv * L_kv * G_kv * d_head
   将 context 沿特征维 reshape:
   context = reshape(context, [B, total_len, S_kv * L_kv, G_kv * d_head])

4. 按 Eq.(3)-(4) 生成各层 KV:
   for l in range(L_kv):
     k_l = RMSNorm(context[:, :, l*S_kv, :])       # Eq.(3)
     if S_kv == 1:
       v_l = k_l                                    # Eq.(4) 共享
     elif S_kv == 2:
       v_l = RMSNorm(context[:, :, l*S_kv+1, :])    # Eq.(4) 分离

5. 存入 context_kv dict

ContextProcessor.get_kv_for_layer:
1. l_kv_idx = floor(layer_id * L_kv / N_layer)  # Eq.(9)
2. 返回 context_kv 中的 k_{l_kv_idx}, v_{l_kv_idx}
```

### 3.3 期望输入/输出

| | 形状 | 说明 |
|---|---|---|
| **输入** user_static_emb | `[B, N_s, d_input]` | d_input 是原始嵌入维度（如 256） |
| **输入** short_term_emb | `[B, T_short, d_input]` | 短期行为序列 |
| **输入** long_term_emb | `[B, T_long, d_input]` | 长期行为序列 |
| **输出** context_kv | `dict` | 含 L_kv 组 (k, v) 对 |

**示例** (1B 模型, L_kv=1, S_kv=1, G_kv=14, d_head=128):
```
d_context = 1 * 1 * 14 * 128 = 1792 = d_model
context = [B, 512, 1792]
k_0 = RMSNorm(context) → [B, 512, 14, 128]
v_0 = k_0 (S_kv=1 共享)
```

### 3.4 错误条件与处理

| 错误条件 | 处理方式 |
|----------|----------|
| d_model % G_kv != 0 | raise ValueError("d_model must be divisible by G_kv") |
| S_kv not in {1, 2} | raise ValueError("S_kv must be 1 (shared) or 2 (separated)") |
| L_kv > N_layer | raise ValueError("L_kv cannot exceed N_layer") |
| 输入 embedding 维度与 Linear 不匹配 | 在 Linear 层自动处理（dense 层会报错） |

### 3.5 单元测试要求

1. **KV 形状正确性**：验证 context_kv 中每对 KV 形状为 [B, total_len, G_kv, d_head]
2. **KV 共享验证**：当 S_kv=1 时，验证 k == v
3. **KV 分离验证**：当 S_kv=2 时，验证 k != v
4. **层映射正确性**：get_kv_for_layer 在不同 layer_id 下返回正确的 KV 对
5. **分片逻辑**：L_kv=3 时，验证 context 被正确分为 3 组
6. **梯度流通**：验证从 context_kv 的 k/v 可以反传梯度到输入 embedding

---

## 4. 模块 2：LazyCrossAttention

### 4.1 精确函数签名

```python
def lazy_cross_attention(queries: tf.Tensor,    # [B*beam, 1, d_model]
                         k_context: tf.Tensor,   # [B, context_len, G_kv, d_head]
                         v_context: tf.Tensor,   # [B, context_len, G_kv, d_head]
                         context_mask: tf.Tensor, # [B, 1, 1, context_len]
                         n_head: int,
                         g_kv: int,
                         d_head: int,
                         cur_beam: int,
                         dropout_rate: float = 0.0,
                         training: bool = False) -> tf.Tensor:
    """
    Lazy Cross-Attention: 无 K/V 投影的跨注意力机制

    论文核心创新（Section 2.2.2 "Lazy Cross-Attention: KV-Sharing"）:
    - K/V 来自 Context Processor 预计算，不需要 w_k/w_v 投影
    - 仅 Q 需要投影 (w_q)
    - 支持 GQA：Q 头数 = n_head, KV 组数 = g_kv

    Args:
        queries: decoder 当前步的输入（已归一化）, [B*beam, 1, d_model]
        k_context: context processor 输出的 key, [B, ctx_len, G_kv, d_head]
        v_context: context processor 输出的 value, [B, ctx_len, G_kv, d_head]
        context_mask: padding mask, [B, 1, 1, ctx_len]
        n_head: query 头数 (H_q)
        g_kv: key-value head 组数
        d_head: 每头维度
        cur_beam: 当前 beam 数（用于 tile context）

    Returns:
        output: [B*beam, 1, d_model]
    """
```

### 4.2 实现逻辑

```
1. Q 投影:
   Q = dense(queries, d_model, use_bias=False, name="w_q")  # [B*beam, 1, d_model]
   Q = reshape(Q, [B*beam, 1, n_head, d_head])
   Q = transpose(Q, [0, 2, 1, 3])  # [B*beam, n_head, 1, d_head]

2. K/V tile 到 beam:
   # k_context: [B, ctx_len, G_kv, d_head] → [B, 1, ctx_len, G_kv, d_head] → tile → [B, cur_beam, ctx_len, G_kv, d_head]
   # reshape → [B*cur_beam, ctx_len, G_kv, d_head]
   # transpose → [B*cur_beam, G_kv, ctx_len, d_head]
   K = tile_and_reshape(k_context, cur_beam)  # [B*beam, G_kv, ctx_len, d_head]
   V = tile_and_reshape(v_context, cur_beam)  # [B*beam, G_kv, ctx_len, d_head]

3. GQA expand K/V:
   # 将 G_kv 组扩展到 n_head 头
   # [B*beam, G_kv, ctx_len, d_head] → [B*beam, n_head, ctx_len, d_head]
   K_expanded = repeat_kv(K, n_head // g_kv)  # 沿 head 维度 repeat
   V_expanded = repeat_kv(V, n_head // g_kv)

4. 缩放点积注意力:
   attn_scores = matmul(Q, K_expanded^T) / sqrt(d_head)  # [B*beam, n_head, 1, ctx_len]

5. Mask:
   # 扩展 context_mask 到 [B*beam, n_head, 1, ctx_len]
   attn_scores = apply_mask_add(attn_scores, context_mask)

6. Softmax + Dropout:
   attn_weights = softmax(attn_scores, axis=-1)
   attn_weights = dropout(attn_weights, rate, training)

7. 加权求和:
   context = matmul(attn_weights, V_expanded)  # [B*beam, n_head, 1, d_head]

8. 合并头 + 输出投影:
   context = transpose(context, [0, 2, 1, 3])  # [B*beam, 1, n_head, d_head]
   context = reshape(context, [B*beam, 1, n_head*d_head])  # = [B*beam, 1, d_model]
   output = dense(context, d_model, name="w_o")  # [B*beam, 1, d_model]

9. 返回 output
```

### 4.3 期望输入/输出

| | 形状 | 说明 |
|---|---|---|
| **输入** queries | `[B*beam, 1, d_model]` | decoder 当前步输入 |
| **输入** k_context | `[B, ctx_len, G_kv, d_head]` | 预计算 key |
| **输入** v_context | `[B, ctx_len, G_kv, d_head]` | 预计算 value |
| **输出** | `[B*beam, 1, d_model]` | cross-attention 输出 |

**示例** (1B 模型, n_head=14, g_kv=2, d_head=128, beam=512, ctx_len=512):
```
Q: [B*512, 14, 1, 128]
K: [B*512, 14, 512, 128]  (G_kv=2 expand 7x to 14)
V: [B*512, 14, 512, 128]
attn_scores: [B*512, 14, 1, 512]
output: [B*512, 1, 1792]
```

### 4.4 错误条件与处理

| 错误条件 | 处理方式 |
|----------|----------|
| n_head % g_kv != 0 | raise ValueError("n_head must be divisible by g_kv") |
| k_context 最后一维 != d_head | raise ValueError("k_context last dim must equal d_head") |
| context_mask 形状不匹配 | 自动 broadcast，若无法 broadcast 则 TF 报错 |
| cur_beam == 0 | raise ValueError("cur_beam must be positive") |

### 4.5 单元测试要求

1. **GQA 行为**：g_kv < n_head 时，验证 K/V 被 repeat 正确次数
2. **无 K/V 投影**：验证函数内不创建 w_k/w_v 变量
3. **Q 投影存在**：验证 w_q 变量被创建
4. **输出投影存在**：验证 w_o 变量被创建
5. **Mask 正确性**：全 1 mask 输出 != 全 0 mask 输出
6. **梯度流通**：验证 Q 的梯度可通过 w_q 回传
7. **参数量对比**：lazy cross-attn 参数 = w_q + w_o，传统 = w_q + w_k + w_v + w_o

---

## 5. 模块 3：LazyDecoderLayer

### 5.1 精确函数签名

```python
class LazyDecoderLayer:
    def __init__(self,
                 name: str,
                 layer_id: int,
                 d_model: int,
                 n_head: int,
                 g_kv: int,
                 d_head: int,
                 hidden_dim: int,
                 dropout_rate: float = 0.1):
        """
        Lazy Decoder Block

        论文 Eq.(6)-(8): Cross-Attn → Self-Attn → FFN, pre-norm
        与现有 DecoderLayer 的关键差异:
        1. 操作顺序: Cross-Attn 先于 Self-Attn
        2. Pre-norm: 归一化在操作之前
        3. Cross-Attn 使用预计算 KV，无 w_k/w_v
        """

    def forward(self,
                x: tf.Tensor,              # [B, 3, d_model] 训练用完整序列
                context_kv: dict,           # ContextProcessor 输出
                context_mask: tf.Tensor,    # [B, 1, 1, ctx_len]
                training: bool = False) -> tf.Tensor:
        """
        训练模式前向传播（并行处理所有 token）
        """

    def step(self,
             x_t: tf.Tensor,            # [B*beam, 1, d_model] 当前步输入
             cur_beam: int,
             layer_id: int,
             context_kv: dict,          # 预计算 KV
             context_mask: tf.Tensor,   # [B, 1, 1, ctx_len]
             cache: dict,              # self-attention KV cache
             training: bool = False) -> tuple:
        """
        推理模式逐步解码

        与现有 DecoderLayer.step 的差异:
        - cross-attention 不使用 w_k/w_v，直接从 context_kv 取
        - 操作顺序为 Cross→Self→FFN
        - 归一化方式为 RMSNorm (pre-norm)
        """
```

### 5.2 实现逻辑

```
LazyDecoderLayer.step (推理模式):
1. Cross-Attention (Eq.(6)):
   h_cross = x_t + LazyCrossAttention(
       RMSNorm(x_t),                           # pre-norm
       context_kv["k_{l_kv_idx}"],             # 预计算 KV
       context_kv["v_{l_kv_idx}"],
       context_mask, n_head, g_kv, d_head, cur_beam
   )

2. Self-Attention (Eq.(7)):
   h_self = h_cross + CausalSelfAttention(
       RMSNorm(h_cross),                       # pre-norm
       cache  # self-attention KV cache
   )
   # (self-attention 逻辑与现有代码相同，使用 w_q/w_k/w_v 投影)

3. FFN (Eq.(8)):
   h_out = h_self + FFN(RMSNorm(h_self))      # pre-norm

4. 更新 cache（self-attention K/V）

5. 返回 (h_out, cache)
```

### 5.3 期望输入/输出

| | 形状 | 说明 |
|---|---|---|
| **输入** x_t | `[B*beam, 1, d_model]` | 当前步解码输入 |
| **输入** context_kv | `dict` | ContextProcessor 输出 |
| **输入** cache | `dict` | self-attention KV cache |
| **输出** h_out | `[B*beam, 1, d_model]` | 当前步输出 |
| **输出** cache | `dict` | 更新后的 KV cache |

### 5.4 错误条件与处理

| 错误条件 | 处理方式 |
|----------|----------|
| layer_id 超出范围 | 由 context_kv 映射自动处理 |
| context_kv 为空 | raise ValueError("context_kv must be populated before step") |
| cache 缺少必要 key | 在 step 中按需初始化（与现有逻辑一致） |

### 5.5 单元测试要求

1. **操作顺序**：验证 cross-attention 在 self-attention 之前执行（可在 variable_scope 中检查操作顺序）
2. **Pre-norm**：验证 RMSNorm 在每个子操作之前调用
3. **残差连接**：验证每个子操作的输出加上了输入（残差）
4. **与现有 DecoderLayer 输出维度一致**：输入 [B, 1, D] 输出 [B, 1, D]
5. **Cache 更新**：验证 step 调用后 cache 中的 T 维度增加了 1

---

## 6. 模块 4：LazyDecoderModel

### 6.1 精确函数签名

```python
class LazyDecoderModel:
    def __init__(self,
                 n_layer: int,
                 d_model: int,
                 n_head: int,
                 g_kv: int,
                 d_head: int,
                 hidden_dim: int,
                 dropout_rate: float = 0.1):
        """
        多层 Lazy Decoder Block 堆叠
        """

    def forward(self,
                decoder_input: tf.Tensor,    # [B, 3, d_model]
                context_kv: dict,
                context_mask: tf.Tensor,
                training: bool = False) -> tf.Tensor:
        """
        训练模式: 并行处理 [BOS, s_1, s_2] → 输出三个 token 的隐状态
        """

    def step(self,
             x_t: tf.Tensor,
             cur_beam: int,
             context_kv: dict,
             context_mask: tf.Tensor,
             cache: dict,
             training: bool = False) -> tuple:
        """
        推理模式: 逐步解码
        """
```

### 6.2 实现逻辑

```
LazyDecoderModel.step:
1. for i, layer in enumerate(self.layers):
     x_t, cache = layer.step(x_t, cur_beam, i, context_kv, context_mask, cache, training)
2. return x_t, cache

# 与现有 DecoderModel.step 的关键差异:
# - 传入 context_kv 而非 enc_output
# - 传入 context_mask 而非 src_mask
# - 每层内部使用 LazyCrossAttention 而非 multi_head_attention
```

### 6.3 单元测试要求

1. **层数正确**：验证 n_layer 个 LazyDecoderLayer 被创建
2. **顺序传播**：验证数据通过所有层依次传播
3. **Cache 累积**：经过 N 层后，cache 中应有 N 组 self-attention KV

---

## 7. 模块 5：LazyMultiInterestModel（训练）

### 7.1 精确函数签名

```python
class LazyMultiInterestModel:
    def __init__(self,
                 feature_emb_dict: dict,
                 feature_emb_size_dict: dict,
                 d_model: int = 1792,
                 n_layer: int = 18,
                 n_head: int = 14,
                 g_kv: int = 14,
                 d_head: int = 128,
                 l_kv: int = 1,
                 s_kv: int = 1,
                 hidden_dim: int = 7168,
                 vocab_sizes: list = [8192, 8192, 8192],
                 dropout_rate: float = 0.1,
                 print_ops: list = None):
        """
        OneRec-V2 Lazy Decoder-Only 推荐模型

        Args:
            d_model: 模型隐藏维度（论文 Table 5: 1B=1792, 0.5B=1408）
            n_layer: decoder 层数（1B=18, 0.5B=14）
            n_head: query 头数（1B=18, 0.5B=14）
            g_kv: GQA KV 组数（默认=n_head，可设为更小值如 1,2,7）
            d_head: 每头维度 = d_model // n_head
            l_kv: context KV 层数（默认 1）
            s_kv: KV 分离系数（默认 1，共享）
            hidden_dim: FFN 隐藏维度（通常 4*d_model）
        """

    def model(self,
              photo_sid: tf.Tensor,            # [B, 3] 全局 ID
              label: tf.Tensor,                # [B, 3] 局部 ID
              photo_semantic_id_int: tf.Tensor  # [B] loss mask
              ) -> tuple:
        """
        训练前向传播

        论文 Eq.(5): h^(0) = Embed([BOS, s_1, s_2])
        论文 Eq.(6)-(8): Lazy Decoder Blocks
        论文 Eq.(10): L_Gen = -1/3 * sum log p(s_i | ...)
        """

    def _build_context(self,
                       feature_emb_dict: dict,
                       feature_emb_size_dict: dict,
                       training: bool = False) -> tuple:
        """
        构建 Context Processor 输入

        替代现有代码中的 encoder_input 构建逻辑:
        - 用户静态特征 → user_static_emb  [B, 1, d_input]
        - 短期点击行为 → short_term_emb   [B, T_short, d_input]
        - 长期行为 → long_term_emb        [B, T_long, d_input]

        Returns:
            (user_static_emb, short_term_emb, long_term_emb,
             short_term_len, long_term_len)
        """
```

### 7.2 实现逻辑

```
LazyMultiInterestModel.model:
1. 构建三路 Context 输入:
   user_static_emb, short_term_emb, long_term_emb = _build_context(...)
   # 替代现有的 user_static_emb + user_click_emb + encoder_input

2. Context Processor:
   context_processor = ContextProcessor(d_model, n_head, g_kv, l_kv, s_kv, n_layer, ...)
   context_kv = context_processor.forward(user_static_emb, short_term_emb, long_term_emb, training)
   context_mask = context_processor.get_context_mask(...)

3. Decoder 输入:
   # 论文 Eq.(5): h^(0) = Embed([BOS, s_1, s_2])
   start_token = tf.constant(total_vocab_size)
   photo_with_bos = tf.concat([start_token, photo_sid[:, :2]], axis=1)  # [B, 3]
   decoder_input = tf.nn.embedding_lookup(vocab_embedding, photo_with_bos)  # [B, 3, d_model]

4. Lazy Decoder:
   decoder_model = LazyDecoderModel(n_layer, d_model, n_head, g_kv, d_head, hidden_dim, dropout_rate)
   decoder_output = decoder_model.forward(decoder_input, context_kv, context_mask, training)  # [B, 3, d_model]

5. 损失计算:
   # 论文 Eq.(10): L_Gen = -1/3 * sum
   for step in range(3):
     with variable_scope('proj_%d' % step):
       h = RMSNorm(decoder_output[:, step, :])  # position-specific RMSNorm
       logits = dense(h, vocab_sizes[step])
       loss_i = softmax_cross_entropy_with_logits(logits, labels[:, step])
   ntp_loss = mean(sum(losses) * loss_mask) / (sum(loss_mask) + eps)  # 注意: 论文用平均而非求和

6. PRM Loss:
   # 保持现有逻辑，但 encoder_output 改为 context_processor 的输出
   # PRM 仍然需要 encoder_output（context 的聚合表示）
   # 可使用 context_kv 中的最后一组 K/V 或对 context 做 pooling
```

### 7.3 与现有代码的关键差异

| 组件 | 现有代码 | Lazy Decoder |
|------|----------|-------------|
| Encoder | `layer_norm(encoder_input)` + `DecoderModel.forward(decoder_input, enc_output, src_mask)` | `ContextProcessor.forward()` → `LazyDecoderModel.forward(decoder_input, context_kv, context_mask)` |
| 特征处理 | `user_static_emb + user_click_emb` 拼接 | 三路独立: `static + short_term + long_term` |
| 归一化 | `layer_norm` (post-norm) | `rms_norm` (pre-norm) |
| Loss | `sum(loss_i)` | `mean(sum(loss_i))` 或按论文 `-1/3 * sum` |
| Decoder 层数 | 2 | 18 (1B) / 14 (0.5B) |
| Cross-Attn | 有 w_k/w_v | 无 w_k/w_v，使用预计算 KV |

### 7.4 单元测试要求

1. **端到端 shape**：输入 [B, 3] 的 SID，输出 [B] 的 loss
2. **无 encoder 变量**：验证不存在 encoder 相关的 trainable variable
3. **Context Processor KV 存在**：验证 context_kv 不为空
4. **参数量验证**：验证参数量符合预期（无 w_k/w_v 减少的参数）
5. **Loss 单调递减**：训练 100 步后 loss 应下降

---

## 8. 模块 6：Beam Search（推理）

### 8.1 精确函数签名

```python
def beam_search_lazy(self,
                     beam_size: int = 512,
                     temperature: float = 1.0) -> tuple:
    """
    Lazy Decoder beam search 推理

    与现有 beam_search_fast 的关键差异:
    1. 无 encoder_output tile — context KV 共享
    2. Cross-attention 使用预计算 KV，无需每层重新计算
    3. Cache 中仅含 self-attention KV + 固定的 context KV 引用

    Returns:
        gen_part_loc: [B, beam_size, 3] 推荐的语义 ID 局部序列
        probs: [B, beam_size] 每条路径的累积概率
    """
```

### 8.2 实现逻辑

```
1. 构建三路 Context + ContextProcessor (与训练相同)
   context_kv = context_processor.forward(...)
   context_mask = context_processor.get_context_mask(...)

2. 初始化 beam:
   seqs = [B, 1, 1]  # <START>
   dec_path_log_probs = zeros([B, 1])
   cache = {}  # self-attention KV only

3. 逐层解码:
   for step, V in enumerate(vocab_sizes):
     # embed 当前 token
     dec_in = embedding_lookup(seqs[:, :, -1])
     dec_in = reshape([B*cur_beam, 1, d_model])

     # decoder step (使用 context_kv 而非 enc_output)
     dec_out, cache = decoder_model.step(
         dec_in, cur_beam, context_kv, context_mask, cache)

     # PRM build_variables (保持变量创建顺序)
     if step == 0:
       prm_model.build_variables()

     # projection + beam 选择 (与现有逻辑相同)
     ...

4. 关键优化: context KV 不需要 tile 到 B*beam
   - lazy cross-attention 内部自动 tile K/V
   - 避免了现有代码中 encoder_output 的 [B*beam, enc_len, dim] 大张量
   - 这就是 Lazy Decoder 节省显存的核心原因
```

### 8.3 与现有 beam_search_fast 的显存对比

| 张量 | 现有代码 | Lazy Decoder |
|------|----------|-------------|
| encoder_output tile | `[B*beam*2, 512, 256]` ≈ 134M floats | **不需要** |
| src_mask tile | `[B*beam*2, 1, 1, 512]` ≈ 0.5M floats | `[B, 1, 1, 512]` 共享 |
| cross-attn K/V (每层) | `[B, 1, H, 512, Dh]` × 2 | `[B, 512, G_kv, Dh]` × 2 (一组，所有层共享) |
| self-attn K/V cache | 相同 | 相同 |

### 8.4 单元测试要求

1. **输出形状**：`gen_part_loc` = `[B, beam_size, 3]`，`probs` = `[B, beam_size]`
2. **无 encoder tile**：验证推理图中不创建 `[B*beam, ctx_len, d_model]` 的大张量
3. **Context KV 不变**：验证 context_kv 在 beam 选择后不改变
4. **概率单调递减**：beam 0 的概率 >= beam 1 的概率 >= ... >= beam beam_size-1
5. **PRM 兼容性**：PRM 仍然可以正常打分（如果启用）

---

## 9. 全量可配置参数汇总

### 9.1 Context Processor 配置

```python
# 在 LazyMultiInterestModel.__init__ 中传入
context_config = {
    "l_kv": 1,          # KV 层数 (1/3/9/18，Table 3)
    "s_kv": 1,          # KV 分离系数 (1=共享, 2=分离)
    "g_kv": 14,         # GQA 组数 (1/2/7/14，Table 4)
    "d_head": 128,      # 每头维度 = d_model // n_head
    "n_static": 1,      # 静态特征 token 数
    "max_short_len": 200,  # 短期行为最大长度
    "max_long_len": 300,   # 长期行为最大长度 (训练) / 2700 (线上)
}
```

### 9.2 Decoder Block 配置

```python
decoder_config = {
    "n_layer": 18,      # 层数 (Table 5)
    "d_model": 1792,    # 隐藏维度
    "n_head": 14,       # query 头数
    "hidden_dim": 7168, # FFN 隐藏维度 (4 * d_model)
    "dropout_rate": 0.1,
}
```

### 9.3 模型规模预设（Table 5）

```python
MODEL_PRESETS = {
    "0.1B": {"d_model": 640,  "n_layer": 12, "n_head": 10, "embed_dim": 325, "lr": 5e-4},
    "0.2B": {"d_model": 896,  "n_layer": 12, "n_head": 14, "embed_dim": 453, "lr": 3.54e-4},
    "0.5B": {"d_model": 1408, "n_layer": 14, "n_head": 11, "embed_dim": 702, "lr": 2.24e-4},
    "1B":   {"d_model": 1792, "n_layer": 18, "n_head": 14, "embed_dim": 890, "lr": 1.58e-4},
    "2B":   {"d_model": 2304, "n_layer": 22, "n_head": 18, "embed_dim": 1151, "lr": 1.12e-4},
    "4B":   {"d_model": 2944, "n_layer": 26, "n_head": 23, "embed_dim": 1472, "lr": 7.91e-5},
    "8B":   {"d_model": 3584, "n_layer": 34, "n_head": 28, "embed_dim": 1792, "lr": 5.59e-5},
}
```

### 9.4 训练配置

```python
training_config = {
    "vocab_sizes": [8192, 8192, 8192],   # 三层语义 ID 词汇表
    "loss_type": "average",              # 论文 Eq.(10): 1/3 * sum (非 V1 的 sum)
    "use_logq_correction": True,        # PRM in-batch 负采样纠偏
    "prm_temperature": 1.0,
    "ntp_temperature": 1.0,
}
```

### 9.5 推理配置

```python
inference_config = {
    "beam_size": 512,
    "temperature": 1.0,
    "use_prm": True,                    # 是否启用 PRM 打分
    "prm_candidate_size": 1024,         # beam_size * 2
    "context_len": 3000,                # 线上 context 长度 (vs 训练时 512)
}
```

---

## 10. 单元测试要求

### 10.1 测试文件结构

```
test_lazy_decoder.py
├── test_rms_norm()
├── test_context_processor()
├── test_lazy_cross_attention()
├── test_lazy_decoder_layer()
├── test_lazy_decoder_model()
├── test_lazy_model_forward()
├── test_beam_search_lazy()
├── test_gqa_repeat_kv()
├── test_kv_sharing()
└── test_parameter_count()
```

### 10.2 关键数值验证

1. **RMSNorm vs LayerNorm**：同一输入，RMSNorm 输出 != LayerNorm 输出
2. **Context Processor KV 形状**：`[B, total_len, G_kv, d_head]`
3. **Lazy Cross-Attention 参数量**：`2 * d_model * d_model`（w_q + w_o），而非 `4 * d_model * d_model`
4. **总参数量**：1B 模型的参数量应接近 1B
5. **FLOPs 估算**：1B Lazy Decoder 应约为 18.9 GFLOPs（论文 Table 2）
6. **Loss 平均 vs 求和**：论文 V2 用平均（1/3），V1 用求和

### 10.3 集成测试

1. **训练→推理一致性**：训练模型保存的变量，推理模型能正确加载
2. **Dense 变量顺序**：训练图和推理图的 dense 变量创建顺序必须一致
3. **Beam search 正确性**：beam_size=1 时结果等同于贪心解码
4. **PRM 打分正确性**：step >= 1 时 PRM 正确参与排序
