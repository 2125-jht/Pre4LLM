# OneRec-V2 Lazy Decoder-Only 架构 — 技术分析与实施对照表

## 1. 论文核心技术组件分解

### 1.1 Context Processor

**论文定义**（Section 2.2.1）：
- 将异构用户特征（用户画像 + 短期行为 + 长期行为）拼接为统一 context 序列
- 每项 context 元素处理到统一维度：`d_context = S_kv * L_kv * G_kv * d_head`
- Context 沿特征维度分割为 `L_kv` 组 key-value 对
- 每层 l 的 K/V 通过 RMSNorm 归一化
- 当 `S_kv = 1` 时 k = v（共享表示），`S_kv = 2` 时 k/v 分别归一化

**对应论文公式**：
- Eq.(1): `d_context = S_kv * L_kv * G_kv * d_head`
- Eq.(2): `Context = [C_0, C_1, ..., C_{S_kv*L_kv-1}]`
- Eq.(3): `k_l = RMSNorm_{k,l}(C_{l*S_kv})`
- Eq.(4): `v_l = RMSNorm_{v,l}(C_{l*S_kv+1})` 或 `v_l = k_l`（当 S_kv=1）

**现有代码对应**：
- `model.py` L86-118: 用户静态特征 + 点击行为拼接 → `encoder_input`
- `model.py` L159: `layer_norm(encoder_input, scope="enc_ln")` → 这是 encoder 的层归一化
- **缺失**：Context Processor 的分片逻辑、RMSNorm、per-layer KV 生成

**核心差异**：
| 组件 | 现有代码 | OneRec-V2 |
|------|----------|-----------|
| 特征处理 | MLP 映射后直接拼接 | 三条 Pathway (Static/Short-term/Long-term) 独立 Linear |
| 归一化 | LayerNorm | RMSNorm（无 beta，无均值中心化） |
| KV 生成 | 每层独立做 K/V 投影（cross-attention 的 w_k/w_v） | Context Processor 一次性预计算所有层的 KV |
| KV 共享 | 无（每层各自计算） | 多层共享同一组 KV（每 N_layer/L_kv 层共享一组） |

### 1.2 Lazy Decoder Block

**论文定义**（Section 2.2.2）：
- Tokenizer: 3 个 semantic ID，训练时用前 2 个 + BOS，推理时逐步生成
- Block 结构：Cross-Attention → Self-Attention → FFN（论文顺序与现有代码不同！）
- Lazy Cross-Attention：无 K/V 投影，直接使用 Context Processor 输出的 KV
- GQA 支持：query heads = H_q，key-value groups = G_kv

**对应论文公式**：
- Eq.(5): `h^(0) = Embed([BOS, s_1, s_2])`
- Eq.(6): `h_cross^(l) = h^(l-1) + CrossAttn(RMSNorm(h^(l-1)), k_{l_kv}, v_{l_kv})`
- Eq.(7): `h_self^(l) = h_cross^(l) + SelfAttn(RMSNorm(h_cross^(l)))`
- Eq.(8): `h^(l) = h_self^(l) + FFN^(l)(RMSNorm(h_self^(l)))`
- Eq.(9): `l_kv = floor(l * L_kv / N_layer)`

**现有代码对应**：
- `modulesV2.py` DecoderLayer: Self-Attention → Cross-Attention → FFN（顺序不同！）
- `modulesV2.py` DecoderLayer.step: 有 self-attention cache + cross-attention cache
- `modulesV2.py` multi_head_attention: 有 w_k/w_v 投影

**核心差异**：
| 组件 | 现有代码 | OneRec-V2 |
|------|----------|-----------|
| Block 内操作顺序 | Self-Attn → Cross-Attn → FFN | Cross-Attn → Self-Attn → FFN |
| Cross-Attn K/V 来源 | 每层独立投影 encoder_output | 直接使用 Context Processor 预计算 KV，无 w_k/w_v |
| 归一化 | LayerNorm（post-norm） | RMSNorm（pre-norm） |
| 残差连接 | Post-norm 式 | Pre-norm 式 |
| GQA | 无 | 支持 G_kv < H_q |
| 输出投影 | 有 w_o | 有 w_o（仅 Q 有投影） |

### 1.3 Output Layer

**论文定义**（Section 2.2.2 末尾）：
- 最后一个 decoder block 的 hidden state 经 position-specific RMSNorm + Linear 预测每个 semantic ID
- 训练 loss：`L_Gen = -1/3 * sum_{i=1}^{3} log p(s_i | BOS, s_{<i}, Context)`

**现有代码对应**：
- `model.py` L247-283: 逐层 proj + softmax + cross-entropy loss
- 基本一致，但需将 LayerNorm 改为 RMSNorm

### 1.4 推理：Beam Search

**论文隐含的推理流程**：
- Context Processor 一次性预计算 KV
- Decoder 逐步生成 s_1, s_2, s_3
- Cross-attention KV 是固定的（来自 context processor）
- Self-attention KV 随生成步数增长

**现有代码对应**：
- `model.py` beam_search_fast / beam_search_fast_no_prm
- Cache 机制已实现（self-attention KV cache + cross-attention KV cache）
- **关键差异**：现有 cross-attention 每层有独立 K/V，lazy decoder 中 K/V 是共享的

## 2. 可配置参数清单

### 2.1 Context Processor 参数

| 参数名 | 论文符号 | 默认值 | 说明 |
|--------|----------|--------|------|
| `L_kv` | L_kv | 1 | context KV 层数（每 N_layer/L_kv 层共享一组 KV） |
| `S_kv` | S_kv | 1 | KV 分离系数（1=共享，2=分离） |
| `G_kv` | G_kv | N_head | key-value head 组数（GQA） |
| `d_head` | d_head | d_model/N_head | 每个注意力头的维度 |
| `context_dim` | d_context | S_kv*L_kv*G_kv*d_head | context 元素统一维度 |
| `n_static_features` | N_s | 1 | 用户静态特征 token 数 |
| `T_short` | T_short | 可变 | 短期行为序列长度 |
| `T_long` | T_long | 可变 | 长期行为序列长度 |

### 2.2 Lazy Decoder Block 参数

| 参数名 | 论文符号 | 默认值 | 说明 |
|--------|----------|--------|------|
| `N_layer` | N_layer | 18 (1B) / 14 (0.5B) | decoder 层数 |
| `d_model` | d_model | 1792 (1B) / 1408 (0.5B) | 模型隐藏维度 |
| `N_head` | H_q | 18 (1B) / 14 (0.5B) | query 头数 |
| `hidden_dim` | — | 4*d_model | FFN 隐藏维度 |
| `dropout_rate` | — | 0.1 | dropout 率 |

### 2.3 模型规模配置（论文 Table 5）

| 参数规模 | d_model | N_layer | N_head | embed_dim | LR |
|----------|---------|---------|--------|-----------|----|
| 0.1B | 640 | 12 | 10 | 325 | 5e-4 |
| 0.2B | 896 | 12 | 14 | 453 | 3.54e-4 |
| 0.5B | 1408 | 14 | 11 | 702 | 2.24e-4 |
| 1B | 1792 | 18 | 14 | 890 | 1.58e-4 |
| 2B | 2304 | 22 | 18 | 1151 | 1.12e-4 |
| 4B | 2944 | 26 | 23 | 1472 | 7.91e-5 |
| 8B | 3584 | 34 | 28 | 1792 | 5.59e-5 |

## 3. 文件修改与新增模块清单

### 3.1 需修改的文件

| 文件 | 修改内容 | 影响范围 |
|------|----------|----------|
| `modulesV2.py` | 新增 RMSNorm、LazyCrossAttention、LazyDecoderLayer、LazyDecoderModel、ContextProcessor | 核心模块 |
| `model.py` | 替换 MultiInterestModel 的 forward/beam_search 逻辑 | 模型主逻辑 |
| `feature_attr_extract.py` | 新增 long-term behavior pathway 配置 | 特征工程 |
| `kai_v2_model.py` | 适配新模型接口 | 训练/推理脚本 |

### 3.2 需新增的模块

| 模块 | 所在文件 | 功能 |
|------|----------|------|
| `RMSNorm` | `modulesV2.py` | Root Mean Square 归一化（替代 LayerNorm） |
| `ContextProcessor` | `modulesV2.py` | 三路特征整合 + 预计算分层 KV |
| `LazyCrossAttention` | `modulesV2.py` | 无 K/V 投影的跨注意力 + GQA |
| `LazyDecoderLayer` | `modulesV2.py` | Cross-Attn → Self-Attn → FFN 的 pre-norm block |
| `LazyDecoderModel` | `modulesV2.py` | 多层 LazyDecoderLayer 堆叠 |
| `LazyMultiInterestModel` | `model.py` | 新模型类，替代原 MultiInterestModel |
