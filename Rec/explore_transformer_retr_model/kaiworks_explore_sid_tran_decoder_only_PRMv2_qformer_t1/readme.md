
256his 512dim 4layers LayerNorm+PostNorm (Decoder-Only + PRM)

## 基线

- based on v3_mask_256_no_enc_8layers

## 模型结构变更

- 改为512dim
- 改回4layers
- 保持去掉enc（Decoder-Only）
- 改回layernorm以及postnorm形式

## 本次修改（PRM 残差连接 + LayerNorm）

### 问题
`PRMModel` 的 forward 流程原来没有残差连接和归一化，直接 cross-attention → MLP → score：
```
target_attn = cross_attention(target_embedding, hidden_states, ...)
score = mlp(target_attn, ...)
```

### 改动
给 PRM 的 cross-attention 输出加上残差连接 + LayerNorm（Post-Norm），与 DecoderLayer 保持一致：
```
out = layer_norm(target_embedding + target_attn, scope="cross_atten_ln")
score = mlp(out, ...)
```

### 涉及文件

| 文件 | 改动 |
|---|---|
| `modulesV2.py` | `PRMModel.forward()`: 在 cross-attention 输出后加 `layer_norm(target_embedding + target_attn, scope="cross_atten_ln")` |
| `modulesV2.py` | `PRMModel.build_variables()`: 新增 `prm_model/cross_atten_ln/layer_norm/gamma` 和 `beta` 变量声明 |
| `uni_retr_server_local_ann/predict/conf_gpu/dnn_model.yaml` | `param` 列表中新增 `prm_model/cross_atten_ln/layer_norm/gamma` 和 `beta` 两项（手动维护） |

### LayerNorm vs RMSNorm 选择

原论文用的 RMSNorm，但整个代码库统一使用 LayerNorm（DecoderLayer、model.py 等），且两者在模型规模下效果相当，因此直接复用已有的 `layer_norm()` 函数，无需新增 `rms_norm`。

### 热启动加载

- **新增 LayerNorm 参数**（gamma/beta）：旧 checkpoint 中不存在，会按 `init_range=0` 初始化（gamma=1, beta=0），训练几步即可收敛
- **Q/K/V/Wo 权重**：shape 不变（均为 `[dim, dim]`），正常加载
- **`dynamic_json_config_gpu.json`**：由 `dsl_gpu.py` 自动生成，无需手动修改，重新运行即可

### PRM heads 数

PRM 的 `num_heads` 保持与 decoder 一致，均为 8。曾考虑改为 1/4（2 heads），但因减少 head 数不减少计算量（总 FLOPs 只与 dim 有关），且语义变化需微调，暂不修改。待下次改，控制变量。

---

## PRM False Negative Mask（in-batch 同路径屏蔽）

### 问题

PRM 训练使用 in-batch 负采样（B=128），`prm_logits[i, j]` = PRM(user_i, path_j)，对角线为正样本。

当 batch 内用户 i 和用户 j 点击了**同一个视频**（即路径完全相同）时，`path_i == path_j`，此时 `prm_logits[i, j]` 的输入与正样本 `prm_logits[i, i]` 的 path 输入完全一致，本质上是把正样本当作负样本来压低，即 **false negative**。

通过 TensorBoard 中 `prm/path_freq_0/1/2` 的分布可以观察到：
- 三层分布几乎一样（因为路径碰撞由「是否点了同一个视频」决定，三层同时碰撞）
- 存在明显的双峰：freq ≈ 1-5（长尾视频）和 freq ≈ 40-65（热门视频）

这意味着约 31-51% 的 batch 样本可能指向同一条热门路径，false negative 非常严重。这类样本的 PRM loss 下界为 log(freq)（约 3.9），无论训多久都降不下去，已有的 logQ 纠偏对此无效。

### 改动

在 `model.py` 的 PRM loss 计算段（`for step in range(len(self._vocab_sizes))` 循环内），将 path_hash 计算提前，并在 logQ 纠偏之前加入 false negative mask：

1. **提前计算 path_hash**（与 logQ 复用同一份 hash，无额外开销）
2. **构造 same_path 矩阵** `[B, B]`，标记 batch 内同路径的所有 (i, j) 对
3. **去掉对角线**（对角线是正样本，必须保留），得到 `false_neg_mask`
4. **把 false negative 位置的 logit 设为 -1e9**，令其对 softmax 无贡献
5. **加监控指标** `prm/valid_neg_count_0/1/2`，记录每行 mask 后剩余的真负样本数

logQ 纠偏在 mask 之后继续执行，已屏蔽位置（-1e9）减去 logQ 后仍极小，不影响 softmax。

### 涉及文件

| 文件 | 改动位置 | 改动说明 |
|---|---|---|
| `model.py` | `model()` 方法，`# === 7-A. PRM loss ===` 段，原 248-265 行 | 重构 path_hash 计算位置，新增 false_neg_mask 屏蔽逻辑，新增 `prm/valid_neg_count_%d` 监控 |

### 新增监控指标

| 指标 | 含义 | 健康范围 |
|---|---|---|
| `prm/valid_neg_count_0/1/2` | 每行 mask 后剩余的真负样本数（均值） | 越接近 B-1=127 越好；若长期低于 50，需考虑调整数据采样策略 |
| `prm/path_freq_0/1/2` | 每个样本所在路径的 batch 内出现频率 | 理想为接近 1；均值越低越好 |

---

## PRM K/V 投影顺序优化（Level 1：先投影再 tile）

### 问题

PRM 的 cross-attention 中，K/V 投影的输入 `hidden_states`（encoder_output）在所有 pair / cand 上是共享的，但原实现先 tile 再投影：

- 训练侧：`pair_encoder_output = tf.tile(encoder_output, [B, B, ...])` 得到 `[B², L_enc, dim]`，再走 `w_k/w_v` 投影
- 推理侧：`prm_encoder_output = tf.tile(enc_out_base, [B, cand, ...])` 得到 `[B×cand, L_enc, dim]`，再走 `w_k/w_v` 投影

这导致同一份 `encoder_output[i]` 的 K/V 投影被重复执行 B 次（训练）/ cand 次（推理），是 PRM FLOPs 的绝对大头。

定量估算（dim=256, L_enc=201, H=8, num_heads=8）：

| 场景 | K/V 投影 FLOPs | 占 PRM 总量 |
|---|---|---|
| 训练 B=512（3 个 step） | 2 × 3 × B² × L × dim² ≈ 20.8 T | ~99% |
| 推理 B=1, cand=1024（2 个 step） | 2 × 2 × cand × L × dim² ≈ 54 G | ~95% |

### 改动（Level 1：先投影再 tile，仅省 FLOPs）

核心思路：把 K/V 投影移到 tile 之前，让 K/V 投影只对 `[B, L_enc, dim]` 做一次，得到 `[B, H, L_enc, Dh]`，再 tile 到所有 pair。

- 训练侧：step 循环前调 `project_kv(encoder_output)` 一次，得到 `prm_K, prm_V`；step 循环内 tile 到 `[B², H, L_enc, Dh]` 后调 `forward_with_kv`
- 推理侧：step==0 调 `project_kv(enc_out_base)` 一次；step>=1 tile 到 `[B×cand, H, L_enc, Dh]` 后调 `forward_with_kv`

### 涉及文件

| 文件 | 改动位置 | 改动说明 |
|---|---|---|
| `modulesV2.py` | L74-119 新增函数 `multi_head_attention_with_kv` | 与 `multi_head_attention` 一致，但跳过 w_k/w_v 投影，直接用传入的 `K_pre/V_pre`（已 split_heads） |
| `modulesV2.py` | L418-437 `PRMModel.project_kv` | 内部先调 `self.build_variables()` 固定 dense bin 顺序，再用 `w_k/w_v`（reuse）投影并 split_heads，返回 `[B, H, L, Dh]` |
| `modulesV2.py` | L439-465 `PRMModel.forward_with_kv` | Q 在内部用 w_q 算，K/V 用传入值，其余与 `forward` 一致 |
| `model.py` | `model()` 方法 L217-248（`# === 7-A. PRM loss ===` 段） | step 循环前调 `project_kv(encoder_output)` 得到 `prm_K/prm_V`；循环内 K/V tile 到 B² 后调 `forward_with_kv`；删除原 `pair_encoder_output` 的 tile |
| `model.py` | `beam_search_fast()` 方法 L458-622（4 处改动） | ① L458-463 初始化 `prm_K/prm_V/prm_K_tiled/prm_V_tiled` 为 None；② L504-505 step==0 调 `project_kv(enc_out_base)`；③ L543-573 step>=1 tile K/V 到 `[B×cand, H, L, Dh]`；④ L616-622 `forward` → `forward_with_kv` |

### 变量建图顺序对齐（关键）

原训练侧 PRM 变量在第一次 `forward` 时懒创建（原 L241），dense bin 顺序由 `multi_head_attention` 内部决定为 `[w_q, w_k, w_v, w_o]`。原推理侧在 step==0 调 `prm_model.build_variables()`（原 L466-467）显式建变量。改写后训练侧改为 `project_kv` 内部调 `build_variables`，与推理侧合并到 `project_kv` 里，**两边都通过 `project_kv` 触发变量创建**，建图顺序天然一致。

`project_kv` 内 `build_variables` 按 `[cross_atten_ln(gamma/beta), w_q, w_k, w_v, w_o, target_score_mlp_0/1/2/final]` 顺序显式建全部 PRM 变量。后续 `tf.layers.dense(name="w_k/w_v/w_q", reuse=AUTO_REUSE)` 都复用，不再新建。这同时修复了原训练侧 `build_variables()` 被注释掉（原 L219）的潜在隐患，使训练/推理建图顺序统一。

### 验证：dump 训练/推理图变量顺序对比

在 `test_model_flop.py` 新增 `dump_prm_variable_order(graph, tag)` 函数（L73-86），打印图中所有 `prm_model/*` 变量的名字、shape 和顺序。训练图 session 末尾（L197）和推理图 session 末尾（L240）各调用一次，输出示例：

```
===== PRM Variable Order [train] =====
  [ 0] prm_model/cross_atten_ln/layer_norm/gamma:0        (256,)
  [ 1] prm_model/cross_atten_ln/layer_norm/beta:0         (256,)
  [ 2] prm_model/multi_head_attention/w_q/kernel:0        (256, 256)
  [ 3] prm_model/multi_head_attention/w_k/kernel:0        (256, 256)
  [ 4] prm_model/multi_head_attention/w_v/kernel:0        (256, 256)
  [ 5] prm_model/multi_head_attention/w_o/kernel:0        (256, 256)
  [ 6] prm_model/multi_head_attention/w_o/bias:0          (256,)
  [ 7] prm_model/target_score_mlp/target_score_mlp_0/kernel:0  (256, 256)
  ...
===== PRM Variable Order [infer] =====
  (同上，顺序应完全一致)
```

**上线检查**：两个 tag 下变量顺序必须完全一致（名字、shape、顺序）。若顺序不一致，线上 dense bin 加载权重会错位，需调整 `build_variables` 的调用时机。

### 预期收益

| 场景 | 原始 PRM FLOPs | Level 1 后 | 降幅 |
|---|---|---|---|
| 训练 B=512 | ~20.8 T | ~422 G | ~50× |
| 推理 B=1, cand=1024 | ~55 G | ~0.83 G | ~66× |

### Level 1 的局限：K/V tile 仍占显存

Level 1 只省 FLOPs，K/V 仍 tile 到 `[B², H, L, Dh]` / `[B×cand, H, L, Dh]`：

| 训练 batch | K/V tile 显存 | 可行性 |
|---|---|---|
| B=64 | ~860 MB | 可行 |
| B=128 | ~3.4 GB | 紧张 |
| B=256 | ~13.7 GB | 风险 |
| B=512 | ~54 GB | OOM |

若实际训练 B 需要更大，Level 1 不可用，需切换 Level 2。

---

## 下一步可改进：Level 2（einsum 替代 tile+matmul）

### 动机

Level 1 在训练大 batch 下 K/V tile 仍会 OOM。Level 2 用 einsum 把「pair (i, j) 用 K[i]」的语义直接表达，K/V 不 tile，同时省 FLOPs 和 K/V 显存。

### 改法

不做 tile，用 einsum 替代 matmul。K shape 保持 `[B, H, L_enc, Dh]`，Q reshape 成 `[B, B, H, 1, Dh]`（i 维对应用户，j 维对应样本）：

```
attn_scores = tf.einsum('ijhd,ihld->ijhl', Q, K)          # 等价于 QKᵀ，[B, B, H, 1, L_enc]
attn_weights = tf.nn.softmax(attn_scores / sqrt(Dh), axis=-1)
context = tf.einsum('ijhl,ihld->ijhd', attn_weights, V)  # 等价于 attn @ V
context = tf.reshape(context, [B*B, H, 1, Dh])
```

推理侧把 B 换成 cand，B(j) 换成 cand，逻辑相同。

### 收益对比

| 场景 | Level 1 | Level 2 | Level 2 K/V 显存 |
|---|---|---|---|
| 训练 B=512 | 422 G FLOPs（K/V tile 54 GB OOM） | 422 G FLOPs | 1 MB |
| 推理 B=1, cand=1024 | 0.83 G FLOPs（K/V tile 108 MB） | 0.83 G FLOPs | 1 MB |

Level 2 在两个场景下 FLOPs 与 Level 1 相同，但 K/V 显存从 54 GB / 108 MB 降到 1 MB。

### 切换 Level 2 的判断标准

Level 1 实施后跑 `test_model_flop.py`，对照以下四条，满足任一即应升级 Level 2：

1. **训练 OOM**：训练 batch 无法提到 ≥64，说明 K/V tile 已是显存瓶颈（硬性触发）
2. **FLOPs 下降 < 10×**：预期 PRM FLOPs 降 ~50×，若实测降不到 10×，说明 XLA 已部分融合 K/V 投影，瓶颈不在这，需重新分析
3. **wall time 不降**：FLOPs 降但 wall time 没降，说明瓶颈是 memory-bound（attn score 的 `[B², H, 1, L]` softmax），Level 2 也救不了 attn weights，需另想方案（分块 softmax）
4. **推理显存仍紧张但不是 K/V**：Level 1 推理 K/V tile ~108 MB 不大，但 attn weights `[B×cand, H, 1, L]` ~210 MB 仍在；若 OOM 主因是 attn weights，Level 2 也救不了

第 1 条是硬性触发，第 2-4 条是「收益不达预期」的信号。建议改完后用 `test_model_flop.py` 跑训练 B=1/64/256 三档 + 推理 beam=512，记录 FLOPs、显存峰值、wall time，对照判断。

### Level 2 的额外复杂度

- 需重写 cross-attention，不再走 `multi_head_attention` 通用路径
- einsum 表达式需严格验证语义：`pair (i, j)` 表示用户 i 对样本 j 路径打分，K 用 i 维，Q 用 ij 两维，对应 `ijhd,ihld->ijhl`
- 训练/推理变量建图顺序仍需对齐（与 Level 1 一致）

详细决策记录见 `.codeflicker/discuss/2026-07-27/prm-kv-projection-order/`。

---

## QFormer 压缩 user context（阶段4优化）

### 背景与动机

方法B（gaiprm）的推理耗时瓶颈在于：Decoder 和 PRM 的 cross-attention 每步都要对 `encoder_output [B, 201, D]` 做 K/V 投影并 beam/candidate 维度 tile，造成：

- **Decoder cross-attn**（beam=512，step×2）：`~100G FLOPs`
- **PRM cross-attn**（cand=1024，step×2）：`~136G FLOPs`
- **合计**：`~236G FLOPs`（Tk=201 是关键因子）

### 架构设计思路（t1 版本：K=1，静态特征直通）

**本目录当前为 K=1 版本**：行为序列压缩为 1 个画像 token，静态特征保留 1 个 token，最终 user context 为 2 个 token。作为 K=4（多兴趣）的对照基线，用于衡量多兴趣设计的增益。

**关键设计：静态特征（1 token）直接保留，只对行为特征（200 tokens）做 QFormer。**

```
user_static（性别/年龄/等级）→ MLP → [B, 1, D]  ← 身份锚点，不经 QFormer，直接保留
user_click（200 条点击行为）→ MLP → [B, 200, D]
                                  ↓ QFormer（4层，K=1 个 learnable query）
                            [B, 1, D]  ← 单个行为画像摘要

enc_compressed = concat([static, portrait]) = [B, 1+1, D] = [B, 2, D]
      ↓ Decoder cross-attn（Tk: 201 → 2）
      ↓ PRM cross-attn（Tk: 201 → 2）
```

**这样设计的理由：**

| 特征 | 语义 | 信息密度 | 处理方式 |
|---|---|---|---|
| 1 个静态 token（user_id/gender/age/level） | "我是谁"（用户身份） | 高，不应被压缩 | **直接保留** |
| 200 个行为 token（点击序列） | "我喜欢什么"（用户兴趣） | 有冗余，需提炼 | **QFormer → 1 个兴趣画像** |

相比 K=1 把 201 tokens 全部压缩为 1 个，新方案：
- 静态特征**零损失**传递给下游
- 1 个 query 专注于从 200 条行为中提炼**单个兴趣画像**，任务更清晰
- 有效信息量从 1 token 提升到 2 tokens（静态 1 + 画像 1），语义分工明确

QFormer 每层只有 cross-attention + FFN，无 self-attention（比 DecoderLayer 轻），一次性对全 B 样本压缩，不随 beam/cand 数量增加而增加开销。

> **K=1 vs K=4 对比**：K=1 无多 query 同质化问题，结构最简；K=4 在 t1_selfattn 目录试过（无 self-attn 同质化严重 >0.8），需加 self-attn 打破（见 t1_selfattn 目录）。本目录作为 K=1 基线，与 K=4 + self-attn 对照。

### 改动说明（t1 相对 t0 的变化）

#### `model.py`

| 位置 | 改动 |
|---|---|
| `__init__` | `self._qformer_query_num = 1`（K=1，单 query 压缩行为序列为 1 个画像 token） |
| `model()` | **行为序列**（`user_click_emb` [B,200,D]）单独做 LayerNorm + QFormer，得到 `click_portrait [B, 1, D]`；**静态特征**（`user_static_emb` [B,1,D]）直通，不经 QFormer；最终 `enc_compressed = concat([user_static_emb, click_portrait]) = [B, 2, D]`；同质化监控仅在 K>1 时计算（K=1 时跳过避免除零） |
| `beam_search_fast()` | 同上架构，推理侧一致 |
| `beam_search_fast_no_prm()` | 同上架构，推理侧一致 |

#### `modulesV2.py`（t0 已有，t1 复用）

| 组件 | 说明 |
|---|---|
| `rms_norm()` | RMSNorm 归一化（QFormer 内部 pre-norm 使用） |
| `QformerLayer` | 单层 QFormer：cross-attention(Q=learnable query, K/V=click_emb_normed) + FFN，无 self-attention |
| `QFormer` | 多层 QformerLayer 堆叠，将 `[B, 200, D]` 压缩为 `[B, K, D]` |

### 参数量说明

| 新增参数 | 数量 | 说明 |
|---|---|---|
| `qformer_queries` | `1 × 256 = 256` | 1 个 learnable query token（K=1）|
| QFormer **4**层参数 | 每层 ~4×256²+2×256² = ~786K | cross-attn 4个线性层 + FFN 2个线性层 + RMSNorm |
| **合计** | **~3.2M** | 4层 QFormer + 1个 query token |

### 预期收益

| 指标 | 原始（Tk=201） | QFormer 后（Tk=1+1=2） | 收益 |
|---|---|---|---|
| Decoder cross-attn FLOPs | ~100G | ~1G | ~**100×** |
| PRM cross-attn FLOPs | ~136G | ~1.4G | ~**100×** |
| 推理端到端耗时 | 基线 | 预期 **~30-100× 降低** | 待实测 |
| QFormer 一次性压缩开销 | — | ~4×B×200×256²×2 ≈ **4.4G** | 与 beam 无关，固定开销 |

> 相比 K=4（Tk=5），K=1（Tk=2）的 cross-attn FLOPs 进一步减半，但牺牲了多兴趣表达能力。

---

### 同质化监控说明（K=1 时自动跳过）

**核心问题**（K>1 时）：K 个 query token 经 QFormer 输出后，若相互之间的余弦相似度过高（趋向同质），则 K 个 token 实际上表达的是同一种兴趣，等效退化为 K=1，多样性优势消失。

**K=1 特例**：只有 1 个 query，不存在多 query 同质化问题。`model()` 的 `5-C` 步骤中已用 `if self._qformer_query_num > 1` 保护，K=1 时跳过监控计算，避免 `num_pairs = K·(K-1) = 0` 除零产生 NaN。

#### K>1 时的监控指标

在 `model()` 的 `5-C` 步骤中，计算并上报（K>1 时）：

| TensorBoard 指标 | 计算方式 | 含义 |
|---|---|---|
| `qformer/query_pairwise_cos_sim` | K 个 query 输出的 pairwise cosine similarity 均值（去掉对角线） | 越低越好，反映 query 间多样性 |

**判断阈值：**

| 指标值范围 | 状态 | 建议 |
|---|---|---|
| < 0.3 | ✅ 健康 | query 间差异明显，多兴趣有效 |
| 0.3 ~ 0.6 | ⚠️ 可接受 | 一定同质化，暂可观察 |
| 0.6 ~ 0.8 | ⚠️ 偏高 | 建议考虑加 self-attention |
| > 0.8 | ❌ 严重同质化 | 需加 query self-attention 打破退化 |

#### 为什么 K>1 没有 self-attention 可能同质化

若 QFormerLayer 只有 **cross-attention + FFN**（无 query self-attention）：
- K 个 query 分别独立对同一个 `click_emb_normed` 做 cross-attention
- 由于没有 query 间的通信（self-attention），K 个 query 会倾向于 attend 到 encoder 里最显著的相同 token（如最近点击的热门 item）
- 结果：K 个 query 的输出趋于相同，信息量等效于 K=1

**实测验证**：在 t1_selfattn 目录用 K=4（无 self-attn）训练后 `qformer/query_pairwise_cos_sim` 持续 > 0.8，触发严重同质化阈值，需加 self-attention 打破。详见 t1_selfattn 目录的 readme。

#### 备用方案：加 self-attention

若 K>1 且 `qformer/query_pairwise_cos_sim` 持续 > 0.6，可在 `QformerLayer.forward()` 中在 cross-attention 之前加一步 query self-attention：

```python
# 在 cross-attention 之前先做 query 间 self-attention
x_norm_0 = rms_norm(x, scope="self_atten_ln")
self_attn_output = multi_head_attention(x_norm_0, x_norm_0, x_norm_0, num_heads, self_mask, ...)
x = x + self_attn_output
# 然后再做原有的 cross-attention + FFN
```

这样 K 个 query 在 attend encoder 之前先互相"商量"，自然会分工捕捉不同兴趣。

> **self-attn 复杂度增量**：cross-attn 是 O(K·L·D)，self-attn 是 O(K²·D)。K=4, L=200 时 self-attn 仅为 cross-attn 的 ~2%，几乎可忽略。参数量增加 ~1.05M（4 层 × 4×256²）。详见 t1_selfattn 目录。

#### 训练前期参考基线

以下是 query 相似度的量级参考（供 K>1 评估用）：
- **随机初始化刚开始训练**：各 query 独立随机初始化，相似度约 0.0~0.2
- **收敛后健康状态**：< 0.3，说明 query 自然分化出了不同兴趣偏好
- **退化状态**：> 0.8，说明 K 个 query 几乎学到了同一个"平均用户兴趣"

---

### LayerNorm 设计调整：static 加 LN，click 去掉外部 LN

#### 背景与问题

原设计中：
- 行为序列 `user_click_emb` 在送入 QFormer 前做了外部 LayerNorm（`enc_ln`）
- 静态特征 `user_static_emb` 不经 QFormer，也未做任何归一化
- 导致 `enc_compressed = concat([user_static_emb, click_portrait])` 中两个 token 尺度不平衡：
  - `click_portrait` 经 QFormer 内部多次归一化，输出 ~O(1)
  - `user_static_emb` 保持 MLP 原始输出尺度，可能偏离 O(1)
- 尺度不平衡会让下游 Decoder/PRM 的 cross-attention 被某一方主导

#### 设计调整

| 张量 | 原设计 | 新设计 | 理由 |
|---|---|---|---|
| `user_click_emb` | 外部 LayerNorm（`enc_ln`）→ QFormer | **直接送 QFormer**（无外部 LN） | QFormer 内部对 query 做 RMSNorm，K/V 侧靠 attention 的 `1/sqrt(d_k)` 缩放即可；去掉外部 LN 可保留 token 间幅度差异（兴趣强度/时近性信号），且 `light_more_final` 目录已验证此方案可行 |
| `user_static_emb` | 不归一化 | **加 LayerNorm（`static_ln`）** | static 不经 QFormer，需外部归一化保证与 `click_portrait` 尺度一致，使 `enc_compressed` 内两 token 公平竞争下游注意力 |

#### 改动位置

- `model.py` `model()` 方法：L107 后加 `static_ln`，L159-165 删除 `enc_ln`，L181-182 QFormer 输入改为 `user_click_emb`
- `model.py` `beam_search_fast()`：L423 后加 `static_ln`，L457-459 删除 `enc_ln`，L471-472 QFormer 输入改为 `user_click_emb`
- `model.py` `beam_search_fast_no_prm()`：L776 后加 `static_ln`，L810-811 删除 `enc_ln`，L822-823 QFormer 输入改为 `user_click_emb`
- 三处路径必须保持一致，避免训练/推理不一致

#### 潜在风险

| 风险 | 说明 | 缓解 |
|---|---|---|
| click_emb 尺度波动 | MLP 输出方差若很大，K/V 进入 attention 后 attention score 过大 | 加监控 `qformer/click_kv_l2_norm`（可选）；若 >10 再考虑加回 LayerNorm |
| static LN 抹掉重要维度 | LayerNorm 把 feature 维归一化，可能损失维度间相对大小 | `gamma/beta` 可学习，能恢复重要维度；static 只有 1 个 token，影响有限 |
| 与热启动 checkpoint 不匹配 | `static_ln` 的 gamma/beta 是新增变量，`enc_ln` 的 gamma/beta 变成孤立变量 | 热启动时 `static_ln` 随机初始化，`enc_ln` 可忽略（不影响其余参数加载） |

---

### QFormer 压缩保真度监控：portrait 与用户行为序列均值的对齐（K=1/K>1 通用）

#### 背景

QFormer 的职责是"压缩用户行为信息成画像"，不是"直接对齐某个 target"。因此 `interest_target_sim`（portrait vs 目标 item 偏好）低值不能直接判定 QFormer 没学好——它本来就不要求直接对齐 target。

更合适的诊断指标是**压缩保真度**：QFormer 把 200 个行为 token 压缩为 1 个画像 token，好的画像应该能"代表"用户行为序列的整体方向。因此监控 portrait 与序列有效 token 均值的余弦相似度。

#### 新增指标

在 `model()` 的 5-D2 段（尺度监控之后），计算 QFormer 输出 `click_portrait` 与用户行为序列有效 token 均值的余弦相似度：

| TensorBoard 指标 | 计算方式 | 含义 |
|---|---|---|
| `qformer/portrait_seq_mean_sim` | `cos_sim(click_portrait[:,0,:], masked_mean(user_click_emb, click_mask))` 的 batch 均值 | QFormer 画像与用户行为序列主方向的压缩保真度 |

**计算细节**：
- `click_mask` 为 0/1 mask（1=有效，0=padding），`click_mask_f = tf.cast(click_mask, tf.float32)`
- `valid_count = sum(click_mask_f, axis=-1, keepdims=True)`，每个样本有效 token 数
- `click_seq_sum = sum(user_click_emb * expand_dims(click_mask_f, -1), axis=1)`，只累加有效 token
- `click_seq_mean = click_seq_sum / (valid_count + 1e-9)`，有效 token 均值
- `portrait` 与 `click_seq_mean` 都 L2 归一化后求内积，得到逐样本余弦相似度
- 取 batch 均值上报

#### 判断阈值

| 训练阶段 | 期望值 | 说明 |
|---|---|---|
| 初始随机初始化 | ≈ 0 | QFormer 参数随机，输出与序列主方向无关 |
| 训练中期 | 单调上升 | QFormer 逐步学到保留序列主方向 |
| 收敛后 | 稳定在正区间（如 0.3~0.6） | 画像保留了序列的主要信息 |
| 异常：长期 ≈ 0 或为负 | ❌ 画像与序列主方向不对齐 | QFormer 可能学到了“平均用户画像”而非当前用户的序列信息，或丢失了信息 |

#### 与 `interest_target_sim` 的配合使用

| 场景 | `portrait_seq_mean_sim` | `interest_target_sim` | 诊断 |
|---|---|---|---|
| QFormer 是好的压缩器，下游负责对齐 target | 高 | 低 | ✅ 正常（本架构设计预期） |
| QFormer 丢失了序列信息 | 低 | 低 | ❌ QFormer 质量不足，需改进结构 |
| QFormer 奇特地直接对齐了 target | 低/高 | 高 | ⚠️ 可疑（可能与训练目标泄漏有关，或刚好目标 item 跟序列主方向一致） |
| 画像与 target 都对齐 | 高 | 高 | ✅ 理想状态 |

**核心观点**：`portrait_seq_mean_sim` 是评估 QFormer 压缩质量的**直接指标**；`interest_target_sim` 是**参考指标**。两者结合可较全面判断 QFormer 是否学到了有效的兴趣表示。

#### 为什么 `interest_target_sim` 仅作参考

QFormer 在本架构中是 **K/V 提供者**，不是 **target predictor**：
- `click_portrait` 作为 K/V 送给 Decoder/PRM，下游学的是"如何从 portrait 提取信息"，不是"让 portrait 对齐 target"
- 真实训练信号是 Decoder/PRM 通过 cross-attn 从 `enc_compressed` 提取信息来完成 NTP/PRM 任务
- 因此 `interest_target_sim` 低不能直接判定 QFormer 没用，应结合下游业务指标（`loss/ntp_loss`、`loss/prm_loss`、recall@k）一起判断

---

### QFormer 输出与目标 item 偏好对齐监控（K=1/K>1 通用）

#### 背景

`qformer/query_pairwise_cos_sim` 仅在 K>1 时计算（监控多 query 间同质化），K=1 时跳过。但 K=1 时仍需监控 QFormer 抽取的兴趣画像是否与目标 item 语义对齐，以判断 QFormer 是否学到了有效的兴趣表示。

#### 新增指标

在 `model()` 的 PRM 段（`sid_embeddings` 定义之后），计算 QFormer 输出 `click_portrait` 与目标 item 偏好向量（3 级语义 ID embedding 求和）的余弦相似度：

| TensorBoard 指标 | 计算方式 | 含义 |
|---|---|---|
| `qformer/interest_target_sim` | `cos_sim(click_portrait[:,0,:], sum(sid_embeddings, axis=1))` 的 batch 均值 | QFormer 兴趣画像与目标 item 偏好的对齐程度 |

#### 判断阈值

| 训练阶段 | 期望值 | 说明 |
|---|---|---|
| 初始随机初始化 | ≈ 0 | QFormer 参数随机，输出与目标无关 |
| 训练中期 | 单调上升 | QFormer 逐步学到与目标 item 对齐的兴趣 |
| 收敛后 | 稳定在正区间（如 0.1~0.4） | 兴趣画像与目标偏好方向一致 |
| 异常：长期 ≈ 0 或为负 | ❌ 未学到对齐 | QFormer 输出与目标 item 无关，需排查结构/训练问题 |

#### 尺度监控（配合 LayerNorm 调整）

为验证 LayerNorm 设计调整的效果，新增尺度监控：

| TensorBoard 指标 | 计算方式 | 含义 |
|---|---|---|
| `qformer/static_l2_norm` | `sqrt(mean(square(user_static_emb[:,0,:])))` 的 batch 均值 | 静态特征 L2 范数 |
| `qformer/portrait_l2_norm` | `sqrt(mean(square(click_portrait[:,0,:])))` 的 batch 均值 | QFormer 输出 L2 范数 |
| `qformer/static_portrait_norm_ratio` | `static_l2_norm / (portrait_l2_norm + 1e-9)` | 两者尺度比，期望 ≈ 1 |

**预期**：调整后 `static_portrait_norm_ratio` 应接近 1，表明 `enc_compressed` 内两 token 尺度平衡。

---

## 推理诊断指标（PRM 退化检测）

### 背景与问题

PRM 剪枝的有效性依赖 beam 候选的多样性。若绝大多数候选来自同一个父 beam，PRM 实际上是在同质候选里选优，区分力接近随机。通过以下指标可以检测这种退化。

### 新增诊断指标

在 `beam_search_fast` 的 step>=1 分支（PRM 打分之后）打印以下指标（通过 `tf.print` 输出，不影响推理逻辑）：

| 指标名 | 含义 | 计算方式 | 健康范围 |
|---|---|---|---|
| `beam_diag/parent_entropy_step1/2` | PRM 候选池中父 beam 分布的信息熵 | `-sum(p_parent * log(p_parent))`，`p_parent` = 各父 beam 被选为候选的频率 | 接近 `log(beam_size) = log(512) ≈ 6.24` 为好；< 1 说明候选来自少数父 beam（退化） |
| `beam_diag/prm_entropy_step1/2` | PRM logits 经 softmax 后的信息熵 | `-sum(prm_prob * log(prm_prob))` | 接近 `log(cand_size)` 说明 PRM 无法区分候选（打分近随机）；**过高**也是问题，理想范围 2~4 |
| `beam_diag/prm_top1_parent_ratio_step1/2` | PRM 保留的 beam_size 条路径中，来自同一父 beam 的最大占比 | `max(parent_count) / beam_size` | 越低越好；> 0.5 说明超半数最终 beam 来自 1 个父 beam（严重退化） |

### 指标解读方法

**PRM 有效的正常情况**：
- `parent_entropy` 接近 `log(beam_size)`（候选来自多个父 beam，decoder 有效扩展）
- `prm_entropy` 明显低于 `log(cand_size)`（PRM 能区分好坏候选，打分集中）
- `prm_top1_parent_ratio` < 0.3（最终 beam 来源分散）

**PRM 退化的信号**：
- `parent_entropy` < 1：decoder 候选池本身就退化了（all beams 来自 1~2 个父 beam），根本原因是 decoder 过于 peaky，与 PRM 无关
- `prm_entropy` 接近 `log(cand_size)`：PRM 打分几乎均匀，无法区分候选；可能原因是 user context 压缩导致 PRM 信息不足
- `prm_top1_parent_ratio` > 0.8：几乎所有 beam 来自同一个父节点，最终输出多样性极差

### 修改位置

`model.py` `beam_search_fast()` 方法，step>=1 分支，`prm_logits_scaled = prm_logits / prm_temperature` 之后、`prm_best_idx sort` 之前。

### 需重新训练（可热启动 Decoder/PRM 权重）

Decoder 和 PRM 的 `w_q/w_k/w_v/w_o` 等权重形状均为 `[dim, dim]`，与 encoder context 序列长度（Tk）**无关**，权重形状不受 Tk 从 201→1 的影响。因此可以：

- **从原 gaiprm checkpoint 热启动** Decoder 和 PRM 的全部参数（`decoder_layer_*/multi_head_attention/w_q` 等）
- **仅 `qformer_queries` 和 QFormer 参数**（`user_qformer/qformer_layer_*/...`）需要随机初始化

热启动可以显著加速 Decoder 和 PRM 的收敛，只有 QFormer 这部分约 1.6M 参数从头学习。

> **注意**：若不做热启动也没问题，从头训练即可，效果上限通常相同，只是收敛更慢。

---

## 正样本定义与 Loss 加权

### 正样本定义（`kai_v2_model.py` `filter_mask_wrapper`）

训练样本经过 `filter_mask_wrapper` 过滤，仅保留满足以下条件的样本。

#### v2（当前）：秒级分桶查表，`tf.gather` 实现

过滤逻辑由阈值表 `PLAYING_TIME_THRESHOLD_MS`（401 个元素，索引对应 duration 秒数 0..400）驱动：

```python
duration_sec = clip(floor(duration_ms / 1000), 0, 400)
threshold    = tf.gather(PLAYING_TIME_THRESHOLD_MS, duration_sec)
mask         = playing_time < threshold   # True → 丢弃
```

**阈值表生成规则：**

| duration 区间 | 阈值策略 | 数据来源 | 设计意图 |
|---|---|---|---|
| [0, 13) s | 固定 20s | 硬编码 | 强卡短视频；暂不让 <13s 视频进入正样本，后续视效果决定是否加分位数 |
| [13, 30) s | 大盘 p90 分位数 | Excel 分桶统计 | 短视频严格 |
| [30, 58) s | 大盘 p85 分位数 | Excel 分桶统计 | 中短视频较严格 |
| [58, 140) s | 大盘 p80 分位数 | Excel 分桶统计 | 中长视频正常 |
| [140, 150) s | 固定 85s | 硬编码 | 长视频梯度过渡 |

| [150, 160) s | 固定 90s | 硬编码 | 长视频梯度过渡 |
| [160, 170) s | 固定 95s | 硬编码 | 长视频梯度过渡 |
| [170, 400] s | 固定 100s | 硬编码 | 长视频宽松；高 duration 桶样本量小、分位数噪声大，用固定阈值更稳 |

精度：分位数阈值取自 Excel 并四舍五入到 0.1s（100ms）。
数据来源：`计算大盘分位数_Snippet 1_32938564.xlsx`，按 `duration_bucket` 列取对应秒级桶的 p80/p85/p90 值（单位 s）× 1000 转为 ms。

#### v1（已废弃）：粗粒度嵌套 tf.where

| duration 区间 | 播放时长阈值 | 说明 |
|---|---|---|
| duration < 5s | playing_time ≥ 5s | 绝对阈值 |
| 5s ≤ dur < 20s | playing_time ≥ 1.0 × duration | 完播 |
| 20s ≤ dur < 40s | playing_time ≥ 0.9 × duration | 90% 完播率 |
| 40s ≤ dur < 80s | playing_time ≥ 0.8 × duration | 80% 完播率 |
| 80s ≤ dur < 120s | playing_time ≥ 0.7 × duration | 70% 完播率 |
| duration ≥ 120s | playing_time ≥ 85s | 长视频绝对阈值 |

v1 已被 v2 替换，保留此处供回溯对比。

### 过滤监控指标（`filter/seg_*/pass_rate` & `filter/seg_*/sample_prop`）

`filter_mask_wrapper` 中同步上报每个 duration 段的 TensorBoard 指标，用于监控过滤效果和样本分布：

| TensorBoard 指标 | 含义 | 说明 |
|---|---|---|
| `filter/seg_0_13s/pass_rate` | 0-13s 视频正样本通过率 | 固定 20s 阈值；预期很低，接近 0 |
| `filter/seg_13_30s/pass_rate` | 13-30s 视频正样本通过率 | p90 过滤；约 10% 通过 |
| `filter/seg_30_58s/pass_rate` | 30-58s 视频正样本通过率 | p85 过滤；约 15% 通过 |
| `filter/seg_58_140s/pass_rate` | 58-140s 视频正样本通过率 | p80 过滤；约 20% 通过 |
| `filter/seg_140_150s/pass_rate` | 140-150s 视频正样本通过率 | 固定 85s |
| `filter/seg_150_160s/pass_rate` | 150-160s 视频正样本通过率 | 固定 90s |
| `filter/seg_160_170s/pass_rate` | 160-170s 视频正样本通过率 | 固定 95s |
| `filter/seg_170s_plus/pass_rate` | 170s+ 视频正样本通过率 | 固定 100s；相对宽松 |
| `filter/seg_0_13s/sample_prop` | batch 中 0-13s 视频占比（过滤前） | 监控短视频曝光分布 |
| `filter/seg_13_30s/sample_prop` | batch 中 13-30s 视频占比（过滤前） | — |
| `filter/seg_30_58s/sample_prop` | batch 中 30-58s 视频占比（过滤前） | — |
| `filter/seg_58_140s/sample_prop` | batch 中 58-140s 视频占比（过滤前） | — |
| `filter/seg_140_150s/sample_prop` | batch 中 140-150s 视频占比（过滤前） | — |
| `filter/seg_150_160s/sample_prop` | batch 中 150-160s 视频占比（过滤前） | — |
| `filter/seg_160_170s/sample_prop` | batch 中 160-170s 视频占比（过滤前） | — |
| `filter/seg_170s_plus/sample_prop` | batch 中 170s+ 视频占比（过滤前） | — |

**健康检查要点：**
- `seg_0_13s/pass_rate` 应接近 0（固定 20s 几乎卡死短视频）
- `seg_13_30s/pass_rate` ~ `seg_58_140s/pass_rate` 应随时长增加而升高（宽松度递增）
- `seg_140_150s` ~ `seg_170s_plus` 的 `pass_rate` 应递增（85s → 90s → 95s → 100s 阈值梯度）
- `sample_prop` 各段加和应为 1；若某段 `sample_prop` 极低，说明该时长视频原始曝光就少

### Loss 加权（`model.py` `model()` → `loss_mask`）

训练 loss 按**播放时长（playing_time）**进行对数加权，播放时间越长的样本获得更高权重：

```
weight = lg(2 + playing_time / 1000)     # lg = log₁₀
loss_mask = valid * weight
```

**加权示例：**

| playing_time | weight | 场景 |
|---|---|---|
| 0 ms | lg(2) ≈ 0.301 | 无播放（最低权重） |
| 5 s | lg(7) ≈ 0.845 | 短播放 |
| 30 s | lg(32) ≈ 1.505 | 中等播放 |
| 120 s | lg(122) ≈ 2.086 | 深度消费 |
| 600 s | lg(602) ≈ 2.780 | 极长播放 |

加权后的 NTP loss 和 PRM loss 计算方式：
```
ntp_loss = Σ(ntp_loss_i × loss_mask) / Σ(loss_mask)
prm_loss = Σ(prm_loss_i × loss_mask) / Σ(loss_mask)
total_loss = ntp_loss + prm_loss
```

### 设计动机

- **正样本定义（v2 查表）**：用大盘实测分位数替代人工比例阈值，粒度从 6 段粗分档细化到秒级，使阈值更贴近真实用户消费分布；短视频严格、长视频宽松的梯度策略，配合 playing_time loss 加权，共同引导模型偏好长播/深消费内容
- **Loss 加权**：用 playing_time（而非 duration_ms）加权，让模型更关注用户真正深度消费的样本，而非仅仅偏好长视频。一个 5 分钟视频播放 5s 的样本权重远低于一个 30s 视频完播的样本
- **双重效果叠加**：正样本过滤 + loss 加权同向作用，共同推动模型偏好长播视频；需注意避免偏置过强，线上需同时监控完播率、点击率等指标，防止单一维度过拟合
