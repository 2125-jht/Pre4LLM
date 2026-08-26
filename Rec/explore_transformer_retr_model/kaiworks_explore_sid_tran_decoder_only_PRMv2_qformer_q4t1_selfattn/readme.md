
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

### 架构设计思路（t1_selfattn 版本：K=4，静态特征直通 + Query Self-Attention）

**本目录是 t1 的结构修改版，与 t1 的唯一区别：QFormer 每层加了 query self-attention。**

**为何另起目录**：修改 `QformerLayer` 结构会新增 `self_attention/*` 变量，从 t1 checkpoint 热启动会报「参数不一致」错误。因此复制 t1 到 `t1_selfattn` 独立训练，原 t1 目录保持可热启动状态不变。

**触发原因**：t1 初版（无 self-attn）训练后 `qformer/query_pairwise_cos_sim` 持续 > 0.8，4 个 query 严重同质化，等效退化为 K=1，多兴趣设计失效。

**关键设计：静态特征（1 token）直接保留，只对行为特征（200 tokens）做 QFormer。**

```
user_static（性别/年龄/等级）→ MLP → [B, 1, D]  ← 身份锚点，不经 QFormer，直接保留
user_click（200 条点击行为）→ MLP → [B, 200, D]
                                  ↓ QFormer（4层，K=4 个 learnable query）
                            [B, K, D]  ← 多兴趣行为画像

enc_compressed = concat([static, portrait]) = [B, 1+K, D] = [B, 5, D]
      ↓ Decoder cross-attn（Tk: 201 → 5）
      ↓ PRM cross-attn（Tk: 201 → 5）
```

**这样设计的理由：**

| 特征 | 语义 | 信息密度 | 处理方式 |
|---|---|---|---|
| 1 个静态 token（user_id/gender/age/level） | "我是谁"（用户身份） | 高，不应被压缩 | **直接保留** |
| 200 个行为 token（点击序列） | "我喜欢什么"（用户兴趣） | 有冗余，需提炼 | **QFormer → K=4 个兴趣画像** |

相比 K=1 把 201 tokens 全部压缩，新方案：
- 静态特征**零损失**传递给下游
- 4 个 query 专注于从 200 条行为中提取**多兴趣画像**，任务更清晰
- 有效信息量从 1 token 提升到 5 tokens（1+K=5）

QFormer 每层结构为 **self-attention + cross-attention + FFN** 三件套（对齐 BLIP-2 QFormer），一次性对全 B 样本压缩，不随 beam/cand 数量增加而增加开销。

### 改动说明（t1 相对 t0 的变化）

#### `model.py`

| 位置 | 改动 |
|---|---|
| `__init__` | `self._qformer_query_num = 4`（K=4，4 个 learnable query 捕捉多兴趣） |
| `model()` | **行为序列**（`user_click_emb` [B,200,D]）单独做 LayerNorm + QFormer，得到 `click_portrait [B, K, D]`；**静态特征**（`user_static_emb` [B,1,D]）直通，不经 QFormer；最终 `enc_compressed = concat([user_static_emb, click_portrait]) = [B, 1+K, D]`；新增同质化监控 `qformer/query_pairwise_cos_sim` |
| `beam_search_fast()` | 同上架构，推理侧一致 |
| `beam_search_fast_no_prm()` | 同上架构，推理侧一致 |

#### `modulesV2.py`

| 组件 | 说明 |
|---|---|
| `rms_norm()` | RMSNorm 归一化（QFormer 内部 pre-norm 使用） |
| `QformerLayer` | 单层 QFormer：**self-attention**(Q=K/V=learnable query) + cross-attention(Q=query, K/V=click_emb_normed) + FFN，pre-norm + residual |
| `QFormer` | 多层 QformerLayer 堆叠，将 `[B, 200, D]` 压缩为 `[B, K, D]` |

> **t1_selfattn vs t1 的唯一差异**：`QformerLayer.forward()` 在 cross-attention 之前多了一步 query self-attention，变量 scope 为 `<layer>/self_attention/...` 和 `<layer>/self_atten_ln/...`。cross-attn / FFN 的变量路径与 t1 完全一致，但与 t1 checkpoint 不兼容（新增 self-attn 参数）。

### 参数量说明

| 新增参数 | 数量 | 说明 |
|---|---|---|
| `qformer_queries` | `4 × 256 = 1K` | 4 个 learnable query token（K=4）|
| QFormer **4**层参数 | 每层 ~6×256²+2×256² = ~1.18M | self-attn 4个线性层 + cross-attn 4个线性层 + FFN 2个线性层 + 3个 RMSNorm |
| **合计** | **~4.7M** | 4层 QFormer（含 self-attn）+ 4个 query token |

> 相比 t1（~3.2M）新增 ~1.6M，全部来自 self-attention 的 4×256²×4层。

### 预期收益

| 指标 | 原始（Tk=201） | QFormer 后（Tk=1+4=5） | 收益 |
|---|---|---|---|
| Decoder cross-attn FLOPs | ~100G | ~2.5G | ~**40×** |
| PRM cross-attn FLOPs | ~136G | ~3.4G | ~**40×** |
| 推理端到端耗时 | 基线 | 预期 **~10-30× 降低** | 待实测 |
| QFormer 一次性压缩开销 | — | ~8×B×200×256²×2 ≈ **8.8G** | 含 self-attn（K×K 开销可忽略），与 beam 无关 |

---

### K=4 同质化监控与判断方法

**核心问题**：K=4 个 query token 经 QFormer 输出后，若相互之间的余弦相似度过高（趋向同质），则 4 个 token 实际上表达的是同一种兴趣，等效退化为 K=1，多样性优势消失。

#### 监控指标

在 `model()` 的 `5-C` 步骤中，计算并上报：

| TensorBoard 指标 | 计算方式 | 含义 |
|---|---|---|
| `qformer/query_pairwise_cos_sim` | K=4 个 query 输出的 pairwise cosine similarity 均值（去掉对角线） | 越低越好，反映 query 间多样性 |

**判断阈值：**

| 指标值范围 | 状态 | 建议 |
|---|---|---|
| < 0.3 | ✅ 健康 | query 间差异明显，多兴趣有效 |
| 0.3 ~ 0.6 | ⚠️ 可接受 | 一定同质化，暂可观察 |
| 0.6 ~ 0.8 | ⚠️ 偏高 | 建议考虑加 self-attention |
| > 0.8 | ❌ 严重同质化 | 需加 query self-attention 打破退化 |

#### 为什么 K=4 没有 self-attention 会同质化

若 QFormerLayer 只有 **cross-attention + FFN**（无 query self-attention）：
- 4 个 query 分别独立对同一个 `click_emb_normed` 做 cross-attention
- 由于没有 query 间的通信（self-attention），4 个 query 会倾向于 attend 到 encoder 里最显著的相同 token（如最近点击的热门 item）
- 结果：4 个 query 的输出趋于相同，信息量等效于 K=1

**实测验证**：t1 初版（无 self-attn）训练后 `qformer/query_pairwise_cos_sim` 持续 > 0.8，触发严重同质化阈值，已确认需要加 self-attention。

### t1_selfattn 修复版：加入 Query Self-Attention

**触发条件**：`qformer/query_pairwise_cos_sim` 训练后持续 > 0.8，达到严重同质化阈值。

**改动**：在 `QformerLayer.forward()` 的 cross-attention 之前加一步 query self-attention，对齐 BLIP-2 QFormer 的三件套结构（self-attn + cross-attn + FFN）：

```python
def forward(self, x, enc_output, src_mask, training=False):
    with tf.variable_scope(f"{self.name}", reuse=tf.AUTO_REUSE):
        # K 个 query 全部有效，self-attn mask 全 1
        B = tf.shape(x)[0]
        K = tf.shape(x)[1]
        self_mask = tf.ones([B, 1, 1, K], dtype=tf.int8)

        # 1. Query Self-Attention（Pre-Norm）—— 打破 K 个 query 同质化
        x_norm_0 = rms_norm(x, scope="self_atten_ln")
        with tf.variable_scope("self_attention", reuse=tf.AUTO_REUSE):
            self_attn_output = self.mha(
                x_norm_0, x_norm_0, x_norm_0,
                self.num_heads, self_mask, self.dropout_rate, training=training)
        out0 = x + self_attn_output

        # 2. Cross-Attention（Pre-Norm）—— query attend encoder_output
        x_norm_1 = rms_norm(out0, scope="cross_atten_ln")
        cross_attn_output = self.mha(
            x_norm_1, enc_output, enc_output,
            self.num_heads, src_mask, self.dropout_rate, training=training)
        out1 = out0 + cross_attn_output

        # 3. FFN（Pre-Norm）
        x_norm_3 = rms_norm(out1, scope="ffn_ln")
        ffn_output = self.ffn(x_norm_3, training=training)
        out2 = out1 + ffn_output
    return out2
```

K 个 query 在 attend encoder 之前先互相"商量"（self-attn），自然会分工捕捉不同兴趣，打破同质化。

#### 变量 scope 设计

为保留对 t1 初版 checkpoint 的潜在热启动能力（虽受限于 Kai 参数对齐机制，本次不能直接热启动），cross-attn / FFN 的变量路径保持不变，仅新增 self-attn 相关变量：

| 组件 | 变量路径 | 与 t1 兼容 |
|---|---|---|
| self-attn（新增） | `<layer>/self_atten_ln/rms_norm/gamma`、`<layer>/self_attention/multi_head_attention/w_q/kernel` 等 | ❌ t1 无此参数 |
| cross-attn（原路径保留） | `<layer>/cross_atten_ln/rms_norm/gamma`、`<layer>/multi_head_attention/w_q/kernel` 等 | ✅ 名称一致 |
| FFN（原路径保留） | `<layer>/ffn_ln/rms_norm/gamma`、`<layer>/feed_forward_network/w_up/kernel` 等 | ✅ 名称一致 |

> **注意**：虽然 cross-attn / FFN 变量路径与 t1 一致，但 Kai v2 框架要求训练图与 checkpoint 参数**完全对齐**，新增 self-attn 参数会导致热启动报错。因此本目录（t1_selfattn）需**从头训练**，不能从 t1 checkpoint 热启动。下个版本若要修复，可考虑用 `my_load_dense_func` 自定义加载逻辑，对新增参数随机初始化、对原有参数照常加载。

#### 预期效果

加入 self-attn 后，`qformer/query_pairwise_cos_sim` 应在训练 1~2 天内逐渐下降至 < 0.5。若仍持续 > 0.6，需考虑：
1. 增大 self-attn 的 dropout（0.1 → 0.2）强制 query 间 diversified 依赖
2. 在 self-attn 输出加 layer_norm 前的 L2 normalize（cosine attention）
3. 检查 query token 初始化是否过于接近（尝试正交初始化）

#### 训练前期参考基线

以下是 query 相似度的量级参考（供评估用）：
- **随机初始化刚开始训练**：各 query 独立随机初始化，相似度约 0.0~0.2
- **收敛后健康状态**：< 0.3，说明 query 自然分化出了不同兴趣偏好
- **退化状态**：> 0.8，说明 4 个 query 几乎学到了同一个"平均用户兴趣"

---

### t1_selfattn 二次修复：正交初始化 + Pairwise Cosine Diversity Loss（D01）

#### 背景：加 SA 后 query 同质化反而加剧

加入 self-attention 后实测 `qformer/query_pairwise_cos_sim` **不降反升**：从无 SA 的约 0.975 升至约 0.985，方向与设计预期（降至 < 0.5）相反；同时 retrieval recall/ndcg 业务指标下降。确认 collapse 损害效果，触发本节修复。

#### 根因分析

| 根因 | 表现 | 机制 |
|------|------|------|
| **起点 collapse** | 4 个 query 初始几乎重合 | `_qformer_queries` 用 `stddev=0.01` 初始化，4 个 query 起点都是近零向量，pairwise cos sim ≈ 1。SA 一上来面对"4 个相同向量"，自然学出"4 个相同的均值" |
| **训练 collapse** | 训练过程 collapse 无法被主任务 loss 纠正 | 下游 `enc_compressed = concat([static, portrait])` 整体作为 Decoder/PRM 的 K/V 池，下游自由挑选 token，任务结构不奖励 4 个 query 分工——退化成 K=1 主任务 loss 一样能降 |

SA 的 smoothing 是默认行为，在 query 缺乏"分化信号"（无 position encoding、K=4 小、loss 不奖励 diversity）时主导了 collapse。SA 只是放大器，不是病因。

#### Framing 转变：从"多兴趣分化"到"压缩保真"

原 framing（"4 个 query 各代表一个兴趣"）被重新框定为**"K 个 query 加起来逼近 201 个行为序列的信息容量"**：

- 论证：decoder-only 用 201 个原始序列能提取不同兴趣 → 压缩成 K 个 query 后 decoder 同样应能提取 → 问题不是"让 query 各代表一个兴趣"，而是"让 K 个 query 信息不冗余"
- diversity loss 的合法性来源：从"让 query 代表不同兴趣"转向"让 query 信息不冗余，最大化总信息容量"
- 评估口径：`query_pairwise_cos_sim` 只是手段指标，需联合看压缩保真度（`portrait_seq_mean_sim`）和业务指标（recall），避免"cos sim 降了但信息覆盖丢了"的 cosmetic 风险

#### 改动（model.py，3 处，不新增变量）

**改动 1：正交初始化**（[model.py#L76-L87](model.py)）

`_qformer_queries` 初始化从 `stddev=0.01` 改为 `tf.orthogonal_initializer()`，消除"起点 collapse"：

```python
# K=4 < D=256 时 orthogonal_initializer 生成行正交矩阵，每行 L2 norm = 1，互相正交（cos sim ≈ 0）
self._qformer_queries = tf.get_variable(
    shape=[self._qformer_query_num, dim],
    name='qformer_queries',
    initializer=tf.orthogonal_initializer(),
    trainable=True
)
```

> 对应 L401 预期效果第 3 条"检查 query token 初始化是否过于接近（尝试正交初始化）"的落地。

**改动 2：diversity loss 计算**（[model.py#L202-L212](model.py)，监控段之后）

复用现有 `query_sim_matrix` 计算 `L_div`，不重复建图：

```python
# L_div = mean_batch mean_{i≠j}(cos_sim(qi, qj)^2)
#   - 平方：对高相似度对打压更狠（梯度 2·cos·∇cos），集中火力压最趋同的 query 对
#   - 不加 stop_gradient：梯度回流整个 QFormer（SA/cross-attn/FFN/query embedding），
#     让网络整体学会产出 diverse 输出，避免只推 query embedding 被 SA 重新 smoothing 抹平
query_sim_sq = tf.square(query_sim_matrix)                                   # [B, K, K]
off_diag_sq_sum = tf.reduce_sum(query_sim_sq, axis=[1, 2]) - K_f            # [B]: 减去 K 个对角线 1^2
diversity_loss = tf.reduce_mean(off_diag_sq_sum / num_pairs)                 # scalar
print_tensor("loss/qformer_diversity_loss", diversity_loss)
```

**改动 3：loss 聚合**（[model.py#L433-L438](model.py)）

```python
ntp_loss = tf.reduce_sum(tf.add_n(losses) * loss_mask) / (tf.reduce_sum(loss_mask) + 1e-9)
prm_loss = tf.reduce_sum(tf.add_n(prm_losses) * loss_mask) / (tf.reduce_sum(loss_mask) + 1e-9)
# Diversity loss 权重：起步 0.05，动态调（cos sim 不降则 0.1，主任务受损则 0.02）
# 不乘 loss_mask：diversity 是 query 结构属性（非逐样本属性），全 batch 等权贡献
diversity_loss_weight = 0.05
loss = ntp_loss + prm_loss + diversity_loss_weight * diversity_loss
```

设计要点：
- diversity loss **只加最后一层** `click_portrait`，中间层自由发挥表达能力（中间层是 means to an end，约束它们等于限制 QFormer 的逐层提炼能力）
- **不加 stop_gradient**：collapse 成因是"整个网络的最简路径"，只推 query embedding 会被 SA 重新 smoothing 抹平，需让整个网络感受分化压力
- **不乘 loss_mask**：diversity 是 query 结构属性，非逐样本属性，全 batch 等权贡献

#### 涉及文件

| 文件 | 改动 |
|------|------|
| `model.py` | 3 处改动：正交初始化（L76-L87）、diversity loss 计算（L202-L212）、loss 聚合（L433-L438） |
| `modulesV2.py` | 无改动（QFormer 结构不变） |
| `beam_search_fast()` | 无需改（`_qformer_queries` 在 `__init__` 共享，正交初始化自动生效；推理不算 loss） |

#### 兼容性说明

- **不新增任何变量**：正交初始化只改 `_qformer_queries` 初始值（shape/name 不变）；L_div 复用 `click_portrait`，无新参数。**不破坏 Kai v2 checkpoint 参数对齐**。
- **热启动行为**：
  - 从现有 t1_selfattn checkpoint 热启动 → 正交初始化被 checkpoint 覆盖（仅 L_div 生效，不验证起点假设）
  - 从头训练 → 正交初始化 + L_div 同时生效（**推荐**，可验证双重效果）

#### 评估判别逻辑

跑完训练 1~2 天后，联合看三个指标，分四种情况定夺：

| `query_pairwise_cos_sim` | `portrait_seq_mean_sim` | recall | 判别 | 下一步 |
|---|---|---|---|---|
| ↓ < 0.5 | ↑ 或持平 | ↑ | ✅ 有效 | 继续观察，考虑微调权重 |
| ↓ < 0.5 | ↓ | 不升或 ↓ | ⚠️ cosmetic | diversity loss 让 query 角度分开但丢了信息覆盖，升级第二梯队（attention overlap loss / position encoding） |
| 不降 | - | - | ❌ 约束不够 | 权重 0.05 → 0.1 |
| ↓ 但 recall 大幅 ↓ | - | 大幅 ↓ | ⚠️ 过度约束 | 权重 0.05 → 0.02 |

**关键监控指标**（模型已有，无需新增）：
- `qformer/query_pairwise_cos_sim`（手段指标）
- `qformer/portrait_seq_mean_sim`（压缩保真度，见下节）
- `loss/ntp_loss`、`loss/prm_loss`（主任务 loss）
- `predict_recall_*`（业务 recall）
- 新增 `loss/qformer_diversity_loss`（diversity loss 原始值）

#### 权重动态调整策略

`diversity_loss_weight` 起步 0.05，根据训练曲线动态调（改 [model.py#L437](model.py) 单行即可）：

- `query_pairwise_cos_sim` 不降 → 升到 0.1
- `loss/ntp_loss` 或 `loss/prm_loss` 异常上升 / recall 下降 → 降到 0.02
- 调整后观察 1 天再决定是否继续调

#### 实施顺序建议

分两步验证，避免一次改太多无法定位问题：

1. **Step 1（验证起点 collapse 假设）**：从头训练，跑几百 step 看 `query_pairwise_cos_sim` 初始值是否 ≈ 0，训练几千步内是否快速回升（预期会，需 L_div 兜底）
2. **Step 2（验证 L_div 效果）**：继续训练 1~2 天，按判别逻辑联合看三指标

#### 后续方向（暂缓，待 D01 结果）

- **K 值 ablation**（K=1,2,4,8 vs 201 baseline）：若 D01 效果不足，先做此实验确认 QFormer 压缩合理性和最小有效 K
- **第二梯队**：query position encoding + cross-attn attention overlap loss（从"输出相似度"约束升级到"attention pattern 不重叠"约束，直接逼 4 个 query 看行为序列的不同子集）
- **第三梯队**：MoE-style 路由 + 负载均衡 loss

> 决策与讨论详情：[.codeflicker/discuss/2026-08-07/qformer-sa-collapse-anomaly/](.codeflicker/discuss/2026-08-07/qformer-sa-collapse-anomaly/outline.md)

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
- `model.py` `beam_search_fast()`：L460 后加 `static_ln`，L494-496 删除 `enc_ln`，L508-509 QFormer 输入改为 `user_click_emb`
- `model.py` `beam_search_fast_no_prm()`：L774 后加 `static_ln`，L847-848 删除 `enc_ln`，L859-860 QFormer 输入改为 `user_click_emb`
- 三处路径必须保持一致，避免训练/推理不一致

#### 与 t1 目录的对齐说明

本目录与 `PRMv2_qformer_t1` 共享 LayerNorm 设计调整，保持控制变量。两目录的唯一差异仍是：
- t1: K=1（单 query），QFormer 无 self-attention
- t1_selfattn（本目录）: K=4（4 个 query），QFormer 有 self-attention

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

更合适的诊断指标是**压缩保真度**：QFormer 把 200 个行为 token 压缩为 K 个画像 token，好的画像应该能"代表"用户行为序列的整体方向。因此监控 portrait 与序列有效 token 均值的余弦相似度。

#### 新增指标

在 `model()` 的 5-D2 段（尺度监控之后），计算 QFormer 输出 `click_portrait` 与用户行为序列有效 token 均值的余弦相似度：

| TensorBoard 指标 | 计算方式 | 含义 |
|---|---|---|
| `qformer/portrait_seq_mean_sim` | `cos_sim(mean(click_portrait, axis=1), masked_mean(user_click_emb, click_mask))` 的 batch 均值 | QFormer 画像与用户行为序列主方向的压缩保真度 |

**计算细节**（K=4 适配）：
- `click_mask` 为 0/1 mask（1=有效，0=padding），`click_mask_f = tf.cast(click_mask, tf.float32)`
- `valid_count = sum(click_mask_f, axis=-1, keepdims=True)`，每个样本有效 token 数
- `click_seq_sum = sum(user_click_emb * expand_dims(click_mask_f, -1), axis=1)`，只累加有效 token
- `click_seq_mean = click_seq_sum / (valid_count + 1e-9)`，有效 token 均值
- **K=4 适配**：`portrait_vec = reduce_mean(click_portrait, axis=1)` → `[B, D]`，把 K 个 token 均值作为画像整体方向
- `portrait_vec` 与 `click_seq_mean` 都 L2 归一化后求内积，得到逐样本余弦相似度
- 取 batch 均值上报

#### 判断阈值

| 训练阶段 | 期望值 | 说明 |
|---|---|---|
| 初始随机初始化 | ≈ 0 | QFormer 参数随机，输出与序列主方向无关 |
| 训练中期 | 单调上升 | QFormer 逐步学到保留序列主方向 |
| 收敛后 | 稳定在正区间（如 0.3~0.6） | 画像保留了序列的主要信息 |
| 异常：长期 ≈ 0 或为负 | ❌ 画像与序列主方向不对齐 | QFormer 可能学到"平均用户画像"而非当前用户的序列信息，或丢失了信息 |

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

`qformer/query_pairwise_cos_sim` 监控多 query 间同质化，但无法反映 QFormer 抽取的兴趣画像是否与目标 item 语义对齐。新增 `interest_target_sim` 作为参考指标，衡量 QFormer 输出与目标 item 偏好向量的对齐程度。

#### 新增指标

在 `model()` 的 5-F 段（PRM 段 `sid_embeddings` 定义之后），计算 QFormer 输出 `click_portrait` 与目标 item 偏好向量（3 级语义 ID embedding 求和）的余弦相似度：

| TensorBoard 指标 | 计算方式 | 含义 |
|---|---|---|
| `qformer/interest_target_sim` | `cos_sim(mean(click_portrait, axis=1), sum(sid_embeddings, axis=1))` 的 batch 均值 | QFormer 兴趣画像与目标 item 偏好的对齐程度（参考指标） |

**计算细节**（K=4 适配）：
- `target_pref_emb = reduce_sum(sid_embeddings, axis=1)` → `[B, D]`，目标 item 的 3 级语义 ID embedding 求和
- **K=4 适配**：`portrait_main = reduce_mean(click_portrait, axis=1)` → `[B, D]`，把 K 个 token 均值作为画像整体方向
- 两侧 L2 归一化后求内积，得到逐样本余弦相似度
- 取 batch 均值上报

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
| `qformer/portrait_l2_norm` | `sqrt(mean(square(click_portrait), axis=[1,2]))` 的 batch 均值 | QFormer 输出 L2 范数（K 个 token 的 RMS） |
| `qformer/static_portrait_norm_ratio` | `static_l2_norm / (portrait_l2_norm + 1e-9)` | 两者尺度比，期望 ≈ 1 |

**K=4 适配说明**：`portrait_l2_norm` 对 K 个 token 的所有元素求 RMS（`axis=[1,2]`），反映 K 个 token 整体尺度，与 t1（K=1）的 `sqrt(mean(square(click_portrait[:,0,:])))` 在 K=1 时等价。

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

### 需重新训练（t1_selfattn 不能从 t1 热启动）

Decoder 和 PRM 的 `w_q/w_k/w_v/w_o` 等权重形状均为 `[dim, dim]`，与 encoder context 序列长度（Tk）**无关**，权重形状不受 Tk 从 201→5 的影响。

但 `t1_selfattn` 新增了 `user_qformer/qformer_layer_*/self_attention/*` 参数，Kai v2 框架要求训练图与 checkpoint 参数完全对齐，**从 t1 checkpoint 热启动会报「参数不一致」错误**。因此：

- **方案 A（推荐）**：从头训练 t1_selfattn，效果上限与热启动一致，只是收敛更慢
- **方案 B（需改造）**：实现 `my_load_dense_func` 自定义加载逻辑，对新增 self-attn 参数随机初始化、对原有参数照常加载 t1 checkpoint（需额外开发）

> 原 t1 目录保持原样（无 self-attn），可继续训练 / 推理，不受本目录影响。

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
