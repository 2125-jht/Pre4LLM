
256his 512dim 4layers LayerNorm+PostNorm

- based on v3_mask_256_no_enc_8layers

- 改为512dim

- 改回4layers

- 保持去掉enc

- 改回layernorm以及postnorm形式

---

## 正样本定义 & Loss 加权优化（2026-07-30）

### 1. 正样本过滤逻辑 (`filter_mask_wrapper`)

**旧逻辑**：基于「播放深度 + 显式互动」组合判断，短视频需完播+互动，长视频任一即可。

**新逻辑**：按视频时长(duration)分档设置播放时长阈值，全部使用绝对阈值：

| Duration 分档 | 播放时长阈值 | 设计意图 |
|---|---|---|
| `< 40s` | `≥ 27s` | 中短视频，需看27s以上 |
| `40s ~ 80s` | `≥ 50s` | 中等视频，需看50s以上 |
| `80s ~ 120s` | `≥ 60s` | 较长视频，需看60s以上 |
| `≥ 120s` | `≥ 70s` | 长视频绝对阈值，看70s已是强正向信号 |

**设计考量**：
- 全部使用绝对阈值，逻辑简洁且避免比例阈值在短视频上的"完播=正向"噪声（短视频完播可能只是放着自动放完）
- `<20s` 短视频需播放满20s才能进入训练，过滤掉"自动播放完但没真看"的噪声，同时保留真正观看短视频的用户兴趣信号
- 关于 per-user 个性化 p80：训练框架中序列特征只有 embedding 形式，无法反查每个历史视频的 playing_time；`photo_emp_watch_time` 是视频级全局均值非个性化值，因此当前使用分档固定阈值。若未来要实现 per-user p80，需在数据生产侧预计算并作为新 user-level dense 特征注入

### 2. Loss 加权逻辑 (`model.model()`)

**旧逻辑**：按 `duration_ms` 加权，`weight = lg(2 + duration_ms/1000)`，长视频权重更高。

**新逻辑**：按 `playing_time` 加权，`weight = lg(2 + playing_time/1000)`，长播放样本权重更高。

| playing_time | weight |
|---|---|
| 5s | lg(7) ≈ 0.85 |
| 20s | lg(22) ≈ 1.34 |
| 85s | lg(87) ≈ 1.94 |
| 300s | lg(302) ≈ 2.48 |

**关键区别**：旧方案下，120s 视频即使用户只看了 5s，权重也高达 2.09；新方案下同一样本权重仅 0.85，更准确反映"播放越久=越正向"。

监控统计（基于 `playing_time`，在 `model.model()` 内通过 `print_tensor` 输出）：

| 指标 | 含义 |
|---|---|
| `loss_mask_max` | 加权后 loss_mask 的最大值 |
| `loss_mask_valid_ratio` | 有效样本（`photo_semantic_id_int > 0`）占比 |
| `loss_mask_valid_mean` | 仅对有效样本求 loss_mask 均值（真正权重分布） |
| `play_short_ratio` / `play_long_ratio` | 有效样本中播放时长 ≤17s / >17s 的比例（两者之和≈1.0） |
| `play_ratio_sum` | 短/长播放比例之和，用于交叉验证 |
| `avg_playing_time` | 有效样本的平均播放时长（毫秒） |

### 3. 修改的文件

- **`kai_v2_model.py`**：`filter_mask_wrapper` 中的 `mask_fn`（正样本过滤）；训练模式下获取 `playing_time` 并传入 `model.model()`
- **`model.py`**：`model()` 函数签名新增 `playing_time` 参数（`duration_ms` 形参已移除）；loss 加权从 `duration_ms` 改为 `playing_time`；监控统计（短/长播放比例、平均播放时长等）全部基于 `playing_time`

---

## PRM 模块优化：残差+LN（先改回来了 否则无法训练） / False Negative Mask / K/V 投影优化 / 诊断指标（2026-07-30）

### 1. PRM 残差连接 + LayerNorm（撤销）

**旧逻辑**：PRM 的 cross-attention 输出直接送入 score MLP，无残差连接、无归一化。

**新逻辑**：在 cross-attention 输出上加残差连接和 LayerNorm：

```
target_attn = cross_attn(target_emb, encoder_output)
target_attn = LayerNorm(target_emb + target_attn)   # 残差 + LN
score = MLP(target_attn)
```

- **`modulesV2.py`**：`PRMModel.forward()` 和 `PRMModel.forward_with_kv()` 均加入残差连接（`target_embedding + target_attn`），LayerNorm 已移除以兼容旧checkpoint
- **`modulesV2.py`**：`PRMModel.build_variables()` 中 `ln_vars("cross_attn_ln", self.dim)` 已注释掉，旧checkpoint无此变量

### 2. False Negative Mask（FNM）

**问题**：in-batch 负采样下，batch 内同路径的不同样本对 (i, j) 被当作负样本，但它们实际上是正样本（共享相同路径前缀），导致对比学习信号被污染。

**修复**：

1. 提前计算 `path_hash`（多项式哈希，与 logQ 复用同一份 hash，无额外开销）
2. 构造 `same_path` 矩阵 `[B, B]`，标记 batch 内同路径的所有 (i, j) 对
3. 去掉对角线（对角线是正样本，必须保留），得到 `false_neg_mask`
4. 将 false negative 位置的 logit 设为 `-1e9`，令其对 softmax 无贡献
5. logQ 纠偏在 mask 之后继续执行，已屏蔽位置（-1e9）减去 logQ 后仍极小，不影响 softmax

**新增监控指标**：

| 指标 | 含义 | 健康范围 |
|---|---|---|
| `prm/valid_neg_count_0/1/2` | 每行 mask 后剩余的真负样本数（均值） | 越接近 B-1=127 越好；若长期低于 50，需考虑调整数据采样策略 |

**改动位置**：`model.py` `model()` 方法的 PRM loss 循环内（`for step in range(len(self._vocab_sizes))` 循环）

### 3. K/V 投影优化：先投影再 tile

**旧逻辑**：先 tile `encoder_output` 到 `[B², L, dim]`（训练）或 `[B×cand, L, dim]`（推理），再对每个副本做 w_k/w_v 投影，同一份 encoder_output 的 K/V 投影被重复执行 B 次（训练）/ cand 次（推理），是 PRM FLOPs 的绝对大头。

**新逻辑**：
1. 对 `encoder_output [B, L, dim]` 做一次 w_k/w_v 投影，得到 `prm_K/prm_V [B, H, L, Dh]`
2. 再 tile 到 `[B², H, L, Dh]`（训练）或 `[B×cand, H, L, Dh]`（推理）

**关键点**：tile 和 src_mask 的构建都在循环外只做一次，避免 Python 静态展开后重复创建相同的 TF op。

**涉及新增函数**：

| 函数 | 文件 | 说明 |
|---|---|---|
| `multi_head_attention_with_kv` | `modulesV2.py` | 与 `multi_head_attention` 一致，但跳过 w_k/w_v 投影，直接用传入的 K_pre/V_pre（已 split_heads） |
| `PRMModel.project_kv` | `modulesV2.py` | 调用 `build_variables()` 固定 dense bin 顺序，再用 w_k/w_v（reuse）投影并 split_heads，返回 `[B, H, L, Dh]` |
| `PRMModel.forward_with_kv` | `modulesV2.py` | Q 在内部用 w_q 算，K/V 用传入值，其余与 forward 一致 |

**训练侧改动**（`model.py` `model()` 方法）：
- step 循环前调 `project_kv(encoder_output)` 一次，得到 `prm_K/prm_V`
- step 循环前统一 tile K/V 和 src_mask 到 `[B², H, L, Dh]`，循环内直接复用
- 循环内调 `forward_with_kv` 替代原 `forward`，删除原 `pair_encoder_output` 的 tile

**推理侧改动**（`model.py` `beam_search_fast()` 方法）：
- `step==0` 时调 `project_kv(enc_out_base)` + 一次性 tile K/V 和 src_mask
- `step>=1` 时直接复用 `prm_K_tiled/prm_V_tiled/prm_src_mask`，调 `forward_with_kv`

### 4. Beam Search 诊断指标

在 `beam_search_fast` 的 `step>=1` 分支（PRM 打分之后）通过 `tf.print` 输出以下指标，不影响推理逻辑：

| 指标 | 含义 | 计算方式 | 健康范围 |
|---|---|---|---|
| `beam_diag/parent_entropy_step1/2` | PRM 候选池中父 beam 分布的信息熵 | -sum(p_parent * log(p_parent)) | 接近 log(beam_size)=log(512)≈6.24 为好；<1 说明候选来自少数父 beam（退化） |
| `beam_diag/prm_entropy_step1/2` | PRM logits 经 softmax 后的信息熵 | -sum(prm_prob * log(prm_prob)) | 接近 log(cand_size) 说明 PRM 无法区分候选；理想范围 2~4 |
| `beam_diag/prm_top1_parent_ratio_step1/2` | PRM 保留的 beam_size 条路径中，来自同一父 beam 的最大占比 | max(parent_count) / beam_size | 越低越好；>0.5 说明超半数最终 beam 来自 1 个父 beam（严重退化） |

### 5. 训练/推理图 PRM 变量顺序对比

**新增函数**（`test_model_flop.py`）：`dump_prm_variable_order(graph, tag)` 打印图中所有 `prm_model/*` 变量的名字、shape 和顺序。

- 训练图 session 末尾调用 `dump_prm_variable_order(g_train, tag="train")`
- 推理图 session 末尾调用 `dump_prm_variable_order(g_infer, tag="infer")`
- 用于验证训练/推理两侧 PRM 变量创建顺序是否一致，避免线上 dense bin 加载时权重错位

### 6. 修改的文件

| 文件 | 改动位置 | 改动说明 |
|---|---|---|
| `modulesV2.py` | L73-118 新增函数 | `multi_head_attention_with_kv`：跳过 w_k/w_v 投影，直接用传入的 K_pre/V_pre |
| `modulesV2.py` | L421-443 `PRMModel.project_kv` | 内部先调 `build_variables()` 固定 dense bin 顺序，再用 w_k/w_v 投影并 split_heads |
| `modulesV2.py` | L445-469 `PRMModel.forward_with_kv` | Q 在内部用 w_q 算，K/V 用传入值，含残差（无LN，与旧checkpoint兼容） |
| `modulesV2.py` | L471-491 `PRMModel.forward` | 原函数也加入残差（无LN，与旧checkpoint兼容） |
| `modulesV2.py` | L392-419 `PRMModel.build_variables` | ~~新增 `ln_vars("cross_attn_ln", self.dim)`~~ 已注释掉，旧checkpoint无此变量 |
| `model.py` | `model()` L227-240 | step 循环前调 `project_kv` 一次；K/V 和 src_mask 统一在循环外 tile |
| `model.py` | `model()` L242-300 step 循环内 | 提前计算 path_hash → 构造 FNM → `forward_with_kv` → FNM 屏蔽 → logQ 纠偏 → 新增 `prm/valid_neg_count` 监控 |
| `model.py` | `beam_search_fast()` L498-517 | step==0 调 `project_kv` + 一次性 tile K/V 和 src_mask |
| `model.py` | `beam_search_fast()` L548-640 step>=1 | 删除旧 tile 逻辑，直接复用已有张量；新增 beam 诊断指标（`tf.print`） |
| `test_model_flop.py` | L73-85 新增函数 | `dump_prm_variable_order` 打印 prm_model/* 变量顺序 |
| `test_model_flop.py` | L197 / L240 | 训练图/推理图 session 末尾各调用一次 |
