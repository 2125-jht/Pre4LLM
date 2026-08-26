
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

**新逻辑**：按视频时长(duration)分档设置播放时长阈值，混合绝对阈值和比例阈值：

| Duration 分档 | 播放时长阈值 | 播放率 | 设计意图 |
|---|---|---|---|
| `< 5s` | `≥ 5s`（绝对） | >100% | <5s视频放着易自动放完但未必真看，用>duration的5s卡掉 |
| `5s ~ 20s` | `≥ 1.0 × duration` | 100% | 完播，短视频最严格 |
| `20s ~ 40s` | `≥ 0.9 × duration` | 90% | 中短视频，需看9成 |
| `40s ~ 80s` | `≥ 0.8 × duration` | 80% | 中等视频 |
| `80s ~ 120s` | `≥ 0.7 × duration` | 70% | 较长视频 |
| `≥ 120s` | `≥ 85s`（绝对） | ≤71% | 长视频绝对阈值，不再按比例 |

**设计考量**：
- 短视频通过完播即可进入训练，保留用户对短视频的长期兴趣信号
- 不再使用全局固定阈值，避免短视频完全无法进入（有损长期兴趣）
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

`duration_ms` 仍传入 `model.model()` 用于 TensorBoard 监控短/长视频比例等统计信息。

### 3. 修改的文件

- **`kai_v2_model.py`**：`filter_mask_wrapper` 中的 `mask_fn`（正样本过滤）；训练模式下获取 `playing_time` 并传入 `model.model()`
- **`model.py`**：`model()` 函数签名新增 `playing_time` 参数；loss 加权从 `duration_ms` 改为 `playing_time`；`duration_ms` 保留用于监控统计

---

## PRM 模块优化：残差+LN / False Negative Mask / K/V 投影优化 / 诊断指标（2026-07-30）

### 1. PRM 残差连接 + LayerNorm

**旧逻辑**：PRM 的 cross-attention 输出直接送入 score MLP，无残差连接、无归一化。

**新逻辑**：在 cross-attention 输出上加残差连接和 LayerNorm：

```
target_attn = cross_attn(target_emb, encoder_output)
target_attn = LayerNorm(target_emb + target_attn)   # 残差 + LN
score = MLP(target_attn)
```

- **`modulesV2.py`**：`PRMModel.forward()` 和 `PRMModel.forward_with_kv()` 均加入 `layer_norm(target_embedding + target_attn, scope="cross_attn_ln")`
- **`modulesV2.py`**：`PRMModel.build_variables()` 新增 `ln_vars("cross_attn_ln", self.dim)` 注册 LayerNorm 的 gamma/beta 变量

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
| `modulesV2.py` | L445-469 `PRMModel.forward_with_kv` | Q 在内部用 w_q 算，K/V 用传入值，含残差+LN |
| `modulesV2.py` | L471-491 `PRMModel.forward` | 原函数也加入残差+LN |
| `modulesV2.py` | L392-419 `PRMModel.build_variables` | 新增 `ln_vars("cross_attn_ln", self.dim)` |
| `model.py` | `model()` L227-240 | step 循环前调 `project_kv` 一次；K/V 和 src_mask 统一在循环外 tile |
| `model.py` | `model()` L242-300 step 循环内 | 提前计算 path_hash → 构造 FNM → `forward_with_kv` → FNM 屏蔽 → logQ 纠偏 → 新增 `prm/valid_neg_count` 监控 |
| `model.py` | `beam_search_fast()` L498-517 | step==0 调 `project_kv` + 一次性 tile K/V 和 src_mask |
| `model.py` | `beam_search_fast()` L548-640 step>=1 | 删除旧 tile 逻辑，直接复用已有张量；新增 beam 诊断指标（`tf.print`） |
| `test_model_flop.py` | L73-85 新增函数 | `dump_prm_variable_order` 打印 prm_model/* 变量顺序 |
| `test_model_flop.py` | L197 / L240 | 训练图/推理图 session 末尾各调用一次 |

---

## PRM 模块精简：dim 减半 + heads 减半（2026-07-30）

### 1. 设计原理

PRM 是辅助评分模块，不需要和主 decoder 同等规模。计算量分析：

| 策略 | dim | num_heads | head_dim | 计算量比 | 参数量比 |
|---|---|---|---|---|---|
| **原始** | 256 | 8 | 32 | 1.0 | 1.0 |
| **dim减半+heads减半** | 128 | 4 | 32 | ~0.25 | ~0.36 |

- Multi-Head Attention 的计算量由 `dim` 决定（`H × Dh = dim`），与 `num_heads` 无关
- 只减半 dim 而不减半 heads 会导致 `head_dim` 从 32 降到 16，每个 head 表示能力不足
- **同时减半 dim 和 heads 保持 `head_dim=32` 不变**，每个 head 仍和原来一样强，只是 head 数量从 8 减到 4（注意力模式多样性减少，但每个模式质量不变）
- 减半 dim 导致计算量降至 ~1/4（投影和 MLP 都是 `O(dim²)`），对 PRM 辅助模块来说够用

### 2. Input Projection 机制

PRM 输入（encoder_output 和 target_embedding）来自主模型的 256-dim 空间，但 PRM 内部在 128-dim 空间运算。新增 `input_proj` 层统一降维：

```
input_proj: [input_dim=256] → [prm_dim=128]  (共享，encoder 和 target 共用同一投影)
w_q / w_k / w_v: [128] → [128]               (attention 内部投影，不跨维度)
w_o: [128] → [128]                            (attention 输出投影)
LayerNorm: 在 prm_dim=128 空间内闭合
score MLP: 128 → 128 → 64 → 16 → 1
```

**残差连接**在 prm_dim 空间内闭合：`LayerNorm(input_proj(target) + cross_attn(target, enc))`，无需回映到 input_dim。

**input_proj 共享**：encoder 输出和 target embedding 处于同一 256-dim 嵌入空间（都源自 `vocab_embedding` 和同一 feature 空间），共享一个投影是合理的。

### 3. 参数量对比

| 组件 | 原始 (dim=256) | 精简 (prm_dim=128) |
|---|---|---|
| input_proj | — | [256, 128] + 128 = 32,896 |
| w_q | [256, 256] = 65,536 | [128, 128] = 16,384 |
| w_k | [256, 256] = 65,536 | [128, 128] = 16,384 |
| w_v | [256, 256] = 65,536 | [128, 128] = 16,384 |
| w_o | [256, 257] = 65,792 | [128, 129] = 16,640 |
| cross_attn_ln | 512 | 256 |
| score MLP | ~102K | ~26K |
| **总计** | **~300K** | **~108K** |

### 4. 变量创建顺序（dense bin 加载对齐）

训练图和推理图（含 `beam_search_fast_no_prm`）的 PRM 变量顺序：

1. `prm_model/input_proj/kernel` [256, 128] — **新增**
2. `prm_model/input_proj/bias` [128] — **新增**
3. `prm_model/multi_head_attention/w_q/kernel` [128, 128]（原 [256, 256]）
4. `prm_model/multi_head_attention/w_k/kernel` [128, 128]
5. `prm_model/multi_head_attention/w_v/kernel` [128, 128]
6. `prm_model/multi_head_attention/w_o/kernel` [128, 128]
7. `prm_model/multi_head_attention/w_o/bias` [128]
8. `prm_model/cross_attn_ln/layer_norm/gamma` [128]（原 [256]）
9. `prm_model/cross_attn_ln/layer_norm/beta` [128]
10. `prm_model/target_score_mlp/...`

⚠️ **此变更不兼容旧 checkpoint**，需从头训练或做权重映射。

### 5. Checkpoint 加载兼容

**问题**：精简后 PRM 变量数从 13（旧 dim=256）变为 17（新 dim=128 + input_proj + cross_attn_ln），且 `w_q/w_k/w_v/w_o/MLP` 的 shape 从 `[256,256]` 变为 `[128,128]`。旧 checkpoint（70 vars）加载到新模型（74 vars）时，框架默认的 `default_load_dense_func` 会因 `len(tf_weight) != len(warmup_weight)` 直接 assert 失败。

报错信息：
```
AssertionError: 参数数量不匹配：74 vs 70，模型多定义了[
  {prm_model/cross_attn_ln/layer_norm/gamma:0,
   prm_model/input_proj/kernel:0,
   prm_model/input_proj/bias:0,
   prm_model/cross_attn_ln/layer_norm/beta:0}
]，weight 多加载了[set()]
```

**两层修复**：

#### 修复 A：启用自定义加载函数（训练路径）

`kai_v2_model.py` L256 启用 `my_load_dense_func`（原先被注释掉）：

```python
config.set_load_dense_func(my_load_dense_func)  # 原为注释状态
```

`my_load_dense_func`（L184-254）处理三种参数差异：

| 类型 | 场景 | 处理方式 | 本次的命中变量 |
|---|---|---|---|
| 【新增参数】 | 新模型有，checkpoint 无 | 使用 TF 初始化值 | `input_proj/kernel`, `input_proj/bias`, `cross_attn_ln/gamma`, `cross_attn_ln/beta` |
| 【修改参数】 | 同名但 size 不匹配 | 随机初始化 `U(-1e-4, 1e-4)` | `w_q/w_k/w_v/w_o/kernel` [256,256]→[128,128], `w_o/bias` [256]→[128], MLP 各层 |
| 【删除参数】 | checkpoint 有，新模型无 | 从 warmup 删除 | 无（本次不涉及） |

**效果**：非 PRM 变量（encoder/decoder/embedding）正常从 checkpoint 加载；PRM 变量全部随机初始化（因为 shape 全部变了），等价于 PRM 从头训练，但其他模块可复用旧权重。

#### 修复 B：手动更新推理 YAML `param` 列表（推理路径）

**问题**：`uni_retr_server_local_ann/predict/conf_gpu/dnn_model.yaml` 的 `param` 列表是手动维护的 dense 变量清单，推理服务按此清单加载变量。PRM 结构变更后，该 YAML 需同步更新，否则推理时变量名/shape 与模型图不匹配。

**更新内容**（13 项旧 PRM → 17 项新 PRM）：

| 变量 | 旧 shape (rown×coln) | 新 shape (rown×coln) | 变更类型 |
|---|---|---|---|
| `prm_model/input_proj/kernel` | — | 256×128 | 新增 |
| `prm_model/input_proj/bias` | — | 128 | 新增 |
| `prm_model/multi_head_attention/w_q/kernel` | 256×256 | 128×128 | shape 修改 |
| `prm_model/multi_head_attention/w_k/kernel` | 256×256 | 128×128 | shape 修改 |
| `prm_model/multi_head_attention/w_v/kernel` | 256×256 | 128×128 | shape 修改 |
| `prm_model/multi_head_attention/w_o/kernel` | 256×256 | 128×128 | shape 修改 |
| `prm_model/multi_head_attention/w_o/bias` | 256 | 128 | shape 修改 |
| `prm_model/cross_attn_ln/layer_norm/gamma` | — | 128 (init_mean=1) | 新增 |
| `prm_model/cross_attn_ln/layer_norm/beta` | — | 128 (init_mean=0) | 新增 |
| `prm_model/target_score_mlp/target_score_mlp_0/kernel` | 256×256 | 128×128 | shape 修改 |
| `prm_model/target_score_mlp/target_score_mlp_0/bias` | 256 | 128 | shape 修改 |
| `prm_model/target_score_mlp/target_score_mlp_1/kernel` | 256×128 | 128×64 | shape 修改 |
| `prm_model/target_score_mlp/target_score_mlp_1/bias` | 128 | 64 | shape 修改 |
| `prm_model/target_score_mlp/target_score_mlp_2/kernel` | 128×32 | 64×16 | shape 修改 |
| `prm_model/target_score_mlp/target_score_mlp_2/bias` | 32 | 16 | shape 修改 |
| `prm_model/target_score_mlp/target_score_mlp_final/kernel` | 32×1 | 16×1 | shape 修改 |
| `prm_model/target_score_mlp/target_score_mlp_final/bias` | 1 | 1 | 不变 |

**变量顺序**与 `PRMModel.build_variables()` 中的创建顺序一致（input_proj → w_q/k/v/o → cross_attn_ln → MLP），确保 dense bin 按序加载不错位。

⚠️ 仅 `predict/conf_gpu/dnn_model.yaml` 含 PRM 变量；`conf/`、`conf_gpu_sample/`、`conf_gpu900/` 下的 YAML 无 PRM 变量，无需更新。

### 6. 修改的文件

| 文件 | 改动位置 | 改动说明 |
|---|---|---|
| `modulesV2.py` | `PRMModel.__init__` | 签名从 `(dim, num_heads, ...)` 改为 `(input_dim, prm_dim, num_heads, ...)` |
| `modulesV2.py` | `PRMModel.build_variables` | 新增 `input_proj` 变量注册；所有 dense_vars 改用 `self.dim`(=prm_dim) |
| `modulesV2.py` | `PRMModel.project_kv` | 先 `input_proj` 降维 256→128，再做 w_k/w_v |
| `modulesV2.py` | `PRMModel.forward_with_kv` | 先 `input_proj` 降维 target，残差在 prm_dim 空间闭合 |
| `modulesV2.py` | `PRMModel.forward` | 先 `input_proj` 降维 target 和 hidden，残差在 prm_dim 空间闭合 |
| `model.py` | L219 `model()` | `PRMModel(input_dim=self._dim, prm_dim=self._dim//2, num_heads=4, ...)` |
| `model.py` | L379 `beam_search_fast()` | 同上 |
| `model.py` | L683 `beam_search_fast_no_prm()` | 同上（仅建变量，不做 PRM 计算） |
| `kai_v2_model.py` | L256 | 启用 `config.set_load_dense_func(my_load_dense_func)`，处理 PRM 变量新增/修改/删除的 checkpoint 兼容 |
| `uni_retr_server_local_ann/predict/conf_gpu/dnn_model.yaml` | PRM param 段 | 13 项旧 PRM (256-dim) → 17 项新 PRM (128-dim + input_proj + cross_attn_ln)，手动维护 |
