
256his 512dim 4layers LayerNorm+PostNorm

- based on v3_mask_256_no_enc_8layers

- 改为512dim

- 改回4layers

- 保持去掉enc

- 改回layernorm以及postnorm形式

---

## 推理优化版（与 t0_2 计算图 & 权重加载对齐）

本目录是 t0_2 的推理优化版本，支持混合精度（FP16）推理和 KV cache，不进行训练。
已同步 t0_2 的以下模型改动，确保计算图和权重加载一致：

### 1. PRM KV Cache（推理：broadcast-based，训练：tile-based）

推理侧和训练侧使用不同的 K/V 缓存策略，**但共享相同的权重变量**：

- **推理 `beam_search_fast`**：使用 `build_encoder_cache` + `forward_with_encoder_cache`
  - `build_encoder_cache`：对 encoder_output 做一次 w_k/w_v 投影 + split_heads，缓存为 `[B, 1, H, L_enc, Dh]`
  - `forward_with_encoder_cache`：每步用 `broadcast_to` 惰性扩展到 `[B, cur_beam, H, L_enc, Dh]`
  - `broadcast_to` 不物化大张量（XLA 可融合到 matmul），内存高效
  - 与 Decoder 交叉注意力的 KV cache 策略一致

- **训练 `model()`**：使用 `project_kv` + `forward_with_kv`
  - `project_kv`：投影一次，返回 `[B, H, L_enc, Dh]`
  - 训练时需 tile 到 `[B², H, L_enc, Dh]`（in-batch 负采样配对），`tile` 物化大张量但训练不需要省内存
  - `forward_with_kv` 使用 `multi_head_attention_with_kv`（跳过 w_k/w_v 投影）

### 2. PRM 残差连接

- `forward_with_encoder_cache`、`forward_with_kv`、`forward` 均加入残差连接
- `target_attn = target_embedding + target_attn`（在 `w_o` 投影之后、`target_score_mlp` 之前）
- 无 LayerNorm（与旧 checkpoint 兼容），`ln_vars("cross_attn_ln", self.dim)` 已注释掉

### 3. False Negative Mask（训练侧）

- `model()` 方法：提前计算 path_hash → 构造 FNM → 屏蔽 false negative 位置 (-1e9) → logQ 纠偏

### 4. Loss 加权逻辑（训练侧）

- `model()` 新增 `playing_time` 参数：按 `lg(2 + playing_time/1000)` 加权，长播放样本权重更高
- `kai_v2_model.py` 训练模式下获取 `playing_time` 并传入 `model.model()`

### 5. Beam Search 推理改进

- `beam_search_fast()`：step>=1 使用 `k=beam_size`（每 beam 扩展 beam_size 条），而非旧的 `k=prm_candidate_size`
- PRM 排序：使用 `tf.sort(prm_best_idx)` 升序保持 decoder 概率序，不再末尾重排
- 新增 beam 诊断指标（parent_entropy, prm_entropy, prm_top1_parent_ratio）
- step==0 cache 扩展改用 `tile_cache_for_first_step` 替代 `gather_cache`
- 使用 `tf.nn.log_softmax` 替代手动 `logsumexp`

### 6. 混合精度（infer_copy 特有，t0_2 不含）

- `modulesV2.py`：`layer_norm` 在 FP32 岛内计算，结果转回原 dtype
- `modulesV2.py`：`multi_head_attention` / `multi_head_attention_with_kv` 在 FP32 岛内做注意力计算
- `modulesV2.py`：`scaled_attention` 在 FP32 岛内做点积、softmax、加权求和
- `modulesV2.py`：PRM 变量使用 `dtype=tf.float16, trainable=False`
- `model.py`：proj 层 `trainable=False`；log_softmax 前 cast 到 FP32
