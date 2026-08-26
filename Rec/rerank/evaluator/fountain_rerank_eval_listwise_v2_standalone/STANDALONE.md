# Standalone List 模型说明

该目录是从 v2 拆出的纯 List 模型，不再构建或训练原来的单点预估分支。

## 计算路径

```text
原始 user / item / history / context 特征
    -> 独立 embedding
    -> list_backbone（用户历史交叉注意力 + 候选集自注意力）
    -> 按候选 List 下标 gather
    -> causal prefix decoder
    -> P(K) / Prefix WT / Prefix EVV / Prefix Interaction
    -> expected_consume_length
       expected_list_watch_time
       expected_list_effective_vv
       expected_list_interaction
```

embedding、`list_backbone` 和 `list_value_branch` 都只接受 List loss 的梯度。
这里保留的 `fullrank_detail_*` 等字段是原始输入特征，不是本模型中的单点预测头。

## 训练边界

- 唯一总损失为 `list_value_loss`。
- 与 `v1_alone` 使用同一事实样本口径：先按 `real_show` 取真实曝光 item，
  再按 `real_show_index` 恢复最终曝光顺序；只有该事实 Prefix 能匹配 30 条
  候选 List 之一且曝光 rank 合法时，才训练所有 List 目标。若有多个匹配
  候选，选择其中旧分最高的一条。
- 训练 mask、主评估 mask 均使用上述 matched factual List；评估另保留
  `_legacy_max_score` 对照口径，便于观察旧 Top1 选样口径带来的指标差异。
- 训练目标包括消费长度、Prefix WT/EVV/Interaction、List 总
  WT/EVV/Interaction 和前缀单调约束。
- 合成换序 Pair 的 preference 实验当前停用：不构造 `Y_w/Y_l`，不做额外
  两条 List 的前向，也没有 preference loss 或相关 TensorBoard 监控。
- 不创建 click、VTR、LTR、WTD 等单点 tower，也不生成对应 loss 或输出。
- 这是新的参数空间，应冷启动训练；不要加载 v2 的 point-wise/共享底座权重。

## 推理输出

模型图只导出：

- `expected_list_watch_time`
- `expected_list_effective_vv`
- `expected_list_interaction`
- `expected_consume_length`

在线兼容字段 `eval_list_scores` 为三个 List 价值输出的线性组合：

```text
WT weight * expected_list_watch_time
+ EVV weight * expected_list_effective_vv
+ Interaction weight * expected_list_interaction
```

三个权重默认均为 0，正式使用前需要通过对应 AB 参数显式配置。四个原始
List 输出也会直接返回，便于离线检查或由下游自行组合。

## 综合互动目标

互动只建一个 List 级目标，不分别创建 like/comment/follow/forward tower。
单 item 标签按当前业务价值的相对比例融合：

```text
interaction_value
= like + 10 * comment + 10 * follow + 2.5 * forward
```

对已曝光位置累加得到 `PrefixInteraction(k)`，模型使用独立输出头在
`log1p` 空间做 Huber 回归；最终输出为：

```text
expected_list_interaction
= sum_k P(K=k) * PrefixInteractionPred(k)
```

互动正样本的 Prefix/List loss 使用 3 倍权重；Prefix 和 List 总互动 loss
系数分别为 0.5 和 0.2，与 EVV 对齐且低于主时长目标。

## 合成换序 Preference 实验

该实验当前已从训练图中停用。历史方案和参数仍可通过版本历史查阅；恢复前
需要先改成基于 `real_show_index` 重建且匹配成功的事实 Prefix 构造 Pair，
不能继续使用旧 Top1 或样本数组前 K 项作为事实曝光顺序。

## 评估与 TensorBoard 口径

- 不带后缀的 List 指标是 matched factual 主口径；`_legacy_max_score` 只作
  旧 Top1 口径对照，不参与训练。
- `expected_list_watch_time_oracle_k` 使用真实 K 选 Prefix WT，只用于拆分
  长度头误差与时长头误差；`expected_list_watch_time_fixed_k6` 始终取完整
  K=6 的 Prefix WT，不使用真实 K，也不乘长度概率。
- TensorBoard 保留匹配覆盖、逐位置继续概率校准与候选敏感度、K1～K6
  长度分布/召回、价值校准、时长全局/分 K/长尾/PWTD 对照和合并后的
  loss contribution；已移除旧 Top1 匹配、K1～K6 request rate、K6 重复
  指标及 preference 相关监控。

## 外部 PWTD 的 List 时长对照

以下两项只用于离线评估，不进入任何 loss：

- `list_wt_from_context_pwtd_sum`：累加候选 List 中全部非 padding item；
- `list_wt_from_context_pwtd_position_decay`：对非 padding item 使用
  `1 / (0.3 + position^0.6)` 的固定位置衰减后求和，与 backbone
  线上 item 聚合方式保持一致。

两项预测都只使用打分时可获得的候选 item、padding mask 和位置，
不使用真实 `show_label` 或事后曝光 K，可与 List 分支作为公平对照。

## 配置生成

代码或输出项变更后，需要在具备内部 `kai`、TensorFlow 和 Dragonfly
依赖的环境中执行：

```bash
bash init_train.sh
bash init_infer.sh
```

从 v2 复制来的历史 `training/`、`infer/` 模型生成物已经移除，避免误启动
旧 point-wise 图；应以上述脚本重新生成的结果为准。
