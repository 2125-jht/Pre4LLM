# v1 List-only 版本说明

该目录只构建和训练 v1 List Value 模型，不再创建原来的 click、VTR、LTR、
WTD 等单点 tower，也不再计算或导出对应的单点 loss、指标和预测值。

## 模型与训练目标

```text
user / item / history 特征
  -> list_backbone
  -> 按候选 List 下标 gather
  -> causal prefix Transformer
  -> P(K=1..6)、Prefix WT、Prefix Engagement
  -> expected_consume_length
     expected_list_watch_time
     expected_list_engagement
```

v1 的长度概率建模保持不变：直接对 6 个消费长度类别做 softmax，不改成
v2 standalone 的 continuation/hazard 建模。总损失只等于 `list_value_loss`，
包含长度分类、Prefix WT/Engagement、List 总 WT/Engagement 和前缀单调约束。

Engagement 标签统一建模有效播放和四类显式互动：

```text
Engagement = EVV + like + 10*comment + 10*follow + 2.5*forward
```

综合标签在 `log1p` 空间使用 Huber loss；包含显式互动的 Prefix/List 样本
使用 3 倍权重，避免稀疏互动被更稠密的 EVV 样本淹没。

消费长度 K 使用请求中 `real_show=1` 的实际个数，不再使用最后一个
`real_show=1` 的物理位置。已确认中间的 `real_show=0` 来自后续样本受端排
影响后的顺序打乱，不能解释为快速划过。

训练只保留前 K 个 item 与真实曝光顺序逐项匹配的候选 List。若多个候选
共享同一个事实 Prefix，则选择其中旧 evaluator 分数最高的一个；完全没有
Prefix 匹配候选的请求不进入 List Value loss 或 matched 离线指标。历史排查
曾观察到约三成请求 Prefix 不匹配，实际比例以
`list_value/match/training_request_drop_rate` 为准。

训练 mask 与评估 mask 已拆分。现有 List 指标名继续表示可信的 Prefix-matched
口径；同时增加 `_legacy_max_score` 指标，使用旧版“每请求旧分最高 List”的
one-hot mask，只用于与历史曲线比较。legacy 口径不保证候选 List 与真实曝光
Prefix 对齐，因此不能替代 matched 口径作为事实效果结论。

`context_info__pwtd_list` 只用于构造外部模型组 PWTD 求和后的 List 级离线
AUC 对照，不参与任何训练 loss。

## 推理输出

本模型自身只导出三个 List 级预测：

- `expected_list_watch_time`
- `expected_list_engagement`
- `expected_consume_length`

`context_*` 是上游 fullrank 分数的原样 gather，属于可选线上融合基线，不是
本模型的单点预测头。backbone 已删除旧模型的 pCTR/pVTR/pLTR/pWTD 依赖，
三个 List 输出通过独立 AB 权重加入 `eval_list_scores`；权重默认均为 0。

## 重新生成配置

复制目录时带入的旧 `training/`、`infer/` 生成物包含 point-wise 图，已经移除。
需要在具备内部 Kai、TensorFlow 和 Dragonfly 依赖的环境中重新生成：

```bash
bash init_train.sh
bash init_infer.sh
```
