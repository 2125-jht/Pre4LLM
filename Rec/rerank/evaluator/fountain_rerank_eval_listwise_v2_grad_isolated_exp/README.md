# List Value 梯度隔离优化实验

## 实验目的

在保持 point-wise 分支和梯度隔离入口不变的前提下，降低消费长度任务对
List Value 共享 adapter/Transformer 的支配，让 WT、EVV 和互动价值获得
更多有效训练容量。

本实验只修改一个变量：

```python
LENGTH_LOSS_WEIGHT = 0.25  # 基线为 1.0
```

不修改样本过滤、标签、其他 loss 权重、学习率、模型结构或线上打分配置。

此外新增两项只读离线监控，不进入任何 loss：

- `list_wt_from_context_pwtd_sum`：外部 item `context_pwtd` 在 List 内直接求和；
- `list_wt_from_context_pwtd_position_decay`：外部 item `context_pwtd` 使用与线上一致的固定位置衰减聚合。

它们与 `expected_list_watch_time` 使用相同的 List 总时长标签和
`listwise_match_mask`，用于补齐外部模型的同口径 List 级基线。

提交训练前需要在具备 Kai/TensorFlow 依赖的环境中执行：

```bash
bash init_train.sh
```

以重新生成 `training/dnn-plugin.yaml`，避免继续使用复制目录中的旧训练配置。

## 基线依据

梯度隔离基线任务 `17298585` 的 242 个完整 pass 和 TensorBoard 显示：

- List Value 总 loss 约为 `1.89`，其中 length contribution 约为 `1.18`，占比约 `62%`。
- Prefix Value contribution 约为 `0.39`，List Total contribution 约为 `0.31`。
- 标签 `K=6` 比例约为 `61%`，预测 `P(K=6)` 均值约为 `61.3%`，平均 K 校准正常。
- 但预测 argmax 为 `K=6` 的比例约为 `95%`，非 `K=6` 准确率只有约 `4%`。
- 最终 List WT、EVV、互动价值 AUC 分别约为 `0.513`、`0.525`、`0.599`。

当前 length loss 会对一条样本的多个已观测 hazard CE 求和，因此其数值尺度
约为单步 loss 的四倍。`0.25` 对应当前平均约四个有效 hazard 决策的尺度归一化，
不是针对多个权重的网格搜索。

## 预期结果

按基线数值估算，length contribution 将从约 `1.18` 降至约 `0.295`，List
Value 总 loss 将从约 `1.89` 降至约 `1.00`，length 占比降至约 `29%`。

主要预期是最终 List WT、EVV、互动价值的 AUC 或 UAUC 提升，而不是要求长度
分类准确率上升。长度任务的校准应基本保持，包括预测/标签平均 K、K=6 概率和
continuation 预测率。

## 验收与回退

至少比较连续 20 个稳定 pass 的均值，重点关注：

- `expected_list_watch_time`、`expected_list_effective_vv`、`expected_list_interaction`；
- `expected_list_interaction_occurrence` 的 AUC/UAUC；
- `predicted_k_mean` 对 `label_k_mean`、预测/标签 K=6 概率；
- continuation AUC 和 point-wise 指标。

如果 Value 指标没有稳定改善，或长度校准明显恶化，则恢复
`LENGTH_LOSS_WEIGHT = 1.0`，不继续扫描更多 length 权重。
