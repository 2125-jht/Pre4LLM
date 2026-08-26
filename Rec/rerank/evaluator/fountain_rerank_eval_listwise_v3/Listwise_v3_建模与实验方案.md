# Listwise v3 建模与实验方案

## 1. 本次改动要回答的问题

v2 已经验证了两个重要结论：

1. 用 continuation 得到的到达概率对逐 item 价值做衰减，方向是成立的；但相对直接累加的提升较小。
2. continuation 的固定位置 AUC 约为 0.70，而去掉同请求公共倾向后的相对 AUC 只有约 0.53～0.55。模型的大部分能力来自用户/请求公共先验，候选 List 内容提供的增量信号较弱。

因此 v3 不继续扩大 `P(K) × PrefixValue` 的复杂度，而是在一次训练中同时生成三个可独立评估、独立上线的输出：

| 输出 | 目的 | 是否新增价值头 |
|---|---|---|
| `reach_pointwise_wt` | 验证“到达概率 × 现有 point-wise WT”的最小改动基线 | 否 |
| `reach_incremental_wt` | 直接学习每个已到达位置贡献的增量 WT | 是 |
| `relative_list_delta` | 学习候选 List 相对本请求公共水平的增量 | 是 |

三个输出不在训练代码中混成一个最终分数，线上也分别配置独立权重，保证实验结果可以归因。

## 2. 方案一：Reach × point-wise WT

沿用现有 continuation 头：

```text
reach_1 = 1
reach_t = product(continue_1 ... continue_(t-1))
```

再使用 point-wise VTR 解码出的逐 item 观看时长：

```text
reach_pointwise_wt = sum_t(reach_t * pointwise_wt_t)
```

这个输出不新增训练损失，作用是提供一条低风险强基线。它与 v2 阶段性实验中的 `reach_weighted_vtr_watch_time` 口径一致，但在 v3 中正式导出，供线上独立开权重。

## 3. 方案二：Reach × Incremental Value

### 3.1 为什么替换累计 Prefix Value

v2 的主要公式为：

```text
ExpectedValue = sum_k P(K=k) * PrefixValue(k)
```

但是 `PrefixValue(k)` 是在所有 `K>=k` 的样本上训练的，而公式实际需要的是 `K=k` 条件下的累计价值。用户是否继续和当前消费质量相关时，这两个条件分布并不相同。

v3 改为逐位置增量分解：

```text
ExpectedValue = sum_t P(reach_t) * E[item_value_t | reach_t]
```

增量 WT 头在 log1p 空间训练，增量 EVV 头使用 sigmoid。两者都只在真实曝光位置训练：

```text
incremental_mask = matched_list_mask * real_show_position_mask
```

最终输出：

```text
reach_incremental_wt  = sum_t(reach_t * incremental_wt_t)
reach_incremental_evv = sum_t(reach_t * incremental_evv_t)
```

累计 Prefix WT/EVV 和原 `expected_vtr_wt` 暂时保留，作为 v2 对照及兼容输出，但不与 v3 三个实验分数混合。

## 4. 方案三：Request Baseline + Relative List Delta

### 4.1 Candidate-set 上下文

当前每条 6-item List 分别经过因果 Transformer。v3 在此基础上增加轻量候选集合摘要：

1. 对每条 List 的有效位置做 masked mean，得到 30 个 `list_embedding`。
2. 对所有有效候选 List 做 masked mean，得到 `candidate_set_mean`。
3. 构造相对特征：

```text
relative_embedding_j =
    list_embedding_j - stop_gradient(candidate_set_mean)
```

候选集合均值在相对分支中停止梯度，避免只有一个真实标签时，把另外 29 条未曝光 List 隐式当成负样本。

### 4.2 公共基线与候选增量

把真实 List WT 的 log1p 价值拆为：

```text
log1p(list_wt) = request_baseline + relative_list_delta
```

- `request_baseline` 使用候选集合公共摘要，负责拟合用户/请求整体消费水平。
- `relative_list_delta` 使用当前 List、相对 List 特征和公共摘要，负责拟合候选间增量。
- `relative_list_delta` 在前向计算时减去请求内有效候选的均值；该均值使用 `stop_gradient`，不会向无标签候选传播伪监督。

训练标签：

```text
delta_label =
    log1p(real_list_wt) - stop_gradient(request_baseline)
```

用于绝对价值评估的重建量为：

```text
relative_list_wt =
    expm1(max(request_baseline + relative_list_delta, 0))
```

线上排序主要使用有正有负的 `relative_list_delta`，而不是把公共 baseline 重复加到所有候选 List 上。

## 5. 损失与监控

v3 新增损失：

```text
incremental_wt_loss
incremental_evv_loss
reach_incremental_list_wt_loss
request_baseline_loss
relative_delta_loss
```

保留的核心监控：

- `v3/loss_contribution/incremental_value`
- `v3/loss_contribution/list_and_relative`
- `v3/loss_contribution/total`
- `v3/calibration/reach_incremental_wt_pred_label_ratio`
- `v3/calibration/reach_incremental_wt_mae`
- `v3/relative/delta_std`

stdout 离线指标：

- `reach_pointwise_list_watch_time`
- `reach_incremental_list_watch_time`
- `relative_list_watch_time`
- `relative_list_delta`

前三个都使用相同的 matched List、相同真实 List WT，能够直接比较；`relative_list_delta` 使用去除 request baseline 后的残差标签。

## 6. 线上控制

新增三个默认值为 0 的动态权重：

```text
fountain_rerank_eval_list_reach_pointwise_wt_weight
fountain_rerank_eval_list_reach_incremental_wt_weight
fountain_rerank_eval_list_relative_delta_weight
```

建议一次只打开一个：

```text
实验 A：旧分 + alpha_1 * reach_pointwise_wt
实验 B：旧分 + alpha_2 * reach_incremental_wt
实验 C：旧分 + alpha_3 * clip(relative_list_delta, -2, 2)
```

`use_order_es=true` 时仍只使用原来的 item 级序融合，不接入本次三个绝对/相对 List 级分数。验证 v3 时应保持 `use_order_es=false`。

## 7. 结果解释边界

本次仍然只有实际曝光 List 有真实回报。因此：

- v3 是更合理的 List-aware outcome model；
- 不能把其余候选 List 当负样本；
- 离线 AUC/MAE 只能比较已曝光 List 的拟合和泛化；
- 候选间真实排序能力最终仍要依赖线上 AB；
- 真正的反事实 Listwise 需要 Top-M 小流量随机选择并记录准确 propensity，后续再使用 IPS/SNIPS/DR。

## 8. 建议的实验顺序

1. 先确认训练收敛、三个输出尺度正常，point-wise 指标没有明显下降。
2. 线上第一天验证 `reach_pointwise_wt`，确定简单 reach 是否有实际收益。
3. 第二天验证 `reach_incremental_wt`，判断新增增量头是否优于 point-wise 价值。
4. 第三天验证 `relative_list_delta`，判断请求内候选增量是否能进一步改善线上指标。
5. 三组实验使用同一份模型，只切换动态权重，不需要重复训练。
