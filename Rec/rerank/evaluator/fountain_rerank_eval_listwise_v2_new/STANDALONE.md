# Standalone List 模型说明

该目录是从 v2 拆出的纯 List 模型，不再构建或训练原来的单点预估分支。
本版本将原来的 EVV 与互动价值头合并为 Engagement，业务价值目标收敛为
WT 和 Engagement 两项；消费长度 `P(K)` 继续作为期望价值计算所需的结构头。

## 计算路径

```text
原始 user / item / history / context 特征
    -> 独立 embedding
    -> list_backbone（用户历史交叉注意力 + 候选集自注意力）
    -> 按候选 List 下标 gather
    -> causal prefix decoder
    -> P(K) / Prefix WT / Prefix Engagement
    -> expected_consume_length
       expected_list_watch_time
       expected_list_engagement
    -> 用户反馈重排 Y_w / 事实顺序 Y_l（仅训练）
    -> WT-primary RankNet preference loss
```

embedding、`list_backbone` 和 `list_value_branch` 都只接受 List loss 的梯度。
这里保留的 `fullrank_detail_*` 等字段是原始输入特征，不是本模型中的单点预测头。

## 训练目标

- 唯一总损失为 `list_value_loss`。
- 每个请求只监督旧分最高的候选 List；上游保证它就是实际曝光 List。
  Prefix 匹配率用于数据诊断，用户反馈偏好样本会额外要求旧 Top1 与事实
  曝光 Prefix 匹配。
- WT 在 Prefix 和 List 总值上分别使用 `log1p` Huber，权重为 `1.0/0.5`。
- Engagement 在 Prefix 和 List 总值上分别使用 `log1p` Huber，权重为
  `0.5/0.2`；不再对互动正样本额外乘 3，因为互动业务价值已经体现在标签幅度中。
- 消费长度 hazard loss 权重为 `1.0`，WT/Engagement 前缀单调约束权重为 `0.1`。
- 不创建 click、VTR、LTR、WTD 等单点 tower，也不生成对应 loss 或输出。
- 这是新的参数空间，应冷启动训练；不要加载旧三头版本的参数。

## Engagement 口径

单 item 标签为：

```text
interaction_value
  = like + 10 * comment + 10 * follow + 2.5 * forward

engagement_value = EVV + interaction_value
```

互动相对权重来自原单点 LTR 样本权重中的 `20/200/200/50`，统一除以 20
以控制尺度。EVV 最大为 1，因此单 item Engagement 上界为 24.5，六条
List 上界为 147。对已曝光位置累加得到 `PrefixEngagement(k)`，最终输出为：

```text
expected_list_engagement
  = sum_k P(K=k) * PrefixEngagementPred(k)
```

合并后不再要求一个共享表示同时校准两个量纲差异较大的独立 head，同时保留
EVV-only 与 interaction-positive 分桶，便于监控稠密基础价值和稀疏高价值行为。

## WT-primary 用户反馈顺序偏好

训练时借鉴 GReF 的用户反馈正负 List 构造思路，但 evaluator 不使用 DPO。
负 List `Y_l` 是旧 evaluator 选中的事实曝光顺序；正 List `Y_w` 只重排
其中真实曝光的 Prefix，未曝光 suffix 的内容和相对顺序保持不变。

每个已曝光 item 的反馈效用为：

```text
wt = clip(log1p(play_time) / log1p(400), 0, 1)
engagement = clip(log1p(engagement_value) / log1p(24.5), 0, 1)

feedback_utility = 0.70 * wt + 0.30 * engagement
personalization_score_i = 1 / position_i + 2 * feedback_utility_i
```

只有 `Y_w != Y_l`、至少真实曝光两个 item、List 无 padding，且旧 Top1
确实匹配事实曝光 Prefix 时才计算 preference loss。训练图在原 30 条候选后
追加 `Y_w/Y_l`，共享同一次 `list_backbone` 前向；两条合成 List 不接受
任何事实回归标签。

模型的偏好价值与样本构造使用同一口径：

```text
V(List)
  = 0.70 * log1p(expected_list_watch_time) / log1p(2400)
  + 0.30 * log1p(expected_list_engagement) / log1p(147)

L_preference = -log sigmoid((V(Y_w) - V(Y_l)) / 0.1)
```

`L_preference` 在总 loss 中的权重为 `0.1`，作为弱顺序先验，不把 `Y_w`
解释成具有真实反事实回报的曝光样本。

## 推理与在线融合

模型图导出两项业务价值及消费长度：

- `expected_list_watch_time`
- `expected_list_engagement`
- `expected_consume_length`

在线兼容字段 `eval_list_scores` 的 List 分支增量项为：

```text
WT weight * expected_list_watch_time
+ Engagement weight * expected_list_engagement
+ Length weight * expected_consume_length
```

对应 AB 参数默认均为 0，正式使用前需要显式配置。原来的
`expected_list_effective_vv(_weight)` 和 `expected_list_interaction(_weight)`
不再导出或参与融合。

## 监控

除消费长度和 WT 的原有监控外，本版本增加或调整了：

- Engagement 整体 pred/label ratio 与 MAE；
- EVV 与互动对 Engagement 标签总价值的占比；
- EVV-only、interaction-positive 两个分桶的 pred/label ratio 与 MAE；
- interaction-positive 样本率；
- 偏好样本中 WT/Engagement 的反馈贡献和预测 margin；
- Prefix/List Engagement 加权 loss 贡献。

离线 target 同时输出 Engagement 总体、EVV-only 和 interaction-positive
线性回归指标，便于判断合并目标是否牺牲某一类行为。

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

当前复制目录中的 `training/`、`infer/` 属于旧三头版本的生成物；在上述脚本
成功执行并覆盖前不可用于部署，本次改动以源代码输出定义为准。
