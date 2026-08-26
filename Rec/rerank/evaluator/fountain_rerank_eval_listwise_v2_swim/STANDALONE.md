# SWIM 事实曝光 Prefix 训练版说明

该目录在原 factual Prefix List 模型上实现 SWIM 时长建模。模型不构建
point-wise tower，保留 `context_pwtd` 等原始 item 输入特征；业务价值头只预测
List watch time 和 EVV。

## 计算路径

```text
user / item / history / context 特征（保留 context_pwtd）
    -> list_backbone（历史 cross attention + 候选集 self attention）
    -> 按候选 List 下标 gather
    -> causal List prefix Transformer
       -> List continuation q_i -> P(K)
       -> 视频内 causal segment continuation h_ij
          -> segment survival -> item WT -> PrefixWT -> expected List WT
       -> item EVV probability -> PrefixEVV -> expected List EVV
```

最终 List WT 为：

```text
sum_k P(K=k) * PrefixWT(k)
```

其中 `PrefixWT(k)` 由前 k 个视频各自的 segment survival 积分后累加得到。
训练监控会同时计算等价形式
`sum_k P(K>=k) * ItemValue(k)`，并记录 WT/EVV 两种写法的最大绝对差，
用于及时发现长度分布、Prefix 累加或张量维度错位。

## 事实训练边界

- 先按 `real_show` 筛出曝光 item，再按 `real_show_index` 恢复真实曝光顺序，
  最后映射到 `fountain_fulllink_rerank_index` 候选坐标。
- Continuation 标签只由恢复后的真实曝光长度 K 构造；
  `fountain_slide_to_next` 不进入 loss。
- 训练图前向 `[真实 Prefix, PAD...]` 和旧分最高 List 两条序列；所有 loss
  只选中第一条。旧分最高 List 只保留历史评估口径。
- K 后的 PAD、未曝光 item、非法 candidate/rank 请求均不参与 WT/EVV loss。
- 完整 expected List WT/EVV 和 P(K) 指标只在 `raw K == 6` 的无 PAD
  factual List 上评估；K<6 仍使用已观测 item、segment 和 Prefix 监督。
- `context_pwtd` 保持原样作为 item 输入，并额外保留 sum/position-decay
  两种 List 基线；它不作为当前模型生成的 point-wise head。

## 视频内 segment 设计

- 前 20 个相对时长片段均匀覆盖视频首播。
- 接着 4 个 quarter-duration 片段覆盖一次 replay。
- 最后使用 `[0.5, 1, 2, 4, 8, 16] * duration` 的指数 tail 片段覆盖重复播放
  长尾；总 horizon 为 `33.5 * duration`。
- 对 segment j，只有真实播放到达其起点才进入 risk set；标签表示是否完整
  看完该 segment。退出后的 segment 不构造额外负样本。
- `play_time >= 33.5 * duration` 视为视频内右截断，训练不虚构退出标签，
  并监控 `right_censored_item_rate`。
- segment logits 使用 causal segment Transformer 并行计算，推理不做逐片段
  autoregressive decode。
- 当前特征池没有视频片段级内容 embedding，因此 segment 输入由 causal List
  item hidden、相对 segment 位置、累计时长比例和原视频 duration 组成；后续若
  接入片段内容 embedding，可在 `segment_input` 处直接拼接。

## Loss

```text
L_total =
    1.0 * L_list_continuation
  + 1.0 * L_segment_focal_BCE
  + 0.5 * L_prefix_WT_log1p_huber
  + 0.5 * L_complete_list_WT_log1p_huber
  + 0.2 * L_item_EVV_BCE
```

Segment loss 在 risk set 内使用 gamma=2 的 Focal BCE。第一版不加入
Polarization loss，避免概率被强推向 0/1 后破坏 survival 积分校准。
PrefixWT 由非负 item WT 累加，天然单调，不再需要旧 A 的单调正则。

## 推理输出

- `expected_list_watch_time`
- `expected_list_effective_vv`
- `expected_consume_length`（内部消费过程的诊断/可选融合输出）
- 原有 `context_pctr/context_pwtd/...` gather 输出

旧 `expected_list_watch_time_a`、`expected_list_watch_time_d` 及 WTD ratio
输出已移除。

## 评估与监控

Continuation：

- 分位置 `pred_label_ratio`
- 全局 Brier score、AUC
- P(K) accuracy、真实类概率、熵、expected K MAE
- P(K) 归一化误差，以及 PMF-Prefix / reach-item 两种聚合的恒等误差

SWIM segment：

- conditional positive rate、pred/label ratio、Brier score
- marginal reach Brier score 与关键 segment 的 reach 校准
- 平均 risk-set 决策数
- replay item rate、segment horizon 右截断率
- segment conditional/reach AUC

Watch time：

- factual item WT 的 pred/label ratio、MAE、WMAPE
- factual PrefixWT、oracle-K WT、完整 List expected WT
- List WT 的 pred/label ratio、MAE、WMAPE、低估率
- `<120s`、`>=120s` 分段回归与 threshold AUC
- 相对 `context_pwtd` sum 的逐请求绝对误差胜率

EVV：

- item positive rate、pred/label ratio、Brier、AUC
- PrefixEVV 与完整 List EVV 的回归校准

所有 loss 均有独立 TensorBoard contribution，便于检查任务权重是否失衡。

## 配置生成

当前 `training/`、`infer/` 是复制目录中的旧生成物。代码变更后需要在具备
内部 Kai、TensorFlow 和 Dragonfly 依赖的环境执行：

```bash
bash init_train.sh
bash init_infer.sh
```

重新生成配置后再发布，不能直接使用旧生成物。
