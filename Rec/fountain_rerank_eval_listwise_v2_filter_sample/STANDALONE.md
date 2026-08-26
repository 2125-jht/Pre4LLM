# 事实曝光 Prefix 训练版 Standalone List 模型说明

该目录是从 v2 拆出的纯 List 模型，不再构建或训练原来的单点预估分支。

## 计算路径

```text
原始 user / item / history / context 特征
    -> 独立 embedding
    -> list_backbone（用户历史交叉注意力 + 候选集自注意力）
    -> 按候选 List 下标 gather
    -> causal prefix decoder
    -> P(K) / Prefix WT(A) / WTD(D) / item EVV probability
    -> expected_consume_length
       expected_list_watch_time_a
       expected_list_watch_time_d
       expected_list_effective_vv
```

embedding、`list_backbone` 和 `list_value_branch` 都只接受 List loss 的梯度。
这里保留的 `fullrank_detail_*` 等字段是原始输入特征，不是本模型中的单点预测头。

## 训练边界

- 唯一总损失为 `list_value_loss`。
- 先按 `real_show` 取真实曝光 item，再按 `real_show_index` 升序恢复事实曝光
  Prefix，并映射到 `fountain_fulllink_rerank_index` 候选坐标。训练不再要求
  该 Prefix 命中 30 条候选 List；30候选匹配率仅作为覆盖诊断。
- 训练图每个请求前向两条 List：`[真实 Prefix, PAD...]` 和旧分
  最高 List。只有第一条参与 loss，第二条仅保留历史评估口径。推理图
  仍一次评分30条完整候选 List；两边共享完全相同的参数和位置级计算。
- 有效训练样本必须至少包含一个曝光 item，曝光 rank 必须为正且严格递增，
  factual candidate index 必须位于 `[1, 60]` 且互不重复；完整候选坐标还必须
  构成 `1…60` 的合法排列，保证按坐标重排特征后事实 index 不会错位。
- Continuation、A 的 Prefix WT、D 的 duration-conditioned WTD ratio
  使用所有合法事实 Prefix；K 后的 PAD 不参与这些 loss。
- EVV 使用 duration-conditioned 有效播放阈值派生单 item 二值标签，
  在所有合法事实曝光位置以 BCE 训练，loss 权重为 `0.2`。
- A 依赖完整六项输入的 expected List WT 总值 loss 只使用
  `raw K == 6` 的完整事实 List。D 保持独立的 WTD ratio 监督，
  不引入 A 的 `log1p` Prefix/List 回归 loss。
- 本轮仅恢复 EVV；互动的预测头、标签、loss、指标及推理输出仍停用。
- 合成换序 Pair 的 preference 实验当前停用：不构造 `Y_w/Y_l`，不做额外
  两条 List 的前向，也没有 preference loss 或相关 TensorBoard 监控。
- 不创建 click、VTR、LTR 等其他单点 tower。D 仅借用 WTD 的
  duration-conditioned 标签编解码，head 仍位于同一个 List causal decoder。
- 这是新的参数空间，应冷启动训练；不要加载 v2 的 point-wise/共享底座权重。

## 推理输出

模型图导出：

- `expected_list_watch_time`（A 的兼容别名）
- `expected_list_watch_time_a`：`sum_k P(K=k) * PrefixWT(k)`
- `expected_list_watch_time_d`：先按 duration-conditioned WTD 解码单 item 时长，
  再以 `sum_k P(K=k) * PrefixWTD_D(k)` 聚合；使用与 A 完全相同的 `P(K)`
- `expected_list_effective_vv`：先累加单 item EVV 概率得到
  `PrefixEVV(k)`，再以 `sum_k P(K=k) * PrefixEVV(k)` 聚合
- `expected_consume_length`

两套时长输出同时生成，便于离线对比或在线 shadow；当前默认线上兼容 key
仍指向 A，不会静默切换线上使用的时长口径。
EVV 在 backbone 中保留独立融合权重，默认为 0。

## 合成换序 Preference 实验

该实验当前已从训练图中停用。历史方案和参数仍可通过版本历史查阅；恢复前
必须以 `real_show_index` 重建后的事实 Prefix 为监督边界，不能继续使用旧
Top1、样本数组前 K 项或补出的反事实 suffix 作为事实曝光顺序。

## 评估与 TensorBoard 口径

- `prefix_match_rate`、`candidate_prefix_miss_rate` 和 `by_k/*_any_prefix_match_rate`
  只衡量30候选对事实 Prefix 的覆盖率，不代表训练保留率。
- `factual_training_request_rate` 才是实际训练请求保留率；主要由 rank 和
  candidate index 数据质量决定。
- Continuation 和 Prefix 指标使用所有合法事实 Prefix；完整 P(K)、expected
  List Value 及外部 PWTD 总值对照只使用无 PAD 的原始 K=6 样本。
- `expected_list_watch_time_oracle_k` 使用真实 K 选 A 的 Prefix WT，只用于拆分
  长度头误差与时长头误差；`expected_list_watch_time_fixed_k6` 始终取完整
  K=6 的 Prefix WT，不使用真实 K，也不乘长度概率。
- A、D 和 `content_pwtd` 对照各自增加 `_lt120` 与 `_ge120` targets：
  二者与未分段的 watch-time target 使用相同连续时长 label 和
  `linear_regression` 评估，只是分别限定在 `<120s` 和 `>=120s` 样本。
  这些才是用来对比模型在两个时长段内的效果指标。
- 后缀 `_threshold_120_auc` 仅衡量能否把 `>=120s` List 排在 `<120s`
  List 之前，保留为次要的跨阈值诊断，不用来代表时长段内的效果。
- 保留后缀为 `_legacy_max_score` 的旧分最高 List 评估 targets，包括
  continuation、消费长度、A/D 时长和 EVV。它们使用
  原始旧口径的 Top1 List 及其 item labels，不参与任何 loss。
- 请求内候选相对分和 candidate continuation 方差监控仍未恢复。

### Slide 与曝光序列排查

训练图额外读取 `context_info__fountain_slide_to_next_list`，但暂不用它
替换当前由真实曝光长度 K 推导的 continuation label。每10个 step 会以
`[exposure_slide_debug]` 为前缀打印 batch 第一条请求，包括：

- 原始 `real_show`、`real_show_index`、`fountain_slide_to_next`、播放时长和
`fountain_fulllink_rerank_index`的60项数组；
- 按 `real_show_index` 恢复后的 factual rank、candidate index、slide、
  播放时长及当前由 K 推导的 continue label；
- rank/candidate/pool 有效性、30候选 Prefix 匹配结果、完整候选 Lists
  与分数、旧分最高 List。

同一 step 还会输出 `[exposure_slide_prediction_debug]`，将 factual 和旧分
最高 List 的逐位 continue 概率、P(K) 和 expected consume length，与
factual slide 及 K 推导标签并排打印。
两类日志均按 `BEGIN/END` 分块，RAW、FACTUAL、VALIDITY 和每条
`CANDIDATE[00..29]` 各占一行；每行都保留日志前缀和 step，便于
`rg` 筛选以及从多 Worker 日志中恢复完整区块。

TensorBoard 同时增加：

- `list_value/watch_time/ad/{a,d,pwtd}/*`：在同一 mask 和真实
  List WT 下对比 pred/label ratio、MAE、WMAPE、低估率、相对 PWTD
  的绝对误差胜率以及 `<120s`/`>=120s` 分段的 pred/label ratio
  和 WMAPE。A 和 D 均只与 `content_pwtd` 对照，不再额外输出 D vs A 指标；
- `list_value/watch_time/a/by_position/pos{1..6}_pred_label_ratio`：仅在真实
  到达该位置的事实 Prefix 上，监控 A 的累计时长预估/真实均值比；
- `list_value/watch_time/ad/d/{decoded_item_wmape,wtd_ratio_mae}`：单 item
  解码误差，用于定位 D 的 WTD 表达层问题。
- `list_value/evv/item/{positive_rate,pred_label_ratio,brier_score}`：EVV 单 item
  分类的样本率、校准和 Brier 分数；
- `list_value/evv/list/{pred_label_ratio,mae}`：完整事实 List 上的 EVV
  聚合校准。

- `slide_debug/slide_binary_valid_rate`：事实曝光位置的 slide 是否为 0/1；
- `slide_debug/slide_vs_k_disagreement_rate`：slide 与 K 推导 continue 标签的不一致率；
- `slide_debug/nonterminal_slide_positive_rate`：事实 Prefix 非末项的 slide=1 比例；
- `slide_debug/terminal_slide_positive_rate`：K<6 时末项的 slide=1 比例。若该值高，
  说明样本 Prefix 结束未必等于用户退出，当前末项 continue=0 可能存在误标。

## 外部 PWTD 的 List 时长对照

以下两项在完整 K=6 事实 List 上评估，同时保留了对应的
`_legacy_max_score` 旧分最高 List 口径；它们都不进入任何 loss：

- `list_wt_from_context_pwtd_sum`：累加候选 List 中全部非 padding item；
- `list_wt_from_context_pwtd_position_decay`：对非 padding item 使用
  `1 / (0.3 + position^0.6)` 的固定位置衰减后求和，与 backbone
  线上 item 聚合方式保持一致。

两项预测只使用事实 List 中的候选 item 和位置，不使用行为 label 本身；限制
到 K=6 后，可与 List 分支的完整 List 输出作同口径对照。

## 配置生成

代码或输出项变更后，需要在具备内部 `kai`、TensorFlow 和 Dragonfly
依赖的环境中执行：

```bash
bash init_train.sh
bash init_infer.sh
```

目录中随复制保留的 `training/`、`infer/` 文件是旧生成物，不包含本次事实
Prefix 训练图变更。使用前必须通过上述脚本重新生成；不要直接发布旧文件。
