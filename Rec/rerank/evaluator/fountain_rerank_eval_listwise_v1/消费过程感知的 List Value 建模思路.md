# 消费过程感知的 List Value 建模思路

[【LR】极速版\_单列\_重排\_Evaluator List-wise序列价值建模实验](https://docs.corp.kuaishou.com/k/home/VeZ8Mb2UiaEU/fcABrmYbe2pfqgdVNgyYE63tm)

[【LR】极速\_单列\_重排\_FlashEvaluator架构下的SWIM序列评估范式](https://docs.corp.kuaishou.com/d/home/fcACe8zXiRZi8fQ8kKaF2dlCn)

现在的 Evaluator 虽然看了 List 上下文，但最终还是给每个 item 分别预测pCTR/pWTD，然后在线上把这些分数人工累加成 List Score。这里的问题是，用户不一定会看完整个 List，而且越靠后的位置能不能被看到，本身就受前面内容和用户退出行为影响。直接把 6 个 item 的价值加起来，实际上没有把这个消费过程说明白。

把目标改成：模型直接判断这个 List 最终能带来多少价值。这个价值要同时考虑两件事，一是用户大概率会消费到什么位置，二是消费到这个位置时，前面整个子序列能产生多少时长、VV 等收益。

准备分两版做。第一版尽量使用现在已经有的样本字段，先验证方向。第二版再模型升级成完整的 step-wise 消费过程建模。

---

## 第一版：先用现有字段做全概率 List Value

现在虽然有曝光结果，但还不能很好地区分“用户主动退出”和“请求、日志或下游链路在这里结束”。如果直接把最后一次曝光当作退出负样本，continue probability 会混进比较大的 Label 噪声。

第一版思路：直接预测最后一个产生 realshow 事实价值的物理位置，再预测到不同位置时整个前缀的价值，最后用全概率公式把它们加起来。

设当前使用的最后价值位置为 K，List 长度为 6。模型先输出：

```plain/text
P(K=1), P(K=2), ... , P(K=6)
```

然后对每个前缀预测整体价值：

```plain/text
V_1, V_1:2, ... , V_1:6
```

最终 List Value 为：

```plain/text
ListValue = sum(P(K=k) * V_1:k)
```

这里的V\_1:k指前k个 item 组成的整个子序列价值，不是第k个 item 自己的价值。第一版先做两个目标：前缀总观看时长和前缀有效 VV。最后分别得到期望时长和期望有效 VV，再按线上权重融合成一个 List Score。

### 第一版需要的样本和 Label

现有字段基本够用。`context\_info\_\_real\_show\_list`可以确定哪些 item 产生了 realshow 事实价值，`context\_info\_\_real\_show\_index\_list`和`fountain\_fulllink\_rerank\_index\_list`用来还原位置顺序（<span style="background-color:#d1f2ff;">或者直接在内流上做</span>），`rerank\_list\_item\_idx\_flat\_list`用来找到旧分选中的 30 个候选 List 之一。stdout 验证已经确认 realshow 可能出现内部 gap，因此不能再把它直接理解为连续物理曝光前缀。

价值方面，`context\_info\_\_playing\_time\_list`可以得到逐位置观看时长，`photo\_info\_\_duration\_ms\_list`可以辅助构造 WTD 或有效 VV。click、like、follow、comment、forward 等行为 Label 也已经有了。

接下来在训练代码里派生下面几个 Label，不需要样本流额外生产：

```plain/text
realshow_count           realshow=1 的数量，仅用于诊断，不能作为位置 K
consume_depth            当前取最后一个 realshow=1 的物理位置 K
prefix_watch_time[k]     前 k 个位置的累计观看时长
prefix_effective_vv[k]   前 k 个位置的累计有效 VV
list_watch_time          realshow 事实 item 的总观看时长
list_effective_vv        realshow 事实 item 的总有效 VV
```

需要先确认有效 VV 的业务口径。如果它就是由播放时长和视频时长按固定规则计算，直接在训练代码里派生即可。如果线上口径还依赖其他行为，就需要样本流直接给一个统一的`effective\_vv\_label\_list`，避免离线训练和线上指标不一致。

实际训练统一选择旧 evaluator 分数最高的候选 List；上游保证它就是实际下发 List。Prefix 匹配仅作为数据质量诊断，不再过滤训练样本或改选候选 List。当前 Prefix 监督使用 `sequence_mask(K_last_realshow)`：K 以内的内部 `realshow=0` 位置保留为零价值增量，K 之后不参与 Prefix 监督。

### 第一版模型

当前 item encoder、用户兴趣和 point-wise PLE 分支都保留。List 分支增加位置 Embedding 和一个 causal prefix encoder。这样第`k`个位置的 hidden 只包含前`k`个 item 的信息，可以用来预测`V\_1:k`。

最后价值位置`P(K=k)`可以从整个 List 表征上输出一个 6 分类 softmax。前缀时长和有效 VV 则从各位置的 prefix hidden 上分别预测。这样模型一次前向就能得到 6 个位置概率和 6 个前缀价值，再按全概率公式得到最终分数。

第一版的 Loss 大致为：

```plain/text
L_v1 = L_pointwise
     + lambda_k * L_consume_depth
     + lambda_prefix * L_prefix_value
     + lambda_list * L_list_total
```

`L\_pointwise`继续使用现有 click 和 WTD 任务，主要用来稳定底层表征。`L\_consume\_depth`以最后一个 realshow 的物理位置为标签，使用多分类交叉熵；它当前表示“最后价值位置”，不能解释为用户真实浏览深度。前缀时长可以使用 Huber Loss（MSE和MAE的折中），有效 VV 可以根据最终定义使用 BCE 或回归 Loss。`L\_list\_total`用预测的最终期望价值约束真实 List 总时长和总 VV，开始时权重不需要太大。

第一版的目标不是把消费过程解释得特别精确，而是先回答一个问题：直接预测“最后价值位置概率 × 前缀价值”，是否比现在人工累加 6 个 item 分数更准。如果离线指标有改善，再继续做第二版。

---

## 2026-08-13 至 2026-08-14 stdout 样本口径验证记录

以下验证均来自训练阶段的 `[jht][list_value][P0_P1_validation]` stdout，只做数据诊断，不写入 TB，也不参与 loss。各比例在不同 batch 间会有小幅波动，因此下面以多批日志范围为主。

### 第一轮：P0 List 总价值 mask 与 P1 前缀假设

最初使用 `K=sum(real_show)`，再用 `sequence_mask(K)` 构造连续前缀。日志结果：

| 指标 | 多批结果 | 结论 |
| --- | ---: | --- |
| `p0_unshown_wt_nonzero_rate` | 0 | 旧分选中 List 内，`real_show=0` 的 item 没有非零 WT |
| `p0_wt_label_abs_delta_mean` | 0 | List WT 是否乘 `real_show`，当前样本数值完全一致 |
| `p0_evv_label_abs_delta_mean` | 0 | List EVV 是否乘 `real_show`，当前样本数值完全一致 |
| `p1_prefix_request_mismatch_rate` | 约 32.2%～35.9% | `sequence_mask(sum(real_show))` 与逐 item realshow mask 经常不一致；这是请求级比例，一个请求只要存在一个 gap 就计为不一致 |
| `p0_show_label_nonbinary_rate` | 0 | realshow label 是合法的 0/1 |
| `selected_list_invalid_request_rate` | 0 | 旧分选中 List 的 item index 合法 |

P0 的结论是：为保证标签语义清晰，可以显式使用 `value * real_show`，但在当前数据上不会改变 WT/EVV 数值。P1 的结论是：realshow 不能直接假设为连续前缀。

### 第二轮：区分“数量不一致”和“位置布局不一致”

在 P1 中进一步拆分 realshow 数量与位置布局：

| 指标 | 多批结果 | 结论 |
| --- | ---: | --- |
| `p1_selected_show_count_mismatch_rate` | 早期约 0.29%～0.59% | 旧分选中 List 的 realshow 数量与请求级数量基本一致 |
| `p1_prefix_layout_mismatch_given_count_match_rate` | 约 31.8%～34.6% | 即使数量一致，realshow 的 1 仍经常不是前 K 位连续布局 |

因此问题主要不在数量，而在把“realshow 数量”误当成“最后物理位置”。例如：

```plain/text
real_show = [1, 1, 0, 1, 0, 1]
sum(real_show) = 4
最后一个 real_show 的物理位置 = 6
```

基于这轮结果，K 从 `sum(real_show)` 改为旧分选中 List 中最后一个 `real_show=1` 的物理位置；原来的 mismatch 指标随后改名为 internal gap 指标。

### 第三轮：realshow 与 WT、EVV 的关系

为了判断 realshow 是否等于“有效播放”，增加反向验证：

| 指标 | 多批结果 | 结论 |
| --- | ---: | --- |
| `p0_shown_zero_wt_rate` | 约 3.9%～4.7% | 少量 `real_show=1` 的 item 仍然 WT=0 |
| `p0_real_show_evv_mismatch_rate` | 约 38.4%～42.2% | realshow 与当前 EVV 定义差异很大，不能把 realshow 解释为 EVV 有效播放 |
| `p0_unshown_wt_nonzero_rate` | 始终为 0 | 当前样本满足 `WT>0 => real_show=1`；反方向不完全成立 |

结合 EVV 阈值始终大于 0，还可以得到 `real_show=0 => EVV=0`。因此 realshow 更像“产生有效事实记录的 real show/播放事件”，口径明显宽于 EVV，但不等同于纯播放时长正样本。

目前只验证了 WT 和 EVV；`real_show=0` 时 click、like、follow、comment、forward 等其他行为是否也为 0，尚未验证，不能外推。

### 第四轮：改用最后 realshow 位置后的内部 gap

K 改成最后一个 realshow 的物理位置后，日志名称和含义变为：

| 指标 | 多批结果 | 结论 |
| --- | ---: | --- |
| `p1_internal_real_show_gap_request_rate` | 约 31.3%～34.0% | 约三分之一请求在 K 内至少有一个 `real_show=0` gap；这不再视为标签错误 |
| `p1_selected_show_count_mismatch_rate` | 约 0.88%～1.27% | 绝大多数请求的选中 List 仍覆盖请求级 realshow 数量 |
| `p1_internal_real_show_gap_given_count_match_rate` | 约 31.2%～33.9% | 排除数量不一致后，内部 gap 依然稳定存在 |

内部 gap 请求不能过滤，也不能把 realshow item 压缩成新序列。对 WT/EVV Prefix 来说，K 以内 gap 是已进入事实序列但价值增量为 0 的位置，应保留成累计价值平台：

```plain/text
位置           1   2   3   4   5   6
real_show      1   1   0   1   0   1
WT             8   3   0   6   0   9
Prefix WT      8  11  11  17  17  26
K_last         6
```

如果只保留 realshow 连续前缀请求，会丢弃约三分之一包含 gap 的真实请求并产生样本选择偏差；如果 Prefix loss 只 mask `real_show=1` 的位置，又会丢失“累计价值在 gap 处应保持不变”的监督。

### 第五轮：使用 `fountain_slide_to_next_list` 验证 gap/continue 猜想

参考实现使用 `context_info__fountain_slide_to_next_list` 作为 continue label。v1 将该字段接入 stdout，但没有加入 loss、target 或 TB。日志结果：

| 指标 | 多批结果 | 结论 |
| --- | ---: | --- |
| `p1_internal_gap_slide_positive_rate` | 0 | 所有内部 `real_show=0` gap 的 slide 都为 0；更可能是该字段在非 realshow 位置无有效值或默认填 0，不能据此否定快速划过 |
| `p1_later_realshow_after_no_slide_rate` | 约 16.8%～18.6% | slide=0 后仍存在后续 realshow，说明 slide=0 不能在全位置域直接解释为用户停止 |
| `p1_last_realshow_slide_positive_rate` | 约 72.7%～74.8% | 最后一个 realshow 后经常仍有下滑行为，K_last_realshow 不是用户真实到达深度 |

这组结果也解释了参考实现为什么用 `list_show_label` mask continue loss：slide label 很可能只在 `real_show=1` 的位置可靠。现有 slide 数据无法跨过 `real_show=0` gap 恢复完整 reach chain，因此当前不能改用 slide 链计算 K。

### 当前最终采用的训练口径

1. 训练 List 仍选择旧 evaluator 分数最高的候选 List。
2. `K` 定义为该 List 中最后一个 `real_show=1` 的物理位置，而不是 `sum(real_show)`。
3. Prefix mask 使用 `sequence_mask(K)`，保留 K 以内的内部 gap。
4. Prefix WT/EVV 按物理位置累加；由于日志已验证 `real_show=0` 时 WT/EVV 均为 0，gap 对应零价值增量。
5. List 总 WT/EVV 显式使用 `real_show` mask；当前数据上与未乘 mask 数值一致。
6. 不过滤约 30%～34% 的 internal-gap 请求，不压缩 realshow item 顺序。
7. `fountain_slide_to_next_list` 当前只用于 stdout 数据诊断，不参与训练。

当前 K 应解释为“最后价值位置”或“最后 realshow 位置”，不是用户真实浏览/到达深度。最后一个 realshow 后可能还存在只滑动、没有新增 WT/EVV 的尾部位置；对于当前 WT/EVV List 总价值，这些尾部零增量不会改变事实总价值，但会使 `expected_consume_length` 低估真实浏览深度。

---

## 第二版：升级成更细粒度消费感知

第一版直接预测`P(K=k)`，实现简单，但 6 个最后价值位置更像 6 个结果桶，没有显式描述用户是怎么一步步消费下去的。进一步改进就是把这个位置分布拆成逐步继续概率。

当用户已经消费到位置t时，模型预测他是否会继续看到位置t+1：

```plain/text
q_t = P(继续看到 t+1 | 已经看到 t)
```

根据这些条件概率，可以递推得到用户到达每个位置的概率：

```plain/text
到达 t+1 的概率
= 到达 t 的概率 × 到达 t 后继续的概率
```

也可以得到用户恰好消费到位置t的概率：

```plain/text
reach_prob[1] = 1
reach_prob[2] = q_1
reach_prob[3] = q_1 * q_2
reach_prob[4] = q_1 * q_2 * q_3
```

最后一个位置使用剩余概率。这样消费长度分布天然来自一套共享的 step-wise 网络，并且到达概率会随位置单调下降。

注：

```plain text
q_t = P(继续看到 t+1 | 已经看到 t)
```

那么到达下一位置的概率就是：

```plain text
到达 t+1 的概率
= 到达 t 的概率 × 到达 t 后继续的概率
```

从第一个位置开始递推：

```plain text
reach_prob[1] = 1
reach_prob[2] = q_1
reach_prob[3] = q_1 * q_2
reach_prob[4] = q_1 * q_2 * q_3
```

### 第二版 continue Label

最初设想从连续 `real_show_list` 直接构造 continue：

```plain/text
当前位置和下一位置都曝光：continue_label[t] = 1
当前位置曝光、下一位置未曝光：continue_label[t] = 0
当前位置未曝光：不参与 continue loss
```

例如真实曝光前缀为：

```plain/text
[1, 1, 1, 0, 0, ...]
```

对应的有效 continue Label 为：

```plain/text
[1, 1, 0]
```

stdout 验证已经否定了“realshow 必然是连续前缀”和“最后一个 realshow 等于停止位置”这两个前提：约 31%～34% 的请求存在内部 realshow gap，并且约 73%～75% 的最后 realshow 位置仍有 `slide=1`。因此不能再从 `real_show_list` 单独派生真实 continue/stop label。

`context_info__fountain_slide_to_next_list`虽然是更直接的 continue 候选标签，但当前样本中内部 `real_show=0` gap 的 slide 全为 0，同时仍有约 17%～19% 的“slide=0 后存在后续 realshow”位置，说明该字段很可能只在 `real_show=1` 的位置有效。第二版启动前需要上游补齐所有真实到达位置的 slide/stop label，或提供可靠的 reached mask；在此之前不应使用现有 slide 链替换第一版 K。

### 第二版的价值怎么预测

仍然是前缀整体价值。causal prefix encoder 在位置`t`输出`List[1:t]`的表示，然后从这个表示上预测消费到该前缀时的整体价值。

对于有效 VV、点赞等二值目标，价值头本身输出的就是行为概率。对于观看时长，可以参考 SWIM 的做法做时长切片，把连续时长转换成多个分类概率，再根据概率还原期望时长。同时增加前缀总时长和整个 List 总时长的回归约束，避免分类结果虽然区分度高但绝对值不准。

最终 List Value 为：

```plain/text
ListValue = sum(P(K=t) * PrefixValue[t])
```

第二版 Loss 大致为：

```plain/text
L_v2 = L_pointwise
     + lambda_continue * L_continue
     + lambda_value * L_prefix_value
     + lambda_list * L_list_total
     + lambda_consistency * L_consistency
```

`L\_continue`在所有真实到达且 continue label 有效的位置上监督继续/停止概率。`L\_prefix\_value`监督各个前缀的时长和有效 VV。`L\_list\_total`约束最终积分后的 List Value。第一版的最后 realshow 位置分类头可以在第二版保留下来作为辅助任务，但它不是实际 reach depth，不能直接用 KL 或交叉熵强制等同于 step-wise 推导出的真实消费长度分布。

做到这些后，我们的目的大致完成：消费概率是逐步预测的；模型能得到完整且单调的消费长度分布；价值头预测的是前缀整体价值；最后用消费长度概率对前缀价值求期望。

## 代码改动

`model.py`主要增加位置编码、causal prefix encoder、消费长度或 continue head、前缀价值 head，以及最终 List Value 聚合。现有 point-wise 分支保持不动。

`kai\_v2\_model.py`主要负责匹配真实 List、派生最后价值位置和前缀 Label、构造各种 mask、计算新 Loss，并在推理时导出 30 个 List 的期望时长、期望 VV 和最终 List Score。
