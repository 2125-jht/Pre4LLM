# 消费过程感知的 List Value 建模思路

> 历史资料：本文形成于原始 v2 阶段，其中“保留 point-wise 分支”的描述不适用于本目录；当前结构见 `STANDALONE.md`。

[【LR】极速版\_单列\_重排\_Evaluator List-wise序列价值建模实验](https://docs.corp.kuaishou.com/k/home/VeZ8Mb2UiaEU/fcABrmYbe2pfqgdVNgyYE63tm)

[【LR】极速\_单列\_重排\_FlashEvaluator架构下的SWIM序列评估范式](https://docs.corp.kuaishou.com/d/home/fcACe8zXiRZi8fQ8kKaF2dlCn)

现在的 Evaluator 虽然看了 List 上下文，但最终还是给每个 item 分别预测pCTR/pWTD，然后在线上把这些分数人工累加成 List Score。这里的问题是，用户不一定会看完整个 List，而且越靠后的位置能不能被看到，本身就受前面内容和用户退出行为影响。直接把 6 个 item 的价值加起来，实际上没有把这个消费过程说明白。

把目标改成：模型直接判断这个 List 最终能带来多少价值。这个价值要同时考虑两件事，一是用户大概率会消费到什么位置，二是消费到这个位置时，前面整个子序列能产生多少时长、VV 等收益。

准备分两版做。第一版尽量使用现在已经有的样本字段，先验证方向。第二版再模型升级成完整的 step-wise 消费过程建模。

---

## 第一版：先用现有字段做全概率 List Value

现在虽然有曝光结果，但还不能很好地区分“用户主动退出”和“请求、日志或下游链路在这里结束”。如果直接把最后一次曝光当作退出负样本，continue probability 会混进比较大的 Label 噪声。

第一版思路：直接预测用户最终会消费几个 item，再预测消费到不同长度时整个前缀的价值，最后用全概率公式把它们加起来。

设实际消费长度为K，List 长度为 10。模型先输出：

```plain/text
P(K=1), P(K=2), ... , P(K=10)
```

然后对每个前缀预测整体价值：

```plain/text
V_1, V_1:2, ... , V_1:10
```

最终 List Value 为：

```plain/text
ListValue = sum(P(K=k) * V_1:k)
```

这里的V\_1:k指前k个 item 组成的整个子序列价值，不是第k个 item 自己的价值。第一版先做两个目标：前缀总观看时长和前缀有效 VV。最后分别得到期望时长和期望有效 VV，再按线上权重融合成一个 List Score。

### 第一版需要的样本和 Label

现有字段基本够用。`context\_info\_\_real\_show\_list`可以确定哪些 item 被曝光，`context\_info\_\_real\_show\_index\_list`和`fountain\_fulllink\_rerank\_index\_list`用来还原曝光顺序（<span style="background-color:#d1f2ff;">或者直接在内流上做</span>），`rerank\_list\_item\_idx\_flat\_list`用来找到真实曝光对应的是 35 个候选 List 中的哪一个。

价值方面，`context\_info\_\_playing\_time\_list`可以得到逐位置观看时长，`photo\_info\_\_duration\_ms\_list`可以辅助构造 WTD 或有效 VV。click、like、follow、comment、forward 等行为 Label 也已经有了。

接下来在训练代码里派生下面几个 Label，不需要样本流额外生产：

```plain/text
realshow_num             实际连续曝光的长度 K
prefix_watch_time[k]     前 k 个位置的累计观看时长
prefix_effective_vv[k]   前 k 个位置的累计有效 VV
list_watch_time          实际曝光前缀的总观看时长
list_effective_vv        实际曝光前缀的总有效 VV
```

需要先确认有效 VV 的业务口径。如果它就是由播放时长和视频时长按固定规则计算，直接在训练代码里派生即可。如果线上口径还依赖其他行为，就需要样本流直接给一个统一的`effective\_vv\_label\_list`，避免离线训练和线上指标不一致。

本目录的实际训练直接使用 `real_show` 筛选、再按 `real_show_index` 排序后恢复的事实曝光 Prefix。训练张量在 K 后补 PAD，不从旧 evaluator Top1 补反事实 suffix。30候选 Prefix 匹配仅作为覆盖诊断，不再过滤训练样本；Continuation 和 Prefix 价值只监督真实已曝光的位置，expected List 总价值只在原始 K=6 的完整事实 List 上训练。旧分最高 List 另外前向一次，仅保留 `_legacy_max_score` 历史评估口径，不会进入 loss。

### 第一版模型

当前 item encoder、用户兴趣和 point-wise PLE 分支都保留。List 分支增加位置 Embedding 和一个 causal prefix encoder。这样第`k`个位置的 hidden 只包含前`k`个 item 的信息，可以用来预测`V\_1:k`。

消费长度`P(K=k)`可以从整个 List 表征上输出一个 10 分类 softmax。前缀时长和有效 VV 则从各位置的 prefix hidden 上分别预测。这样模型一次前向就能得到 10 个消费长度概率和 10 个前缀价值，再按全概率公式得到最终分数。

第一版的 Loss 大致为：

```plain/text
L_v1 = L_pointwise
     + lambda_k * L_realshow_num
     + lambda_prefix * L_prefix_value
     + lambda_list * L_list_total
```

`L\_pointwise`继续使用现有 click 和 WTD 任务，主要用来稳定底层表征。`L\_realshow\_num`使用多分类交叉熵。前缀时长可以使用 Huber Loss（MSE和MAE的折中），有效 VV 可以根据最终定义使用 BCE 或回归 Loss。`L\_list\_total`用预测的最终期望价值约束真实 List 总时长和总 VV，开始时权重不需要太大。

第一版的目标不是把消费过程解释得特别精确，而是先回答一个问题：直接预测“消费长度概率 × 前缀价值”，是否比现在人工累加 10 个 item 分数更准。如果离线指标有改善，再继续做第二版。

---

## 第二版：升级成更细粒度消费感知

第一版直接预测`P(K=k)`，实现简单，但 10 个消费长度更像 10 个结果桶，没有显式描述用户是怎么一步步消费下去的。进一步改进就是把这个消费长度分布拆成逐步继续概率。

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

`real\_show\_list`的最后一个曝光位置可以认为用户没有继续消费（可以这样认为吗？），因此可以直接构造：

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

`L\_continue`在所有真实到达位置上继续/停止概率。`L\_prefix\_value`监督各个前缀的时长和有效 VV。`L\_list\_total`约束最终积分后的 List Value。第一版的`realshow\_num`分类头可以在第二版保留下来作为辅助任务，用 KL 或交叉熵约束它与 step-wise 推导出的消费长度分布一致。

做到这些后，我们的目的大致完成：消费概率是逐步预测的；模型能得到完整且单调的消费长度分布；价值头预测的是前缀整体价值；最后用消费长度概率对前缀价值求期望。

## 代码改动

`model.py`主要增加位置编码、causal prefix encoder、消费长度或 continue head、前缀价值 head，以及最终 List Value 聚合。现有 point-wise 分支保持不动。

`kai\_v2\_model.py`主要负责匹配真实 List、派生消费长度和前缀 Label、构造各种 mask、计算新 Loss，并在推理时导出 35 个 List 的期望时长、期望 VV 和最终 List Score。
