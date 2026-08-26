# 局部成对探索与 List Advantage 学习方案

> 状态：保留方案，作为后续 Listwise 方向的候选主线；当前不代表已经决定实施。  
> 目标：解决现有 factual List 价值模型缺少候选间反事实监督的问题，直接学习“把旧 Top1 List 换成另一条候选 List 会带来多少增量”。

---

## 1. 背景与问题重述

v2/v3 的 List Value 模型主要在学习：

```text
E[Y | request, exposed_list]
```

其中 `Y` 可以是 List 总观看时长、有效 VV 或其他消费指标。

当前训练数据里，每个请求虽然有多条候选 List，但只有实际曝光的一条 List 有真实回报。其余候选没有被展示，因此不存在可信的 outcome label。

可以把实际回报粗略拆成：

```text
Y = B(user, request) + T(list | user, request) + noise
```

- `B`：用户和请求公共消费倾向，例如用户本身是否活跃、当前是否愿意继续浏览。
- `T`：选择某条 List 相对其他候选产生的增量。
- `noise`：偶然行为及未观测因素。

已有实验显示，请求公共信息远强于候选 List 内容信号：

- continuation 的绝对 AUC 较高；
- 去除同请求公共倾向后，候选内容的相对 AUC 只剩约 0.53～0.55；
- 不同 List WT 方案长期停留在约 0.51～0.52 AUC；
- 增加更复杂的 Prefix、Incremental 或 Relative Value 头，没有获得稳定增益。

因此，继续优化绝对 List outcome predictor 时，模型很容易主要学习 `B`，却不必真正学会候选间的 `T`。

本方案把目标改为直接学习 List Advantage：

```text
Delta(x, Li, L0)
    = E[Y(Li) - Y(L0) | x]
```

其中：

- `x` 是展示前已知的用户、请求和候选集合信息；
- `L0` 是旧策略选出的 Top1 List；
- `Li` 是待比较的候选 List；
- `Delta > 0` 表示用 `Li` 替换 `L0` 预计有收益。

---

## 2. 为什么无法从当前确定性日志直接构造 Advantage Label

对同一个请求，理论上存在两个潜在结果：

```text
Y(L0)    展示旧 Top1 时的回报
Y(Li)    展示候选 Li 时的回报
```

但线上只能展示其中一条，因此每个请求最多观察到一个结果：

```text
只展示 L0 -> 只能观察 Y(L0)
只展示 Li -> 只能观察 Y(Li)
```

单个请求的真实 Advantage：

```text
Y(Li) - Y(L0)
```

永远无法被直接观察。这是反事实缺失，不是通过构造新 loss、请求内中心化或增加模型容量可以弥补的。

因此 Advantage 不是普通的单样本 label。它的监督由下面三部分共同组成：

```text
随机处理变量 Z
实际结果 Y
准确展示概率 p
```

模型通过大量随机对照样本估计条件平均增量，而不是试图为每条样本伪造一个 `delta_label`。

---

## 3. 如何获得 Advantage 监督

### 3.1 构造局部候选 Pair

对每个进入探索流量的请求：

1. 取旧策略 Top1 为锚点 `L0`。
2. 从其余候选中选择一条局部相似的 `L1`。
3. 在线随机展示 `L0` 或 `L1`。
4. 记录实际选择、展示概率和真实消费结果。

例如：

```text
L0 = [A, B, C, D, E, F]
L1 = [A, B, X, D, E, F]
```

二者只在第 3 位发生一次替换。

随机处理：

```text
P(show L1) = p
P(show L0) = 1 - p
```

最简单的方案是 `p = 0.5`。

每条探索样本至少记录：

```text
request_id
user/session/request features x
完整候选集合
L0 的 item 序列
L1 的 item 序列
两条 List 的差异位置和差异类型
实际展示变量 Z：0 表示 L0，1 表示 L1
展示 L1 的概率 p
实际逐位置曝光和播放结果
List 总 WT、有效 VV、退出等最终结果
```

一条样本仍然只有一个 outcome，但随机化保证展示 `L0` 和展示 `L1` 的两组请求在统计上具有相同的用户/请求分布。因此积累足够样本后：

```text
E[Y | Z=1, x] - E[Y | Z=0, x]
```

可以识别在条件 `x` 下用 `L1` 替换 `L0` 的因果增量。

不要求同一个用户、同一个请求重复两次。随机化负责平衡两组请求之间的消费倾向，模型负责根据特征在不同请求间泛化。

### 3.2 为什么使用局部 Pair，而不是 Top-M 均匀随机

不建议直接在 30 条差异很大的候选 List 之间均匀随机。这样会带来：

- 较大的线上风险；
- 较高的 outcome 方差；
- 很难判断收益来自哪个 item 或哪个顺序变化；
- 需要更多样本才能学习稳定。

优先选择与 `L0` 局部相似、旧分接近的候选：

```text
单 item 替换
[A, B, C, D, E, F] -> [A, B, X, D, E, F]

相邻位置交换
[A, B, C, D, E, F] -> [A, C, B, D, E, F]

相同 item 的位置调整
[A, B, C, D, E, F] -> [A, B, D, C, E, F]
```

推荐的候选筛选条件：

1. 与 `L0` 共享尽可能长的前缀。
2. 编辑距离优先为 1，其次为 2。
3. 旧策略分数接近，避免明显劣质候选。
4. 优先覆盖 position 2～5 的 swap/replace。
5. 分桶覆盖不同时长、作者、主题、多样性和重复度变化。
6. 对低置信或高风险请求不做探索。

局部探索把高维的“整条 List 哪个更好”转化为低方差问题：

```text
其他内容基本不变时，这一次局部编辑带来了什么边际影响？
```

---

## 4. Reward 与归因方式

### 4.1 基础 Reward

可以先定义统一 List Utility：

```text
Y = total_watch_time
    + alpha * effective_vv
    + beta  * long_view
    - gamma * negative_feedback
```

最终口径需与业务目标和线上 guardrail 对齐。训练时也可以分别预测 WT、有效 VV 和退出率，线上再做约束融合。

### 4.2 使用分叉点之后的 Suffix Reward

如果 `L0` 和 `L1` 前两位相同、从第 3 位开始不同，那么前两位产生的回报与此次 List 编辑无关。可以使用：

```text
Y_suffix = sum(reward_t, t >= divergence_position)
```

代替整个 List 总回报，以降低方差。

例如：

```text
L0 = [A, B, C, D, E, F]
L1 = [A, B, X, D, E, F]
                ^
             第 3 位分叉
```

使用第 3 位及其之后的总 WT、有效 VV 和退出结果作为主要 reward。

如果用户没有消费到第 3 位，则本次替换没有产生实际影响，`Y_suffix=0` 是合法的 intention-to-treat 样本。

由于两条 List 在分叉位置之前完全一致，是否到达分叉位置由共同前缀决定，不受第 3 位选择影响。必要时还可以在“已到达分叉位置”的样本上估计条件增量，同时保留全流量 intention-to-treat 指标作为最终政策价值。

### 4.3 分解直接收益与后续外部性

一次局部编辑可能同时改变：

```text
ImmediateEffect
    当前 item 自身的 WT/有效播放变化

ContinuationEffect
    当前 item 改变用户继续消费后续位置的概率

DownstreamEffect
    后续 item 因 reach 变化而产生的总价值变化
```

模型可以输出：

```text
Delta_total
Delta_immediate
Delta_continuation
```

其中 `Delta_total` 是主优化目标，另外两项作为解释和辅助任务，不强制三者使用独立的伪标签。

---

## 5. Advantage 模型

### 5.1 反对称打分结构

模型不直接输出 `E[Y | x, L]`，而是为每条 List 输出一个可比较的候选增量表示：

```text
g_theta(x, L)
```

Pair Advantage 定义为：

```text
Delta_theta(x, L1, L0)
    = g_theta(x, L1) - g_theta(x, L0)
```

该结构天然满足：

```text
Delta(L1, L0) = -Delta(L0, L1)
Delta(L0, L0) = 0
```

请求公共信息如果对两条 List 作用相同，会在做差时自动抵消，模型被迫关注 List 之间的差异。

### 5.2 输入表示

除了两条完整 List 的 embedding，显式提供局部编辑特征：

```text
共同前缀长度
首次分叉位置
编辑距离
替换/交换类型
两条 List 的 point-wise 价值差
逐位置 item 特征差
时长结构差
作者/主题/内容重复度差
多样性差
相邻 item transition 特征差
两条 List 的预测 reach 差
两条 List 的 Survival × Reward 差
```

这样 Advantage 头主要学习现有结构模型无法解释的剩余增量，而不是重新从头预测用户消费水平。

---

## 6. Orthogonal / R-Learner 训练

### 6.1 Nuisance Baseline

先用展示前信息训练一个 outcome baseline：

```text
m(x) = E[Y | x]
```

它负责吸收：

- 用户整体活跃度；
- 请求时机；
- session 深度；
- 公共候选集合质量；
- 其他与具体选 L0 还是 L1 无关的消费倾向。

baseline 不负责解释 List Pair 差异。训练 Advantage 时应冻结或使用 cross-fitting 预测，避免 baseline 与 Advantage 共同移动、互相抢占目标。

### 6.2 R-Learner 形式

设：

```text
Z = 1  实际展示 L1
Z = 0  实际展示 L0
p = P(Z=1 | x, L0, L1)
```

训练关系：

```text
Y - m(x)
    ≈ (Z - p) * Delta_theta(x, L1, L0)
```

可以最小化：

```text
L_adv =
    weight *
    Huber(
        Y - m(x),
        (Z - p) * Delta_theta
    )
```

当 `p=0.5` 时：

```text
展示 L1：Z-p = +0.5
展示 L0：Z-p = -0.5
```

如果展示 `L1` 后 outcome 高于该请求的 baseline，梯度推动 `Delta` 变正；如果展示 `L0` 后 outcome 更高，则推动 `Delta` 变负。

用户本身“特别爱看”或“完全不想看”的公共波动主要由 `m(x)` 吸收，不再直接成为 List 差异信号。

### 6.3 Doubly Robust 评估

离线政策评估优先使用 Doubly Robust：

```text
DR =
    mu_1(x) - mu_0(x)
    + Z / p       * (Y - mu_1(x))
    - (1-Z)/(1-p) * (Y - mu_0(x))
```

同时保留 IPS/SNIPS 作为交叉检查：

```text
IPS、SNIPS、DR
```

需要监控 propensity 的最小值、有效样本量和权重分布，避免极端权重造成高方差。

---

## 7. Survival × Distributional Reward 基础模型

Advantage 分支可以建立在结构化 List Value 基线之上：

```text
V_base(L)
    = sum_t P(reach_t | x, prefix_t)
            * E[WT_t | reach_t, x, prefix_t]
```

但 WT 头不应继续使用单一的 `log1p + Huber + expm1` 点估计直接近似原空间期望。

原因是：

```text
E[Y | x] != exp(E[log(1+Y) | x]) - 1
```

log-space Huber 更接近预测稳健中心或中位水平，而线上 List Value 需要算术期望。异方差情况下，这不仅会低估均值，也可能改变候选排序。

推荐使用分布式 WT 头，例如：

```text
p_play              是否产生有效播放
mu, sigma_square    正 WT 的 lognormal 参数
```

原空间期望：

```text
E[WT]
    = p_play * (exp(mu + 0.5 * sigma_square) - 1)
```

也可以使用 Tweedie/Gamma deviance 等直接对期望友好的目标。需要结合 WT 是否大量为零、是否被视频时长截断以及重复播放口径最终选择。

最终候选打分可以是：

```text
Score(L)
    = OriginalScore(L)
    + lambda_base * V_base(L)
    + lambda_adv  * g_theta(x, L)
```

其中：

- `V_base` 提供可解释的 factual 结构价值；
- `g_theta` 提供由随机对照学出的候选因果 Advantage；
- 线上权重逐步放量，避免一次性替换原策略。

---

## 8. 离线指标

现有 List AUC 只能衡量 factual outcome 的跨样本拟合，不能作为候选 rerank 能力的主指标。

Advantage 方案优先监控：

### 8.1 因果政策指标

```text
IPS policy value
SNIPS policy value
DR policy value
effective sample size
propensity weight distribution
```

### 8.2 Advantage 一致性

```text
预测 Delta 分桶后的实际 uplift 单调性
Delta > 0 样本的真实 uplift
新旧策略 disagree 样本的真实 uplift
不同编辑位置的 uplift
不同编辑类型的 uplift
不同 shared-prefix 长度的 uplift
```

### 8.3 结构与稳定性

```text
Delta(L1,L0) + Delta(L0,L1) 接近 0
Delta(L0,L0) 接近 0
同请求候选 Delta 的分布和离群点
Top1 change rate
候选分数方差
WT、有效 VV、退出率等 guardrail
```

普通 List AUC、MAE 可以保留为辅助监控，但不再承担模型选型职责。

---

## 9. 线上探索的风险控制

### 9.1 流量与候选限制

- 只在小流量实验桶开启。
- 只探索旧分接近的候选 Pair。
- 排除高风险用户、强意图请求和明显低质候选。
- 设置候选最低质量阈值与最大旧分差。
- 初始只允许单 item swap/replace。

### 9.2 Propensity 要求

必须记录实际展示概率，而不是事后猜测：

```text
pair_selection_probability
arm_probability
final_action_probability
```

如果 Pair 由确定性规则选出，只随机 `L0/L1`，则核心 arm propensity 可以直接记录为 `p`。如果 Pair 本身也随机选择，需要同时记录 Pair 选择概率。

### 9.3 Guardrail

至少同步观察：

```text
总 WT
有效 VV
首条/前缀消费
退出率
负反馈
稳定性和延迟
旧策略分数损失
```

当候选之间旧分差过大、预测 Advantage 置信度不足或 propensity 异常时，回退旧 Top1。

---

## 10. 与 v2/v3 的核心差异

| 维度 | v2/v3 factual List Value | 本方案 |
|---|---|---|
| 学习目标 | 预测已曝光 List 的绝对 outcome | 预测替换候选后的增量 |
| 每请求监督 | 一条 List outcome | 随机 Pair assignment + outcome + propensity |
| 请求公共倾向 | 容易成为主要信号 | 通过做差和 orthogonal residual 消除 |
| 未曝光候选 | 无 label，依靠模型外推 | 在探索流量中获得真实曝光结果 |
| Relative label | 从同一 factual outcome 人工残差化 | 从随机对照统计识别 |
| 离线主指标 | AUC/MAE | DR/SNIPS policy value 与 uplift |
| 候选学习粒度 | 整条 List 绝对价值 | 局部编辑的边际价值 |

---

## 11. 最小可行数据闭环

如果后续决定实施，最小闭环建议为：

1. 从 30 条候选中确定 `L0` 与一个局部候选 `L1`。
2. 小流量内以已知概率随机展示一条。
3. 记录 Pair、处理变量、propensity、逐位置 outcome 和最终 utility。
4. 使用共同前缀后的 suffix reward 降低方差。
5. 冻结或 cross-fit 请求 baseline `m(x)`。
6. 使用 R-Learner 训练反对称 Advantage。
7. 使用 DR/SNIPS 评估新策略价值。
8. 仅在 `Delta` 高置信、旧分差受控的请求上逐步增加 Advantage 权重。

这条闭环的关键不是增加模型复杂度，而是让训练目标第一次真正包含“候选 List 之间的可识别增量监督”。

---

## 12. 实施前需要确认的问题

1. 当前线上是否已经存在可恢复 propensity 的探索、打散或实验流量。
2. 30 条候选 List 中，单编辑或双编辑候选的覆盖率有多高。
3. List 的实际下发与曝光链路能否准确记录随机选择结果。
4. 总 WT、有效 VV 和退出的统一 utility 口径是什么。
5. 分叉位置后的 suffix outcome 是否能在样本流中稳定还原。
6. 是否可以保证随机选择发生在所有候选特征生成之后，避免处理变量泄漏。
7. 探索流量允许的最大旧分损失和业务 guardrail 是什么。
8. 模型最终是作为旧分增量层，还是逐步替换当前 List Score。

在上述问题明确之前，本方案保留为候选主线，不直接进入代码开发。
