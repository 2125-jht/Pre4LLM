# v2 当前 List 分支结构笔记

> 按 2026-07-30 的代码简单记一下。核心代码主要在 `model.py`、`kai_v2_model.py` 和 `backbone.py`。

## 先记结论

当前 v2 不是把原来的 pointwise 模型替掉，而是在同一套 shared bottom 上并行加了一条 List Value 分支：

```text
user/item/历史序列特征
        ↓
shared bottom
        ↓
按候选 List gather 成 [30, 6, 64]
        ├── 原 pointwise 分支：List self-attention → PLE → ltr/vtr/click/wtd
        └── 新 List 分支：位置编码 → causal Transformer
                                      ├── continuation / P(K)
                                      ├── Prefix WT（三种并行参数化）
                                      └── Prefix Effective VV
```

训练时两边 loss 直接相加，共享底层特征；推理时原 item 分数照常生成，
List 分支输出三种 WT、消费长度和 Effective VV。

## 原 pointwise 分支现在是什么

原来有四个 item 级 head：

| Head | 大概在学什么 |
|---|---|
| `ltr` | 行为价值加权的 click |
| `click` | 时长 advantage 加权的 click |
| `vtr` | 上游归一化播放程度，可查表解码成秒 |
| `wtd` | 按视频时长选择播放时长桶，预测桶位置后插值成秒 |

这里虽然叫 pointwise，但并不是每个 item 完全独立预测。每条候选 List 会先做一次不带 causal mask 的 self-attention，所以某个位置能看到整条候选 List，再分别输出四个 item 分数。更准确地说是 **List-aware 的 item 级预估**。

pointwise 训练样本继续沿用 base 的规则：只取旧 evaluator 分数最高的候选 List，并且只对其中真实曝光的 item 算 loss，尽量不改变原任务的数据分布。

## 新 List 分支怎么接进来

pointwise 和 List 分支共用前面的 user/item/历史序列表征。生成候选 List 后，将对应 item embedding gather 成：

```text
[batch, list_num=30, list_size=6, dim=64]
```

List 分支给六个位置加位置编码，然后过一层 causal Transformer。在这层 Transformer 里，第 $k$ 个位置只能读取 `Prefix[1:k]`，不能直接读取后面的 List item。需要注意，进入这里之前的 item embedding 已经经过全候选 candidates-aware 编码，所以 causal 限制针对的是当前 List 序列层。

因此它学的是：

```text
Prefix 1 → 用户看到第 1 个 item 后的消费状态和累计价值
Prefix 2 → 用户看到前 2 个 item 后的消费状态和累计价值
...
Prefix 6 → 用户看到完整 List 后的累计价值
```

## List 分支具体预估哪些东西

### 1. continuation 和消费长度

前五个位置分别预测：

$$
q_t=P(K>t\mid K\ge t,\ Prefix_{1:t})
$$

也就是已经消费到第 $t$ 个 item 后，是否还会继续看下一个。

再把五个 continuation 概率转成最终消费长度分布：

$$
P(K=k)=\prod_{i<k}q_i(1-q_k),\quad k<6
$$

$$
P(K=6)=\prod_{i<6}q_i
$$

K=6 当作右截断，只表示看到了最后一位，不额外构造“第六位之后停止”的标签。

最后输出预期消费长度：

$$
expected\_consume\_length=\sum_{k=1}^{6}P(K=k)\cdot k
$$

### 2. 每个 Prefix 的累计观看时长

真实标签直接对已曝光 item 的播放时长做累加：

$$
PrefixWT(k)=\sum_{i=1}^{k}watch\_time_i
$$

当前并行训练三种 Prefix WT 参数化，三者共享同一 causal Prefix 表征和
`P(K)`；同时将每个 Prefix 的 `log1p(累计 duration)` 显式接入三种
WT Head：

1. `log_space`：直接回归 `log1p(PrefixWT)`，通过 `softplus` 保证非负，
   再用 `expm1` 还原成秒。
2. `duration_normalized`：预测
   `PrefixWT / max(PrefixDuration, 1)`，再乘 Prefix 累计 duration
   还原成秒。
3. `duration_bucketed`：根据 Prefix 累计 duration 选择 List 级 WT
   分桶，预测连续桶位置后插值还原成秒。

每种方式都用相同的长度分布聚合：

$$
expected\_list\_watch\_time_m
=\sum_{k=1}^{6}P(K=k)\cdot \widehat{PrefixWT_m}(k)
$$

### 3. 每个 Prefix 的累计 Effective VV

先从真实播放时长构造单 item EVV 标签：

$$
y_i^{EVV}=
\mathbf{1}[watch\_time_i\ge threshold(duration_i)]
$$

再做累计：

$$
PrefixEVV(k)=\sum_{i=1}^{k}y_i^{EVV}
$$

List 分支直接预测累计值，不依赖独立的 pointwise pEVV head。因为前 $k$ 个位置最多有 $k$ 次有效播放，所以输出写成：

$$
\widehat{PrefixEVV}(k)=k\cdot sigmoid(logit_k)
$$

最终 List Effective VV：

$$
expected\_list\_effective\_vv
=\sum_{k=1}^{6}P(K=k)\cdot \widehat{PrefixEVV}(k)
$$

## List 标签和 mask 怎么取

真实日志只知道已经曝光的 `Prefix[1:K]`，不知道未曝光后缀的反事实结果。

当前做法是：

1. 直接选择旧 evaluator 分数最高的候选 List；上游保证它就是实际曝光 List。
2. Prefix 匹配率只作为数据质量诊断，不再过滤训练样本或改选候选 List。
3. Prefix WT/EVV 只监督真实已经曝光的位置，K 后面的 item 不打标签。

pointwise 和 List Value 现在使用同一个候选 List：

```text
pointwise：旧分最高 List
List Value：旧分最高 List
```

## 当前 List loss 组成

当前权重定义：

| Loss | 作用 | 权重 |
|---|---|---:|
| `length_loss` | continuation 的右截断 NLL | 1.0 |
| 三种 `prefix_wt_*_loss` | log-Huber / duration-ratio Huber / bucket-position BCE | 各 1/3 |
| `prefix_evv_loss` | 每个已曝光 Prefix 的累计 EVV Huber | 0.5 |
| 三种 `list_wt_*_loss` | 各自 expected List WT 的统一 log-Huber | 各 1/6 |
| `list_evv_loss` | 最终 expected List EVV 的 Huber | 0.2 |
| `monotonic_loss` | 三种 WT 的 log 单调约束取均值，再与 EVV 单调约束相加 | 0.1 |

具体来说：

$$
L_{length}
=-\sum_{t\in observed}
\left[y_t\log q_t+(1-y_t)\log(1-q_t)\right]
$$

三种 Prefix WT 原生 loss 分别为：

$$
L_{prefixWT}^{log}
=Huber\left(\log(1+PrefixWT),\ \widehat{\log(1+PrefixWT)}\right)
$$

$$
L_{prefixWT}^{normalized}
=Huber\left(
\frac{PrefixWT}{PrefixDuration},\
\widehat{ratio}
\right)
$$

$$
L_{prefixWT}^{bucketed}
=BCE(soft\_bucket\_position,\ \widehat{bucket\_position})
$$

$$
L_{prefixEVV}
=Huber\left(PrefixEVV,\ \widehat{PrefixEVV}\right)
$$

三种方式解码成秒并按同一个 $P(K)$ 聚合后，都使用相同的 List 总
时长校准 loss：

$$
L_{listWT}^{m}
=Huber\left(
\log(1+ListWT),\
\log(1+expected\_list\_watch\_time_m)
\right)
$$

$$
L_{listEVV}
=Huber\left(
ListEVV,\
expected\_list\_effective\_vv
\right)
$$

单调约束只罚预测下降的部分：

$$
L_{mono}
=\sum_k ReLU(\widehat{V}_{k-1}-\widehat{V}_k)
$$

所以新增 List 总 loss 是：

$$
\begin{aligned}
L_{list}={}&
1.0L_{length}
+\frac{1}{3}\sum_m L_{prefixWT}^{m}
+\frac{0.5}{3}\sum_m L_{listWT}^{m}\\
&+0.5L_{prefixEVV}
+0.2L_{listEVV}
+0.1L_{mono}
\end{aligned}
$$

其中 `duration_normalized` 的原生 Huber 先乘固定尺度 10，以补偿
`[0,1]` 比例空间中 loss 数值明显偏小的问题；代码会分别监控三种方法的
真实加权贡献，供后续按训练统计校准。

最终联合训练目标：

$$
L_{total}
=L_{ltr}+L_{vtr}+L_{click}+L_{wtd}+L_{list}
$$

## 推理和原打分怎么结合

推理时模型同时输出：

```text
原 item 级：
pctr / pltr / pvtr / pwtd / context_*

新增 List 级：
expected_list_watch_time                  # log_space 兼容别名
expected_list_watch_time_log_space
expected_list_watch_time_duration_normalized
expected_list_watch_time_duration_bucketed
expected_list_effective_vv
expected_consume_length
```

原 item 分数先按原逻辑聚合成 `original_score`。打开 Listwise fusion 后，在当前请求的真实候选 List 内做 z-score：WT 会先 `log1p`，EVV 使用原值，最后都截断到 `[-3,3]`。融合增量再乘原分数的标准差控制尺度：

$$
\Delta score_i
=std(original)\cdot
\left(
\alpha_{wt}z(WT_i)+
\alpha_{evv}z(EVV_i)
\right)
$$

$$
final\_score_i=original\_score_i+\Delta score_i
$$

当前开关和两个 alpha 默认都是 0，需要通过 AB 参数打开。`expected_consume_length` 目前主要用于观测，没有直接进这条融合公式。

## 容易混淆但不参与训练的指标

代码里还算了下面几组 List WT：

```text
VTR/WTD item WT 直接 sum
VTR/WTD item WT × 固定 position decay
VTR/WTD item WT × 模型 reach probability
```

它们主要用于离线对照，**不进入 loss，也不是当前三种 List WT Head 的
最终公式**。三种正式输出都使用：

$$
\boxed{
\sum_k P(K=k)\cdot PrefixWT_m(k)
}
$$

continuation 的固定位置 AUC、请求内相对 AUC也只是诊断项，不参与训练和线上融合。
