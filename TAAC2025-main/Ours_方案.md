# TAAC2025 适配版 OneRec 方案：协作式 SID + Decoder-only 生成推荐

## 1. 方案定位

本方案面向 **TAAC2025 全模态生成式推荐任务**，目标是在比赛数据条件下构建一个尽量接近 **OneRec 核心范式**、同时又适配官方数据特点的生成式推荐模型。

核心设计如下：

1. **SID 构造阶段**：完整保留技术创新奖方案的思路，先通过专用 `Decoder-only Transformer + InfoNCE` 学习包含协作信号的 item embedding，再基于该 embedding 构造低碰撞、高分辨率的两层 Semantic ID。
2. **主推荐模型阶段**：采用 **Decoder-only** 结构，将用户历史行为序列作为前缀上下文，直接自回归生成目标 item 的 `action → sid1 → sid2`，而不是生成用户向量后再进行 ANN 检索。
3. **关键增强策略**：只保留与该任务最匹配、收益最确定的 trick：
   - `action-conditioning`
   - 时间特征
   - 合法 SID 约束解码
   - 历史行为过滤
4. **大负样本库**：仅用于 **SID 构造阶段** 的 InfoNCE 训练，不引入主推荐模型，因为主模型本质是 SID 生成式交叉熵训练，而非向量检索式对比学习。

---

## 2. 设计背景与出发点

### 2.1 为什么采用 OneRec 范式

传统推荐系统通常是多阶段级联：召回、粗排、精排分别训练，目标不统一。OneRec 的关键变化是将推荐重写为一个统一的生成问题：

> 给定用户历史，直接生成目标 item 的 Semantic ID。

其核心不在于某个具体的 Encoder-Decoder 结构，而在于以下三点：

1. **item 被离散化为层级 SID**；
2. **模型直接生成目标 SID**；
3. **推理时通过生成的 SID 映射回真实 item**。

这种范式可以将推荐问题从“检索一个相似向量”改写为“生成下一交互 item 的离散语义路径”，更接近大模型中的 next-token prediction。

### 2.2 为什么不能直接照搬原版 OneRec

原版 OneRec 面向快手工业场景，具备以下条件：

- 原始视频内容、caption、tag、OCR、ASR、封面、多帧图像；
- 超长用户序列；
- 在线 reward system 与 RL post-training；
- 工业级千万至十亿级物品空间。

而 TAAC2025 官方数据具有不同特征：

- 用户历史最长仅 100；
- 官方提供的是多模态 embedding，而非原始内容；
- 数据中显式包含 `exposure / click / conversion`；
- 决赛目标中 conversion 价值更高；
- 无在线 reward、无超长 lifelong 序列。

因此，本方案保留 OneRec 的**核心生成范式**，但对模型结构和 SID 构造过程进行比赛适配。

---

## 3. 整体架构

```text
阶段 A：协作式 SID 构造
用户历史序列 + item 特征
        ↓
专用 Decoder-only Transformer + InfoNCE
        ↓
协作式 item embedding
        ↓
两级 RQ-KMeans
        ↓
二级碰撞解决
        ↓
item_id ↔ (sid1, sid2)

阶段 B：主推荐模型训练
[user_token, item_1_token, ..., item_T_token, BOS]
        ↓
Decoder-only Transformer
        ↓
自回归生成 action → sid1 → sid2

阶段 C：推理
历史序列作为 prefix
        ↓
联合生成 action → sid1 → sid2
        ↓
合法 SID 约束解码
        ↓
SID 反查 item
        ↓
历史行为过滤
        ↓
Top-10 推荐结果
```

---

## 4. 阶段 A：协作式 SID 构造

### 4.1 为什么 SID 构造不直接使用原始多模态 embedding

TAAC2025 官方提供了多种文本和视觉 embedding，但不同 embedding 的覆盖率并不完全一致；如果直接对单一模态 embedding 做聚类，容易造成 SID 覆盖不足或高碰撞。

技术创新奖方案的关键做法是：

> 不直接拿原始 embedding 做 K-Means，而是先训练一个专用生成模型，通过协作行为学习 item embedding，再对这个协作式 embedding 进行量化。

这样得到的表示不仅包含内容语义，还包含“哪些 item 被相似用户共同交互”的协同信息，更适合作为推荐系统中的离散 token。

### 4.2 专用协作模型

#### 4.2.1 输入

对于用户历史序列中的每个 item，构造 token：

```text
item token =
item sparse features
+ official multimodal embeddings
+ historical action embedding
+ time features
```

#### 4.2.2 模型

使用一个轻量的 **Decoder-only Transformer** 作为协作建模器：

```text
[item_1, item_2, ..., item_t] → 预测 item_{t+1}
```

#### 4.2.3 训练目标

采用 InfoNCE：

\[
\mathcal{L}_{\text{InfoNCE}}
=
-\log
\frac{\exp(\operatorname{sim}(h_t, e_{i^+})/\tau)}
{\exp(\operatorname{sim}(h_t, e_{i^+})/\tau)+\sum_{j \in \mathcal{N}}\exp(\operatorname{sim}(h_t,e_j)/\tau)}
\]

其中：

- \(h_t\)：历史序列最后位置的用户兴趣表示；
- \(e_{i^+}\)：真实下一 item 的 embedding；
- \(\mathcal{N}\)：负样本集合；
- \(\tau\)：温度系数。

#### 4.2.4 大负样本库

大负样本库适合放在这一阶段，因为这一阶段本质是 **contrastive learning / retrieval representation learning**：

- 负样本越丰富，item embedding 的区分能力越强；
- 更强的协作 embedding 会直接提升后续 SID 的质量；
- 但主推荐模型是纯生成式 CE 训练，因此不在主模型中引入大负样本库。

### 4.3 两层 SID 构造

本方案采用两层 Semantic ID：

\[
\text{SID}(i)=(s_i^1,s_i^2)
\]

#### 4.3.1 第一级量化

对协作式 item embedding 做 K-Means：

\[
s_i^1=\arg\min_k\left\|e_i-c_k^{(1)}\right\|
\]

其中 \(c_k^{(1)}\) 是第一层聚类中心。

#### 4.3.2 第二级量化

先计算残差：

\[
r_i=e_i-c_{s_i^1}^{(1)}
\]

再对残差做第二级 K-Means：

\[
s_i^2=\arg\min_k\left\|r_i-c_k^{(2)}\right\|
\]

### 4.4 为什么采用两层，而不是三层

当前方案采用：

```text
sid1 + sid2
```

而不是三层，主要原因有三点：

1. **容量已经足够**  
   若两层 codebook 均为 16K，则组合空间约为：
   \[
   16384^2 \approx 2.68\times 10^8
   \]
   已显著大于 TAAC2025 决赛约 1749 万个广告 item 的规模。

2. **更适合当前比赛数据**  
   三层 SID 的主要优势在于更大规模、更复杂物品空间下的层级表达；而 TAAC2025 的主要瓶颈不是空间不足，而是 SID 质量、碰撞率和生成稳定性。

3. **生成更稳**  
   两层只需要：
   \[
   sid1 \rightarrow sid2
   \]
   三层则需要：
   \[
   sid1 \rightarrow sid2 \rightarrow sid3
   \]
   额外层级会增加误差传播和 beam search 复杂度。

### 4.5 二级碰撞解决

标准 RQ-KMeans 可能会让多个 item 映射到同一组 `(sid1, sid2)`，这对直接生成 SID 的系统非常不利，因为一个 SID pair 可能对应多个 item。

因此保留技术创新方案中的二级碰撞解决机制：

1. 先按最近质心为每个 item 分配 `(sid1, sid2)`；
2. 检测是否存在多个 item 使用相同 SID pair；
3. 对发生冲突的 item，在第二级码本中继续搜索下一个最近但尚未占用的中心；
4. 直到尽量实现：

```text
一个 item ↔ 一个唯一 SID pair
```

---

## 5. 阶段 B：主推荐模型

### 5.1 为什么采用 Decoder-only

虽然原版 OneRec 使用 Encoder-Decoder，但对于 TAAC2025，本方案采用 **Decoder-only** 更合适：

1. 用户历史长度较短，最大仅 100，不需要复杂多路 Encoder；
2. Decoder-only 更接近“历史前缀 → 继续生成目标 token”的语言模型式建模；
3. 后续若继续扩展 action 生成、条件生成、多 token item 表示，都更自然；
4. 技术创新奖方案本身也验证了 Decoder-only 在该赛题中的可行性。

### 5.2 输入序列设计

主模型输入为：

```text
[user_token, item_1_token, item_2_token, ..., item_T_token, BOS]
```

其中，`user_token` 表示用户静态特征，`item_t_token` 表示第 \(t\) 个历史交互 item。

#### 5.2.1 user token

```text
user_token =
user sparse features
```

#### 5.2.2 item token

```text
item_token =
sid1 embedding
+ sid2 embedding
+ item sparse features
+ multimodal features
+ action-conditioned representation
+ time features
```

这里历史 item 使用 SID 表示，而不是单独维护大规模 item ID embedding 表。这样可以让输入和输出共享同一语义空间，也更贴近 OneRec 的设计思想。

### 5.3 Action-conditioning

TAAC2025 决赛中，历史序列与目标都包含 `exposure / click / conversion`，且 conversion 在评估中具有更高价值。因此，不同行为不应被简单混成同一种交互。

本方案采用 **per-position action-conditioning**：

- 对历史中每个 item token，根据其 action type 进行调制；
- 让模型区分“这个 item 只是曝光过”“用户点击过”“用户转化过”。

可以采用如下形式：

#### 5.3.1 Gated Fusion

\[
g_t=\sigma(W_g[x_t;a_t])
\]

\[
\hat{x}_t=g_t\odot x_t+(1-g_t)\odot W_a a_t
\]

#### 5.3.2 FiLM 调制

\[
\gamma_t,\beta_t = \operatorname{MLP}(a_t)
\]

\[
\tilde{x}_t=\gamma_t\odot \hat{x}_t+\beta_t
\]

其中：

- \(x_t\)：原始 item 表示；
- \(a_t\)：action embedding；
- \(\tilde{x}_t\)：经过行为条件调制后的 item token。

这样可以让同一个 item 在不同历史行为下具有不同语义。

### 5.4 时间特征

只保留最有效且稳定的两类：

1. **相对时间间隔**
   \[
   \Delta t_t = t_t-t_{t-1}
   \]
   经 log-bucket 后 embedding 化。

2. **绝对时间特征**
   - `hour-of-day`
   - `day-of-week`

这些特征能够表达：

- 用户当前兴趣是否刚刚被激活；
- 浏览行为是否具有会话连续性；
- 广告交互是否存在日周期、周周期。

### 5.5 输出序列

模型输出定义为：

```text
action → sid1 → sid2
```

即：

\[
P(a,s_1,s_2|H)
=
P(a|H)\cdot P(s_1|H,a)\cdot P(s_2|H,a,s_1)
\]

其中：

- \(a\)：目标行为类型；
- \(s_1,s_2\)：目标 item 的两层 SID；
- \(H\)：用户历史序列。

这样做有两个好处：

1. 模型显式学习“下一步用户更可能产生哪种行为”；
2. SID 生成是在目标 action 条件下进行的，更有利于区分 click 与 conversion 对应的 item 分布。

### 5.6 训练目标

训练时完整序列可写为：

```text
[user, item_1, ..., item_T, BOS, action*, sid1*, sid2*]
```

但 loss 只计算目标端三个 token：

\[
\mathcal{L}
=
\lambda_a \mathcal{L}_{action}
+
\lambda_1 \mathcal{L}_{sid1}
+
\lambda_2 \mathcal{L}_{sid2}
\]

默认可先取：

\[
\lambda_a=\lambda_1=\lambda_2=1
\]

若后续发现 action 预测过强或过弱，再做权重调整。

---

## 6. 阶段 C：推理流程

### 6.1 联合生成

推理时，给定历史前缀：

```text
[user, item_1, ..., item_T, BOS]
```

模型依次生成：

```text
action → sid1 → sid2
```

联合分数为：

\[
\log P(a,s_1,s_2|H)
=
\log P(a|H)
+
\log P(s_1|H,a)
+
\log P(s_2|H,a,s_1)
\]

### 6.2 合法 SID 约束解码

由于离线阶段已经获得完整的：

```text
sid1 → 合法 sid2 集合
```

因此推理时在生成 `sid2` 时，只允许选择真实存在的合法后缀，避免非法 SID pair。

这一步不改变模型训练，但能显著提升生成稳定性。

### 6.3 SID 反查 item

对合法 `(sid1, sid2)` 进行反查：

```text
(sid1, sid2) → item_id
```

### 6.4 历史行为过滤

在候选结果中，过滤用户历史已经交互过的 item：

```text
若 item_id ∈ user_history，则丢弃
```

该策略能够避免重复推荐历史物品，是比赛场景中非常高性价比的推理优化。

### 6.5 最终输出

经过联合生成、合法路径约束、SID 反查与历史过滤后，输出 Top-10 item。

---

## 7. 当前版本明确不加入的内容

为了保持方案主线清晰，当前版本暂不引入以下机制：

1. `InfoNCE auxiliary head + rerank`
2. MoE 主干
3. OneRec 的 RL / reward system
4. lifelong pathway
5. GNN 图增强
6. 多窗口热度特征
7. random-k / SID dropout

这些方法并非无效，而是会让当前方案从“清晰的 OneRec-like 直接生成路线”变成更复杂的混合系统，不适合作为第一版主方案。

---

## 8. 本方案与已有方案的关系

### 8.1 与原版 OneRec 的关系

保留：

- item 离散为 SID；
- 历史可使用 SID 表示；
- 直接自回归生成目标 SID；
- 推理时由 SID 映射回 item。

调整：

- 原版 OneRec 的 Encoder-Decoder 改为 Decoder-only；
- 原版三层 SID 改为适配 TAAC 数据规模的两层 SID；
- 原版工业多路 Encoder、reward system、RL 不保留。

### 8.2 与技术创新奖方案的关系

保留：

- 专用 Decoder-only Transformer + InfoNCE 学习协作 embedding；
- 用协作 embedding 构造 SID；
- 二级碰撞解决；
- Decoder-only 生成建模思想。

调整：

- 主模型不采用其完整的“SID 生成 + 排序统一框架”；
- 当前只保留必要的 action 生成，不引入更多工程增强。

### 8.3 与冠军方案的关系

借鉴：

- action-conditioning；
- 时间特征；
- 大负样本库，但仅用于 SID 构造阶段。

不采用：

- 生成 user vector + ANN 检索的主路线；
- random-k；
- 过多复杂特征工程。

---

## 9. 最终一句话概括

> 本方案首先利用专用 Decoder-only Transformer 与 InfoNCE 学习包含协作信号的 item embedding，再通过两级 RQ-KMeans 与二级碰撞解决构造唯一化 Semantic ID；随后采用 Decoder-only 生成模型，将用户历史作为前缀上下文，结合 action-conditioning 与时间特征，直接自回归生成目标 item 的 `action → sid1 → sid2`，并在推理阶段通过合法 SID 约束解码与历史行为过滤得到最终 Top-10 推荐结果。

---

## 10. 参考依据

1. **OneRec Technical Report**：OneRec 将推荐转化为直接生成 Semantic ID 的端到端生成式范式，并验证了 SID 作为历史输入表示的可行性。
2. **The Tencent Advertising Algorithm Challenge 2025: All-Modality Generative Recommendation**：总结了冠军、季军及技术创新奖方案，包括 action-conditioning、时间特征、技术创新奖的 decoder-only + 协作 SID 构造等关键设计。
3. **OnePiece: The Great Route to Generative Recommendation**：说明了两层 collaborative tokenizer、16K × 16K SID 空间、重分配降低碰撞的可行性。
