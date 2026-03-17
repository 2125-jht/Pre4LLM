# LLM通用推理增强项目 - 面试问答整理

> 基于 RLPR (Reinforcement Learning with Probability Reward) 的大模型推理能力增强项目

---

## 一、项目背景与技术架构

### Q1: verl 框架是什么？训练时为什么要用 vLLM？

**核心要点：**

verl (Vulkan Engine RL) 是一个**解耦式 RLHF 训练框架**，它将训练过程分为两个主要部分：

1. **Actor/Policy 训练**：使用 PyTorch/DeepSpeed/FSDP 进行标准的模型参数更新
2. **Rollout（采样）**：使用 vLLM 或 SGLang 等高效推理引擎生成 responses

**为什么训练时需要 vLLM：**

| 方面 | 说明 |
|-----|------|
| 问题背景 | 在 RL 训练（特别是 GRPO/PPO）中，每步训练需要：① Rollout 阶段生成 responses ② Advantage 计算 ③ Policy 更新 |
| 传统方案 | 如果直接用 PyTorch 做 generation，吞吐量很低 |
| verl 的解决方案 | 将当前 policy 的权重同步到 vLLM 引擎，vLLM 以高并发方式生成所有 responses（利用 Continuous Batching 和 PagedAttention），生成后再传回训练流程计算 loss |

**配置示例：**
```yaml
actor_rollout_ref:
  rollout:
    name: vllm  # 关键配置：使用 vLLM 作为 rollout 引擎
    tensor_model_parallel_size: 1
    gpu_memory_utilization: 0.6
```

**面试标准回答：**
> "verl 框架在训练时确实会调用 vLLM，但仅限于 rollout 阶段。具体来说：① Rollout：使用 vLLM 加载当前策略模型的权重，高效生成8个（group size）候选 responses，利用其 continuous batching 提升吞吐；② Training：生成完成后，vLLM 的 outputs 会传给 actor model 计算 log probs 和 GRPO loss，这部分回到 PyTorch 训练流程；③ 权重同步：每轮 policy update 后，verl 会将最新的权重同步到 vLLM 的 worker 进程中。这种设计解耦了 training 和 generation，既享受 vLLM 的高吞吐，又保持训练的灵活性。"

**注意事项：**
- ⚠️ **不是** "训练时用了 vLLM 做 backward" - vLLM 只负责 forward generation（生成 tokens），backward 梯度计算仍在 PyTorch 中
- 权重同步开销：每次 policy update 后需要同步权重，如果模型很大（如70B），这部分 overhead 需要考虑
- 替代方案：verl 也支持其他 rollout 引擎（如 SGLang、原生 PyTorch），但 vLLM 是最常用的默认选项

---

## 二、RLPR 核心技术：概率奖励机制的三大创新

> **重要澄清**：以下三个技术**不是**为了解决 Reward Hacking，而是为了让"概率作为奖励信号"这个核心思路能够稳定工作。Reward Hacking 主要通过 GRPO 的组内对比 + KL=0 配合其他技巧来解决。

### 2.1 平均概率（解决高方差问题）

**问题本质：**
- 传统方法用概率乘积（Sequence Likelihood）计算奖励：$r = \prod p_i$ 或几何平均
- **致命缺陷**：只要有一个 token 概率很低（如 0.01），整个奖励就会崩塌，即使其他 token 概率很高（0.9）
- 例子：$(0.01, 0.9, 0.9)$ 和 $(0.001, 0.9, 0.9)$ 的乘积差10倍，但实际上只差第一个 token 的微小概率

**RLPR 的解决：**
- 改用**算术平均**：$r = \frac{1}{|y|} \sum \log p_i$
- **优势**：对异常 token 不敏感，奖励分布更稳定，与答案质量相关性更高（AUC 提升 2-20%）

**面试话术：**
> "我们发现直接用概率乘积（likelihood）作为奖励时，如果参考答案里有一个生僻词导致模型给的 probability 很低，整个奖励就会骤降，哪怕推理过程是对的。所以我们改用平均概率，这样单个 token 的波动不会主导整体奖励，训练更稳定。"

---

### 2.2 奖励去偏策略（Debiasing）

**问题本质：**
- 模型对参考答案的解码概率受两个因素影响：
  1. **推理质量**（你想要的）：推理过程越好，对答案概率越高
  2. **题目本身难度**（偏差）：有的题目天生难（如高数题），无论怎么推理，概率都低；有的答案本身生僻，概率天然低

**Debiasing 做法：**
- 计算基线概率 $r'$：把推理过程去掉，直接问模型"这个问题的答案是什么"的概率
- 最终奖励：$\hat{r} = clip(0, 1, r - r')$
- **直观理解**：奖励只保留"有了这段推理比没有推理提升了多少"，消除题目本身难度的干扰

**面试话术：**
> "直接拿概率当奖励会有偏差——难题本身概率就低，简单题概率就高。所以我们计算一个基线：让模型不经过推理直接猜答案的概率，最终奖励是用"有推理的概率减去没推理的概率"，这样只保留推理带来的增量收益。"

---

### 2.3 自适应标准差过滤（StdDev Filtering）

**问题本质：**
- RL 训练中，有些样本太简单（8个回答全对，奖励都很高）或太难（8个回答全错，奖励都很低）
- 这些样本梯度信号弱（没有区分度），训练浪费计算，甚至导致模型退化

**过滤逻辑：**
- 对每个 prompt 采样 8 个回答，计算这8个奖励的标准差 $\sigma$
- 标准差低 → 8个奖励差不多（太简单或太难）→ 过滤掉
- **自适应**：不用固定阈值，而是用指数移动平均（EMA）动态调整：$\beta = 0.5 \times EMA(\sigma)$，适应训练过程中模型能力的变化

**面试话术：**
> "训练中发现有些题要么太简单（所有回答奖励都很高），要么太难（都很低），这样样本对模型学习没帮助。我们计算每个 prompt 生成8个回答的奖励标准差，标准差太低的就过滤掉，而且这个阈值是动态调整的，随着模型变强，标准差会相应提高，形成自适应课程学习。"

---

### 2.4 防追问清单

如果被追问细节，必须能回答：

| 问题 | 答案 |
|-----|------|
| 为什么不用中位数而用均值？ | → 均值计算简单，且实验表明与质量相关性足够好 |
| Debiasing 后奖励会是负数吗？ | → 不会，用 clip(0,1) 裁剪到 [0,1] 区间 |
| 过滤掉多少数据？ | → 通常 10-30%，具体看 EMA 动态阈值 |
| 这三个技术哪个最重要？ | → **Probability Reward 是基础**，Debiasing 和 StdDev Filtering 是为了让它在 RL 训练中稳定工作 |

**建议：** 把这三个概念用大白话讲给非技术朋友听，如果他听懂，面试时你就稳了。

---

## 三、Reward Hacking 是如何解决的？

**澄清：** 以下才是项目中解决 Reward Hacking 的核心机制，不要与上面的三个技术创新混淆。

### 3.1 Reward Hacking 是什么？

Reward Hacking 指模型 exploit reward model 的缺陷，找到捷径获得高奖励但实际能力退化。在 RLPR 中，可能表现为：
- 模型学会生成看似合理但实际错误的推理过程
- 利用概率计算的漏洞（如生成极短答案）

### 3.2 项目的解决方案

| 方案 | 机制 | 作用 |
|-----|------|------|
| **GRPO 组内相对优势** | 同一问题的8个回答归一化（减去均值除以标准差） | 限制极端行为，防止单一回答骗取高分 |
| **KL = 0 + clip-high** | 不约束与参考模型的KL，但裁剪范围设为 [0.73, 1.27] | 允许探索但防止过度优化 |
| **Entropy 奖励** | `entropy_coef=1e-3` | 鼓励输出多样性，防止模式坍缩 |
| **StdDev Filtering** | 过滤低方差样本 | 间接作用，确保训练数据有区分度 |

**注意：** StdDev Filtering 主要是为了提高训练效率（过滤无效样本），对防止 Reward Hacking 有间接帮助但不是主要目的。

---

### 3.3 其他主流解决方法（面试扩展知识）

除了本项目使用的方案，业界还有以下主流方法应对 Reward Hacking：

#### 1. Reward Model Ensembling（奖励模型集成）

**原理**：训练多个不同的 Reward Model，综合它们的打分（如投票、平均、加权）

**优势**：
- 单一 RM 的缺陷可能被其他 RM 发现
- 提高奖励信号的鲁棒性

**代表工作**：OpenAI 的 Constitutional AI、Anthropic 的 RLHF 实践

**面试话术**：
> "我们项目用概率奖励天然避免了 RM 的 hack，但如果用传统 RLHF，一个常见做法是训练多个 RM 做 ensemble，这样模型很难同时骗过所有打分器。"

---

#### 2. Process-based Reward（过程奖励）

**原理**：不仅奖励最终答案，还奖励正确的推理步骤

**实现方式**：
- **Outcome Reward Model (ORM)**：只判断最终答案对错（传统方法）
- **Process Reward Model (PRM)**：判断每一步推理是否正确

**优势**：
- 防止模型靠"蒙"得到正确答案
- 更细粒度的监督信号

**代表工作**：OpenAI 的 Let's Verify Step by Step (2023)

**局限**：
- 需要人工标注每一步的正确性，成本极高
- 步骤边界难以定义

**面试话术**：
> "OpenAI 提过 Process Reward，就是不仅看最终答案对不对，还看每一步推理是否正确。但这需要大量人工标注步骤级别的标签，成本很高。我们项目通过 Debiasing 技巧，用概率差来间接衡量'推理带来的提升'，也算是一种轻量级的过程监督。"

---

#### 3. Constrained RL / Cost Model（约束强化学习）

**原理**：除了主奖励外，增加约束条件（如安全约束、格式约束）

**实现**：
- **CPO (Constrained Policy Optimization)**：在优化奖励的同时满足约束条件
- **Cost Model**：单独训练一个成本模型惩罚不良行为

**代表工作**：
- DeepSeek-R1 在推理任务中设置格式奖励（要求按 `<think>...</think><answer>...</answer>` 格式输出）
- Anthropic 的 Constitutional AI 中的约束条件

**面试话术**：
> "除了主奖励，很多工作会加约束条件。比如 DeepSeek-R1 除了答案正确性，还会给格式奖励，强制模型按 `<think>` 标签输出推理过程。这也是一种防止 Reward Hacking 的手段——如果模型不展示推理过程，格式奖励就是0。"

---

#### 4. Red Teaming / Adversarial Training（红队测试/对抗训练）

**原理**：主动寻找模型的弱点，用对抗样本训练提高鲁棒性

**流程**：
1. 用红队模型生成可能导致 Reward Hacking 的输入
2. 人工或自动标注这些输入的正确奖励
3. 将对抗样本加入训练集重新训练 RM 或 Policy

**代表工作**：Anthropic 的 Red Teaming 实践、DeepSeek 的对抗训练

**面试话术**：
> "大厂会做 Red Teaming，就是专门找模型漏洞。发现模型 exploit 奖励的套路后，把这些案例加入训练集再训练。这是一个持续迭代的过程。"

---

#### 5. Human-in-the-loop（人工介入）

**原理**：在训练循环中引入人工审核，及时纠正模型的 Hack 行为

**应用场景**：
- 发现模型输出看似合理但实际错误时，人工标注正确偏好
- 定期人工抽检模型输出质量

**局限**：成本高，难以规模化

**面试话术**：
> "终极方案还是人工介入。当发现模型找到奖励漏洞时，人工标注正确的偏好对，重新训练 RM。但这对人力要求很高，我们项目通过概率奖励的设计，减少了对人工偏好标注的依赖。"

---

#### 6. Reward Model Regularization（奖励模型正则化）

**原理**：在训练 Reward Model 时加入正则化项，防止其过拟合到表面的相关性

**技术**：
- **Dropout**：在 RM 中加入 dropout 防止过拟合
- **Label Smoothing**：软化标签，避免模型过度自信
- **Confidence Penalty**：惩罚过于自信的预测

**面试话术**：
> "训练 RM 本身也有很多技巧，比如 dropout、label smoothing，防止 RM 学到表面的 hack 模式。但根本上，RM 还是可能被 exploit，所以 DeepSeek-R1、Kimi 这些前沿工作都转向了 RLVR 或概率奖励，绕开 RM 这个'中间商'。"

---

### 3.4 方法对比总结

| 方法 | 核心思想 | 优点 | 缺点 | 本项目是否使用 |
|-----|---------|------|------|--------------|
| **GRPO 组内对比** | 同一问题的多个回答相互比较 | 无需外部参考，自包含 | 需要多次采样，计算量增加 | ✅ 使用 |
| **KL 约束** | 限制与参考模型的偏离 | 简单有效 | 可能限制探索 | ❌ 设为0，用 clip 替代 |
| **Entropy 奖励** | 鼓励输出多样性 | 防止模式坍缩 | 需要调参平衡 | ✅ 使用 |
| **RM Ensemble** | 多个 RM 投票 | 提高鲁棒性 | 训练成本高 | ❌ 本项目无 RM |
| **Process Reward** | 奖励每一步推理 | 细粒度监督 | 标注成本极高 | ❌ 未使用 |
| **Cost Model** | 增加约束条件 | 可控性强 | 需要设计约束 | ✅ 格式奖励隐含 |
| **Red Teaming** | 对抗训练 | 主动发现漏洞 | 需要持续投入 | ❌ 未涉及 |
| **Human-in-the-loop** | 人工纠正 | 最可靠 | 不可规模化 | ❌ 本项目减少人工依赖 |

**面试总结话术**：
> "Reward Hacking 是 RLHF 的核心难题，业界有多种应对思路。传统方法是加 KL 约束、做 RM Ensemble；更先进的思路是绕开 RM 直接用规则验证（RLVR）或概率奖励（RLPR）；或者像 OpenAI 那样用 Process Reward 给每一步打分。我们项目选择了**无 RM 的概率奖励路线**，配合 GRPO 组内对比和 entropy 奖励，从源头上减少了 Hack 的空间。"

---

## 四、RLPR 与其他方法的区别

### Q3: 提升模型通用推理能力，现在一般都用什么技术路径？相比其他方法，RLPR 有什么优势？

#### 主流技术路径全景

提升模型通用推理能力的主流技术路径目前主要分为以下几类，RLPR 属于 RLVR（可验证奖励强化学习）路线在开放域/不可验证领域的关键扩展：

| 路径 | 做法 | 局限 |
|-----|------|------|
| **纯监督微调 (SFT) 路线** | 用高质量的 CoT（思维链）数据直接微调模型，如 DeepSeek-R1 的冷启动阶段、OpenAI 早期的 InstructGPT | 数据标注成本高（需要人工写长推理过程），模型容易过拟合特定格式，难以探索新的推理模式 |
| **传统 RLHF（基于奖励模型）** | 训练 Reward Model (RM) 评估回答质量，用 PPO 算法优化 | RM 需要大量偏好对比数据（哪个回答更好），且 RM 本身是"黑盒"，在复杂推理上容易给流畅但错误的推理高分（reward hacking），难以处理自由文本答案 |
| **RLVR（可验证奖励强化学习）** —— 当前最火路线 | 使用规则验证器判断答案对错，给二元奖励（对=1，错=0） | 仅限可验证领域（数学、代码），无法扩展到开放域问答（如"解释光合作用"） |
| **Test-time Scaling（推理时扩展）** | 不训练模型，而是让模型在推理时生成多条路径（如 CoT sampling + Self-consistency/Voting） | 仅提升推理时表现，不改进模型本身能力；计算成本高 |

#### RLPR（平均解码概率奖励）的核心优势

RLPR 属于 RLVR 路线的突破性扩展，它解决了上述路径的关键瓶颈：

| 优势 | 说明 |
|-----|------|
| **相比传统 RLHF：无需偏好数据，信号更客观** | RLHF：需要人工标注 "A 比 B 好" 的偏好对，成本高且主观性强；RLPR：只需要问题+参考答案（类似 SFT 数据），奖励是模型对参考答案的客观概率，不是学出来的 RM 打分，避免 reward hacking |
| **相比规则验证器 RLVR：打破领域限制** | 规则 RLVR：只能做数学/代码（有确定答案），"解释光合作用"这类问题无法验证；RLPR：通过概率奖励（模型对正确答案的信度）间接评估推理质量，天然支持开放域自由文本，是第一个真正将 RLVR 扩展到通用领域的方法 |
| **相比模型验证器：极简架构，零额外成本** | General Reasoner：需要单独训练 1.5B 参数的 Verifier 模型，训练和推理都更复杂；RLPR：无需任何外部验证器，奖励计算就是一次前向传播（算概率），架构简单，没有级联错误风险 |
| **技术独特性：Mean Probability vs Likelihood** | 传统方法（如 VeriFree）：用序列似然（概率乘积）做奖励，对长答案极不稳定（一个生僻词导致奖励崩盘），只能处理短答案（<7 tokens）；RLPR：用平均解码概率（Mean Token Prob），对答案长度鲁棒，支持长文本推理（如定理论的长解释），且与答案质量相关性更高（AUC 提升 2-20%） |
| **数据效率极高** | 不需要像 RLHF 那样选偏好对，也不需要像训练 Verifier 那样标注验证数据；只需要通用知识库（如 WebInstruct）的问题+答案，数据获取成本与 SFT 相当，但能获得 RL 的探索能力 |

#### 面试时的标准回答框架

如果面试官问："现在提升推理能力有哪些方法？你们这个 RLPR 有什么优势？"

**推荐回答：**
> "目前主要有三条路：
> 1. **SFT**：直接用高质量 CoT 数据教，但容易过拟合，且标注贵；
> 2. **RLHF**：用 Reward Model 打分，但 RM 容易被 fooled by fluency，且需要大量偏好数据；
> 3. **RLVR**：用验证器给奖励，这是 DeepSeek-R1 的路，但传统 RLVR 只能做数学/代码（有规则验证），开放域做不了。
> 
> 我们做的 **RLPR 属于第四条路**：它继承了 RLVR 的强探索能力，但通过模型对参考答案的内在概率作为奖励，不需要任何外部验证器，既保留了 RL 的训练效率，又突破了只能做可验证任务的限制，在 MMLU-Pro、TheoremQA 这些开放域 benchmark 上超过了用 1.5B Verifier 的 General Reasoner。"

**关键记忆点：** RLPR = RLVR 的训练效率 + 开放域的通用性 + 零额外模型成本

---

## 五、关于 Verifier 的深入探讨

### Q4: 你们提到用概率做奖励，这是之前已有的方法吗？VerifyFree 之类的工作你们有参考吗？

是的，基于模型内在概率的奖励信号在 RLPR 之前确实被探索过，但用于替代 verifier 进行通用域 RLVR 训练是 RLPR（及同期工作 VeriFree）的开创性贡献。

#### 先前探索：概率作为奖励的雏形

在 RLPR 之前，已有一些相关工作使用模型概率作为信号：

- **Self-Consistency / Confidence Estimation**：早期工作发现模型对同一问题的多次采样一致性（或概率阈值）可以反映答案质量，但这主要用于推理时筛选（test-time），而非训练时优化
- **Self-Reward / Self-Play**：如 Zhao et al. (2025) 等尝试用模型自举率提升一个答案的权重，但主要用于无参考答案的场景（通过多数投票生成伪标签），而非利用参考答案的概率进行有监督的 RL 优化

#### 同期工作：VeriFree 的 Sequence Likelihood

**VeriFree** (Zhou et al., 2025，与 RLPR 同期) 是第一个明确尝试用参考答案概率替代 verifier 进行 RL VR 的工作：

- **方法**：使用 Sequence Likelihood（概率乘积 $P(y) = \prod p_i$）作为奖励
- **致命局限**：由于乘积的高方差特性，必须限制参考答案长度 < 7 tokens，否则奖励信号会崩塌（一个生僻词导致整体概率极低）
- **结果**：只能处理短答案（如选择题选项、简短填空），无法支持通用领域的长文本推理（如科学解释、定理论证）

#### RLPR 的关键突破：Mean vs Product

RLPR 的核心创新在于改进了概率计算方式：

| 方法 | 计算方式 | 局限 | RLPR 改进 |
|-----|---------|------|----------|
| VeriFree | Sequence Likelihood（概率乘积） | 对长答案极不稳定，长度限制 < 7 tokens | 改用 **Mean Probability**，对答案长度鲁棒 |
| RLPR | Mean Token Probability | 支持长文本推理，与质量相关性更高 | - |

**论文依据（RLPR Paper p.8）：**
> "We compare our per-token probability-based reward with naive sequence likelihood as the reward signal... (Zhou et al., 2025) proves this instability by filtering out prompts whose reference answers exceed seven tokens. In contrast, using the mean per-token probability is much more robust."

#### 面试时的标准回答

如果面试官问："用概率做奖励是不是你们首创？"

**推荐回答：**
> "模型内在概率做奖励的概念之前有过探索（比如 confidence-based filtering），但用于替代 verifier 做 RLVR 训练是 RLPR 和同期 VeriFree 的开创性工作。
> 
> 关键区别在于：
> - **VeriFree** 用的是序列似然（概率乘积），导致奖励方差极大，只能处理 7 个 token 以内的短答案，无法做通用推理
> - **我们用的** 是平均概率，解决了高方差问题，第一次让这种无需 verifier 的方法支持长文本推理（如 TheoremQA 的长解释），在 7 个 benchmark 上验证了有效性
> 
> 核心记忆点：**RLPR 不是第一个用概率的，但是第一个用 mean probability 解决高方差、突破长度限制、真正支持通用域长推理的。**"

---

## 六、KL coef = 0 的设计

### Q5: 为什么 RLPR 把 KL 系数设为0？这是什么考虑？

$KL\_coef = 0$ 表示在训练过程中完全移除了 KL 散度约束，这是 RLPR（以及 DeepSeek-R1、Kimi k1.5 等近期工作）的关键工程选择。

#### 常规用法（KL coef > 0）

在标准 PPO/RLHF 中，KL 约束用于限制当前策略（policy）与参考模型（reference model，通常是初始 SFT 模型）的差异：

- **目的**：防止模型为了迎合奖励信号而偏离原始分布太远（避免遗忘通用知识或产生 gibberish）
- **机制**：在 loss 中加入 $\beta \cdot KL(\pi_\theta || \pi_{ref})$，$\beta$ 越大，惩罚越重

#### 为什么 RLPR 设为 0（KL coef = 0）

RLPR 论文（Table 7: Implementation Details）明确设置 `kl_coef=0`，原因如下：

##### A. GRPO 的组间对比自带稳定性

GRPO 使用**组内相对优势**（group relative advantage），即把同一问题生成的 8 个回答的奖励归一化（减去均值除以标准差）。这种**相对比较机制本身就限制了极端行为**，不需要额外的 KL 惩罚。

##### B. 鼓励充分探索（Exploration）

- 有 KL 约束：模型被"拴"在初始模型附近，不敢偏离太远，可能错过更优的推理模式
- **KL=0**：允许模型充分探索新的推理路径（如自我修正、多步验证），这是涌现长思维链（long CoT）的关键

##### C. 与 DeepSeek-R1 的实验一致

DeepSeek-R1（你冷启动提到的范式）也采用了 **KL-free** 的设置，实践证明在推理任务中，去掉 KL 约束反而能激发更强的推理能力（代价是需要配合 clip-high 等技巧防止 entropy collapse）。

#### 风险与配套措施

不设 KL 约束的风险是模型可能模式崩溃（mode collapse）或生成乱码。RLPR 的解决方案：

- **Clip-high 技巧**：裁剪上限设为 1.27（而非 PPO 标准的 1.2），允许更大更新步长但仍防止过度优化
- **Entropy 奖励**：设置 `entropy_coef=1e-3`，直接鼓励策略保持输出多样性
- **StdDev Filtering**：过滤奖励方差过低的样本（太简单/太难），确保每批数据都有有效梯度信号

#### 面试标准回答

如果面试官问："KL=0 不会导致模型崩溃吗？"

**推荐回答：**
> "传统 PPO 设 KL 约束是为了防止策略偏离参考模型太远，但 **GRPO 通过组内相对优势已经提供了稳定性**。在 RLPR 中我们设 KL=0，配合 clip-high=1.27 和 entropy 奖励，目的是充分释放模型的探索能力。实践证明，去掉 KL 约束后模型能学到更复杂的推理模式（如自我反思），最终性能超过了带 KL 约束的基线。
> 
> 关键记忆点：**KL=0 不是偷懒，而是主动选择**——用 GRPO 的组内归一化替代 KL 约束，换取更大的探索空间，这是当前推理模型训练的主流趋势（DeepSeek-R1/Kimi 同款设置）。"

---

## 七、总结：面试核心记忆点

| 主题 | 一句话总结 |
|-----|-----------|
| **verl + vLLM** | 解耦训练与生成，vLLM 只做 rollout，训练回 PyTorch |
| **平均概率** | 解决乘积高方差，支持长文本，AUC 提升 2-20% |
| **Debiasing** | $r - r'$ 消除题目难度干扰，只保留推理增量 |
| **StdDev Filtering** | EMA 动态阈值过滤低方差样本，自适应课程学习 |
| **Reward Hacking 解决** | GRPO组内归一化 + KL=0 + clip-high + entropy奖励 |
| **RLPR vs 其他** | = RLVR 效率 + 开放域通用性 + 零额外模型成本 |
| **KL=0** | GRPO 组内归一化替代 KL，换取更大探索空间 |

**关键区分（不要混淆）：**
- ✅ 平均概率、Debiasing、StdDev Filtering → **概率奖励机制的技术创新**
- ✅ GRPO组内对比、KL=0、clip-high、entropy奖励 → **防止 Reward Hacking 的机制**

**最后建议：** 把这些概念用自己的话讲 3 遍，直到能脱稿讲给非技术朋友听明白，面试就稳了。
