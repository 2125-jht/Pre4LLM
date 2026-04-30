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

#### 1. Reward Model Ensembling（奖励模型集成）
> "我们项目用概率奖励天然避免了 RM 的 hack，但如果用传统 RLHF，一个常见做法是训练多个 RM 做 ensemble，这样模型很难同时骗过所有打分器。"

#### 2. Process-based Reward（过程奖励）
> "OpenAI 提过 Process Reward，就是不仅看最终答案对不对，还看每一步推理是否正确。但这需要大量人工标注步骤级别的标签，成本很高。我们项目通过 Debiasing 技巧，用概率差来间接衡量'推理带来的提升'，也算是一种轻量级的过程监督。"

#### 6. Reward Model Regularization（奖励模型正则化）
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

**面试总结话术**：
> "Reward Hacking 是 RLHF 的核心难题，业界有多种应对思路。传统方法是加 KL 约束、做 RM Ensemble；更先进的思路是绕开 RM 直接用规则验证（RLVR）或概率奖励（RLPR）；或者像 OpenAI 那样用 Process Reward 给每一步打分。我们项目选择了**无 RM 的概率奖励路线**，配合 GRPO 组内对比和 奖励去偏策略，从源头上减少了 Hack 的空间。"

---

## 四、RLPR 与其他方法的区别

### Q3: 提升模型通用推理能力，现在一般都用什么技术路径？相比其他方法，RLPR 有什么优势？
> "目前主要有三条路：
> 1. **SFT**：直接用高质量 CoT 数据教，但容易过拟合，且标注贵；
> 2. **RLHF**：用 Reward Model 打分，但 RM 容易被 fooled by fluency，且需要大量偏好数据；
> 3. **RLVR**：用验证器给奖励，这是 DeepSeek-R1 的路，但传统 RLVR 只能做数学/代码（有规则验证），开放域做不了。
> 
> 我们做的 **RLPR 属于第四条路**：它继承了 RLVR 的强探索能力，但通过模型对参考答案的内在概率作为奖励，不需要任何外部验证器，既保留了 RL 的训练效率，又突破了只能做可验证任务的限制，在 MMLU-Pro、TheoremQA 这些开放域 benchmark 上超过了用 1.5B Verifier 的 General Reasoner。"

### Q4: 你们提到用概率做奖励，这是之前已有的方法吗？
> "模型内在概率做奖励的概念之前有过探索（比如 confidence-based filtering），但用于替代 verifier 做 RLVR 训练是 RLPR 和同期 VeriFree 的开创性工作。
> 
> 关键区别在于：
> - **VeriFree** 用的是序列似然（概率乘积），导致奖励方差极大，只能处理 7 个 token 以内的短答案，无法做通用推理
> - **我们用的** 是平均概率，解决了高方差问题，第一次让这种无需 verifier 的方法支持长文本推理（如 TheoremQA 的长解释），在 7 个 benchmark 上验证了有效性
> 
> 核心记忆点：**RLPR 不是第一个用概率的，但是第一个用 mean probability 解决高方差、突破长度限制、真正支持通用域长推理的。**"

### Q5: 为什么 RLPR 把 KL 系数设为0？这是什么考虑？
> "传统 PPO 设 KL 约束是为了防止策略偏离参考模型太远，但 **GRPO 通过组内相对优势已经提供了稳定性**。在 RLPR 中我们设 KL=0，配合 clip-high=1.27 和 entropy 奖励，目的是充分释放模型的探索能力。实践证明，去掉 KL 约束后模型能学到更复杂的推理模式（如自我反思），最终性能超过了带 KL 约束的基线。
> 
> 关键记忆点：**KL=0 不是偷懒，而是主动选择**——用 GRPO 的组内归一化替代 KL 约束，换取更大的探索空间，这是当前推理模型训练的主流趋势（DeepSeek-R1/Kimi 同款设置）。"

---

**关键区分（不要混淆）：**
- ✅ 平均概率、Debiasing、StdDev Filtering → **概率奖励机制的技术创新**
- ✅ GRPO组内对比、KL=0、clip-high、entropy奖励 → **防止 Reward Hacking 的机制**