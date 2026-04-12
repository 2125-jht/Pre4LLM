# SFT构造数据的一些经验

> 作者：沐波

---

## 〇、SFT的定位变了，先把这事儿说清楚

以前我们对 post-training 的理解很朴素：SFT 教模型干活，RLHF 教模型做人。DeepSeek-R1 出来之后这个认知被打破了——R1-Zero 连 SFT 都没做，直接拿 base model 跑 RL，推理能力就涌现了，自我验证、反思、长链推理都有。

但你要是因此觉得 SFT 可以不做了，那会摔得很惨。后面大量复现实验都指向同一个结论：**SFT 不是必须的，但没有它 RL 会非常难训，不稳定，收敛慢，还容易 reward hack。**

所以现在 SFT 的角色其实是三件事：

**第一，给 RL 一个好的起点。** 与其让 RL 在 base model 的混沌分布里大海捞针，不如先用 SFT 把模型拉到一个靠谱的初始位置。

**第二，把输出格式定下来。** 思维链怎么写、`<think>` 标签怎么用、分步推理的结构是什么样——这些东西 RL 不擅长从零学起，SFT 定好了 RL 才能在上面优化。

**第三，覆盖那些 RL 搞不定的能力。** 写作、多语言、角色扮演、该拒绝的时候拒绝——这些任务没法自动验证对错，RL 没有好的 reward signal，只能靠 SFT 硬灌。

搞清楚这个定位再去构造数据，很多决策就顺了。

---

## 一、Long-CoT：绕不过去的话题

过去一年 SFT 领域如果只能选一个关键词，那就是 Long-CoT。

### 蒸馏来的比手写的好得多

这个结论现在基本没争议了：从 DeepSeek-R1 这类强推理模型蒸馏出来的 CoT 轨迹，训练效果远好于人工设计的 CoT 模板。原因也不复杂——蒸馏出来的轨迹里包含了真实的回溯、分支探索、走错路再纠正这些行为，人写的模板通常只是在格式上"长得像"长链推理，内核是空的。

有一个很有意思的实验：把 CoT 里的关键词（比如 "wait"、"let me reconsider"）替换成别的词，效果几乎不变；但如果改变底层的推理结构和行为模式，效果立刻掉下来。说明 SFT 学到的是推理的"骨架"，不是"皮肤"。

所以别费劲手写思维链模板了。拿强模型蒸馏，哪怕生成的东西看着啰嗦，训出来的效果就是比精心设计的短 CoT 好。

### 但也不能一味追求长

长 CoT 有一个很烦人的副作用：模型在简单问题上也会启动"深度思考"模式，1+1=2 也要推理半天。Draft-Thinking、LS-Mixture SFT 这些工作都在解决这个问题，核心思路就是数据里要有长有短：

- **难题**给完整的长链推理，10K token 以上都行
- **中等难度**的问题，把推理链压缩一下，去掉冗余步骤
- **简单问题**就直接给答案，训练模型学会"这题不用想"

这个"难度自适应"现在已经是标准做法了。你要是数据集里全是长 CoT，训出来的模型推理一道小学算术题都要输出三千 token，用户体验会很差。

### Long-CoT SFT 是 RL 的前置条件

还有一个反复被验证的事情：先在 long-CoT 数据上做过 SFT 的模型，后续 RL 训得又快又稳。反过来，直接在 base model 上做 RL（所谓的冷启动），CoT 长度经常不受控地增长，模型会学会用重复来凑长度骗 reward，而不是真的在推理。

所以现在主流 pipeline 就是 **Long-CoT SFT → RL**。SFT 负责"学会推理的形式"，RL 负责"强化推理的实质"。做数据的时候也要想清楚哪些数据是给 SFT 阶段用的，哪些是给 RL 阶段留的。

---

## 二、数据选择：不是选"最好的"，是选"最合适的"

### 一条反直觉的结论

GRAPE 这篇论文（NeurIPS 2025 spotlight）给了我蛮大的冲击。它做的事情很简单：同一条指令有多个候选回答（来自不同模型），不是选人类觉得最好的那个，而是选目标模型自己觉得 log probability 最高的那个——也就是跟模型预训练分布最接近的回答。

结果呢？用 1/3 的数据量就超过了全量 SFT 的效果。

这意味着什么？意味着**"最好的回答"不存在，只有"对这个模型最合适的回答"**。同一条指令，Qwen 需要的最佳回答和 LLaMA 需要的最佳回答可能完全不同。一直以来大家默认"从最强 Teacher 蒸馏就对了"，这个假设是错的。

具体怎么做也不复杂：有多个候选回答的时候，跑一次 forward pass 算个 perplexity，选最低的那个。几乎零成本，但实测提升明显。

### 在长链推理上要多加注意

不过 GRAPE 的方法到了长链推理数据（10K-32K token）上就有点打折扣了。原因也好理解：student 模型在预训练阶段很少见过这么长的连贯推理，它对长序列的 log probability 估计本身就不太靠谱。

目前看到的几种补救方案：
- 用滑动窗口算局部 perplexity 而不是全局 log probability
- 按推理步骤来做匹配而不是序列级别的匹配
- 或者先用一小批 long-CoT 数据做一轮预热，让模型对长链推理有个基本认知后再做精选

---

## 三、Agent 轨迹数据

Agent 场景对 SFT 数据的要求跟传统场景完全不一样。传统 SFT 数据是一问一答的 pair，Agent 数据是一整条交互轨迹：推理 → 选工具 → 调工具 → 拿到结果 → 再推理 → 可能还要纠错 → 最终回答。一条数据几千 token、多次工具调用是常态。

### 真实轨迹 vs 拼接的轨迹

这个区别比很多人想象的大。在真实环境中端到端采集的 agent 轨迹上做 SFT，效果远好于把各个环节拼接起来的合成轨迹。4B 参数的小模型在真实轨迹数据上训出来的效果，可以超过之前 32B 模型的水平。

### 怎么采集

Agent 轨迹数据是目前最贵的 SFT 数据类型。几种主流做法：

**一种是让强模型在 Docker 化的真实环境里执行任务**，采集完整轨迹后做质量过滤。Nvidia 的 Nemotron-Terminal 就是这么干的——在 terminal 环境里让教师模型跑任务、录轨迹，拿这些轨迹训 8B 模型，效果能打过之前百 B 级别的模型。

**另一种是从非结构化数据（网页、PDF、图片）里合成原子任务**，再通过深度扩展（多跳串联）和宽度扩展（子任务聚合）组合成复杂任务，用 judge LLM 做质量把关。

**还有一种分阶段的思路**，实测比一步到位效果好：第一阶段让模型理解每个工具是干什么的，第二阶段学什么时候该用什么工具，第三阶段通过自反思样本学会从自己的错误中总结经验。

### 别忘了负样本

只给正样本训出来的 Agent 模型会有一个毛病：过于乐观。它默认每一步都会成功，一旦遇到报错或者意外结果就不知道怎么办。

数据里一定要有"不该调工具的时候不调""调错工具后怎么恢复""环境返回异常结果后怎么处理"这类样本。错误恢复轨迹可能是 Agent SFT 数据里最稀缺也最有价值的部分。

---

## 四、多模态 SFT 数据

纯文本 SFT 数据只是故事的一半。VLM 的 SFT 有自己的一套问题。

### 文本域的经验搬不过来

在文本域上效果很好的 Long-CoT SFT → RL pipeline，搬到多模态场景上效果不太稳定。有些工作发现 SFT 和 RL 在 VLM 上甚至会相互干扰。

一个有意思的现象：纯文本的 long-CoT 数据做 SFT，居然能一定程度上提升 VLM 的多模态推理能力——推理结构是跨模态的。但要真正做好，还是得有专门的多模态 CoT 数据：图表的结构化描述、空间关系的推理、时序推理等等。

### 领域专家这笔钱省不了

多模态场景下这一点尤其突出。一个放射科医生标注的医学影像数据，顶得上一百个众包标注员写的东西。道理也简单——不懂的人写出来的"分析"，模型学到的不是"如何分析"，而是"如何自信地说废话"。这就是幻觉的数据根源。

垂直领域多模态 SFT，宁可少但找对人，别贪多但找错人。

---

## 五、SFT 记忆，RL 泛化

"SFT memorizes, RL generalizes" 这个说法现在被引用得很多。简单说就是：

SFT 数据里有什么，模型就会忠实地复现什么——好的坏的都学。你数据里有幻觉，它学幻觉；有翻译腔，它学翻译腔；有"作为一个 AI 语言模型"，它也学。

RL 能在 SFT 画好的框架里面探索和优化，找到更好的解法，但它很难突破 SFT 定下来的格式和行为边界。SFT 阶段推理格式有毛病，RL 只会在这个有毛病的格式里"卷"。

这对做数据的人意味着几件事：

- **SFT 数据要广不要深**——尽量覆盖多的任务类型和推理模式，别在单个任务上堆量。
- **SFT 数据的错误率要极低**——因为错误会被忠实记忆，比"不够好"危害大得多。
- **SFT 和 RL 的数据要配合着来**——比如那些包含"aha moment"（顿悟、自我反思）的高挑战性样本，更适合留给 RL 而不是 SFT，因为 RL 更擅长从这种样本中学习。

---

## 六、数据清洗

### 推理模式级别的去重

以前做去重，用 embedding 相似度过滤语义重复就够了。现在不行了。两条 long-CoT 数据，问题不同、答案不同，但推理模式一模一样（都是"先分类讨论→逐一排除→综合结论"），这种情况 embedding 去重抓不住。

CoTP 这篇工作提了一个思路：用 reasoning pattern chain（推理模式链）来描述每条 CoT 的推理范式，然后在 pattern 层面做去重。比纯语义去重精细一个量级，能有效避免数据集里推理模式的同质化。

### 格式污染和身份泄露

现在 SFT 数据大量靠模型生成，"模型指纹"污染很普遍。常见的有：蒸馏数据里残留的 `<think>` 标签，"作为一个 AI 语言模型" 开头，Markdown 格式滥用——以及一个容易被忽略的问题：**身份泄露**。模型生成的回答里可能无意间说出"我是 DeepSeek"或"我是 ChatGPT"。如果你拿这些数据去训自己的模型，它也会这么说。

清洗 pipeline 里加一道身份泄露检测，成本很低但能省不少麻烦。Nemotron-Terminal 的论文里专门提到了这个过滤步骤。

### 数据配比别再拍脑袋了

传统做法是凭经验手调各类数据的比例。现在有更好的选择：gradient alignment、DRO-based reweighting 之类的自动化方法可以根据训练动态来调整各数据域的采样权重。虽然这些方法最初是给预训练设计的，但思路完全可以迁移到 SFT 配比上。

退一步讲，就算不用这些方法，至少也要维护一组 eval set，每次动了配比就跑一遍评估。别猜，看数字。

---

## 七、数据壁垒

一个不太好听但很现实的事情：**高质量 SFT 数据已经成了竞争壁垒。**

现在行业实际上分三层：
- 顶级闭源模型（GPT-5、Gemini 2.5、Claude 这些）有高度精炼的私有 SFT 数据
- 半开放模型放出权重但 data recipe 保密
- 完全开放的模型和数据集，性能差距明显

开源社区缺的不是算力，不是架构，是数据——尤其是**可迭代的数据构造 pipeline**。大多数开源数据集都是一次性发布的静态资源，背后的清洗代码、过滤逻辑、prompt 设计都不公开。

几个建议：别只盯着数据集本身看，多关注 curation pipeline 的可复现性。与其用一个大而全的公开数据集，不如拿 GRAPE 之类的方法从多个数据集里针对你的 base model 做定制化选择。垂直领域，花钱找领域专家标注仍然是 ROI 最高的选择，没有之一。

---

## 八、零散的 Tips

### Cold-start 数据要精不要多

R1 的训练里只用了几千条 cold-start CoT 数据，但每条都是精心打磨的——few-shot 引导生成，人工精修可读性，从 R1-Zero 的输出里筛选好的样本。这几千条数据决定了后面整个 RL 的上限。

### 蒸馏时优先保留"最短正确解"

同一道题让教师模型生成多个推理轨迹，留最短的那个正确解。推理质量有了，冗余步骤砍了，模型不容易学到"注水"的习惯。

### System prompt 的多样性

训练数据里如果 system prompt 只有一种写法（比如永远是"你是一个有用的 AI 助手"），模型到了新的 system prompt 下表现会断崖式下降。尤其 Agent 场景，system prompt 千差万别，数据构造时要刻意覆盖。

### 训练模型"不调工具"

Agent 数据里如果全是调用工具的正样本，模型会形成"收到消息就调工具"的条件反射。要刻意放一些"判断不需要调工具、直接回答"的样本进去。

### Targeted Human Feedback 能省标注预算

RLTHF 的思路是先用 LLM 做粗对齐，再用 reward model 找出"模型不确定"的难样本，只对这些做人工标注。据报道 6-7% 的人工标注量就能达到全量标注的效果。这个思路在 SFT 数据的人工验证环节也能用。

### 版本管理

最后一条也是最老生常谈但最多人做不到的一条。每个版本的 SFT 数据集：来源、筛选标准、配比、评估结果、跟上个版本的 diff，全部记下来。没有这些记录，三个月后 v23 和 v24 的区别你不会记得的。

---

## 总结

如果让我用一句话概括现在做 SFT 数据应该想什么，那就是：

> **SFT 不是教模型"什么是对的"，而是给 RL 搭一个好的搜索空间。**

"好数据"的定义变了。以前是"答案正确、格式规范"就行，现在还得看推理结构是否合理、跟目标模型的分布是否匹配、跟后续 RL 阶段是否能配合。

一年前的最佳实践今天可能已经过时。保持 pipeline 的可迭代性比任何单一技巧都重要。

---

## 参考文献

[1] DeepSeek-AI. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv:2501.12948, 2025.

[2] Dylan Zhang, Qirun Dai, Hao Peng. The Best Instruction-Tuning Data are Those That Fit (GRAPE). NeurIPS 2025 Spotlight. arXiv:2502.04194.

[3] Edward Yeo, Yuxuan Tong, Morry Niu, Graham Neubig, Xiang Yue. Demystifying Long Chain-of-Thought Reasoning in LLMs. arXiv:2502.03373, 2025.

[4] CoTP: Enriching Reasoning Patterns for Foundation Models via Chain-of-Thought Data Selection. ICLR 2026. arXiv:2509.21124.

[5] Bee/Honey-Data: Open-Source Multimodal LLM Data Curation at Scale. ICLR 2026.

[6] Nvidia. On Data Engineering for Scaling LLM Terminal Capabilities (Nemotron-Terminal). arXiv:2602.21193, 2026.

[7] Draft-Thinking: Learning Efficient Reasoning in Long Chain-of-Thought LLMs. arXiv:2603.00578, 2026.

[8] Yu et al. LS-Mixture SFT: Long-Short CoT Mixture for Adaptive Reasoning. 2025.

[9] SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-Training. arXiv:2501.17161, 2025.

[10] SFT or RL? An Early Investigation into Training R1-Like Reasoning Large Vision-Language Models. arXiv:2504.11468, 2025.

[11] RLTHF: Reinforcement Learning from Targeted Human Feedback. 2025.

[12] The Molecular Structure of Thought: Mapping the Topology of Long Chain-of-Thought Reasoning. arXiv:2601.06002, 2026.
