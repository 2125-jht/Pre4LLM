# 基于 VeRL 框架和 DAPO 算法的多模态 RL Agent 训练项目

## 一、项目概述

### 1.1 项目名称
**训练多模态 RL Agent - VeRL框架+DAPO训练代码** (Think with Images)

### 1.2 核心目标
- 对视觉-语言模型（VLM）进行**在线强化学习训练**
- 在 **ChartQA** 图表问答任务上提升模型性能
- 使模型能够稳定生成可验证的答案
- 通过**工具调用（Tool Use）**增强视觉理解能力

### 1.3 技术路线
1. 使用 DAPO 强化学习让模型执行**工具调用**（生成 Python 代码编辑图像）
2. 执行工具得到**编辑后图像**（Observation）
3. 将编辑结果拼入**二次 Prompt**，促使模型在增强视觉输入上回答

### 1.4 使用模型
- **基础模型**: Qwen2.5-3B（经过 SFT 后有强指令遵循能力）

---

## 二、核心技术栈

| 技术模块 | 具体实现 |
|---------|---------|
| RL 框架 | VeRL（支持 GRPO/DAPO/GSPO 等主流算法） |
| 分布式训练 | Ray 资源编排、FSDP、Tensor Parallel、CPU Offload |
| RL 算法 | DAPO（相比 GRPO 改进的在线 RL 算法） |
| 奖励机制 | Rule-based Reward + Model-based Reward（LLM as Judge） |
| 工具增强 | Agent RL 工具调用（6 种图像编辑工具） |

---

## 三、数据准备

### 3.1 数据来源
- **数据集**: HuggingFaceM4/ChartQA
- **规模**: 
  - 9.6K 人工编写问题（human）
  - 23.1K 机器生成问题（machine）
- **特点**: 
  - 非模板生成（答案不来自固定词汇表）
  - 包含复杂推理问题

### 3.2 数据处理
基于 **ReFocus** 数据处理框架：

1. **格式转换**: 将 bbox 列表转换为可索引字典（value → bbox）
2. **Metadata 生成**: 每条样本包含：
   - 图类型（h_bar/v_bar/table）
   - 图整体 bbox
   - x/y 值到 bbox 的映射（供 tooluse 使用）

3. **预处理命令**:
```bash
cd data
bash preprocess_data.sh
```

---

## 四、模型训练详解

### 4.1 Tool Use 配置（工具调用机制）

#### 6 种核心工具函数
分为 3 类，每类针对 columns 和 rows：

| 工具类型 | 功能描述 | 适用场景 |
|---------|---------|---------|
| highlight | 半透明红色高亮指定列/行 | 需要提醒注意特定区域，同时保留上下文 |
| mask | 遮罩专注以外区域，仅保留指定列/行 | 强过滤干扰信息，对表格更有效 |
| draw | 对指定列/行画红色边框 | 轻量标注，强调位置但不遮挡内容 |

#### 推理流程（Two-round Rollout）

**第一轮**:
- 模型接收 Question + Image
- 输出 Thought + Action（Python 代码调用工具）
- 执行图片编辑：运行 Python 代码生成编辑后的图像（如高亮特定行）

**第二轮**:
- 将编辑后的图像插入 Prompt
- 模型基于增强视觉输入生成 Final Answer

**示例**:
```
Query: "How many cyber espionage incidents involved phishing?"
Action: focus_on_rows_with_draw(image_1, ["Social - Phishing"], rows_bbox)
Final Answer: 181
```

### 4.2 奖励建模（Reward Modeling）

#### 4.2.1 Rule-based Reward（规则奖励）

**特点**:
- 无需训练 Reward Model
- 无需调用 Judge LLM

**机制**:
- 输出归一化分数奖励（outcome reward）r ∈ [0, 1]，映射到 token 维度

**匹配逻辑**:
- 从模型输出用正则抽取最终答案：`FINAL ANSWER:\s*(.*?)(?=\.\s|$|\.\?$)`
- 支持多答案（GT 用 `|||` 分隔，预测用 `||` 分隔）

**评分方式**:
- **数字答案**: 使用数值相似度
  ```
  sim(u, v) = 1 - |u - v| / max(|u|, |v|)
  ```
- **非数字答案**: 精确匹配

**最终得分**:
```
R(y, g) = max(m, 1) * S(A, G)
```
（按 ground-truth 个数归一化）

#### 4.2.2 Model-based Reward（模型奖励）

- 使用 **DeepSeek V3.2 API** 作为 LLM Judge
- 对提取的最终答案进行 YES/NO 二分类打分

### 4.3 DAPO 算法配置

DAPO 相对 GRPO 的两大改进：

#### (1) 更高截断（Clip-Higher）

**目标函数**:
```
J_DAPO(θ) = E[(q,a)~D, {o_i}_{i=1}^G ~ π_θ_old(·|q)] [
    Σ_{i=1}^G (1/|o_i|) Σ_{t=1}^{|o_i|} min(
        r_{i,t}(θ) * Â_{i,t},
        clip(r_{i,t}(θ), 1-ε_low, 1+ε_high) * Â_{i,t}
    )
]
```

**Clip-Higher 策略**:
- 提高 token 的概率**提升**幅度（允许更大探索空间，鼓励低概率 token 获得高优势）
- 限制 token 的概率**降低**幅度（稳定训练，防止熵坍塌）

#### (2) 动态采样（Dynamic Sampling）

**机制**: 保证每次采样的回答 reward 不全是 0 或 1，如果全是则重新采样

**在线过滤（Online Filtering）**:
- 每个 prompt 生成 n 个回答并计算 reward
- 按阈值 `filter_low < mean < filter_high` 过滤掉太差或太好的样本组
- 反复生成直到凑够 `rollout_batch_size`

**训练命令**:
```bash
bash train.sh
```

---

## 五、实验结果

### 5.1 训练监控指标

| 指标 | 说明 |
|-----|------|
| val/tool_call_ratio | 工具调用成功率（随训练逐步提升至接近 1.0） |
| val/reward_score | 奖励分数（持续提升） |
| val/overall_reward | 整体奖励（持续提升） |
| response_length | 回答长度变化（监控模型生成行为） |

### 5.2 关键结论

1. **Tool Use 有效性**: 相比非工具调用的 Baseline，有 tooluse 的 RL 训练效率更高，证明 Agent RL 中工具的重要性

2. **RL 训练有效性**: 相比裸模型直接调用 Agent（RL step 0），经过 RL 训练后的模型效果更好，证明 RL 训练的必要性

3. **DAPO 优势**: Clip-Higher 策略有效缓解了熵坍塌（Entropy Collapse），保持模型探索能力

---

## 六、项目资源

### 6.1 代码下载
- **文件名**: mini_chartqa.zip

### 6.2 核心依赖
- VeRL 框架
- Ray
- PyTorch FSDP

### 6.3 显存优化
支持通过以下参数调整：
- `worker.actor.micro_batch_size_per_device_for_update`
- `data.rollout_batch_size`

极端情况可启用 CPU Offload

---

## 七、项目总结

本项目提供了一个完整的从数据预处理、工具定义、奖励函数设计到 RL 训练（DAPO 算法）的全链路解决方案，适用于视觉-语言模型在结构化图表理解任务上的深度优化。

### 核心创新点
1. **工具增强的视觉理解**: 通过图像编辑工具（highlight/mask/draw）增强模型对图表的注意力
2. **Two-round Rollout**: 先执行工具调用，再基于增强视觉输入回答问题
3. **DAPO 算法优化**: 使用 Clip-Higher 和 Dynamic Sampling 提升训练稳定性和效率
4. **混合奖励机制**: 结合 Rule-based 和 Model-based 奖励，兼顾效率和准确性

### 应用场景
- 图表问答（ChartQA）
- 文档理解
- 视觉推理任务
- 需要工具调用的多模态 Agent 场景
