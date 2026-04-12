# RL+大模型项目经验调优策略分享

> 基于RL的多任务多阶段强化学习优化

---

## 一、项目背景

### 1.1 单阶段RL后训练的局限性

- **任务目标冲突**：安全对齐与推理能力呈负相关，单一reward信号无法编码多维偏好，多任务混合训练导致各能力维度相互干扰
- **奖励稀疏与Reward Hacking**：模型在单阶段长期优化中学会exploiting reward model缺陷，RM泛化不足导致reward曲线虚高但实际能力退化
- **策略坍缩**：单阶段PPO/GRPO训练中输出多样性急剧下降，KL约束不足时策略漂移至退化模式，KL过强时策略被锁死无法有效学习

---

## 二、技术栈

### 2.1 基础训练框架

- **分布式训练框架**：Megatron-Bridge + Megatron-Core
- **模型转换**：通过 `AutoBridge.from_hf_pretrained()` 将 HuggingFace 模型转换为 Megatron 格式
- **并行配置**：支持 GPTModelProvider 统一管理模型并行配置（TP/PP/CP/EP）
- **配置管理**：Hydra 配置系统管理全部超参

### 2.2 阶段化任务解耦

将后训练RL拆分为 **K=4个功能正交阶段**：

```
Instruction Following → Reasoning → Code → Safety
```

每阶段配备：
- 独立 Reward Model
- KL预算与 Stage-Gate 自动切换条件
- 基座模型可选 ERNIE-4.5-21B-A3B（MoE 64专家, top-6 routing, sigmoid score）或 Qwen3-VL-2B（多模态Dense）

---

## 三、核心技术创新

### 3.1 Curriculum-Aware Reward Shaping (CARS)

阶段内部按数据难度分级：
- **早期**：给简单样本额外 reward bonus 加速策略建立
- **后期**：bonus 衰减至0使模型直面原始 reward signal
- **数据加载**：通过 Energon DataFlow 的 `packing_buffer_size` 与 `shuffle_buffer_size` 控制样本组织和随机化

### 3.2 Progressive Policy Optimization

- **Reference Policy**：采用上一阶段最终 checkpoint（通过 CheckpointConfig 的 load 参数加载，`ckpt_format="torch_dist"` 格式，支持 fully_parallel_load）
- **KL预算**：跨阶段递减
- **算法选择**：
  - Stage 1-3：GRPO（无Critic）
  - Stage 4：Constrained PPO（含cost value head）

### 3.3 Cross-Stage EWC参数保护

- 阶段切换时计算 Fisher Information Matrix 对角近似
- 后续阶段优化中增加 Fisher 加权正则项防止灾难性遗忘
- 仅保留 top-10% 重要参数的 Fisher 信息降低存储与计算开销

### 3.4 模型健康监控

**Internal Medicine 系统**实时监控模型健康：

| 监控模块 | 监控指标 |
|---------|---------|
| QKStatsMonitor | Attention QK统计（entropy/max logits/sink weights） |
| MoESpecialistMonitor | MoE健康指标（router entropy/expert bias/Jaccard相似度/expert norms） |

- 每个训练 step 通过 `training_logs.gather_and_aggregate()` 全局聚合并写入 TensorBoard

---

## 四、工作亮点

### 4.1 MSRL多阶段框架设计

- 提出 **Multi-Stage RL** 框架，将RL后训练解耦为4个功能正交阶段
- 每阶段有独立reward信号、KL约束和 Stage-Gate 自动切换机制
- 基于 Megatron-Bridge 的 PretrainingTrainer 进行阶段化训练编排
- 通过 `CheckpointConfig(load=prev_stage_ckpt)` 实现跨阶段 checkpoint 接力

### 4.2 CARS奖励整形机制

- 设计 Curriculum-Aware Reward Shaping
- 在每阶段内部引入难度感知的时变 reward bonus 函数
- 配合 Energon DataFlow 的 `packing_buffer_size=100` 实现高效样本组织和 curriculum scheduling
- **效果**：在推理类任务（MATH）上单项贡献 **+3.1分** 提升

### 4.3 MoE专家健康监控集成

- 在RL训练各阶段集成 MoESpecialistMonitor
- 实时监控64个MoE专家的：
  - 路由熵
  - expert bias 均值/方差
  - Bias-Affinity Jaccard 相似度
- **告警机制**：当 Jaccard < 0.3 时触发 "SEVERE" 级别告警，表示 bias 强行扭转了大部分路由决策
- **目标**：确保RL训练过程中MoE专家不退化为专家坍缩

### 4.4 Attention稳定性监控

- 在RL训练各阶段集成 QKStatsMonitor
- 追踪每层 Attention 的：
  - QK max logits
  - entropy 分布（entropy_min/max/std 检测坏头）
  - sink weights
- **性能优化**：通过 Triton 优化内核（Online Softmax）实现零额外训练开销监控
- TP场景下通过 all_gather 跨 rank 聚合 per-head entropy

---

## 五、实验效果

### 5.1 整体效果

基于 **Qwen2.5-7B** 在8个 benchmark 上验证：

| 对比方案 | 平均分 | 提升 |
|---------|-------|------|
| 单阶段 PPO | 62.4 | - |
| 单阶段 GRPO | 63.3 | - |
| **MSRL（本方案）** | **68.5** | **+8.3% vs PPO / +5.2% vs GRPO** |

- Reward Hacking 发生率降低 **62%**
- MoE训练过程中 router entropy 保持稳定（std < 0.05）
- expert bias Jaccard 始终 > 0.7，表明路由决策未被 bias 过度扭转

### 5.2 分项提升

| Benchmark | 基线 | MSRL | 提升 |
|-----------|------|------|------|
| GSM8K | 72.1 | 82.6 | **+10.5** |
| MATH | 38.2 | 49.3 | **+11.1** |
| HumanEval | 62.8 | 71.5 | **+8.7** |
| MBPP | 58.4 | 67.8 | **+9.4** |
| IFEval | 68.3 | 76.8 | **+8.5** |
| TruthfulQA | 52.1 | 63.2 | **+11.1** |

---

## 六、技术关键词

`强化学习` `RLHF` `PPO` `GRPO` `MoE` `Megatron` `多阶段训练` `Reward Shaping` `EWC` `模型监控` `分布式训练` `大模型调优`
