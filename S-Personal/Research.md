# 科研方向整理

## 高价值研究切入点

基于文献分析，以下三个研究空缺最大、可行性高的切入点：

---

### 切入点1：不确定性驱动的自适应工具调用

**题目示例**：Uncertainty-Gated Tool Calling: Adaptive Decision Making in LLM Agents

**核心创新**：将SAUP的不确定性传播与工具调用决策结合，构建"不确定性阈值 → 工具调用策略"的动态映射

**对标缺陷**：解决One Tool Is Enough和OTC缺乏成本感知的问题

---

### 切入点2：风险感知的验证预算分配

**题目示例**：Risk-Aware Verification Budgeting for Code Agents

**核心创新**：基于任务风险等级（高/中/低）和当前不确定性，动态分配验证资源（调用多少工具、验证多少轮）

**对标缺陷**：解决Self-Evolving Code Agents中验证固定的问题

---

### 切入点3：工具结果的可信度验证机制

**题目示例**：Don't Trust, Verify: Bayesian Validation of Tool Outputs for LLM Agents

**核心创新**：Agent不盲目信任工具返回，而是基于先验知识和观测不一致性评估工具输出可信度

**对标缺陷**：填补"Tool Verification"的空白（现有工作几乎都没有）

---

### 切入点4：错误累积的早期检测与回滚机制

**题目示例**：Early Detection of Error Accumulation in Multi-Step LLM Agents

**核心创新**：在多步决策过程中实时监测认知漂移（cognitive drift），当检测到"已走偏"信号时触发智能回滚，支持长程任务（网页操作、数据分析流程等）的可靠执行

**对标缺陷**：解决当前Agent在长程任务中错误累积不可逆的问题，填补早期干预与状态恢复的空白
