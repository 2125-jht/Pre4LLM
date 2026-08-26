# 决策：采用 Lazy Decoder-Only 架构

**决策时间**：#R1
**状态**：✅ 已确认
**关联大纲**：[返回大纲](../outline.md)

---

## 📋 背景

### 问题/需求
现有代码库采用 Encoder-Decoder 架构（OneRec-V1 风格），其中 encoder 处理用户行为序列，decoder 通过 cross-attention 从 encoder 输出中读取上下文。OneRec-V2 论文指出此架构存在严重的计算资源分配不均：context encoding 占 97.66% 的 FLOPs，而 target decoding 仅占 2.34%。

### 约束条件
- 需保持与现有 TensorFlow 1.x + Kai v2.0 框架的兼容性
- 需保持训练和推理的变量创建顺序一致性
- 需支持现有 beam search 推理流程的改造
- 需保持 PRM 打分逻辑的兼容性

---

## 🎯 目标

将现有 Encoder-Decoder 架构改造为 Lazy Decoder-Only 架构，实现：
1. 消除 encoder 的计算瓶颈
2. 实现 Context Processor 预计算 KV
3. 实现 Lazy Cross-Attention（无 K/V 投影，KV 共享）
4. 支持 GQA（Grouped Query Attention）
5. 减少 94% 计算量和 90% 训练资源

---

## 📊 方案对比

| 方案 | 描述 | 优势 | 劣势 | 决策 |
|------|------|------|------|------|
| A | 保留 Encoder-Decoder | 无需改动代码 | 计算瓶颈无法解决 | ❌ |
| B | Lazy Decoder-Only | 94% 计算减少，可扩展到 8B | 架构改动较大 | ✅ |
| C | Naive Decoder-Only | 比 Encoder-Decoder 简单 | 仍需 context encoding，计算量约为 Lazy 的 35x | ❌ |

---

## ✅ 最终决策

### 选定方案
方案 B：Lazy Decoder-Only 架构

### 决策理由
1. 论文实证：Lazy Decoder 在同等 loss 下比 Encoder-Decoder 减少 94% FLOPs
2. 可扩展性：支持从 0.1B 到 8B 的模型规模，且收敛 loss 遵循 scaling law
3. 架构简洁：消除 encoder，仅保留 decoder + context processor

### 预期效果
- 训练 FLOPs 从 296.4 GFLOPs (1B Enc:Dec=1:1) 降至 18.9 GFLOPs
- 激活内存从 17.63B 降至 1.24B
- 收敛 loss 保持可比（3.27 vs 3.28）

---

## ❌ 被否决的方案

### 方案 A（Encoder-Decoder）
- **否决原因**：无法解决 context encoding 计算瓶颈，97.66% FLOPs 浪费在 context 上
- **重新考虑条件**：仅当 context 长度极短（<10 tokens）时可能考虑

### 方案 C（Naive Decoder-Only）
- **否决原因**：虽然比 Encoder-Decoder 简单，但仍需完整 context encoding，计算量为 634.8 GFLOPs（1B 模型）
- **重新考虑条件**：如果 context 不需要参与 decoder 的 full self-attention
