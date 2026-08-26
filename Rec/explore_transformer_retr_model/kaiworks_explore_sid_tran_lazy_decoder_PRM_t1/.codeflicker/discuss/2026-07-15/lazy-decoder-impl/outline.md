# 讨论：OneRec-V2 Lazy Decoder-Only 架构实施

> 状态：进行中 | 轮次：R2 | 日期：2026-07-16

## 🔵 当前焦点

- **Step 1: 实现 LazyCrossAttention + LazyDecoderLayer + LazyDecoderModel + LazyMultiInterestModel**
  - 核心改动：去掉 cross-attention 的 w_k/w_v 投影，改为 context LayerNorm 预计算 KV
  - 保持 2 层 decoder、Self→Cross→FFN 顺序、LayerNorm

## ⚪ 待讨论

- [ ] Step 2: Beam Search 推理（context KV 共享缓存）
- [ ] Step 3: ContextProcessor（三路特征整合 + 预计算分层 KV）
- [ ] Step 4: RMSNorm + GQA 辅助函数
- [ ] Step 5: Long-term behavior pathway + 操作顺序切换

## ✅ 已确认

- 采用 Lazy Decoder-Only 架构替代现有 Encoder-Decoder 架构 → [D01-architecture-choice](./decisions/D01-architecture-choice.md) (#R1)
- 保持 2 层 decoder 结构（现有无真正 transformer encoder） → (#R2)
- 操作顺序先保持 Self-Attn → Cross-Attn → FFN，后续对比 → (#R2)
- 归一化先保持 LayerNorm，后续对比 RMSNorm → (#R2)
- GQA 先不实现，后续添加 → (#R2)
- 新旧模块并存，通过 `--use_lazy_decoder` 配置切换 → (#R2)
- 不实现 RL 模块（GBPO/Duration-Aware Reward），聚焦架构 → (#R2)

## ❌ 已否决

- 一次性全量替换旧代码（风险不可控）→ (#R2)
- 同步实现 MoE FFN（不在当前范围）→ (#R2)

## 📁 归档

| 问题 | 结论 | 详情 |
|------|------|------|
| 架构选择 | Lazy Decoder-Only | [→ 决策](./decisions/D01-architecture-choice.md) |
| 技术对照分析 | 现有代码 vs 论文 6 大差异 | [→ 分析](./notes/tech-analysis.md) |
| 完整实施指南 | 含函数签名/逻辑/测试 | [→ 指南](./notes/implementation-guide.md) |
| 分步实施计划 | 5 步渐进式改造 | [→ 计划](./notes/step-by-step-implementation.md) |
