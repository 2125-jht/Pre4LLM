# MoE模型与MCP协议

## 一、MoE模型

### Q: 有了解过MoE模型吗？
**A:** MoE(Mixture of Experts)核心思想：模型包含多个专家网络(Expert)，每次推理只激活其中少数几个，而非全部参数参与计算。

结构：在Transformer的FFN层替换为多个Expert FFN + 一个Gate/Router网络。Gate根据输入token决定分配给哪几个Expert(通常Top-K, K=2)。

### Q: 为什么激活参数小很多？优势是什么？
**A:** 关键优势：
1. **总参数量大但激活参数少**——如DeepSeek-V3总参671B但每token只激活37B(约5.5%)，获得大模型的容量但计算成本接近小模型
2. **训练阶段**可以扩大模型容量而不成比例增加FLOPs
3. **推理阶段**计算效率高，同等效果下比Dense模型快很多

代价：需要专家负载均衡(load balancing loss防止某些expert被冷落)、通信开销(分布式下expert可能在不同GPU上)。

### Q: MoE的关键挑战？
**A:**
1. **负载不均衡**：部分expert被频繁选择而其他expert闲置，用auxiliary loss或expert capacity限制来缓解
2. **路由坍塌**：所有token都选同一个expert，需要正则化
3. **分布式通信**：expert分布在不同GPU上，All-to-All通信开销大
4. **训练不稳定**：router的离散选择导致梯度估计困难

DeepSeek-V3用了无auxiliary loss的负载均衡策略是一个创新点。

---

## 二、MCP与Skill

### Q: MCP是什么？
**A:** MCP(Model Context Protocol)是Anthropic提出的开放协议，作用类似AI界的USB-C。它定义了LLM连接外部工具和数据源的标准接口。

三个角色：
- **MCP Server**（提供工具/数据的服务端，如Gmail Server）
- **MCP Client**（调用工具的AI客户端）
- 统一通信协议

好处：开发者只需写一次MCP Server，所有支持MCP的AI都能直接调用，避免每个工具单独对接。

### Q: Skill是什么？
**A:** Skill本质是AI的操作手册/SOP，以Markdown文档形式存在（如SKILL.md）。里面写好了针对某类任务的最佳实践：用什么工具库、格式规范、常见坑、具体流程。

模型在执行任务前先读Skill文件，就像厨师做菜前先看菜谱——能力有了，菜谱保证稳定发挥。用户也可以自定义Skill模板化自己的工作流。

### Q: MCP和Skill的关系？
**A:**

| 概念 | 解决的问题 | 类比 |
|------|------------|------|
| MCP | 「能连什么」(能力扩展) | 给机器人装上各种手臂/传感器 |
| Skill | 「怎么做好」(质量保证) | 教它如何优雅高效地使用这些部件完成任务 |

MCP扩展了AI可以调用的外部工具范围，Skill则沉淀了使用这些工具的最佳实践。
