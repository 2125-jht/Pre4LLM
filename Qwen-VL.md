# Qwen 系列多模态模型面试速查（Qwen2.5-VL / Qwen3-VL / Qwen3.5）

> **适用场景**：向面试官清晰阐述 Qwen2.5-VL、Qwen3-VL、Qwen3.5 三代视觉语言模型的架构设计与演进逻辑。三个模型均为**视觉语言模型（VLM）**，处理文本、图像、视频三种模态。

---

## 一、总览：三代模型的核心定位与演进脉络

| 模型 | 核心定位 | 架构关键词 | 最大上下文 | 旗舰规模 |
|------|---------|-----------|-----------|---------|
| **Qwen2.5-VL** | 精细化视觉语言理解 | 自研 ViT + 窗口注意力、MLP Merger、MRoPE 绝对时间 | 32K | 72B Dense |
| **Qwen3-VL** | 深度推理与 Agent 级视觉交互 | SigLIP-2、Interleaved-MRoPE、DeepStack、MoE 底座 | 256K | 235B-A22B MoE |
| **Qwen3.5** | 超长上下文原生多模态 Agent 底座 | Hybrid MoE、Gated DeltaNet 线性注意力、原生合一训练 | 256K → 1M | 397B-A17B MoE |

**演进逻辑一句话**：Qwen2.5-VL 解决了"视觉看得清、时空定得准"的问题；Qwen3-VL 解决了"深度推理 + Agent 交互"的问题，同时将视觉编码器升级为更强大的 SigLIP-2；Qwen3.5 解决了"超长上下文下高效推理并作为原生 Agent 底座"的问题。

---

## 二、Qwen2.5-VL：精细化视觉语言模型

### 2.1 核心架构（三段式经典 VLM）

```
图像/视频 → [自研 ViT Vision Encoder] → [MLP-based PatchMerger] → [Qwen2.5 LLM]
                 ↓                              ↓                        ↓
            原生动态分辨率                  2×2 Patch 空间合并              MRoPE 绝对时间编码
            窗口注意力 + 2D-RoPE            两层 MLP 维度对齐                (t, h, w 顺序分块)
```

### 2.2 视觉编码器 → Merger：单层特征 + 空间压缩注入

Qwen2.5-VL 的多模态注入遵循**"单点注入 + 空间压缩"**的设计哲学。视觉信息仅在 ViT 的最后一层被提取，经过 Merger 压缩后一次性注入 LLM 的输入层。

**Step 1：ViT 特征提取（仅最后一层）**
- 自研 ViT 共 32 层，使用窗口注意力（28 层窗口 + 4 层全局）
- 仅取**第 31 层（最后一层）**的输出特征作为视觉表示
- Patch size 14×14，stride 14，输入 resize 到 28 的倍数
- 输出特征维度：1280（ViT hidden size）

**Step 2：PatchMerger 空间压缩（核心压缩机制）**
- 将空间相邻的 **2×2 共 4 个 patch 特征**沿通道维度拼接
- 通过**两层 MLP** 投影到 LLM 的文本嵌入维度
- Merger 结构：`fc → GELU → fc`，使用 **RMSNorm**
- 压缩效果：视觉 token 数量减少为原来的 **1/4**（如 256 个 patch → 64 个视觉 token）

**Step 3：Token 注入 LLM（masked_scatter）**
- 压缩后的视觉 token 通过 `masked_scatter` 操作注入到统一序列中
- 即在 `input_ids == image_token_id` 的位置，将视觉 embedding 替换到对应位置
- 视觉 token 与文本 token 在输入层拼接后，一并送入 LLM 解码器
- **只在 LLM 第 0 层（输入层）注入一次**，后续由自注意力机制进行跨模态融合

**设计特点总结**：
- 优点：结构简单、计算高效、易于训练和部署
- 局限：仅使用 ViT 最后一层特征，浅层的纹理/边缘等低级视觉信息在深层传播中被稀释；只在 LLM 输入层注入，视觉信息需经过多层自注意力才能充分与文本交互

### 2.3 面试一句话描述

> **Qwen2.5-VL 采用"自研 ViT（原生动态分辨率 + 窗口注意力）+ MLP PatchMerger（2×2 空间压缩）+ MRoPE 绝对时间编码"的三段式架构，视觉特征仅在 ViT 最后一层提取、经两层 MLP 压缩 4× 后通过 masked_scatter 注入 LLM 输入层，以绝对像素坐标和秒级时间编码实现细粒度的视觉 grounding 与长视频理解。**

---

## 三、Qwen3-VL：深度推理与 Agent 级视觉交互

Qwen3-VL 在延续三段式架构的基础上，对**多模态注入机制**进行了系统性升级：从"单点注入"进化为"多层深度注入（DeepStack）"，并配套升级了 Merger 结构和位置编码。

### 3.1 核心架构（三段式 + DeepStack 深度注入）

```
图像/视频 → [SigLIP-2 Vision Encoder] → [Multiple PatchMergers] → [Qwen3 LLM]
                 ↓                    (DeepStack, 3 个独立 Merger)          ↓
            全自注意力 2D-RoPE                 ↓                      Interleaved-MRoPE
            从多层 [8,16,24] 提取特征    分别注入 LLM 前 3 层 [0,1,2]          (t/h/w 交错排列)
```

### 3.2 视觉编码器 → Merger：DeepStack 多层深度注入（核心升级）

Qwen3-VL 的多模态注入遵循**"分层提取 + 多点注入"**的设计哲学。这是相比 Qwen2.5-VL 最根本的架构创新。

**Step 1：ViT 多层特征提取（DeepStack 的核心）**
- 使用 SigLIP-2 ViT（24 层，hidden size 1024 或 1280）
- **不止取最后一层**，而是从 ViT 的**三个中间层**提取特征：
  - 标准配置：ViT 第 **[8, 16, 24]** 层（对应低、中、高三级特征）
  - 2B 模型配置：ViT 第 **[5, 11, 17]** 层
- 每层提取的特征保留各自的分辨率和语义粒度：
  - **浅层（如第 8 层）**：保留高分辨率空间信息、边缘、纹理等**低级视觉特征**
  - **中层（如第 16 层）**：包含局部语义、物体部件等**中级视觉特征**
  - **深层（如第 24 层）**：包含全局场景理解、高级语义概念等**高级视觉特征**

**Step 2：Multi-PatchMerger 独立投影（3 个专用 Merger）**
- 为每个提取层配备**独立的 PatchMerger**，各有一套独立的可学习参数
- 每个 Merger 结构：`Linear(fc → LayerNorm → GELU → fc)`
- 相比 Qwen2.5-VL 的两层 MLP，Qwen3-VL 的 Merger 增加了 **LayerNorm**（从 RMSNorm 改为 LayerNorm）
- 同样执行 **2×2 patch 空间压缩**，将视觉 token 数量减少为 1/4
- 每个 Merger 将对应层的视觉特征投影到 LLM 的文本嵌入维度

**Step 3：多层 scatter-add 注入 LLM（与 Qwen2.5-VL 的关键差异）**
- Qwen2.5-VL：仅在 LLM **第 0 层**注入（替换输入 embedding）
- **Qwen3-VL：分别注入到 LLM 的前 3 层**（decoder layers 0, 1, 2）
- 注入方式从 `masked_scatter`（替换）改为 **`scatter_add`**（累加）：
  ```python
  for layer_idx, layer in enumerate(self.layers):
      hidden_states = layer(hidden_states, ...)
      if layer_idx < len(deepstack_visual_embeds):
          mask = (input_ids == image_token_id)
          indices = cumsum(mask) - 1
          scattered = gather(deepstack_embeds[layer_idx], indices)
          hidden_states += where(mask, scattered, 0.0)  # 累加而非替换
  ```
- 效果：LLM 的每一层在前向传播过程中都能接收到对应粒度的视觉信息，浅层处理低级视觉特征、深层处理高级语义特征

**DeepStack 的设计动机与效果**：
- **问题**：传统 VLM 只把 ViT 最后一层特征注入 LLM，浅层的纹理、边缘等低级视觉信息在深层传播中被稀释，导致细粒度感知能力下降
- **解决方案**：DeepStack 让 LLM 的浅层直接接收 ViT 的浅层特征，深层接收深层特征，实现**视觉-语言的逐层精细化对齐**
- **效果**：在 OCR、文档解析、细粒度 grounding 等任务上，视觉信息的保真度显著提升
- **代价**：3 个独立 Merger 增加了参数量和计算量，但相对整个模型可忽略

### 3.3 Interleaved-MRoPE：从顺序分块到交错排列

- Qwen2.5-VL 的 MRoPE：按 **时间块 → 高度块 → 宽度块** 的顺序排列
  - `[t1,t2,...,tn, h1,h2,...,hn, w1,w2,...,wn]`
  - 问题：时间维度集中在高频位置，高度和宽度在低频位置，导致频率谱不平衡
- **Qwen3-VL 的 Interleaved-MRoPE**：将 t/h/w 维度**交错排列**
  - `[t1,h1,w1, t2,h2,w2, ..., tn,hn,wn]`
  - 每个位置都同时包含时间、高度、宽度三种信息，确保频率谱平衡
- MRoPE section 从 `[16,24,24]` 调整为 `[24,20,20]`（时间维度容量增大）

### 3.4 面试一句话描述

> **Qwen3-VL 通过 DeepStack 机制从 SigLIP-2 ViT 的 [8,16,24] 中间层分层提取低/中/高级视觉特征，经 3 个独立 PatchMerger 投影后通过 scatter-add 分别注入 LLM 的前 3 层，配合 Interleaved-MRoPE 的交错位置编码，实现了视觉信息从底层纹理到高层语义的逐层精细化对齐，支撑起 Agent 级视觉交互与深度推理能力。**

---

## 四、Qwen3.5：超长上下文原生多模态 Agent 底座

Qwen3.5 在多模态注入上的核心创新不在于注入机制本身（继承 DeepStack），而在于**训练范式**的根本变革——从"先训语言再拼视觉"的两段式，进化为"视觉-语言从第一天起一起训练"的原生多模态合一训练。

### 4.1 核心架构（Hybrid MoE + 原生多模态合一）

```
图像/视频 → [SigLIP-2 Vision Encoder] → [Multi PatchMergers] → [Qwen3.5 Hybrid MoE LLM]
                 ↓                     (DeepStack 继承)                 ↓
            与 LLM 同步预训练         早期融合 Early Fusion          Gated DeltaNet + Attention
            （非冻结，联合更新）                                          3:1 交替
```

### 4.2 多模态注入的核心变革：原生合一训练（Native Multimodal Training）

Qwen3.5 的多模态注入机制在结构上继承了 Qwen3-VL 的 DeepStack，但**训练范式**发生了根本性变化。这是三代模型中最大的范式跃迁。

**传统两段式训练（Qwen2.5-VL / Qwen3-VL）的问题**：
1. **阶段一**：先大规模预训练纯文本 LLM（数万亿文本 token）
2. **阶段二**：冻结 LLM 权重，接入 ViT + Merger，用视觉-文本对训练视觉注入模块
3. **根本缺陷**：LLM 在阶段一已经"固化"了文本世界的表示空间，阶段二 ViT 被迫去**补偿/适配**一个冻结的文本空间，而非与文本空间**共同演化**。这导致：
   - 视觉编码器的表示能力被人为限制（要去适配一个不再变化的文本空间）
   - 视觉-语言的对齐停留在"浅层拼接"层面，而非深层语义融合
   - 视觉信息的利用效率低下，模型"看"的能力受限于"语言模板"的约束

**Qwen3.5 的原生合一训练（核心突破）**：

**Step 1：从零开始的联合预训练（Early Fusion）**
- ViT 视觉编码器、PatchMerger、LLM **三个组件从预训练第一天起就一起训练**
- 所有参数同步更新，没有"冻结 LLM 训 ViT"的阶段
- 训练数据从第一天就包含**文本 + 图像 + 视频**三种模态
- 数据配比（约数万亿 token）：
  - 40% 高质量 STEM 文本和代码
  - 30% 多语言网页文本（覆盖 201 种语言）
  - 20% 合成视觉-文本对（通过自蒸馏生成）
  - 10% Agent 轨迹数据（模拟环境中收集）

**Step 2：异构基础设施支持**
- 视觉和语言组件采用**解耦的并行策略**：
  - ViT（轻量级，约 400M 参数）：采用数据并行（Data Parallelism）
  - LLM（重型 MoE，397B 参数）：采用张量并行（Tensor Parallelism）+ 专家并行
- 利用 MoE 的稀疏激活特性，实现跨组件计算的**流水线重叠**
- 在混合文本-图像-视频数据上，训练吞吐达到纯文本基线的 **~100%**
- **FP8 端到端训练管道**：激活显存减少约 50%，训练速度提升 10%+

**Step 3：视觉-语言深层对齐的效果**
- 视觉特征与语言语义空间**从底层共同演化**，而非后期拼接
- ViT 学到的视觉表示天然适配 LLM 的语义空间，无需"补偿式"适配
- 在万亿级多模态 token 上联合训练后，视觉 grounding 和语言推理深度融合
- 早期融合 + 万亿 token 训练 → 视觉理解能力超越同等规模的 Qwen3-VL（后期拼接式）

**注入机制的技术细节**：
- 结构上**继承 Qwen3-VL 的 DeepStack**：多层特征提取 + 多层 scatter-add 注入
- 额外引入 **576 个 image token 直接注入 Transformer 第 1 层**（早期融合点）
- 在 512×512 图像上，视觉 token 与文本 token 从第 1 层起就深度交织
- 这种设计在空间推理基准测试上比后期融合方案高出 **12-18 分**

### 4.3 推理效率：Gated DeltaNet 使超长上下文可行

- **Gated DeltaNet**：线性注意力机制，将 KV Cache 压缩为固定大小的状态向量
- 复杂度从 O(N²) 降到 **O(N)**，KV Cache 减少 **~4 倍**
- 32K 上下文解码吞吐是 Qwen3-Max 的 **8.6 倍**，256K 下达到 **19 倍**
- 配合 512 专家 MoE（每 token 仅激活 17B），397B 参数模型可在 8×H100 上运行
- MTP（Multi-Token Prediction）投机解码进一步加速 **2-3 倍**

### 4.4 面试一句话描述

> **Qwen3.5 继承 DeepStack 多层注入架构，核心突破在于"原生多模态合一训练"——ViT、Merger 与 Hybrid MoE LLM 从预训练第一天起就在万亿级多模态 token 上联合更新，视觉与语言表示从底层共同演化而非后期拼接；配合 Gated DeltaNet 线性注意力将 397B 参数模型的长序列推理成本压缩至实用水平，支撑起 1M token 上下文下的原生多模态 Agent 能力。**

---

## 五、三代模型多模态注入机制对比（核心面试表）

### 5.1 多模态注入技术对比

| 维度 | Qwen2.5-VL | Qwen3-VL | Qwen3.5 |
|------|-----------|----------|---------|
| **特征提取** | 仅 ViT **最后一层** | DeepStack：**[8,16,24] 三层** | 继承 DeepStack 多层提取 |
| **Merger 数量** | **1 个**共享 Merger | **3 个**独立 Merger（各层专用） | 继承多 Merger |
| **Merger 结构** | `fc → GELU → fc` + RMSNorm | `fc → LayerNorm → GELU → fc` | 继承 LayerNorm 结构 |
| **空间压缩** | 2×2 patch 合并（压缩 4×） | 2×2 patch 合并（压缩 4×） | 继承 2×2 压缩 |
| **注入 LLM 位置** | **仅第 0 层**（输入层） | **前 3 层** [0,1,2] | 前 3 层 + 第 1 层早期融合 |
| **注入操作** | `masked_scatter`（**替换**） | `scatter_add`（**累加**） | scatter_add + 早期融合 |
| **训练范式** | 两段式（先 LLM 后视觉） | 两段式（先 LLM 后视觉） | **原生合一训练（Early Fusion）** |
| **LLM 是否冻结** | 阶段二冻结 LLM | 阶段二冻结 LLM | **全程联合更新，不冻结** |
| **ViT 类型** | 自研 ViT（NaViT 风格） | SigLIP-2（So400M） | SigLIP-2（原生融合） |
| **位置编码** | MRoPE 顺序分块 [16,24,24] | Interleaved-MRoPE 交错 [24,20,20] | 继承 Interleaved-MRoPE |
| **Patch size** | 14×14 | 16×16 | 16×16 |
| **训练数据规模** | 数十亿视觉-文本对 | 数十亿视觉-文本对 | **数万亿多模态 token** |
| **视觉-语言对齐深度** | 浅层拼接 | 浅层拼接 | **深层共生（Early Fusion）** |

### 5.2 演进逻辑（面试叙述线）

**第一代（Qwen2.5-VL）："单点注入 + 空间压缩"**
- 仅在 ViT 最后一层提取特征，经单层 Merger 压缩 4× 后一次性注入 LLM 输入层
- 结构简单高效，但浅层视觉信息丢失，视觉-语言对齐停留在浅层
- 窗口注意力 + MRoPE 绝对时间编码实现了高效的视觉编码和精准的时间定位

**第二代（Qwen3-VL）："多层深度注入（DeepStack）"**
- 从 ViT [8,16,24] 三层提取低/中/高级特征，经 3 个独立 Merger 分别注入 LLM 前 3 层
- 从 `scatter`（替换）改为 `scatter_add`（累加），实现视觉-语言的逐层精细化对齐
- 配合 SigLIP-2 更强的全局特征提取和 Interleaved-MRoPE 频率谱平衡
- 但训练范式仍是两段式，视觉-语言对齐深度受限

**第三代（Qwen3.5）："原生合一训练（Early Fusion）"**
- 结构上继承 DeepStack，但**训练范式发生根本变革**
- ViT + Merger + LLM 从预训练第一天起联合训练，不冻结任何组件
- 在万亿级多模态 token 上共同演化，视觉-语言对齐从"后期拼接"进化为"深层共生"
- Gated DeltaNet 线性注意力使 397B 参数 + 1M 上下文成为实用配置

---

## 六、高频面试追问与回答要点

### Q1：Qwen2.5-VL 的 Merger 具体做了什么？为什么需要 2×2 patch 合并？

**答**：Qwen2.5-VL 的 PatchMerger 是一个两层 MLP（fc→GELU→fc），有两个作用：① **空间压缩**——将 2×2 相邻的 4 个 patch 特征拼在一起再投影，token 数量减少为 1/4，降低 LLM 的计算负担；② **维度对齐**——将 ViT 输出的 1280 维特征投影到 LLM 的文本嵌入维度（如 3072 或 3584）。2×2 合并是在不损失太多空间信息的前提下最有效的压缩方式，比简单的线性投影保留了更多的局部空间结构。

### Q2：DeepStack 的具体实现是什么？从哪些层提取？注入到哪里？

**答**：DeepStack 是 Qwen3-VL 的核心创新。具体实现：① **提取点**：从 SigLIP-2 ViT 的 [8, 16, 24] 层（或 2B 模型的 [5, 11, 17]）提取中间特征；② **投影层**：为每个提取点配备独立的 PatchMerger（共 3 个），结构为 fc→LayerNorm→GELU→fc；③ **注入点**：3 组视觉特征分别通过 scatter_add 注入 LLM decoder 的前 3 层（layers 0, 1, 2）。浅层 ViT 特征注入 LLM 浅层（处理边缘纹理），深层 ViT 特征注入 LLM 深层（处理语义概念），实现逐层对齐。

### Q3：Qwen3.5 的"原生多模态合一训练"和 Qwen3-VL 的两段式训练到底差在哪里？

**答**：关键差异在于**LLM 是否冻结**。Qwen3-VL 的训练：阶段一预训练纯文本 LLM（数万亿 token）→ 阶段二冻结 LLM，接入 ViT+Merger 训练视觉对齐。这意味着 ViT 要去适应一个已经固化的文本空间，视觉表示能力被限制。Qwen3.5 的训练：ViT+Merger+LLM **从第一天起一起训练**，所有参数同步更新，在万亿级文本+图像+视频混合数据上共同演化。效果是视觉和语言表示从底层深度共生，而非后期拼接，空间推理能力提升 12-18 分。代价是训练基础设施更复杂（需要异构并行策略解耦视觉和语言组件）。

### Q4：为什么 Qwen3-VL 的 Merger 从 RMSNorm 改成了 LayerNorm？

**答**：这是配合 DeepStack 的多层注入做出的调整。LayerNorm 相比 RMSNorm 多了可学习的平移参数（beta），在多层注入的场景下提供了更强的表示灵活性。当 3 个独立的 Merger 需要将不同深度的视觉特征投影到同一 LLM 嵌入空间时，LayerNorm 的可学习平移有助于更好地对齐不同粒度的视觉表示。此外，SigLIP-2 ViT 本身也使用 LayerNorm，保持归一化方式的一致性有助于训练稳定性。

### Q5：scatter_add 和 masked_scatter 有什么区别？为什么 Qwen3-VL 要改？

**答**：`masked_scatter` 是用视觉特征**替换**对应位置的 hidden state，`scatter_add` 是用视觉特征**累加**到 hidden state 上。Qwen3-VL 改为 scatter_add 的原因是：① 在多层注入场景下，如果每层都替换，深层注入会覆盖浅层的信息；累加可以保留所有层的视觉信息。② 累加操作让视觉特征和文本特征在同一位置共存，更自然地促进跨模态融合。③ 对于 DeepStack 来说，3 组不同粒度的视觉特征需要在 LLM 的同一位置叠加，scatter_add 是天然的选择。

### Q6：Qwen3.5 的 Early Fusion 具体在模型的哪个位置体现？

**答**：Qwen3.5 的 Early Fusion 体现在两个层面：① **训练层面**：ViT+Merger+LLM 从预训练第 1 天起就联合训练，没有"冻结 LLM 只训 ViT"的阶段；② **架构层面**：576 个 image token 直接注入 Transformer 的第 1 层，视觉 token 和文本 token 从模型最早期就深度交织。这与传统 VLM"只在输入层拼接"的做法不同，Qwen3.5 的视觉信息在 LLM 的每一层都通过 DeepStack 持续参与计算，实现了真正意义上的深度融合。

---

## 七、附录：核心术语速查

| 术语 | 含义 |
|------|------|
| **PatchMerger** | 视觉-语言投影模块，将 ViT patch 特征压缩并投影到 LLM 嵌入维度 |
| **DeepStack** | Qwen3-VL 的多层视觉特征注入机制，从 ViT 多中间层提取特征注入 LLM 多层 |
| **MRoPE** | Multimodal Rotary Position Embedding，(t, h, w) 三维解耦位置编码 |
| **Interleaved-MRoPE** | 交错式 MRoPE，t/h/w 维度交错排列，确保频率谱平衡 |
| **masked_scatter** | 在指定位置替换 embedding 的注入操作（Qwen2.5-VL） |
| **scatter_add** | 在指定位置累加特征的注入操作（Qwen3-VL / 3.5） |
| **Early Fusion** | 视觉和语言组件从预训练早期就联合训练的范式 |
| **Native Multimodal Training** | 原生多模态合一训练，所有模态组件同步更新 |
| **Gated DeltaNet** | 线性注意力变体，O(N) 复杂度，KV Cache 与序列长度无关 |
| **Hybrid MoE** | Gated DeltaNet 与传统 Attention 3:1 交替的混合架构 |
| **MTP** | Multi-Token Prediction，多 token 预测，支持投机解码 |
| **YaRN** | RoPE 扩展方法，用于外推至训练时未见过的长上下文 |
