### <strong>1. LLM 八股</strong>

#### <strong>1.1 请详细解释一下 Transformer 模型中的自注意力机制是如何工作的？它为什么比 RNN 更适合处理长序列？</strong>


* <strong>参考答案：</strong>
    自注意力（Self-Attention）机制是Transformer模型的核心，它使得模型能够动态地衡量输入序列中不同单词之间的重要性，并据此生成每个单词的上下文感知表示。

    <strong>工作原理如下：</strong>

    1.  <strong>生成Q, K, V向量：</strong> 对于输入序列中的每一个词元（token）的嵌入向量，我们通过乘以三个可学习的权重矩阵 $W^Q, W^K, W^V$ ，分别生成三个向量：查询向量（Query, Q）、键向量（Key, K）和值向量（Value, V）。
        * <strong>Query (Q):</strong> 代表当前词元为了更好地理解自己，需要去“查询”序列中其他词元的信息。
        * <strong>Key (K):</strong> 代表序列中每个词元所“携带”的，可以被查询的信息标签。
        * <strong>Value (V):</strong> 代表序列中每个词元实际包含的深层含义。

    2.  <strong>计算注意力分数：</strong> 为了确定当前词元（由Q代表）应该对其他所有词元（由K代表）投入多少关注，我们计算当前词元的Q与其他所有词元的K的点积。这个分数衡量了两者之间的相关性。
        <div align="center">
        $$\text{Score}(Q_i, K_j) = Q_i \cdot K_j$$
        </div>

    3.  <strong>缩放（Scaling）：</strong> 将计算出的分数除以一个缩放因子 $\sqrt{d_k}$（ $d_k$ 是K向量的维度）。这一步是为了在反向传播时获得更稳定的梯度，防止点积结果过大导致Softmax函数进入饱和区。
        <div align="center">
        $$\frac{Q \cdot K^T}{\sqrt{d_k}}$$
        </div>

    4.  <strong>Softmax归一化：</strong> 将缩放后的分数通过一个Softmax函数，使其转换为一组总和为1的概率分布。这些概率就是“注意力权重”，表示在当前位置，每个输入词元所占的重要性。
        <div align="center">
        $$\text{AttentionWeights} = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right)$$
        </div>

    5.  <strong>加权求和：</strong> 最后，将得到的注意力权重与每个词元对应的V向量相乘并求和，得到最终的自注意力层输出。这个输出向量融合了整个序列的上下文信息，且权重由模型动态学习得到。
        <div align="center">
        $$\text{Output} = \text{AttentionWeights} \cdot V$$
        </div>

    <strong>为什么比RNN更适合处理长序列？</strong>

    1.  <strong>并行计算能力：</strong> 自注意力机制在计算时，可以一次性处理整个序列，计算所有位置之间的关联，是高度并行的。而RNN（包括LSTM、GRU）必须按照时间顺序依次处理每个词元，无法并行化，导致处理长序列时速度非常慢。
    2.  <strong>解决长距离依赖问题：</strong> 在自注意力中，任意两个位置之间的交互路径长度都是O(1)，因为可以直接计算它们的注意力分数。而在RNN中，序列首尾两个词元的信息传递需要经过整个序列的长度，路径为O(N)，这极易导致梯度消失或梯度爆炸，使得模型难以捕捉长距离的依赖关系。

---

#### <strong>1.2 什么是位置编码？在 Transformer 中，为什么它是必需的？请列举至少两种实现方式。</strong>

   
* <strong>参考答案：</strong>
    <strong>什么是位置编码？</strong>
    位置编码（Positional Encoding, PE）是一个与词嵌入维度相同的向量，其目的是向模型注入关于词元在输入序列中绝对或相对位置的信息。它会与词元的词嵌入（Token Embedding）相加，然后一同输入到Transformer的底层。

    <strong>为什么它是必需的？</strong>
    Transformer的核心机制——自注意力，在计算时处理的是一个集合（Set）而非序列（Sequence）。它本身不包含任何关于词元顺序的信息，是 <strong>置换不变（Permutation-invariant）</strong> 的。这意味着，如果打乱输入序列中词元的顺序，自注意力层的输出也会相应地被打乱，但每个词元自身的输出向量（在不考虑softmax归一化的情况下）是相同的。这显然不符合自然语言的特性，因为语序至关重要（例如“我打你”和“你打我”含义完全相反）。因此，必须通过一种外部机制，将位置信息显式地提供给模型，这就是位置编码的作用。

    <strong>至少两种实现方式：</strong>

    1.  <strong>正弦/余弦位置编码（Sinusoidal Positional Encoding）：</strong>
        这是原始Transformer论文《Attention Is All You Need》中使用的方法。它使用不同频率的正弦和余弦函数来生成位置编码，其公式如下：
        <div align="center">
        $$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{\text{model}}})$$
        </div>

        <div align="center">
        $$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{\text{model}}})$$
        </div>

        其中， $pos$ 是词元在序列中的位置， $i$ 是编码向量中的维度索引， $d_{\text{model}}$ 是嵌入维度。
        * <strong>优点：</strong>
            * <strong>可外推性：</strong> 能够处理比训练中最长序列还要长的序列。
            * <strong>相对位置信息：</strong> 模型可以轻易地学习到相对位置关系，因为对于任何固定的偏移量 $k$ ， $PE_{pos+k}$ 都可以表示为 $PE_{pos}$ 的一个线性函数，这使得模型更容易捕捉相对位置的依赖。

    2.  <strong>可学习的绝对位置编码（Learned Absolute Positional Encoding）：</strong>
        这种方法将位置编码视为模型参数的一部分，通过训练学习得到。具体来说，会创建一个形状为 `(max_sequence_length, embedding_dimension)` 的位置编码矩阵。在处理序列时，根据每个词元的位置索引，从这个矩阵中查找对应的编码向量，并加到词嵌入上。BERT和GPT-2等模型采用了这种方式。
        * <strong>优点：</strong> 模式更加灵活，可以让模型自己学习出最适合数据的位置表示。
        * <strong>缺点：</strong> 无法泛化到超过预设 `max_sequence_length` 的长度。如果需要处理更长的序列，就需要对位置编码进行微调或采用其他策略。

---

#### <strong>1.3 请你详细介绍ROPE，对比绝对位置编码它的优劣势分别是什么？</strong>

* <strong>参考答案：</strong>
    <strong>RoPE (Rotary Position Embedding) 介绍</strong>
    RoPE，全称旋转位置编码，是目前大语言模型（如Llama系列、Qwen等）中最主流的位置编码方案之一。它是一种将位置信息融入自注意力机制的创新方法。

    其核心思想是：<strong>通过向量旋转的方式，将绝对位置信息编码到Query和Key向量中，从而使得模型在计算注意力分数时，能够自然地利用相对位置信息。</strong>

    <strong>工作原理：</strong>
    RoPE不再像传统位置编码那样直接将位置向量加到词嵌入上。它的操作发生在生成Q和K向量之后、计算注意力分数之前：
    1.  <strong>维度分组：</strong> 将Q和K向量的 $d$ 维特征两两一组，视为 $d/2$ 个二维向量。
    2.  <strong>构造旋转矩阵：</strong> 对于序列中的位置 $m$，构造一个与位置相关的旋转矩阵 $R_m$。这个矩阵在二维空间中表示一个旋转操作。
    3.  <strong>旋转Q和K：</strong> 将每个二维向量组通过对应的旋转矩阵 $R_m$ 进行旋转。

    数学上，这个过程等价于将每个二维向量 $(x_m, x_{m+1})$ 看作一个复数，然后乘以一个复数 $e^{im\theta}$，其中 $m$ 是位置， $\theta$ 是一个预设的、与维度相关的常数。这个操作只会改变向量的相位（方向），而不改变其模（长度）。

    <strong>关键特性：</strong>
    RoPE的巧妙之处在于，经过旋转后的两个位置 $m$ 和 $n$ 的Query向量 $q_m$ 和Key向量 $k_n$ 进行点积运算时，其结果只与它们的相对位置 $(m-n)$ 有关，而与它们的绝对位置 $m$ 和 $n$ 无关。这使得自注意力机制天然地具备了对相对位置的感知能力。

    <strong>对比绝对位置编码的优劣势：</strong>

    <strong>RoPE的优势：</strong>
    1.  <strong>内置相对位置建模：</strong> 这是其最大的优势。RoPE使得注意力分数直接依赖于词元间的相对距离，这更符合自然语言中语法和语义依赖通常是相对的这一特性。
    2.  <strong>良好的外推能力：</strong> 由于其数学性质，RoPE在处理比训练时更长的序列时表现出色，具有很强的长度泛化能力，这也是长序列LLM偏爱它的重要原因。
    3.  <strong>不引入额外可训练参数：</strong> RoPE是一种函数式的、固定的编码方式，不需要像可学习位置编码那样占用模型参数。
    4.  <strong>随着距离增加，依赖性衰减：</strong> 旋转的性质使得距离越远的词元，其内积关系会呈现周期性的衰减，符合语言中距离越远相关性越弱的直觉。

    <strong>RoPE的劣势：</strong>
    1.  <strong>理论理解相对复杂：</strong> 其背后的数学原理（复数、欧拉公式、旋转矩阵）比直接相加的绝对位置编码更抽象。
    2.  <strong>对绝对位置信息的表征可能较弱：</strong> 虽然RoPE从绝对位置导出，但其在注意力机制中的核心作用是体现相对位置。对于那些强依赖绝对位置信息的特定任务（例如，判断一个词是否在句子开头），它的效果可能不如直接使用绝对位置编码直观。

---
#### <strong>1.4 你知道MHA，MQA，GQA的区别吗？详细解释一下。</strong>



* <strong>参考答案：</strong>
    MHA、MQA和GQA是Transformer模型中三种不同的注意力机制变体，它们的主要区别在于如何组织和共享Query、Key和Value的“头”（Head），核心目标是在模型效果和推理效率（特别是显存占用）之间做出不同的权衡。

    #### <strong>1. MHA (Multi-Head Attention)</strong>
    这是原始Transformer论文中提出的标准注意力机制。
    * <strong>工作原理：</strong>
        1.  将输入的Q、K、V向量分别通过 $N$ 个独立的线性变换，得到 $N$ 组不同的 $Q_i, K_i, V_i$ 头（ $i=1, ..., N$ ）。
        2.  这 $N$ 组头在各自的子空间中并行地计算注意力（Scaled Dot-Product Attention）。
        3.  将 $N$ 个头计算得到的输出向量拼接（Concatenate）起来。
        4.  最后通过一个线性变换将拼接后的向量映射回原始维度。
    * <strong>结构：</strong> $N$ 个Query头， $N$ 个Key头， $N$ 个Value头。
    * <strong>优点：</strong> 效果最好，模型能力最强。每个头可以在不同的表示子空间中学习到不同的信息。
    * <strong>缺点：</strong> 推理成本高。在自回归生成任务中，需要缓存每一层的Key和Value（即KV Cache），MHA的KV Cache大小与头的数量$N$成正比，显存占用非常大，限制了长序列的生成。

    #### <strong>2. MQA (Multi-Query Attention)</strong>
    为了解决MHA在推理时的显存瓶颈而被提出。
    * <strong>工作原理：</strong>
        1.  与MHA一样，有 $N$ 个独立的Query头。
        2.  <strong>核心区别：</strong> 所有的 $N$ 个Query头共享<strong>同一个</strong>Key头和<strong>同一个</strong>Value头。
    * <strong>结构：</strong> $N$ 个Query头，<strong>1个</strong>Key头，<strong>1个</strong>Value头。
    * <strong>优点：</strong> 极大地降低了推理成本。KV Cache的大小不再依赖于头的数量 $N$ ，相比MHA减小了 $N$ 倍，显著降低了显存占用，并加快了推理速度。
    * <strong>缺点：</strong> 可能会导致模型性能的下降。因为所有Query头被迫从同样的一组Key和Value中提取信息，模型的表达能力受到了一定的限制。

    #### <strong>3. GQA (Grouped-Query Attention)</strong>
    GQA是MHA和MQA之间的一个折中方案，旨在平衡性能和效率。
    * <strong>工作原理：</strong>
        1.  将 $N$ 个Query头分成 $G$ 组。
        2.  <strong>核心区别：</strong> 每组内的Query头共享一个Key头和一个Value头。总共有 $G$ 个Key头和 $G$ 个Value头。
    * <strong>结构：</strong> $N$ 个Query头，<strong>G个</strong>Key头，<strong>G个</strong>Value头。（通常 $1 < G < N$ ）。
    * <strong>说明：</strong>
        * 当 $G=N$ 时，GQA等价于MHA。
        * 当 $G=1$ 时，GQA等价于MQA。
    * <strong>优点：</strong> 在推理效率上远超MHA，同时在模型性能上优于MQA。它提供了一个灵活的旋钮，可以根据具体需求在效率和效果之间进行调整。Llama 2等模型就采用了GQA。

    <strong>总结：</strong>
    | 特性         | MHA (Multi-Head Attention) | MQA (Multi-Query Attention) | GQA (Grouped-Query Attention) |
    | :----------- | :------------------------- | :-------------------------- | :---------------------------- |
    | <strong>结构</strong>     | N个Q头, N个K头, N个V头     | N个Q头, 1个K头, 1个V头      | N个Q头, G个K头, G个V头        |
    | <strong>模型质量</strong> | 最高                       | 可能下降                    | 接近MHA，优于MQA              |
    | <strong>推理效率</strong> | 最低 (KV Cache大)          | 最高 (KV Cache小)           | 居中，远好于MHA               |
    | <strong>应用</strong>     | BERT, GPT-3                | PaLM                        | Llama 2, Mixtral              |

---

#### <strong>1.5 请比较一下几种常见的 LLM 架构，例如 Encoder-Only, Decoder-Only, 和 Encoder-Decoder，并说明它们各自最擅长的任务类型。</strong>



* <strong>参考答案：</strong>
    LLM的架构主要可以分为三类，它们的核心区别在于使用了Transformer的哪些部分以及注意力机制的类型，这直接决定了它们各自擅长的任务。

    #### <strong>1. Encoder-Only 架构 (例如 BERT, RoBERTa)</strong>
    * <strong>结构：</strong> 由多个Transformer Encoder层堆叠而成。
    * <strong>核心机制：</strong> <strong>双向自注意力机制</strong>。在处理序列中的任何一个词元时，模型都可以同时关注到它左边和右边的所有词元。这使得模型能够获得非常丰富的上下文表示。
    * <strong>最擅长的任务类型：自然语言理解 (NLU)</strong>。
        * <strong>具体任务：</strong>
            * <strong>分类任务：</strong> 情感分析、文本分类。
            * <strong>序列标注：</strong> 命名实体识别 (NER)。
            * <strong>句子关系判断：</strong> 自然语言推断 (NLI)。
            * <strong>完形填空：</strong> 像BERT的Masked Language Model (MLM) 预训练任务本身。
        * <strong>原因：</strong> 这些任务的核心是<strong>理解</strong>输入文本的深层含义，而双向上下文对于准确理解至关重要。这类模型的输出通常是固定的标签或类别，而非自由生成的长文本。

    #### <strong>2. Decoder-Only 架构 (例如 GPT系列, Llama, Qwen)</strong>
    * <strong>结构：</strong> 由多个Transformer Decoder层堆叠而成，但移除了其中的Encoder-Decoder交叉注意力部分。
    * <strong>核心机制：</strong> <strong>单向（因果）自注意力机制 (Causal Self-Attention)</strong>。在预测第 `t` 个词元时，模型只能关注到位置 `1` 到 `t-1` 的词元，不能看到未来的信息。这种自回归的特性天然适合生成任务。
    * <strong>最擅长的任务类型：自然语言生成 (NLG)</strong>。
        * <strong>具体任务：</strong>
            * <strong>开放式文本生成：</strong> 写文章、故事、诗歌。
            * <strong>对话系统/聊天机器人：</strong> 如ChatGPT。
            * <strong>代码生成：</strong> 如Copilot。
            * <strong>上下文续写 (In-context Learning)。</strong>
        * <strong>原因：</strong> 语言的生成过程是顺序的、从左到右的，Decoder-Only架构的单向注意力完美地模拟了这一过程。目前绝大多数的通用大语言模型都采用此架构。

    #### <strong>3. Encoder-Decoder 架构 (例如 T5, BART, 原始Transformer)</strong>
    * <strong>结构：</strong> 包含一个完整的Encoder栈和一个完整的Decoder栈。
    * <strong>核心机制：</strong> Encoder部分使用<strong>双向注意力</strong>来编码整个输入序列，形成一个全面的上下文表示。Decoder部分在生成输出时，一方面使用<strong>单向注意力</strong>处理已生成的序列，另一方面通过<strong>交叉注意力 (Cross-Attention)</strong>机制来关注Encoder的输出，确保生成内容与输入相关。
    * <strong>最擅长的任务类型：序列到序列 (Seq2Seq)</strong>。
        * <strong>具体任务：</strong>
            * <strong>机器翻译：</strong> 将一种语言（输入序列）翻译成另一种语言（输出序列）。
            * <strong>文本摘要：</strong> 将一篇长文章（输入序列）概括成几句话（输出序列）。
            * <strong>问答：</strong> 将问题（输入序列）转换为答案（输出序列）。
        * <strong>原因：</strong> 这类任务需要首先对源序列有一个完整的、全局的理解（由Encoder完成），然后基于这个理解有条件地生成一个目标序列（由Decoder完成）。

---

#### <strong>1.6 什么是Scaling Laws？它揭示了模型性能、计算量和数据量之间的什么关系？这对LLM的研发有什么指导意义？</strong>



* <strong>参考答案：</strong>
    <strong>什么是Scaling Laws？</strong>
    Scaling Laws（尺度定律）是由OpenAI、DeepMind等机构通过大量实验发现的一系列经验性规律。它揭示了大型语言模型的性能（通常以交叉熵损失函数Loss来衡量）与三个关键资源要素——<strong>模型参数规模（N）</strong>、<strong>训练数据集大小（D）</strong>和<strong>训练所用的计算量（C）</strong>——之间存在着可预测的<strong>幂律关系（Power-Law Relationship）</strong>。

    <strong>揭示了什么关系？</strong>
    1.  <strong>性能的可预测性：</strong> Scaling Laws表明，模型的性能损失会随着N、D、C的增加而平滑地、可预测地下降。这种关系可以用一个幂律公式来描述，例如，当数据和计算量足够时，模型损失 L 与模型参数量 N 的关系大致为： $L(N) \propto N^{-\alpha}$ ，其中 $\alpha$ 是一个小的正指数。这意味着我们可以通过在小规模模型上的实验结果，来外推（predict）更大规模模型可能达到的性能。
    2.  <strong>瓶颈效应：</strong> 模型的最终性能会被N、D、C中最受限的那个因素所制约。如果仅仅增加模型大小而不增加数据量，性能提升会很快达到瓶颈；反之亦然。为了有效提升模型性能，必须协同扩展这三个要素。
    3.  <strong>资源的最优分配：</strong> 对于一个给定的计算预算（FLOPs），存在一个最优的模型大小（N）和数据量（D）的组合。DeepMind的Chinchilla论文是一个里程碑式的发现，它修正了早期认为应该优先扩大模型规模的观点，指出<strong>为了达到计算最优，模型参数量和训练数据量应该近似1:20的比例进行扩展</strong>。例如，训练一个70B参数的模型，大约需要1.4万亿个token的数据。

    <strong>对LLM研发的指导意义：</strong>
    1.  <strong>科学指导项目规划：</strong> 在投入数百万甚至数千万美元进行一次大规模训练之前，研究机构可以先通过小规模实验拟合出自己数据集和模型架构下的Scaling Law。这使得他们能够科学地预测最终模型的性能，评估项目的投资回报率，并合理申请计算资源。
    2.  <strong>优化资源配置，避免浪费：</strong> Scaling Laws，特别是Chinchilla定律，为如何高效使用计算预算提供了明确的指导。它告诉我们，与其训练一个参数巨大但数据不足的模型（over-trained），不如用同样的算力去训练一个参数稍小但数据更充分的模型（under-trained），后者效果可能更好。这促使业界从单纯追求“大参数”转向“大参数与大数据的平衡”。
    3.  <strong>强调数据的重要性：</strong> Scaling Laws的发现，让学术界和工业界都更加深刻地认识到，高质量、大规模的训练数据和模型参数规模同等重要，甚至在某些阶段更为关键。这推动了数据工程、数据清洗和高质量合成数据生成等领域的发展。

---

#### <strong>1.7 在LLM的推理阶段，有哪些常见的解码策略？请解释 Greedy Search, Beam Search, Top-K Sampling 和 Nucleus Sampling (Top-P) 的原理和优缺点。</strong>


* <strong>参考答案：</strong>
    在LLM的推理（或称解码）阶段，模型会生成一个词元概率分布，解码策略决定了如何从这个分布中选择下一个词元。常见的策略可以分为确定性和随机性两类。

    #### <strong>1. Greedy Search (贪心搜索)</strong>
    * <strong>原理：</strong> 在每个时间步，总是选择当前概率分布中概率最高的那个词元作为输出。
    * <strong>优点：</strong>
        * <strong>速度快：</strong> 计算开销最小，实现最简单。
    * <strong>缺点：</strong>
        * <strong>局部最优：</strong> 每一步的“贪心”选择可能导致整个序列不是全局最优的。一个高概率的词后面可能跟着一系列低概率的词，最终序列的总概率反而不高。
        * <strong>缺乏多样性：</strong> 输出是完全确定的，对于同一个输入，每次生成的结果都一样，内容往往比较呆板、重复。

    #### <strong>2. Beam Search (集束搜索)</strong>
    * <strong>原理：</strong> 这是对贪心搜索的改进。它在每个时间步会保留 $k$ 个（ $k$ 称为 "beam width" 或 "beam size"）最有可能的候选序列。在下一步，它会从这 $k$ 个候选序列出发，生成所有可能的下一个词元，然后从所有这些扩展出的新序列中，再次选出累计概率最高的 $k$ 个。最后，从最终的 $k$ 个完整序列中选择最优的一个。
    * <strong>优点：</strong>
        * <strong>质量更高：</strong> 通过探索更广的搜索空间，通常能找到比贪心搜索概率更高、质量更好的序列。
    * <strong>缺点：</strong>
        * <strong>计算成本高：</strong> 需要维护 $k$ 个候选序列，计算和内存开销是贪心搜索的 $k$ 倍。
        * <strong>仍然倾向于安全和高频：</strong> 优化目标是全局概率，这使得它还是倾向于生成常见、安全的句子，可能缺乏创造性，并且在长文本生成中容易出现重复。

    #### <strong>3. Top-K Sampling (Top-K 采样)</strong>
    * <strong>原理：</strong> 这是一种随机采样策略。在每个时间步，不再是选择最优的，而是：
        1.  从整个词汇表的概率分布中，筛选出概率最高的 $K$ 个词元。
        2.  将这 $K$ 个词元的概率进行归一化（使它们的和为1）。
        3.  在这 $K$ 个词元中，根据新的概率分布进行随机采样。
    * <strong>优点：</strong>
        * <strong>增加多样性：</strong> 引入了随机性，使得生成内容更加丰富、有趣和不可预测。
        * <strong>避免低概率词：</strong> 通过限制在Top-K范围内，过滤掉了那些概率极低、可能不通顺或奇怪的词元。
    * <strong>缺点：</strong>
        * <strong>K值固定：</strong> $K$ 是一个固定的超参数。当概率分布很尖锐时（模型非常确定下一个词），一个大的K可能会引入不相关的词；当概率分布很平坦时（模型不确定），一个小的K可能会限制模型的选择。

    #### <strong>4. Nucleus Sampling / Top-P Sampling (核心采样)</strong>
    * <strong>原理：</strong> 这是对Top-K采样的改进，它使用一个动态的候选词元集。
        1.  将所有词元按概率从高到低排序。
        2.  从概率最高的词元开始，逐个累加它们的概率，直到总概率之和超过一个预设的阈值 $p$（例如 $p=0.95$）。
        3.  这个累加过程中包含的所有词元构成了“核心（Nucleus）”候选集。
        4.  然后，在这个动态大小的候选集中，根据它们的原始概率进行归一化和随机采样。
    * <strong>优点：</strong>
        * <strong>自适应候选集：</strong> 候选集的大小会根据上下文动态变化。当模型对下一个词非常确定时，概率分布尖锐，可能只有一两个词的概率和就超过了 $p$，候选集就很小，生成更精确；当模型不确定时，概率分布平坦，需要包含更多词才能达到 $p$，候选集就变大，允许更多探索。
        * <strong>兼顾质量与多样性：</strong> 相比Top-K，它是一种更原则性和鲁棒性的方法，是目前大多数LLM应用默认的采样策略。

---

#### <strong>1.8 什么是词元化？请比较一下 BPE 和 WordPiece 这两种主流的子词切分算法。</strong>


* <strong>参考答案：</strong>
    <strong>什么是词元化（Tokenization）？</strong>
    词元化是将原始的文本字符串分解成一个个独立的单元（称为“词元”或“token”），并将这些词元映射到唯一的整数ID的过程。这是自然语言处理模型处理文本的第一步，因为模型只能处理数字输入。

    现代大型语言模型普遍采用 <strong>子词（Subword）</strong> 词元化算法，它介于按词切分和按字符切分之间。这样做的好处是：
    1.  <strong>有效处理未登录词（OOV）：</strong> 任何罕见词或新词都可以被拆解成已知的子词组合，避免了“未知”标记。
    2.  <strong>平衡词表大小与序列长度：</strong> 相比于词级别，词表规模大大减小；相比于字符级别，生成的序列长度又不会过长，兼顾了效率。
    3.  <strong>保留形态信息：</strong> 像 "running", "runner" 这样的词可以共享 "run" 这个子词，使得模型能够理解词根和词缀的关系。

    <strong>BPE vs. WordPiece</strong>

    BPE和WordPiece是两种最主流的子词切分算法，它们构建词表的过程相似，但在合并子词的决策标准上有所不同。

    #### <strong>BPE (Byte Pair Encoding)</strong>
    * <strong>工作原理：</strong>
        1.  <strong>初始化：</strong> 词汇表由语料库中出现的所有基本字符组成。
        2.  <strong>迭代合并：</strong> 重复以下步骤直到达到预设的词表大小：
            a.  在整个语料库中，统计所有相邻词元对的出现频率。
            b.  找出<strong>频率最高</strong>的那个词元对（例如 `('e', 's')`）。
            c.  将这个词元对合并成一个新的、更长的词元（`'es'`），并将其加入词汇表。
            d.  在语料库中，用新词元替换所有出现的该词元对。
    * <strong>应用模型：</strong> GPT系列、Llama等。
    * <strong>特点：</strong> 算法思想简单直观，完全基于数据中符号对的出现频率。

    #### <strong>WordPiece</strong>
    * <strong>工作原理：</strong>
        1.  <strong>初始化：</strong> 与BPE一样，词汇表也从所有基本字符开始。
        2.  <strong>迭代合并（核心区别）：</strong> WordPiece在选择合并哪两个子词时，不是基于频率，而是基于<strong>语言模型的似然（Likelihood）</strong>。它会尝试所有可能的合并，并选择那个能够<strong>最大程度提升训练数据似然值</strong>的合并操作。
        * 可以通俗地理解为：如果把语料库看作一个语言模型，每次合并都应该让这个语言模型产生当前语料库的概率变得最大。它倾向于合并那些内部凝聚力更强的字符组合。
    * <strong>应用模型：</strong> BERT, DistilBERT, Electra。
    * <strong>特点：</strong> WordPiece在切分时，通常会在单词的非起始部分子词前加上特殊符号（如`##`），例如 "tokenization" 可能会被切分为 `("token", "##ization")`。

    <strong>主要区别总结：</strong>
    | 特性             | BPE (Byte Pair Encoding)                     | WordPiece                                                  |
    | :--------------- | :------------------------------------------- | :--------------------------------------------------------- |
    | <strong>合并决策标准</strong> | <strong>频率驱动</strong>：合并出现次数最多的相邻子词对。 | <strong>似然驱动</strong>：合并能最大化提升语料库语言模型似然的子词对。 |
    | <strong>理论基础</strong>     | 数据压缩算法，简单高效。                     | 概率语言模型，理论上更优。                                 |
    | <strong>应用代表</strong>     | GPT, Llama, RoBERTa                          | BERT, T5                                                   |

---

#### <strong>1.9 你觉得NLP和LLM最大的区别是什么？两者有何共同和不同之处？</strong>


* <strong>参考答案：</strong>
    NLP（自然语言处理）和LLM（大型语言模型）之间是领域与技术、一般与具体的关系。LLM是NLP发展至今最前沿、最具影响力的一项技术范式，它在很大程度上重塑了NLP领域。

    <strong>共同之处：</strong>
    * <strong>最终目标一致：</strong> 两者的根本目标都是实现人工智能对人类语言的理解、生成、和运用，即所谓的“人工智能皇冠上的明珠”。
    * <strong>技术根基相通：</strong> 现代NLP和LLM都建立在深度学习，特别是神经网络的基础上。Transformer架构是连接两者的关键桥梁，从BERT到GPT，都是其思想的延伸和发展。

    <strong>最大的区别与不同之处：</strong>

    最大的区别在于<strong>研究和应用范式</strong>的根本性转变，从“为每个任务训练一个模型”转向“用一个模型解决所有任务”。

    具体可以从以下几个维度来看：

    1.  <strong>任务处理范式 (Task-Handling Paradigm)：</strong>
        * <strong>传统NLP：</strong> 奉行“分而治之”的策略。研究者会针对每一个具体的NLP任务（如机器翻译、情感分析、命名实体识别）设计特定的模型架构、损失函数和训练数据集，遵循`Pre-train -> Fine-tune`的流程。每个模型都是一个“专家”。
        * <strong>LLM：</strong> 追求“大一统”的通用模型。通过在海量数据上进行大规模预训练，一个LLM基础模型就具备了解决多种任务的潜力。用户通过设计不同的 <strong>提示（Prompt）</strong> 或提供 <strong>上下文示例（In-context Learning）</strong> 来引导模型完成任务，大大简化了开发流程，甚至实现了 <strong>零样本（Zero-shot）</strong> 和 <strong>少样本（Few-shot）</strong> 学习。

    2.  <strong>模型能力与“涌现” (Model Capabilities & Emergence)：</strong>
        * <strong>传统NLP：</strong> 模型的能
        力是明确且有限的，通常与其训练目标直接相关。
        * <strong>LLM：</strong> 当模型规模（参数、数据、算力）跨越某个阈值后，会表现出小模型上不存在的 <strong>“涌现能力” (Emergent Abilities)</strong> 。例如，复杂的逻辑推理（思维链, Chain-of-Thought）、代码生成、遵循复杂指令等。这些能力不是被直接训练的，而是从海量数据中自发学习到的。

    3.  <strong>规模 (Scale)：</strong>
        * <strong>传统NLP：</strong> 模型参数量通常在百万级到几亿级（例如，BERT-base约1.1亿）。
        * <strong>LLM：</strong> 参数量从百亿（Billion）起步，发展到千亿甚至万亿级别。训练数据和所需计算资源也比传统NLP模型高出几个数量级。

    4.  <strong>交互与应用方式 (Interaction & Application)：</strong>
        * <strong>传统NLP：</strong> 通常以API形式被集成到软件中，输入输出格式相对固定。
        * <strong>LLM：</strong> 催生了以<strong>对话</strong>和<strong>指令</strong>为核心的全新交互方式（如ChatGPT），使得AI更加平易近人。应用也从后端工具演变为可以直接面向用户的产品。

    <strong>总结：</strong> 如果说传统NLP是在打造一支由各种“工具专家”组成的工具箱，那么LLM则是在努力打造一个“瑞士军刀”式的通用智能工具，它可能在某些特定任务上不如专用工具精细，但其通用性、灵活性和强大的涌现能力是前所未有的。

---

#### <strong>1.10 L1和L2正则化分别是什么，什么场景适合使用呢？</strong>


* <strong>参考答案：</strong>
    L1和L2正则化都是在机器学习和深度学习中用于防止模型过拟合的常用技术。它们通过在模型的损失函数（Loss Function）中添加一个代表模型复杂度的惩罚项来实现这一目标。

    #### <strong>L1 正则化 (L1 Regularization / Lasso)</strong>
    * <strong>定义：</strong> L1正则化添加的惩罚项是模型所有权重参数 $w_i$ 的<strong>绝对值之和</strong>，乘以一个正则化系数 $\lambda$。
        <div align="center"> 
        $$\text{Loss}_{L1} = \text{Original Loss} + \lambda \sum_{i} |w_i|$$
        </div>
        
    * <strong>核心作用：产生稀疏性 (Sparsity)</strong>。
        在梯度下降优化过程中，L1惩罚项会驱使那些对模型贡献不大的特征的权重最终变为<strong>精确的0</strong>。这相当于从模型中完全移除了这些特征。
    * <strong>适用场景：特征选择 (Feature Selection)</strong>。
        当你的数据集中包含大量特征，但你怀疑其中许多特征是冗余或无用的时，L1正则化非常有用。它能够自动地“筛选”出最重要的特征，简化模型，提高解释性。

    #### <strong>L2 正则化 (L2 Regularization / Ridge / Weight Decay)</strong>
    * <strong>定义：</strong> L2正则化添加的惩罚项是模型所有权重参数 $w_i$ 的<strong>平方和</strong>，乘以一个正则化系数 $\lambda$。
        <div align="center">
        $$\text{Loss}_{L2} = \text{Original Loss} + \lambda \sum_{i} w_i^2$$
        </div>
        
    * <strong>核心作用：权重衰减 (Weight Decay)</strong>。
        L2正则化会惩罚大的权重值，它会促使模型的权重参数尽可能小，<strong>趋近于0但通常不会等于0</strong>。这使得模型的权重分布更加平滑和分散，避免模型过度依赖少数几个高权重的特征。
    * <strong>适用场景：通用性的过拟合防治</strong>。
        L2是更常用、更通用的正则化方法。当特征之间可能存在相关性（共线性），或者你认为绝大多数特征都对预测有或多或少的贡献时，L2是首选。它能有效地提高模型的泛化能力，使其在未见过的数据上表现更好。在深度学习中，“权重衰减”通常就是指L2正则化。

    <strong>总结对比：</strong>
    | 对比项       | L1 正则化                              | L2 正则化                |
    | :----------- | :------------------------------------- | :----------------------- |
    | <strong>惩罚项</strong>   | 权重的绝对值之和 (L1范数)              | 权重的平方和 (L2范数)    |
    | <strong>效果</strong>     | 权重稀疏化，部分权重为0                | 权重平滑化，权重趋近于0  |
    | <strong>主要用途</strong> | 特征选择，简化模型                     | 防止过拟合，提升泛化能力 |
    | <strong>解的特性</strong> | 不稳定，数据微小变动可能导致特征集变化 | 稳定，解是唯一的         |

---

#### <strong>1.11 “涌现能力”是大型模型中一个备受关注的现象，请问你如何理解这个概念？它通常在模型规模达到什么程度时出现？</strong>


* <strong>参考答案：</strong>
    <strong>对“涌现能力”的理解：</strong>
    “涌现能力”（Emergent Abilities）是指那些<strong>在小型模型中不存在或表现不佳，但当模型规模（包括参数量、训练数据和计算量）达到某个临界点后，突然出现并显著超越随机水平的能力</strong>。

    它的核心特征是<strong>非线性和不可预测性</strong>：
    * <strong>非线性增长：</strong> 这种能力的性能表现并不随着模型规模的增加而平滑、线性地提升。相反，它会在某个规模区间内发生“相变”式的跃迁，性能从接近随机猜测的水平迅速提升到非常高的水平。
    * <strong>非直接训练：</strong> 这些高级能力通常不是通过特定的监督学习目标直接训练出来的。例如，我们没有直接教模型如何“一步一步思考”，但当模型足够大时，它通过学习海量文本中的逻辑关系，自发地获得了这种能力。

    <strong>典型的涌现能力例子包括：</strong>
    1.  <strong>思维链（Chain-of-Thought, CoT）：</strong> 在面对需要多步推理的数学或逻辑问题时，通过提示模型“一步一步地思考”，大模型可以生成一个连贯的推理过程并得出正确答案。小模型则无法利用这种提示。
    2.  <strong>上下文学习（In-context Learning）：</strong> 无需更新模型权重，仅在Prompt中提供几个任务示例（Few-shot），大模型就能“学会”并执行这个新任务。
    3.  <strong>执行复杂指令：</strong> 理解并遵循包含多个步骤、约束和否定逻辑的复杂人类指令。

    <strong>出现的模型规模：</strong>
    涌现能力出现的具体规模<strong>没有一个固定的数值</strong>，它取决于能力本身、模型架构、数据质量和评估任务的复杂性。

    然而，根据Google等机构的标志性研究，许多引人注目的涌现能力，例如<strong>思维链推理</strong>，通常是在模型参数规模达到<strong>百亿（tens of billions）到千亿（a hundred billion）</strong> 级别时开始出现的。
    * 例如，在Google PaLM模型的实验中，思维链推理能力在<strong>62B参数</strong>的模型上开始显现，而在8B和16B的模型上则完全无效。这种能力随着模型增长到<strong>540B</strong>时变得更加强大和稳定。

    总而言之，“涌现能力”是“量变引起质变”在大型模型领域的生动体现，它表明单纯地扩大规模可以解锁全新的、更高级的认知能力，这也是当前LLM研究持续推动模型规模增长的核心驱动力之一。

---

#### <strong>1.12 激活函数有了解吗，你知道哪些LLM常用的激活函数？为什么选用它？</strong>

* <strong>参考答案：</strong>
    是的，我了解激活函数。激活函数是神经网络中至关重要的一环，它的主要作用是<strong>为网络引入非线性（non-linearity）</strong>。如果没有激活函数，多层神经网络本质上等同于一个单层的线性模型，无法学习和拟合复杂的数据模式。

    在现代大型语言模型（Transformer架构）中，最常用的激活函数主要有两个：<strong>GeLU</strong> 和 <strong>SwiGLU</strong>。

    1.  <strong>GeLU (Gaussian Error Linear Unit):</strong>
        * <strong>简介：</strong> GeLU曾是Transformer模型中的主流激活函数，被BERT、GPT-2等经典模型采用。它的数学形式是 $x \cdot \Phi(x)$，其中 $\Phi(x)$ 是高斯分布的累积分布函数。
        * <strong>为什么选用它？</strong>
            * <strong>平滑性：</strong> GeLU是ReLU的一个平滑近似。相比于ReLU在0点的突变，GeLU的平滑特性使其在优化过程中梯度更稳定，更有利于模型收敛。
            * <strong>随机正则化思想：</strong> GeLU可以看作是综合了Dropout和ReLU的思想。它根据输入的数值大小，对其进行随机的“归零”或“保留”，但这个过程是确定性的。输入越小，其输出被“归零”的概率越高。

    2.  <strong>SwiGLU (Swish-Gated Linear Unit):</strong>
        * <strong>简介：</strong> SwiGLU是目前<strong>最先进、最主流</strong>的选择，被Llama、PaLM、Mixtral、Gemma等一系列现代LLM广泛采用。它属于<strong>门控线性单元（Gated Linear Unit, GLU）</strong> 家族的变体。
        * <strong>工作原理：</strong> 它将前馈网络（FFN）的第一个线性层的输出 $X$ 分成两部分， $A$ 和 $B$ 。然后通过公式 $Swish(A) \otimes B$ 计算输出，其中 $Swish(x) = x \cdot \sigma(x)$ ， $\sigma$ 是Sigmoid函数， $\otimes$ 是逐元素相乘。
        * <strong>为什么选用它？</strong>
            * <strong>门控机制（Gating Mechanism）：</strong> SwiGLU的核心优势在于其“门控”设计。 $B$ 部分可以被看作一个动态的“门”，它可以根据输入内容，控制 $Swish(A)$ 中的信息哪些可以通过、哪些需要被抑制。这种机制<strong>显著增强了模型的表达能力</strong>，使得FFN层可以更灵活地处理信息。
            * <strong>实证效果优越：</strong> Google在PaLM论文中的实验发现，使用SwiGLU替换标准的GeLU或ReLU，可以<strong>显著提升模型的性能</strong>（降低困惑度）。尽管SwiGLU会增加FFN层的参数量（因为需要两个矩阵而不是一个），但其带来的性能增益被证明是值得的。

---

#### <strong>1.13 混合专家模型（MoE）是如何在不显著增加推理成本的情况下，有效扩大模型参数规模的？请简述其工作原理。</strong>

* <strong>参考答案：</strong>
    混合专家模型（Mixture of Experts, MoE）是一种模型架构，它的核心思想是通过 <strong>“稀疏激活”（Sparse Activation）</strong> 的策略，来解决模型规模与计算成本之间的矛盾。它允许模型拥有巨大的总参数量，但在处理任何一个输入时，只动用其中一小部分参数，从而在不显著增加推理成本（FLOPs）的情况下，大幅提升模型容量。

    <strong>工作原理如下：</strong>

    1.  <strong>用“专家”替换FFN层：</strong>
        * 在标准的Transformer架构中，计算量最大的部分之一是前馈网络（Feed-Forward Network, FFN）层。
        * MoE架构将模型中的部分或全部FFN层替换为<strong>MoE层</strong>。一个MoE层由两部分组成：
            * <strong>N个“专家”（Experts）：</strong> 每个专家本身就是一个独立的、规模较小的FFN。
            * <strong>1个“门控网络”或“路由器”（Gating Network / Router）：</strong> 这是一个小型的神经网络，通常是一个简单的线性层。

    2.  <strong>动态路由决策：</strong>
        * 当一个词元（token）的向量表示来到MoE层时，它首先被送入<strong>路由器</strong>。
        * 路由器的作用是 <strong>“决策”</strong> ，判断这个token应该由哪些专家来处理最合适。它会输出一个包含N个分数的向量，代表该token与N个专家的“匹配度”。

    3.  <strong>Top-K稀疏激活：</strong>
        * 路由器输出的分数经过Softmax归一化后，系统并<strong>不会</strong>激活所有的专家。相反，它只选择分数<strong>最高的Top-K个专家</strong>（K通常很小，比如1或2）。
        * 这就是“稀疏激活”的关键：对于每一个token，只有极少数（K个）专家被激活并进行计算，其余的（N-K个）专家则完全不参与，不产生任何计算成本。

    4.  <strong>加权输出：</strong>
        * 被选中的K个专家分别对输入的token向量进行处理，得到K个输出向量。
        * 最终的输出是这K个输出向量的<strong>加权和</strong>，权重同样由路由器的输出分数决定。

    <strong>如何实现“参数大但成本低”？</strong>
    * 假设一个模型有8个专家（N=8），并且每次只激活2个（K=2），如Mixtral-8x7B模型。
    * <strong>总参数量：</strong> 模型的总参数量是所有共享部分（如注意力层）的参数量，加上<strong>所有8个专家</strong>的参数量之和。这使得模型的总参数规模可以非常大（例如达到47B）。
    * <strong>推理成本：</strong> 但在进行一次前向传播（推理）时，对于任意一个token，实际参与计算的只有共享部分和<strong>被激活的2个专家</strong>。因此，其计算量（FLOPs）约等于一个规模小得多的“稠密”模型（例如一个13B的模型）。
    * <strong>结论：</strong> MoE成功地将<strong>总参数量</strong>（代表模型的知识容量）和<strong>单次推理的计算量</strong>（代表模型的速度和成本）<strong>解耦</strong>，从而实现了“用小模型的成本，获得大模型的知识”。

---

#### <strong>1.14 在训练一个百或千亿参数级别的 LLM 时，你会面临哪些主要的工程和算法挑战？（例如：显存、通信、训练不稳定性等）</strong>

* <strong>参考答案：</strong>
    训练百亿或千亿参数级别的LLM是一个巨大的系统工程，涉及硬件、软件和算法的深度协同。其挑战主要体现在以下三个方面：

    <strong>1. 显存挑战 (Memory Wall):</strong>
    * <strong>问题：</strong> 一个千亿参数的模型，其模型参数、梯度、优化器状态（如Adam中的动量和方差）加起来需要数TB的存储空间，远远超出了任何单张GPU的显存（目前最先进的H100也只有80GB）。
    * <strong>解决方案（3D并行）：</strong>
        * <strong>数据并行 (Data Parallelism, DP):</strong> 最基础的并行方式。在每张卡上都保留一份完整的模型副本，但将数据切分成多个batch，每张卡处理一个batch。计算完成后通过All-Reduce操作同步梯度。这种方式<strong>不能解决单卡显存不足</strong>的问题。
        * <strong>流水线并行 (Pipeline Parallelism, PP):</strong> 将模型的层（layers）进行垂直切分，不同的GPU负责模型的一部分（例如，GPU-1负责1-16层，GPU-2负责17-32层）。这<strong>可以有效降低单卡显存</strong>，但会引入“流水线气泡”（pipeline bubbles），即部分GPU在等待上下游数据时会处于空闲状态。
        * <strong>张量并行 (Tensor Parallelism, TP):</strong> 将模型中的单个大算子（如大的权重矩阵）进行水平切分，放到不同的GPU上协同计算。例如，将一个大的矩阵乘法分解到多张卡上。这也能<strong>降低单卡显存</strong>，但会引入<strong>非常高</strong>的通信开销。
        * <strong>ZeRO (Zero Redundancy Optimizer):</strong> 由微软DeepSpeed提出的显存优化技术。它在数据并行的基础上，将<strong>优化器状态、梯度、甚至模型参数</strong>也进行切分，分布到所有GPU上。每个GPU只保留自己需要计算的那一部分，极大地降低了单卡的显存冗余，是目前大规模训练的标配。

    <strong>2. 通信挑战 (Communication Bottleneck):</strong>
    * <strong>问题：</strong> 上述所有并行策略都引入了大量的GPU间通信。例如，DP需要同步梯度，PP需要传递激活值，TP需要在每次前向和后向传播中交换计算结果。当GPU数量巨大时，通信所需的时间可能超过计算本身，成为整个训练的瓶颈。
    * <strong>解决方案：</strong>
        * <strong>硬件层面：</strong> 使用高速互联技术，如单机内的<strong>NVLink</strong>和跨节点的<strong>InfiniBand</strong>网络。
        * <strong>软件层面：</strong> 开发高效的通信算法（如Ring All-Reduce），并设计调度策略来将<strong>计算和通信操作重叠（overlap）</strong>，以隐藏通信延迟。

    <strong>3. 训练不稳定性挑战 (Training Instability):</strong>
    * <strong>问题：</strong> 训练如此巨大的模型在数值上非常脆弱。由于计算层数极深、数据量极大，训练过程中很容易出现<strong>梯度爆炸或消失</strong>，导致损失（Loss）突然飙升为NaN（Not a Number），使得数小时甚至数天的训练成果毁于一旦。
    * <strong>解决方案：</strong>
        * <strong>数值精度：</strong> 普遍采用 <strong>BF16 (BFloat16)</strong> 混合精度训练。BF16相比FP16有更大的动态范围，能有效避免梯度下溢，同时保持FP32的稳定性。同时，关键部分（如优化器的master weights）仍保留FP32以保证精度。
        * <strong>稳定的模型架构：</strong> 采用更稳定的架构设计，如<strong>Pre-LayerNorm</strong>（在自注意力和FFN之前进行层归一化），以及使用更平滑的激活函数如<strong>GeLU/SwiGLU</strong>。
        * <strong>梯度裁剪 (Gradient Clipping):</strong> 设定一个梯度的范数上限，如果计算出的梯度超过这个阈值，就将其缩放到阈值以内，这是防止梯度爆炸最直接有效的方法。
        * <strong>学习率调度与预热 (Learning Rate Scheduling & Warmup):</strong> 采用精心设计的学习率调度策略，如在训练初期使用一个较小的学习率并逐渐增大的“预热”阶段，有助于模型在训练早期稳定下来。

---

#### <strong>1.15 开源框架了解过哪些？Qwen，Deepseek的论文是否有研读过，说一下其中的创新点主要体现在哪？</strong>

* <strong>参考答案：</strong>

    <strong>开源框架：</strong>
    * <strong>基础框架：</strong> <strong>PyTorch</strong> 是目前大模型研究和开发的事实标准，提供了灵活的张量计算和自动微分能力。
    * <strong>模型与生态：</strong> <strong>Hugging Face Transformers</strong> 是最重要的模型库和生态系统，它极大地降低了使用和分享模型的门槛。
    * <strong>大规模训练：</strong> <strong>DeepSpeed</strong> (微软) 和 <strong>Megatron-LM</strong> (英伟达) 是进行大规模分布式训练的核心框架，它们实现了上述的3D并行、ZeRO等关键技术。
    * <strong>高效推理：</strong> <strong>vLLM</strong>, <strong>TensorRT-LLM</strong> 等框架专注于优化LLM的推理速度和吞吐量，通过PagedAttention等技术来解决KV Cache的显存瓶颈。

    <strong>Qwen系列（可以参考开源论文自行回答，Qwen2.5，Qwen3系列）</strong>

    <strong>Deepseek系列（可以参考开源论文自行回答，如GRPO）</strong>


---

#### <strong>1.16 最近读过哪些LLM比较前沿的论文，聊一下它的相关方法，针对什么问题，提出了什么方法，对比实验有哪些？</strong>

* <strong>参考答案：</strong>
    <strong>(这是一个开放性问题，回答时应选择1-2篇自己真正理解的、有影响力的近期论文。)</strong>