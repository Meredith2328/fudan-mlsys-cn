> 原文：[第 8 章](https://memxlife.github.io/books/mlsys/chapter8.html)
> 说明：内容由原网页整理翻译为中文 Markdown，公式与章节结构尽量保持不变。

# 第 8 章 Transformer 基础

## 1. Transformer 生态：鸟瞰

当前 AI 领域由大语言模型（LLM）定义——这些大规模系统需要深厚的架构知识和工程专业能力，才能将计算分布到成百上千个加速器上。

Transformer 架构已经渗透到现代 AI 的每个角落。Hugging Face 的 `transformers` 库提供了统一接口，只需几行代码即可调用数千个预训练模型，覆盖文本任务（生成、分类、摘要、翻译、特征提取）、视觉任务（图像到文本、分类、目标检测）和音频任务（语音识别、文本到语音、音频分类）。

`pipeline()` 函数是最高层抽象，它将一个预训练模型与其所需的预处理（如分词器）和后处理步骤连接成一个即用管道。一个极简的情感分析示例：

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier("I've been waiting for a machine learning systems course my whole life.")
# 输出: [{'label': 'POSITIVE', 'score': 0.9598}]
```

这行简洁代码背后，隐藏着三个阶段：

1. **预处理：** 分词（tokenization）将原始文本转换为数值表示。
2. **模型推理：** Transformer 执行数十亿次浮点运算。
3. **后处理：** 原始输出被解码为人类可读的标签和分数。

默认情况下，如果本地没有缓存，系统会自动下载预训练模型。这反映了一个核心理念：充分利用大规模预训练，并针对特定任务进行最小化适配。

---

## 2. Transformer 模型简史

当我们追溯模型规模的成长轨迹时，分布式训练的重要性便凸显出来。2017 年，Vaswani 等人在论文 *"Attention is All You Need"* 中提出了 Transformer，证明了仅靠注意力机制就足以实现最先进的机器翻译。

**时间线：**

| 时期 | 关键进展 |
|---|---|
| 2017 | 原始 Transformer |
| 2018–2019 | GPT、BERT、GPT-2、T5 |
| 2020 | GPT-3（175B 参数）——真正的大语言模型出现 |
| 2021–2022 | InstructGPT、FLAN、ChatGPT——对齐与指令遵循 |
| 2023–2024 | GPT-4、LLaMA、LLaMA-3.1（405B）、GPT-4o |
| 2025 | OpenAI-o1、DeepSeek-V3、DeepSeek-R1 |

核心的启示是**规模**。模型已经从数百万参数增长到数千亿参数。GPT-3 拥有 1750 亿参数；LLaMA-3.1 达到 4050 亿参数。没有任何单张 GPU 能装下或训练这样的模型，这使得分布式训练成为刚需。

---

## 3. Transformer 的预处理

### 3.1 因果语言模型

因果语言模型（causal language model）按顺序预测 token，只能看到过去的 token，永远不能看向未来。给定 "My,"，模型预测 "name"；给定 "My name,"，模型预测 "is"；给定 "My name is,"，模型预测 "Sylvain"。训练在数万亿 token 上重复此过程，每个预测与真实标签进行对比以完成反向传播。这一家族包括 GPT-3、GPT-4、LLaMA 以及大多数现代生成式 AI 系统。

### 3.2 非因果语言模型

非因果模型通过双向上下文读取整个序列。它们使用**掩码语言建模（Masked Language Modeling）**：将某个词替换为 `[MASK]` token，模型根据周围上下文预测原始词语：

$$\text{"My [MASK] is Sylvain."} \rightarrow \text{预测: "name"}$$

BERT 是其经典代表。无论哪种范式，庞大的矩阵运算都要求分布式基础设施的支持。

### 3.3 从文本到 Token：分词

神经网络处理的是数字，而非文本。**分词（tokenization）** 使用子词单元的词汇表，将原始文本转换为整数 ID。常见词映射到单个 token；罕见词被拆分为多个子词片段：

- `"bought"` → 单个 token
- `"indivisible"` → `"indiv"` + `"isible"`
- `"."` → 自己的 token

一个包含 50,000 个 token 的词汇表意味着文本被转换为 $[0, 49{,}999]$ 范围内的整数序列。

### 3.4 独热编码

每个整数 ID $i$ 变成一个长度为 $V$（词汇表大小）的稀疏向量，其中仅有第 $i$ 个位置为 1：

$$\mathbf{v} = [\underbrace{0, 0, \ldots, 0}_{3686\text{ 个 }}, \underbrace{1}_{\text{第 }3687\text{ 位}}, 0, \ldots, 0] \in \mathbb{R}^{50{,}000}$$

这种表示极其低效——每个向量中 49,999 个位置都是零。

### 3.5 Token 嵌入

**嵌入层（embedding layer）** 解决了稀疏性问题。一个嵌入矩阵 $W_E \in \mathbb{R}^{V \times d}$ 通过索引查找，将独热向量转换为稠密的 $d$ 维向量：

$$\mathbf{e} = \mathbf{v} \cdot W_E \in \mathbb{R}^{d}$$

通过训练，嵌入空间学会了语义关系——"dog" 比 "car" 更靠近 "cat"。

### 3.6 位置编码

注意力机制是置换不变的——它们不知道 token 的顺序。**位置编码（positional encoding）**（可以是固定的或可学习的）被加到嵌入向量上，使模型能够区分 "Tom likes Jerry" 和 "Jerry likes Tom"。

---

## 4. 自注意力机制

自注意力让每个 token 都能关注到其他所有 token，捕获 RNN 难以处理的长程依赖。

### 4.1 查询、键和值

每个嵌入后的 token $\mathbf{x}_i$ 被投影为三种角色：

$$\mathbf{q}_i = \mathbf{x}_i W_Q, \quad \mathbf{k}_i = \mathbf{x}_i W_K, \quad \mathbf{v}_i = \mathbf{x}_i W_V$$

其中 $W_Q, W_K \in \mathbb{R}^{d \times d_k}$，$W_V \in \mathbb{R}^{d \times d_v}$。

- **查询（Query, $\mathbf{q}$）：** 该 token 正在寻找什么。
- **键（Key, $\mathbf{k}$）：** 该 token 包含什么信息。
- **值（Value, $\mathbf{v}$）：** 当查询与键匹配时，传递出去的含义。

按矩阵形式堆叠：

$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$

### 4.2 缩放点积注意力

注意力输出为：

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$$

步骤分解：

1. **$Q K^\top$** 产生一个 $N \times N$ 的相似度分数矩阵，其中元素 $(i, j)$ 衡量 token $i$ 对 token $j$ 的关注强度。
2. **除以 $\sqrt{d_k}$ 缩放**，防止点积值过大，稳定梯度。直观原因：假设 $\mathbf{q}$ 和 $\mathbf{k}$ 是均值为 0、方差为 1 的独立随机变量，则其点积 $\mathbf{q} \cdot \mathbf{k} = \sum_{i=1}^{d_k} q_i k_i$ 的方差为 $d_k$。除以 $\sqrt{d_k}$ 使方差回到 1，避免 softmax 落入梯度极小的饱和区。
3. **Softmax** 将分数转换为行向概率分布，每行求和为 1。
4. **乘以 $V$**，得到值向量的加权和。

$Q K^\top$ 的计算需要 $\mathcal{O}(N^2)$ 的内存。当上下文窗口达到 100,000–200,000 个 token 时，这个二次方成本成为严重瓶颈——现代框架会避免一次性计算完整的注意力矩阵。

### 4.3 多头注意力

单个注意力只能捕获一种关系类型。**多头注意力（multi-head attention）** 并行运行 $h$ 个独立的注意力计算，每个都有自己的投影矩阵，从而让模型能够在不同的表示子空间中同时关注不同位置的信息。以 $h=8$ 为例，不同头可能分别学会关注句法关系、语义关联、位置邻近性或其他模式。

如果隐藏维度为 $d$，每个头在降维空间 $d_k = d/h$ 上操作：

$$\text{head}_i = \text{Attention}(Q W_Q^{(i)},\ K W_K^{(i)},\ V W_V^{(i)})$$

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_0, \text{head}_1, \ldots, \text{head}_{h-1}) \cdot W_O$$

其中 $W_Q^{(i)}, W_K^{(i)} \in \mathbb{R}^{d \times d_k}$，$W_V^{(i)} \in \mathbb{R}^{d \times d_v}$，$W_O \in \mathbb{R}^{h d_v \times d}$。所有头的输出被拼接起来，再通过 $W_O$ 进行线性投影，融合来自所有表示子空间的上下文信息。

### 4.4 解码器中的因果掩码

自回归生成要求模型永远不能关注未来的 token。在 softmax 之前，将一个掩码矩阵 $M$ 加到注意力分数上：

$$\text{MaskedAttention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}} + M\right) V$$

下三角（过去和当前）位置的值为 $0$；上三角（未来）位置的值为 $-\infty$，因为 $e^{-\infty} = 0$：

$$M = \begin{pmatrix} 0 & -\infty & -\infty & -\infty & -\infty \\ 0 & 0 & -\infty & -\infty & -\infty \\ 0 & 0 & 0 & -\infty & -\infty \\ 0 & 0 & 0 & 0 & -\infty \\ 0 & 0 & 0 & 0 & 0 \end{pmatrix}$$

这使得 Transformer 从双向阅读器转变为单向生成器，架构无需任何改变——仅靠一个掩码。

---

## 5. 前馈网络

每个 token 在经过自注意力之后，都会通过一个按位置独立作用的前馈网络（FFN）：

$$\text{FFN}(\mathbf{x}) = \text{ReLU}(\mathbf{x} W_1 + \mathbf{b}_1) W_2 + \mathbf{b}_2$$

隐藏维度扩展为输入的 4 倍：$W_1 \in \mathbb{R}^{d \times 4d}$ 将输入 $d$ 扩展到 $4d$；经过 ReLU 后，$W_2 \in \mathbb{R}^{4d \times d}$ 再映射回 $d$。

在标准稠密 Transformer 中，大约三分之二的参数位于 FFN 块中。对于一个 700 亿参数的模型，数百亿个权重位于 $W_1$ 和 $W_2$ 中。

---

## 6. 层堆叠与架构家族

每个 Transformer 块包含多头自注意力后接 FFN，每部分都包裹着残差连接和层归一化。完整模型堆叠许多这样的块：

$$\mathbf{x}^{(l+1)} = \text{TransformerBlock}(\mathbf{x}^{(l)})$$

现代 LLM 堆叠几十到上百个块（例如，每个块约 10 亿参数 × 80 层 ≈ 800 亿参数）。存在三种架构家族：

### 6.1 仅编码器模型

无掩码的自注意力块，具有双向上下文，构建丰富的输入表示。适用于理解类任务：情感分析、分类、语义搜索。**BERT** 是其原型。

### 6.2 仅解码器模型

仅有掩码自注意力——每个 token 只能关注前面的 token，支持自回归生成。顶部的线性投影 + softmax 给出词汇表概率，用于下一 token 采样。**GPT-3**、**GPT-4** 和 **LLaMA** 是其原型。这是目前占主导地位的大规模生成式架构。

### 6.3 编码器-解码器模型

2017 年原始论文为序列到序列任务设计的形式。编码器从完整输入生成上下文表示 $\mathbf{e}_1, \ldots, \mathbf{e}_n$。解码器使用对这些表示的交叉注意力，加上自身的掩码自注意力，来生成输出。**T5** 是其原型。

---

## 7. 生成输出

最后一个块的最终隐藏向量 $\mathbf{x} \in \mathbb{R}^d$ 必须被转换为一个词。

**步骤 1：线性投影到 Logits**

将 $\mathbf{x}$ 与 $W \in \mathbb{R}^{d \times V}$（通常与 $W_E$ 共享权重）相乘：

$$\text{Logits} = \mathbf{x} \cdot W^\top + \mathbf{b}$$

结果：$V$ 个原始分数（logits）——高正值表示可能性大的词；大负值表示不太可能的词。

**步骤 2：Softmax 转换为概率分布**

原始 logits 通过 Softmax 转换：

$$P_{\text{word}_i} = \frac{e^{u_i}}{\sum_{j=1}^{V} e^{u_j}}$$

所有概率非负，且总和为 1。

**步骤 3：解码 / 采样**

两种主要策略：

- **贪心解码（greedy decoding）：** 选择概率最高的词。确定性但容易产生重复文本。
- **采样（sampling）：** 将概率视为加权骰子，从 top 候选中进行采样。引入更多变化，产生更自然的输出。

---

## 8. 分布式训练的必然性

训练单个 Transformer 层需要存储：

- **模型权重**
- **梯度**（与权重一样大）
- **优化器状态**（例如 Adam 的一阶/二阶矩估计——额外的数倍）
- **为梯度计算保留的中间激活值**

对于一个 800 亿参数的模型，内存需求达到 TB 级别——远超常见 GPU 约 80 GB 的显存容量。后续章节讨论的分布式技术——数据并行、集合通信库和内存优化——正是应对这一约束的工程解决方案。
