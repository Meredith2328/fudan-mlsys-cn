> 原文：对应课程第 12–13 周（LLM Inference）。原站这两周只有课程幻灯片，无网页教材，本章依据公开论文与官方文档编写。
> 说明：本章不在原网站页教材范围内，属于体系补全章节，用于衔接第 8–10 章（Transformer 与训练）与部署现实。

# 第 11 章 LLM 推理系统

::: tip 本章在体系中的位置
前几章都在讲**训练**：Transformer 怎么算、怎么并行、怎么存。但一个模型训练完，真正赚钱的是**推理**——每秒服务多少用户、每次请求多快。训练和推理的目标完全不同：训练拼吞吐与收敛，推理拼**延迟、成本、并发**。本章把「一个 token 是怎么被服务出来的」从头讲一遍。
:::

## 11.1 为什么推理和训练是两种问题

训练时你有一整批数据，可以自由排列、重算、并行——吞吐是王道。推理时用户是**逐个**来的：发一个 prompt，等一个回复。这里有两件事同时发生：

1. **延迟（latency）**：用户等不得。哪怕生成 100 个 token 也能接受，但第一个 token 要快。
2. **吞吐（throughput）**：同时服务几千个请求，GPU 才不闲着。

矛盾就在这：**用 GPU 做单个请求是暴殄天物**——一个生成请求算力用不满；但把几千个请求塞进一块 GPU 又会互相拖慢。推理系统的一切设计，都在这对矛盾里找平衡。

## 11.2 自回归：为什么推理不能并行

LLM 是**自回归（autoregressive）**模型：生成第 $t+1$ 个 token 时，要以前面所有 $t$ 个 token 为输入。这意味着**一个请求内部天然是串行的**——生成 100 个 token 至少 100 次前向，中间没法并行。

这带来三个推理特有的问题：

- **算力浪费**：每次前向都在重新计算前面所有 token 的表示；
- **KV Cache 爆发**：为了避免重算，推理系统把前面 token 的 key/value 缓存起来——缓存随序列长度增长，占显存；
- **内存墙**：自回归推理的瓶颈往往不是算力，而是**从缓存里读 KV 的带宽**。

::: tip 一个直觉
自回归生成像「一个人写长信」：每写一个字都要回头读一遍之前写的所有字（KV Cache 就是他的草稿纸）。训练是「很多人各自写一封信，互不干扰」，推理是「所有人挤在一个房间，还得共用纸」。
:::

## 11.3 KV Cache 与注意力

**KV Cache（key-value cache）** 是推理系统的核心数据结构：每层注意力都会缓存已生成 token 的 $K$ 和 $V$，下一个 token 只需计算新的 Q，并和缓存的 K 做注意力。省掉了重复计算，但代价是显存：模型是 7B 参数、上下文 32K 时，KV cache 可能比模型本身还大。

KV cache 的显存 = $2 \times n_{layers} \times d_{head} \times n_{heads} \times seq\_len \times 2$ bytes。对一个大模型 + 长上下文，这是几 GB 起步。

于是推理系统要想尽办法压缩 KV cache：

- **GQA / MQA**：多头注意力里，让多组 query 共享 key/value（Multi-Query Attention，Shazeer 2019；Grouped-Query Attention，Ainslie 2023）——把 KV 显存砍掉一个量级，这是如今主流 LLM 的标配；
- **量化**：KV 用 int8 甚至更低精度存储；
- **MLA（Multi-head Latent Attention）**：DeepSeek-V2 提出，把 KV 投影到低维隐空间再缓存，显存又减一个量级；
- **PagedAttention**：下一节的主角。

## 11.4 PagedAttention 与 vLLM

**PagedAttention（Kwon et al., 2023）** 是推理系统史上的关键一页。它的灵感来自操作系统的**虚拟内存分页**：

传统推理里，KV cache 为每个请求预留**连续**的显存块——但请求长度是动态的，预留多了浪费、少了溢出。PagedAttention 把 KV cache 切成**固定大小的块（block）**，像分页内存一样按需分配，用**块表（block table）** 记录逻辑块到物理块的映射。一个请求的 KV 可以散落在物理显存的不同位置，逻辑上仍连续。

好处：

- **零碎片浪费**：不再为「可能变长的请求」预留整块连续显存；
- **共享**：多个请求可以共享同一个 prompt 的 KV 前缀（比如多轮对话、few-shot），只存一份；
- **按需增长**：序列多长，块就多长，内存利用率接近 100%。

基于 PagedAttention 的 **vLLM** 成为开源推理事实标准。它把调度、分页、批处理（continuous batching）整合成一个系统。

## 11.5 Continuous Batching：让 GPU 不空转

传统批处理（static batching）是**同步**的：一批请求必须一起开始、一起结束。但生成式请求长度差异巨大——一个答「OK」的请求可能 5 个 token，一个写论文的请求要 1000 个 token。静态批处理下，GPU 会一直等最慢的那个请求，快的早就结束了。

**Continuous batching（持续批处理）** 打破这个同步：**任何一个请求完成就立刻从批里移除，新请求立刻补进来**。GPU 上永远在处理「当前需要算的 token」，而不是等一批统一结束。

这是推理吞吐的关键技术——vLLM、TensorRT-LLM、SGLang 等都实现了它。配合调度策略（先到先服务 vs 抢占式），连续批处理让 GPU 利用率从传统批处理的 40% 以下拉到 80%+。

## 11.6 解码策略与 Speculative Decoding

自回归生成默认逐个 token，但**投机解码（speculative decoding）** 改变这个假设（Leviathan et al., 2023；Chen et al., 2023）：

1. 用一个**小模型**（draft model）先「猜」出接下来 K 个 token；
2. 大模型（target model）一次性并行验证这 K 个 token 的对错；
3. 对的接受，错的拒绝并回退——**正确性不变，但每轮大模型前向处理了 K 个 token**。

因为大模型的验证是并行的（一个前向算 K 个位置），而草稿是便宜的，投机解码能把生成速度提升 2–3 倍。类似思路还有 n-gram 草稿、自草稿（self-speculative）等。

::: warning 什么时候投机解码不划算
投机解码的收益取决于「草稿猜中的概率」。如果生成内容高度不可预测（比如代码里的大段随机数字），猜中率低，收益就没了。它是「赌一把」——赌对了翻倍，赌错了白跑。
:::

## 11.7 推理系统的整体架构

现代推理栈从下到上：

1. **模型引擎**：内核级优化——FlashAttention（融合注意力，省显存提速度）、量化推理（GPTQ/AWQ）、内核融合（算子合并减少访存）；
2. **KV 管理**：PagedAttention 分页 + 前缀共享 + KV 量化；
3. **调度器**：continuous batching、优先级、抢占；
4. **服务层**：OpenAI 兼容 API、并发控制、多模型路由；
5. **分布式推理**：张量并行（跨卡切一个模型）、前缀缓存、模型分片。

代表系统：

- **vLLM**：PagedAttention + continuous batching（开源事实标准）；
- **TensorRT-LLM**（NVIDIA）：工程化内核 + 图优化，生产首选；
- **SGLang**：RadixAttention（前缀树缓存），长上下文/多轮场景极强；
- **TGI**（Hugging Face）：生态整合最顺。

## 11.8 推理的三个目标与一个权衡

推理系统最终要平衡三件事：

| 目标 | 手段 |
| :--- | :--- |
| 低延迟 | 投机解码、内核融合、小模型蒸馏 |
| 高吞吐 | continuous batching、分页 KV、量化 |
| 低成本 | 量化、KV 压缩、批处理摊薄 |

三个目标互相拉扯：把批调大吞吐上去了但单个请求延迟变高；投机解码提速但多花小模型算力。**推理系统工程，就是在这三角里按业务需求找切点。**

## 延伸阅读

- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023（vLLM 论文）
- Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, NeurIPS 2022（推理内核基础）
- Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need*, 2019（MQA）
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, EMNLP 2023
- DeepSeek-AI, *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*, 2024（MLA）
- Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*, ICML 2023
- Chen et al., *Accelerating Large Language Model Decoding with Speculative Sampling*, 2023
- NVIDIA, *TensorRT-LLM* 文档；SGLang、vLLM 官方文档
