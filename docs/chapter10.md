> 原文：对应课程第 10–11 周（Model Parallelism / Expert Parallelism）。原站这两周只有课程幻灯片，无网页教材，本章依据公开论文与官方文档编写。
> 说明：本章不在原网站页教材范围内，属于体系补全章节，用于衔接第 9 章（数据并行）与第 12 章（训练基础设施）。

# 第 10 章 模型并行与专家并行

::: tip 本章在体系中的位置
第 9 章讲了数据并行（data parallelism）：把数据切给多张卡，每张卡持有完整模型副本，梯度同步靠 all-reduce。但数据并行有个天花板——**模型大到单卡装不下**。本章回答：怎么把模型本身切开？切法分两类——**沿层切**（流水线并行）、**沿张量切**（张量并行）；再加一个异类——**把 MoE 的专家拆开**（专家并行）。这些切法会在第 12 章拼成真实的训练集群。
:::

## 10.1 为什么数据并行会到头

先回顾第 9 章的结论：训练一个百亿参数模型，仅参数+梯度+优化器状态就可能吃掉几百 GB——远超单卡（哪怕 H100 的 80 GB 也要好几张卡）。数据并行用「每卡一份完整副本 + 梯度 all-reduce」解决算力问题，但它有个致命前提：**模型必须能装进一张卡**。

一旦模型大到单卡放不下，数据并行就失效了。这时候的思路是反过来的——**把模型切开**，让每张卡只持有模型的一部分。这就是模型并行（model parallelism）的出发点。

::: tip 一个直觉
数据并行像「印好几份同一本书，每人各读一段，读完互相核对笔记」；模型并行像「把一本书拆成好几册，一人拿一册，大家接力朗读」。前者每人都要整本书，后者每人只有一册——能读更厚的书，但每读一页都要等人传话。
:::

## 10.2 张量并行：把一层切开

**张量并行（tensor parallelism，TP）** 把单个算子的权重矩阵切开，分布到多张卡上，让它们协作完成一次矩阵乘法。

最著名的实现是 Megatron-LM（NVIDIA，2019）提出的方案。以线性层 $Y = XW$ 为例，其中 $X \in \mathbb{R}^{b \times d}$、$W \in \mathbb{R}^{d \times h}$：

- **按列切权重**：把 $W$ 沿输出维切成 $t$ 块 $W_1, \dots, W_t$（$t$ 为 TP 并行度），每张卡只算 $Y_i = XW_i$。$Y_i$ 拼接起来就是完整输出——但注意，每张卡**都需要完整的 $X$**。
- **按行切权重**：把 $W$ 沿输入维切，每张卡拿 $W_i \in \mathbb{R}^{d/t \times h}$，算 $Y_i = X_i W_i$，然后**跨卡做一次 all-reduce** 把部分和相加得到 $Y = \sum_i X_i W_i$。

按行切需要 all-reduce，通信量大；按列切只需要一次拼接。Megatron 的精妙之处在于 transformer 的层结构天然规避了不必要的通信：

- **QKV 投影**用列切（每卡各自算出一部分 Q/K/V，不需要通信）；
- **注意力输出投影**用行切——它正好消费所有卡算出的注意力结果，一次 all-reduce 把碎片合成完整输出，同时把数据「换手」到下一层需要的形态；
- **MLP 的第一个线性层**列切、**第二个**行切，同样只在两处各做一次 all-reduce。

于是每个 transformer 层只需要**两次 all-reduce**，而不是每个算子都通信。这是张量并行的关键设计：**把通信压缩到算子边界**，而不是算子内部。

::: warning 张量并行的代价
张量并行通信量大，且要求参与卡在**同一节点内**（因为依赖 NVLink/NVSwitch 的高带宽低延迟）。跨节点张量并行几乎不可行——这决定了后面会讲到的「TP 组内、PP 组间」的分层结构。
:::

张量并行把单个矩阵乘法加速了 $t$ 倍（受通信限制），但它解决不了**层数**带来的问题——切 1000 层怎么切？

## 10.3 流水线并行：沿层切

**流水线并行（pipeline parallelism，PP）** 把模型的不同层分到不同设备：第 1 层在设备 0，第 2 层在设备 1，以此类推。数据像流水线一样依次流过各设备。

朴素的做法是「一个 batch 走完全部设备才算完」——这造成**空闲**：设备 0 算第 1 层时，设备 1 在等，利用率只有 $1/n$。这就是为什么朴素流水线并行性能很差。

现代流水线并行的核心是**微批次（micro-batch）**：把大 batch 拆成多个 micro-batch，让它们流水式地穿过各阶段。GPipe（Google，2019）提出**同步流水**：每个 micro-batch 依次前向，攒够再统一反向；PipeDream（2020）提出**异步/交错调度**：让前向与反向交叠，进一步压缩空闲气泡（bubble）。

气泡（bubble）是流水线并行的天生浪费：阶段数越多、micro-batch 越少，气泡越大。经验法则——**单卡算力再快，PP 也只适合「不得不切」的场合**（比如层太多、单卡装不下），它不会带来算力加速，反而引入气泡与跨阶段通信。

::: tip 为什么还要用 PP
既然 PP 有气泡、又慢，为什么还要用？因为它是**唯一不依赖节点内高带宽**的切法——层与层之间只传激活，通信量小，可以跨节点。实际训练里常见组合是：节点内用 TP（快），节点间用 PP（省带宽），再叠加数据并行 DP。这就是所谓的 3D 并行。
:::

## 10.4 序列并行与上下文切分

训练超长序列（比如 128K token 的上下文）时，**激活张量本身**会大得离谱：序列长度直接乘进激活的形状。第 9 章提到激活重计算（activation recomputation）来省内存，但长序列还有另一种解法——**序列并行（sequence parallelism）**。

- **上下文并行（context parallelism）** 把序列沿长度维切开，分到多张卡，每张卡只持有序列的一部分；注意力里的跨卡部分通过 ring all-reduce 之类的通信完成。
- **Ring Attention（Liu et al., 2023）** 是代表性方案：让序列分块在卡间循环传递，每张卡依次处理不同块，用分块 softmax（flash attention 的技巧）得到精确结果——把 O(序列长度) 的显存需求摊到多张卡上。

序列并行让「单卡显存放不下一个 token 的完整激活」成为历史——代价是通信随序列长度增长。

## 10.5 专家并行：把 MoE 的专家拆开

**混合专家（Mixture of Experts，MoE）** 把 transformer 的 MLP 层替换成一组专家（expert），每个 token 由路由器（router）挑几个专家处理。MoE 的参数量可以巨大，但每个 token 只激活一小部分专家——**用稀疏激活换参数量**。

MoE 给分布式训练带来新机会：既然每个 token 只用部分专家，那可以把专家分布到不同卡上，token 按路由结果被送往持有对应专家的卡——这就是**专家并行（expert parallelism）**。

专家并行靠 **All-to-All 通信**：路由器把 token 分发给各专家所在卡，各卡算完再把结果送回。All-to-All 是每张卡都向每张其他卡发送数据——通信量与并行度**线性**增长，是专家并行最贵的开销。

::: warning MoE 的工程难点
1. **负载不均衡**：热门专家可能被塞爆。解决思路：负载均衡损失（auxiliary loss）惩罚不均衡路由、专家容量上限、令牌丢弃策略。
2. **通信昂贵**：All-to-All 让跨机架通信量巨大——DeepSeek 等系统因此刻意把专家并行控制在节点内，或用共享专家吸收通用 token 减少跨节点流量。
3. **路由决策影响收敛**：路由器的选择是离散的、不可微的，训练技巧多。
:::

代表系统：

- **GShard（Lepikhin et al., 2020）**：最早的大规模 MoE 训练框架，提出 top-2 路由、辅助负载均衡损失。
- **Switch Transformer（Fedus et al., 2021）**：简化到 top-1 路由（每个 token 只去一个专家），验证了稀疏激活的规模化收益。
- **DeepSeekMoE（2024）**：细粒度专家 + 共享专家——共享专家处理通用模式，细粒度专家负责专业化，大幅降低路由与通信开销。

## 10.6 并行策略的组合：3D/4D 并行

现实中极少只用一种并行。主流是**组合拳**：

- **数据并行（DP）**：复制模型副本，切数据；
- **张量并行（TP）**：节点内切层内算子（靠 NVLink）；
- **流水线并行（PP）**：节点间切层（靠跨节点网络）；
- **专家并行（EP）**：MoE 场景下切专家。

它们套在一起就是「DP + PP + TP」（3D 并行），再加 EP 就是 4D。Megatron 生态的 `tensor_model_parallel_size`、`pipeline_model_parallel_size` 等参数就是让你组合这些维度。

组合的原则：

1. **TP 最贵，放节点内**；PP 次之，可跨节点；DP 最便宜，可以跨最远。
2. 通信量排序：TP（每次前向/反向多次 all-reduce，量级 = 激活×t）> EP（All-to-All，量级 = token×专家数）> PP（只传层间激活）> DP（只在反向结束同步梯度，且有梯度压缩/异步等手段）。
3. 目标是让**最贵通信最短距离**：TP 组内用 NVLink，PP 组间用 InfiniBand，DP 世界甚至可以走以太网。

::: tip 一个具体例子（理解用）
一个 8 节点集群、每节点 8 卡，共 64 卡。训练 175B 模型：TP=8（整节点内切成 8 份，NVLink 全连接）、PP=8（8 个节点各放一段层）、DP=1 或适度放大——这就是经典 GPT-3 规模的部署方式。再加数据并行就变成 4D。
:::

## 10.7 现代大规模训练的切分现实

近年大模型训练的切分越来越讲究。几个趋势：

- **超大规模 MoE 用「数据并行为主 + 专家并行」**：DeepSeek-V3 把 MoE 专家分布到多节点，同时用 DSA（Dual-path Sparse Attention）与节点内共享专家压通信——它在 2048 块 H800 上训练，靠的就是精细的并行组合。
- **序列并行成为标配**：超长上下文让 context parallelism 从可选项变成必选项。
- **自动并行**：工程上，把「用哪种并行、并行度多少」交给工具搜索，是编译器研究的热点（如自动 TP/PP 划分）。

## 延伸阅读

- Shoeybi et al., [*Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism*](https://arxiv.org/abs/1909.08053), arXiv 2019（张量并行的奠基）
- Narayanan et al., [*Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM*](https://arxiv.org/abs/2104.04473), SC 2021（3D 并行 + 序列并行）
- Huang et al., [*GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism*](https://arxiv.org/abs/1811.06965), NeurIPS 2019
- Narayanan et al., [*PipeDream: Fast and Efficient Pipeline Parallel DNN Training*](https://arxiv.org/abs/1806.03377), arXiv 2018
- Lepikhin et al., [*GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding*](https://arxiv.org/abs/2006.16668), ICLR 2021
- Fedus et al., [*Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*](https://arxiv.org/abs/2101.03961), JMLR 2022
- Liu et al., [*Ring Attention with Blockwise Transformers for Near-Infinite Context*](https://arxiv.org/abs/2310.01889), ICLR 2024
- DeepSeek-AI, [*DeepSeek-V3 Technical Report*](https://arxiv.org/abs/2412.19437), arXiv 2024（DSA、节点内共享专家）
- NVIDIA, [*Megatron-DeepSpeed*](https://github.com/microsoft/Megatron-DeepSpeed) 与 [*NVIDIA Collective Communication Library (NCCL)*](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/) 文档
