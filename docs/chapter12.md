> 原文：对应课程第 14–16 周（LLM Training Real World Study / Model Optimization）。原站这几周只有课程幻灯片与 DeepSeek 专题，本章依据公开资料编写。
> 说明：本章不在原网站页教材范围内，属于体系补全章节，讲「真实训练集群长什么样」。

# 第 12 章 分布式训练基础设施

::: tip 本章在体系中的位置
第 9、10 章讲了并行策略（DP/TP/PP/EP），但那些策略最终要跑在**真实的集群**上。本章回答一个工程问题：一个几千卡的大模型训练集群，除了 GPU 还有什么？——网络、存储、调度、检查点、故障恢复。学完你会理解为什么「训练一个 GPT 级别模型」是个系统工程，而不只是「把并行度调大」。
:::

## 12.1 从单卡到集群：差的不只是数量

单卡训练，你的敌人是显存和算力。集群训练，敌人变成**带宽、可靠性、调度**。量变引起质变：

- **网络**：梯度同步、激活传输、checkpoint 落盘，全看网络；
- **可靠性**：几千块 GPU 里总有一块坏、一个进程挂、一次网络抖动——集群越大，故障越常态；
- **调度**：谁在什么时候用哪些卡，需要资源管理器；
- **存储**：训练数据、检查点、日志，PB 级吞吐。

这一章按「一台机器 → 一个节点 → 一个集群」的顺序展开。

## 12.2 单节点：NVLink 与 NVSwitch

第 10 章讲过张量并行需要节点内高带宽。节点内部是怎么连的？

- **NVLink**：NVIDIA 的点对点高速互连，带宽远超 PCIe（H100 时代单链路 900 GB/s，双向）。GPU 之间直接 NVLink 连接，绕开 CPU 和 PCIe 瓶颈；
- **NVSwitch**：节点内多 GPU 的**全连接交换**。8 卡 H100 节点用 2 个 NVSwitch 让任意两张卡之间都有满带宽路径——TP=8 时的梯度同步就在这上面跑；
- **NCCL**（NVIDIA Collective Communication Library）：把 NVLink/NVSwitch/InfiniBand 抽象成 all-reduce 等集合通信原语——分布式训练实际调用的就是 NCCL。

::: tip 一个直觉
NVSwitch 相当于把 8 张卡的「内部总线」做成一个交换机：任何两张卡通信都走满带宽。张量并行在这上面跑，就像多个核心共享同一颗芯片的缓存——通信损耗极小。
:::

## 12.3 跨节点：InfiniBand 与网络拓扑

跨节点通信用的不再是 NVLink，而是**网络**。主流是 **InfiniBand（IB）**——为 HPC 设计的高速低延迟网络，常见 200/400 Gbps；以太网（RoCE，RDMA over Converged Ethernet）是更便宜的可选方案。

跨节点网络的关键概念：

- **fat-tree 拓扑**：把交换机排成树，带宽逐层汇聚——保证「任意节点到任意节点」都有足够带宽；
- **RDMA**（Remote Direct Memory Access）：绕过 CPU，网卡直接读写远端内存——低延迟、高吞吐，IB 和 RoCE 都支持；
- **拥塞控制**：多对多通信（如 all-reduce 的跨节点部分）容易互相拖慢，需要网卡级的拥塞控制。

一个训练节点（如 8 卡）通常用**多个 IB 端口**上连，避免单链路成为瓶颈。

## 12.4 集合通信再深入：All-to-All 与拓扑感知

第 9 章讲了 all-reduce（梯度同步）、第 10 章讲了 all-to-all（专家并行）。这里补两点工程现实：

1. **通信量计算**：一次全量梯度 all-reduce，通信量 = 模型参数量 × 2（reduce + broadcast）。175B 模型、fp16 梯度 = 350 GB 数据要跨集群同步——这就是为什么训练不可能每步都全量同步，也催生了梯度压缩、延迟同步（局部同步 SGD）等技巧。
2. **拓扑感知（topology-aware）通信**：NCCL 会先探测「哪些卡在一个节点内、哪些跨节点」，把通信拆成「节点内 NVLink 先 reduce → 节点间 IB 再 reduce」的分层结构，避免跨节点传不必要的量。这就是为什么你经常看到「两次 all-reduce」的梯度同步优化。

## 12.5 调度与资源管理

集群不是随便插上 GPU 就能训练。调度器负责「谁在什么时候用哪些卡」：

- **Slurm**：HPC 领域事实标准，训练任务以 job 形式排队；
- **Kubernetes（k8s）**：云原生调度，配合 **KubeFlow**、**Volcano** 等做 GPU 调度；
- **任务亲和性**：TP 组必须共节点（依赖 NVLink），调度器要懂这个约束——这是「拓扑感知调度」；
- **弹性训练**：节点坏了自动腾挪、任务自动恢复。

调度不只是「分配资源」，还关系到**利用率的连续性**——大模型训练动不动几百小时，中途调度抖动一次就是几千块 GPU 时浪费。

## 12.6 检查点与故障恢复

训练 30 天，第 29 天节点炸了——没有 checkpoint 等于从头再来。检查点（checkpoint）是分布式训练的命根子：

- **频率与成本**：175B 模型一个全量 checkpoint 可能 1 TB+，落盘要分钟级。折中：每 N 步存一次全量 + 高频存「增量」；
- **异步 checkpoint**：把模型状态快照到内存再后台落盘，避免阻塞训练；
- **碎片化 checkpoint**：TP/PP 下每张卡存自己的分片，恢复时按拓扑重组；
- **故障检测**：心跳超时、错误日志、看门狗；
- **重启策略**：从最近 checkpoint 恢复，重放没同步的梯度。

::: warning 集群训练的铁律
**任何一步都可能死。** 设计训练框架时，「挂了能恢复」不是可选项——节点故障、OOM、网络抖动、宿主机重启都是常态。把「恢复路径」当一等公民设计，而不是事后补救。
:::

## 12.7 训练数据管道

集群训练的数据流动也是系统工程：

- **数据加载**：几千 GB 训练集从对象存储（S3/OSS）或并行文件系统流式读入；
- **数据预处理**：tokenize、shuffle、过滤，通常离线完成；
- **WebDataset / streaming**：把数据切成 tar 包、随机取块，避免整份载入内存；
- **数据并行读取**：每个 data-parallel rank 读不同的数据分片，靠 seed 控制重复；
- **持久化**：检查点、日志、指标存到分布式存储。

一个容易忽视的点：**数据 I/O 和训练计算重叠**。训练循环里数据加载慢一拍，GPU 就空转——所以主流框架都用预取（prefetch）和多 worker 加载。

## 12.8 集群训练的最佳实践清单

结合前几章，一个生产级训练任务的检查清单：

1. **先算账**：参数量 → 内存需求 → 卡数 → 网络需求（用 9.1 的公式）；
2. **选并行**：TP 节点内、PP 跨节点、DP 兜底，MoE 加 EP；
3. **网络确认**：NVLink 满带宽、IB 端口够、无拥塞；
4. **checkpoint 就绪**：频率、存储、恢复演练（真演练一次，别等出事）；
5. **监控**：GPU 利用率、显存、网络带宽、温度——像看仪表盘一样盯训练；
6. **混沌测试**：故意杀一个进程、断一次网，验证恢复逻辑。

## 延伸阅读

- Narayanan et al., [*Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM*](https://arxiv.org/abs/2104.04473), SC 2021（3D 并行 + 真实集群）
- NVIDIA, [*NCCL Documentation*](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)（集合通信原语、拓扑感知）
- NVIDIA, [*NVLink / NVSwitch*](https://www.nvidia.com/en-us/data-center/nvlink/) 白皮书
- Smith et al., [*Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B*](https://arxiv.org/abs/2201.11990), arXiv 2022（530B 训练实践）
- Jacobs et al., [*Deep Learning in Production*](https://www.kubeflow.org/docs/)（KubeFlow 生态）
- [Slurm 官方文档](https://slurm.schedmd.com/) / [Kubernetes 官方文档](https://kubernetes.io/docs/)
