# MLSys 中文整理

> 在线阅读：<https://meredith2328.github.io/fudan-mlsys-cn/>

复旦大学 [Machine Learning Systems (MLSys)](https://memxlife.github.io/courses/mlsys.html)
（2026 春，尚笠老师 / 徐跃东老师）课程网页教材的中文整理。

这里不只是把英文教材翻译成中文——逐章翻译之外，还给每一章补上了它在
MLSys / AI Infra 体系中的位置：跟经典教材、论文、官方文档的对应关系，
以及「为什么这一章要在那一章后面讲」。

## 内容地图

| 章 | 主题 | 原站材料 |
| :--- | :--- | :--- |
| [第 1 章](chapter1.md) | 机器学习系统：问题、约束与垂直整合 | [chapter1](https://memxlife.github.io/books/mlsys/chapter1.html) |
| [第 2 章](chapter2.md) | CPU 基础、GPU 的兴起与吞吐计算的逻辑 | [chapter2](https://memxlife.github.io/books/mlsys/chapter2.html) |
| [第 3 章](chapter3.md) | GPU 架构与机器学习系统 | [chapter3](https://memxlife.github.io/books/mlsys/chapter3.html) |
| [第 4 章](chapter4.md) | CUDA 编程从架构开始 | [chapter4](https://memxlife.github.io/books/mlsys/chapter4.html) |
| [第 5 章](chapter5.md) | CUDA 编程作为软硬件协同优化 | [chapter5](https://memxlife.github.io/books/mlsys/chapter5.html) |
| [第 6 章](chapter6.md) | Claude Code 内部机制：重建与解读 | [ma-infra](https://memxlife.github.io/books/mlsys/ma-infra.html) |
| [第 7 章](chapter7.md) | 深度学习编译器：从通用性到专用化，再回到学习式搜索 | [chapter7](https://memxlife.github.io/books/mlsys/chapter7.html) |
| [第 8 章](chapter8.md) | Transformer 基础 | [chapter8](https://memxlife.github.io/books/mlsys/chapter8.html) |
| [第 9 章](chapter9.md) | 分布式训练与数据并行 | [chapter9](https://memxlife.github.io/books/mlsys/chapter9.html) |
| [第 10 章](chapter10.md) | 模型并行与专家并行 | 课程 slides（周 10–11） |
| [第 11 章](chapter11.md) | LLM 推理系统 | 课程 slides（周 12–13） |
| [第 12 章](chapter12.md) | 分布式训练基础设施 | 课程 slides（周 14–16） |
| [DeepSeek 专题](deepseek.md) | DeepSeek 技术路线图：效率护城河 | [deepseek](https://memxlife.github.io/books/mlsys/deepseek.html) |
| [项目 1](project1.md) | GPU 性能分析与硬件探测 | [Phase 1](https://memxlife.github.io/books/mlsys/project.html) |
| [项目 2](project2.md) | LoRA 算子的 Agentic 优化 | [Phase 2](https://memxlife.github.io/books/mlsys/project_phase2.html) |
| [项目 3](project3.md) | 自动化 LLM 推理运行时 | [Phase 3](https://memxlife.github.io/books/mlsys/project_phase3.html) |
| [附录](tutorial.md) | Profiling 教程 | — |

## 说明

- 原始网页版权归原作者所有，本仓库仅做学习整理与中文查阅；
- 章节内容保留原有公式结构；第 8 章之后（含第 8 章）的大模型系统部分，
  在翻译之外补充了大量与经典论文/官方文档的对照，方便接轨业界体系。
