# MLSys 中文整理

在线阅读： [MLSys 中文整理](https://meredith2328.github.io/fudan-mlsys-cn/)

本仓库整理了复旦大学 [Machine Learning Systems (ML Sys)](https://memxlife.github.io/courses/mlsys.html) （2026春，尚笠老师/徐跃东老师）课程网页教材的中文翻译与整理。

除了逐章翻译原文，每章还补充了它在 **MLSys / AI Infra 体系**中的位置：与经典论文、官方文档的对照，以及概念之间的衔接关系。原站没有网页教材的周次（模型并行/专家并行/LLM 推理/训练基础设施），也按公开资料补写了对应章节。

## 当前内容

| Week | Topic | Materials | 中文翻译 |
| :--- | :--- | :--- | :--- |
| 1 | Introduction: The Intellectual Map of Machine Learning Systems | [Book Chapter 1](https://memxlife.github.io/books/mlsys/chapter1.html) | [第 1 章：机器学习系统：问题、约束与垂直整合](docs/chapter1.md) |
| 2 | CPU Foundations and GPU Emergence | [Book Chapter 2](https://memxlife.github.io/books/mlsys/chapter2.html) | [第 2 章：CPU 基础、GPU 的兴起与吞吐计算的逻辑](docs/chapter2.md) |
| 3 | GPU Architecture for Machine Learning Systems | [Book Chapter 3](https://memxlife.github.io/books/mlsys/chapter3.html) | [第 3 章：GPU 架构与机器学习系统](docs/chapter3.md) |
| 4 | CUDA Programming Begins with Architecture | [Book Chapter 4](https://memxlife.github.io/books/mlsys/chapter4.html) | [第 4 章：CUDA 编程从架构开始](docs/chapter4.md) |
| 5 | CUDA Programming as Hardware-Software Co-Optimization | [Book Chapter 5](https://memxlife.github.io/books/mlsys/chapter5.html) | [第 5 章：CUDA 编程作为软硬件协同优化](docs/chapter5.md) |
| 6 | Inside Claude Code, Reconstructed and Interpreted | [Machine Learning Infrastructure](https://memxlife.github.io/books/mlsys/ma-infra.html) | [第 6 章：Claude Code 内部机制：重建与解读](docs/chapter6.md) |
| 7 | Deep Learning Compilers — From Generality to Specialization, and Back to Learned Search | [Book Chapter 7](https://memxlife.github.io/books/mlsys/chapter7.html) | [第 7 章：深度学习编译器：从通用性到专用化，再回到学习式搜索](docs/chapter7.md) |
| 8 | Transformer Foundations | [Book Chapter 8](https://memxlife.github.io/books/mlsys/chapter8.html) | [第 8 章：Transformer 基础](docs/chapter8.md) |
| 9 | Distributed Training and Data Parallelism | [Book Chapter 9](https://memxlife.github.io/books/mlsys/chapter9.html) | [第 9 章：分布式训练与数据并行](docs/chapter9.md) |
| 10–11 | Model Parallelism / Expert Parallelism | 课程 slides | [第 10 章：模型并行与专家并行](docs/chapter10.md) |
| 12–13 | LLM Inference | 课程 slides | [第 11 章：LLM 推理系统](docs/chapter11.md) |
| 14–16 | LLM Training Real World Study / Model Optimization | 课程 slides | [第 12 章：分布式训练基础设施](docs/chapter12.md) |
| ★ | DeepSeek Technical Roadmap: The Efficiency Moat | [Original Article](https://memxlife.github.io/books/mlsys/deepseek.html) | [DeepSeek 技术路线图：效率护城河](docs/deepseek.md) |
| Project 1 | GPU Profiling & Hardware Probing | [Phase 1](https://memxlife.github.io/books/mlsys/project.html) | [课程项目1：GPU 性能分析与硬件探测](docs/project1.md) |
| Project 2 | Agentic Optimization of a LoRA Operator | [Phase 2](https://memxlife.github.io/books/mlsys/project_phase2.html) | [课程项目2：LoRA 算子的 Agentic 优化](docs/project2.md) |
| Project 3 | Automated LLM Inference Runtime | [Phase 3](https://memxlife.github.io/books/mlsys/project_phase3.html) | [课程项目3：自动化 LLM 推理运行时](docs/project3.md) |

## 技术栈

- 静态站点：**[VitePress](https://vitepress.dev/zh/)** + `markdown-it-mathjax3`（公式渲染）
- 本地预览：`npm run docs:dev`
- 构建：`npm run docs:build`（输出到 `docs/.vitepress/dist`）
- 部署：`.github/workflows/deploy.yml` 自动构建并发布到 GitHub Pages

## 说明

- 原始网页版权归原作者所有，本仓库仅做学习整理与中文查阅
- 章节内容保留原有公式结构，适合直接在线阅读
- 第 10–12 章为体系补全章节（原站只有 slides），内容依据公开论文与官方文档编写，非翻译
- 数学公式统一使用 `$...$`（行内）与 `$$...$$`（行间）
