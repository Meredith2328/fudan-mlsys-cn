# MLSys 中文整理

这个站点整理了 [Machine Learning Systems (ML Sys)](https://memxlife.github.io/courses/mlsys.html) 网页教材的中文翻译，并使用 docsify 构建为可直接部署到 GitHub Pages 的静态站点。

## 当前内容

| Week | Topic | Materials | 中文翻译 |
| :--- | :--- | :--- | :--- |
| 1 | Introduction: The Intellectual Map of Machine Learning Systems | [Book Chapter 1](https://memxlife.github.io/books/mlsys/chapter1.html) | [第 1 章：机器学习系统：问题、约束与垂直整合](chapter1.md) |
| 2 | CPU Foundations and GPU Emergence | [Book Chapter 2](https://memxlife.github.io/books/mlsys/chapter2.html) | [第 2 章：CPU 基础、GPU 的兴起与吞吐计算的逻辑](chapter2.md) |
| 3 | GPU Architecture for Machine Learning Systems | [Book Chapter 3](https://memxlife.github.io/books/mlsys/chapter3.html) | [第 3 章：GPU 架构与机器学习系统](chapter3.md) |
| 4 | CUDA Programming Begins with Architecture | [Book Chapter 4](https://memxlife.github.io/books/mlsys/chapter4.html) | [第 4 章：CUDA 编程从架构开始](chapter4.md) |
| 5 | CUDA Programming as Hardware-Software Co-Optimization | [Book Chapter 5](https://memxlife.github.io/books/mlsys/chapter5.html) | [第 5 章：CUDA 编程作为软硬件协同优化](chapter5.md) |
| 6 | Inside Claude Code, Reconstructed and Interpreted | [Machine Learning Infrastructure](https://memxlife.github.io/books/mlsys/ma-infra.html) | [第 6 章：Claude Code 内部机制：重建与解读](chapter6.md) |
| 7 | Deep Learning Compilers — From Generality to Specialization, and Back to Learned Search | [Book Chapter 7](https://memxlife.github.io/books/mlsys/chapter7.html) | [第 7 章：深度学习编译器：从通用性到专用化，再回到学习式搜索](chapter7.md) |
| 8 | Transformer Foundations | — | [第 8 章：Transformer 基础](chapter8.md) |
| 9 | Distributed Training and Data Parallelism | — | [第 9 章：分布式训练与数据并行](chapter9.md) |
| ★ | DeepSeek Technical Roadmap: The Efficiency Moat | [Original Article](https://memxlife.github.io/books/mlsys/deepseek.html) | [DeepSeek 技术路线图：效率护城河](deepseek.md) |
| Project 1 | GPU Profiling & Hardware Probing | [Phase 1](https://memxlife.github.io/books/mlsys/project.html) | [课程项目1：GPU 性能分析与硬件探测](project1.md) |
| Project 2 | Agentic Optimization of a LoRA Operator | [Phase 2](https://memxlife.github.io/books/mlsys/project_phase2.html) | [课程项目2：LoRA 算子的 Agentic 优化](project2.md) |
| Project 3 | Automated LLM Inference Runtime | [Phase 3](https://memxlife.github.io/books/mlsys/project_phase3.html) | [课程项目3：自动化 LLM 推理运行时](project3.md) |

## 导航

- [AI 提供的一二章总结](summary-ch1-ch2.md)
- [AI 提供的三四章总结](summary-ch3-ch4.md)
- [AI 提供的五六章总结](summary-ch5-ch6.md)
- [第 1 章：机器学习系统：问题、约束与垂直整合](chapter1.md)
- [第 2 章：CPU 基础、GPU 的兴起与吞吐计算的逻辑](chapter2.md)
- [第 3 章：GPU 架构与机器学习系统](chapter3.md)
- [第 4 章：CUDA 编程从架构开始](chapter4.md)
- [第 5 章：CUDA 编程作为软硬件协同优化](chapter5.md)
- [第 6 章：Claude Code 内部机制：重建与解读](chapter6.md)
- [第 7 章：深度学习编译器：从通用性到专用化，再回到学习式搜索](chapter7.md)
- [第 8 章：Transformer 基础](chapter8.md)
- [第 9 章：分布式训练与数据并行](chapter9.md)
- [★ DeepSeek 技术路线图：效率护城河](deepseek.md)
- [课程项目1：GPU 性能分析与硬件探测](project1.md)
- [课程项目2：LoRA 算子的 Agentic 优化](project2.md)
- [课程项目3：自动化 LLM 推理运行时](project3.md)

## 说明

- 站点入口文件是 [index.html](index.html)
- 章节 Markdown 与导航文件都位于 `docs/` 目录
- 翻译生成脚本位于仓库根目录 `scripts/translate_mlsys.py`
- 原始网页版权归原作者所有，本仓库仅做学习整理与中文查阅
