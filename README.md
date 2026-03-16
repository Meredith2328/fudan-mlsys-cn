# MLSys 中文整理

本仓库整理了 Machine Learning Systems 网页教材的中文翻译，并按 GitHub Pages 直接部署的方式组织成 `docs/` 目录发布结构。

## 仓库结构

- [docs/index.html](docs/index.html)：docsify 站点入口
- [docs/README.md](docs/README.md)：站点首页
- [docs/chapter1.md](docs/chapter1.md)：第 1 章中文整理
- [docs/chapter2.md](docs/chapter2.md)：第 2 章中文整理
- [docs/conclusion.md](docs/conclusion.md)：前两章总结
- [scripts/translate_mlsys.py](scripts/translate_mlsys.py)：翻译生成脚本

## GitHub Pages 部署

1. 将仓库推送到 GitHub。
2. 在仓库 `Settings -> Pages` 中选择从分支部署。
3. 选择 `main` 分支，并把发布目录设为 `/docs`。
4. 保存后，GitHub Pages 会直接使用 `docs/` 下的静态文件发布站点。

## 当前内容

| Week | Topic | Materials | 中文翻译 |
| :--- | :--- | :--- | :--- |
| 1 | Introduction: The Intellectual Map of Machine Learning Systems | [Book Chapter 1](https://memxlife.github.io/books/mlsys/chapter1.html) | [第 1 章：机器学习系统：问题、约束与垂直整合](docs/chapter1.md) |
| 2 | CPU Foundations and GPU Emergence | [Book Chapter 2](https://memxlife.github.io/books/mlsys/chapter2.html) | [第 2 章：CPU 基础、GPU 的兴起与吞吐计算的逻辑](docs/chapter2.md) |

## 说明

- 原始网页版权归原作者所有，本仓库仅做学习整理与中文查阅
- 章节内容保留原有公式结构，适合直接在线阅读
- 后续若重跑翻译脚本，默认输出也会写入 `docs/`
