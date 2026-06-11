> 原文：[Phase 3: Automated LLM Inference Runtime](https://memxlife.github.io/books/mlsys/project_phase3.html)
> 说明：内容由原网页整理翻译为中文 Markdown，并按课程项目文档的阅读习惯做了适度重组与润色。

# MLSYS 课程项目 · 第三阶段：自动化 LLM 推理运行时

**开始日期：2026 年 5 月 26 日**
**首次提交截止：开始后 2 周内**
**最大提交次数：开始后 3 周内 2 次**

本阶段你要构建一个能自动生成 LLM 推理运行时的 agent。生成的运行时必须能从给定的配置和权重中加载一个 decoder-only 模型、维护请求状态，并高效地执行 prefill 和 decode。运行时将以黑盒形式被评测：我们会将其输出 logits 与参考实现对比以验证正确性，然后用服务式请求轨迹来驱动它，测量吞吐量和内存行为。

**推理正确性是硬性要求。**未通过正确性检查的提交不会获得吞吐量分数。

---

## 1. 任务

你的任务是实现一个 agent，为一个小型 LLaMA 式 decoder-only 模型生成推理运行时。生成的运行时必须支持：

- 从给定的权重目录加载模型权重
- 根据 `model_config.json` 构建运行时行为
- 对 prompt token 执行 prefill
- 对每个活跃请求生成一个新 token 的 decode
- 在多次调用间维护请求状态
- 移除已完成的请求
- 返回与官方参考实现匹配的 logits

你应该设计你的运行时，使其能够处理不同的 batch size、prompt 长度、decode 长度和请求顺序。官方评测轨迹不会提前公开。

---

## 2. 提交内容

你的提交必须包含：

- `run.sh`
- 你的 agent 实现及其所需的任何文件

`run.sh` 执行完毕后，你的 agent 必须生成：

- `workspace/engine.py`
- `workspace/results.log`

不要把 `workspace/engine.py` 当作手工提交的静态方案。它是你的 agent 生成出来的产物。日志文件不参与评分，仅用于你自己在提交后排查失败原因，比如 agent 错误、代码生成错误、编译错误或本地自测失败。

### 提交规约

评测系统会进入提交根目录并执行：

```bash
bash run.sh
```

`run.sh` 执行完毕后，评测系统会从同一目录导入：

```
workspace/engine.py
```

并运行官方的正确性和吞吐量 harness。

你的 `run.sh` 应调用你的 agent。如果生成的运行时需要自定义扩展、生成文件或本地自测，请在此过程中准备好。评测器不会使用你日志文件中的自报结果；它会直接调用生成的运行时。

---

## 3. 提供的输入

模型配置文件位于：

```
/target/model_config.json
```

在公开骨架中对应路径为：

```
target/model_config.json
```

该文件描述模型结构，包括 hidden size、层数、注意力头数、key-value 头数、词表大小等相关参数。你的运行时不应硬编码这些值，而应根据传入 `create_engine(...)` 的 `model_config` 参数动态构建引擎。

模型权重目录位于：

```
/target/weights
```

在公开骨架中权重文件为：

```
target/weights/model.pt
```

公开骨架使用单个 PyTorch state dict。隐藏评测将通过同样的 `weight_dir` 参数提供权重。

---

## 4. 运行时接口要求

`workspace/engine.py` 必须定义：

```python
def create_engine(model_config: dict, weight_dir: str, device: str = "cuda"):
    return Engine(...)
```

返回的对象必须支持：

```python
class Engine:
    def prefill(self, request_ids, input_ids):
        ...

    def decode(self, request_ids, token_ids):
        ...

    def remove(self, request_ids):
        ...
```

### `prefill(request_ids, input_ids)`

**输入：**

- `request_ids`：请求 ID 列表，例如 `[0, 1, 2]`
- `input_ids`：由 1D `torch.Tensor` token 序列组成的列表，每个请求一个序列

**输出：**

- 形状为 `[batch_size, vocab_size]` 的 logits 张量
- 第 `i` 行必须包含 `request_ids[i]` 的最后一个 token 的 logits

对某个请求调用 `prefill(...)` 应为该请求创建或替换其状态。不应清除无关请求的状态。

### `decode(request_ids, token_ids)`

**输入：**

- `request_ids`：已有请求 ID 的列表
- `token_ids`：形状为 `[batch_size]` 的 1D `torch.Tensor`，每个请求一个新 token

**输出：**

- 形状为 `[batch_size, vocab_size]` 的 logits 张量
- 第 `i` 行必须包含将 `token_ids[i]` 追加到 `request_ids[i]` 后的最后一个 token 的 logits

### `remove(request_ids)`

**输入：**

- `request_ids`：已完成请求 ID 的列表

此方法不需要返回任何内容。它应释放或删除与这些 ID 关联的请求状态。

---

## 5. 正确性检查

官方评测器将使用相同的隐藏模型配置和权重，以 PyTorch 参考实现做对比。我们比较的是 logits，而非生成文本。

正确性检查公式：

\[|y_{\mathrm{student}} - y_{\mathrm{ref}}| \leq \mathrm{atol} + \mathrm{rtol} \cdot |y_{\mathrm{ref}}|\]

公开骨架使用：

\[\mathrm{atol}=10^{-2}, \quad \mathrm{rtol}=10^{-2}\]

公开正确性测试使用：

```python
torch.allclose(student_logits, ref_logits, atol=1e-2, rtol=1e-2)
```

正确性测试覆盖：

- 单请求 prefill
- 单请求 decode
- 多请求 prefill
- 多请求 decode
- 插入新请求
- 移除请求后继续对其他请求 decode

若某用例未通过正确性检查，该用例不获得吞吐量分数。

---

## 6. 吞吐量评测

官方评测器将直接驱动你的引擎：

```python
engine = create_engine(model_config, weight_dir, device)
engine.prefill(...)
engine.decode(...)
engine.remove(...)
```

计时区间包含对以下方法的调用：

- `prefill(...)`
- `decode(...)`
- `remove(...)`

计时区间**不包含** `create_engine(...)` 或初始权重加载。如果你在计时区间内做了懒编译或昂贵的初始化，这些时间会被算进去。

吞吐量报告为：

\[\mathrm{tokens/s}=\frac{\mathrm{prefill\ tokens}+\mathrm{decode\ tokens}}{\mathrm{elapsed\ seconds}}\]

Decode 吞吐量报告为：

\[\mathrm{decode\ tokens/s}=\frac{\mathrm{decode\ tokens}}{\mathrm{elapsed\ seconds}}\]

公开 benchmark 包含三类用例：

- `prefill`：批量化长 prompt prefill
- `decode`：多个活跃请求，重复 decode 步
- `mixed`：服务式轨迹，包含 prefill、decode 和 remove 操作

隐藏评测将使用相同的接口和评测风格，但使用隐藏的模型尺寸、权重、batch size、prompt 长度、decode 步数和请求轨迹。

---

## 7. 评分策略

**正确性是硬性要求。**

未通过正确性检查的提交不会获得吞吐量分数。

对于通过正确性检查的提交，最终分数为：

- **70% 吞吐量**
- **30% Agent 实现 / 工程方法论**

### 吞吐量

吞吐量评分基于官方 benchmark 轨迹。评测器会在适当位置使用 warmup、多次重复测量和中位计时。

Benchmark 会考虑 prefill、decode 和混合服务行为。你应该优化引擎的整体运行时表现，而不仅是某个孤立的调用模式。

### Agent 实现 / 工程方法论

这部分奖励展示了真实工程 workflow 的提交，考察因素包括：

- 清晰的运行时组织
- 对照参考实现的本地正确性测试
- 基于 benchmark 和 profiling 的决策
- 迭代改进
- 对不同模型配置和请求模式的鲁棒处理
- 通过 `run.sh` 和日志确保可复现性

这个项目不是让你交一个手工写的、只对公开玩具用例有效的静态方案。强提交应使用公开输入来验证接口，然后构建一个能够泛化到隐藏用例的运行时。

---

## 8. 允许的优化方向

你可以使用以下技术优化运行时：

- 真正的逐层 KV cache
- 批量 prefill 和 decode
- PyTorch SDPA 或其他 PyTorch 原语
- Triton kernel
- C++/CUDA 扩展
- 针对 RMSNorm、RoPE、attention、MLP 或 cache 操作的自定义 kernel
- 更好的内存布局和请求状态管理

你应避免直接依赖完整的推理框架作为最终运行时实现。评测器期望你的 `engine.py` 直接实现所需接口。

---

## 9. 公开骨架

如果公开权重文件缺失，用以下命令重新生成：

```bash
python3 scripts/generate_toy_weights.py \
  --config target/model_config.json \
  --output target/weights/model.pt
```

运行公开正确性测试：

```bash
python3 evaluator/test_correctness.py \
  --engine workspace/engine.py \
  --model-config target/model_config.json \
  --weight-dir target/weights \
  --device auto
```

运行公开吞吐量 benchmark：

```bash
python3 evaluator/benchmark_throughput.py \
  --engine workspace/engine.py \
  --model-config target/model_config.json \
  --weight-dir target/weights \
  --device auto
```

或同时运行两者：

```bash
bash scripts/run_public_tests.sh
```

如果你的默认 `python3` 不含 PyTorch，请指定 Python 解释器：

```bash
PYTHON=/path/to/python-with-torch bash scripts/run_public_tests.sh
```

---

## 10. 基线

公开骨架在 `workspace/engine.py` 中已经包含了一个生成好的示例产物，方便你立刻运行评测器。该文件是一个最小化的 PyTorch 基线实现：它为每个请求存储完整 token 序列，每次 decode 都重新计算完整序列。这很慢，但它展示了所需的接口和正确的请求语义。在你自己的提交中，你的 agent 必须在 `run.sh` 启动后生成 `workspace/engine.py`。

重要的优化方向包括：

- 实现真正的逐层 KV cache
- 让 `decode(...)` 只计算新 token
- 跨请求批量处理工作
- 降低 Python 开销
- 优化 attention、MLP、RMSNorm、RoPE 和 cache 操作
- 根据 `model_config.json` 自适应实现选择

---

## 11. 总结

在这个项目中，你要构建一个能自动为 decoder-only 语言模型生成推理运行时的 agent。

你的提交应该：

- 提供 `run.sh`
- 从 `run.sh` 调用你的 agent
- 生成 `workspace/engine.py`
- 实现 `create_engine(...)`
- 支持 `prefill(...)`、`decode(...)` 和 `remove(...)`
- 使用 request ID 维护独立的请求状态
- 匹配参考 logits
- 在服务式轨迹上优化吞吐量

只有正确的实现才会获得吞吐量分数。在正确的提交中，评分基于：

- **70% 吞吐量**
- **30% Agent 实现 / 工程方法论**
