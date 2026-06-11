> 原文：[Course Project Phase 2](https://memxlife.github.io/books/mlsys/project_phase2.html)
> 说明：内容由原网页整理翻译为中文 Markdown，并按课程项目文档的阅读习惯做了适度重组与润色。

# MLSYS 课程项目 · 第二阶段：LoRA 算子的 Agentic 优化

**截止时间：2026 年 5 月 12 日上午 8:00**
**评测设备：NVIDIA GeForce RTX 3090**

第一阶段里，你搭建了一个能探测 GPU 属性并推理性能的 agentic profiling 工具。
第二阶段，你要为一个 LoRA 式算子构建一个**优化 agent**。目标不是交一个手工写的静态 kernel，而是构建一个能迭代式生成、测试、profiling 并改进 CUDA 实现的 **agent 系统**。

---

## 1. 任务

你的目标算子是：

\[Y = W X + A(B^T X)\]

其中：

\(W \in \mathbb{R}^{d \times d}\)，
\(X \in \mathbb{R}^{d \times d}\)，
\(A \in \mathbb{R}^{d \times r}\)，
\(B \in \mathbb{R}^{d \times r}\)，
\(r = 16\)。

所有张量均以 **`.pt` 文件**存储，可通过 `torch.load` 加载。
所有隐藏评测张量使用 **`float32`**。

### 尺寸范围

隐藏评测会在以下范围内选多个尺寸：

- **隐藏维度 $d$** 从 [3584, 4608] 中选取。
- 本项目公开范围如下：

\[d \in [3584, 4608]\]

评测时我们会选多个 $d$ 在此区间内的测试用例。因此你设计的 agent 和生成的 CUDA 代码应能处理**此区间内的多种尺寸**，而不是只对某一个精确矩阵形状过拟合。

---

## 2. 提交内容

你的提交必须至少包含：

- `run.sh`

执行过程中，你的系统必须维护一个文件：

- `optimized_lora.cu`

### 提交规约

- 评测系统会进入**提交根目录**
- 执行：

```bash
bash run.sh
```

- 随后从**同一目录**读取：

```
./optimized_lora.cu
```

### 时间预算

你的 agent 可以运行**最多 30 分钟**。

超时或正常结束时，评测系统会读取提交根目录下**最终的** `optimized_lora.cu`，并用官方 harness 对其做基准测试。

因此：

- 你必须**始终保持一个可编译的最新版本** `optimized_lora.cu`
- 不要等到最后一刻才产出第一个能跑的版本

---

## 3. Agent 的预期行为

你提交的必须是一个**真正在做优化的 agent**。

一个合格的 agent 应该能够：

- 生成候选 CUDA 实现
- 编译并测试它们
- 进行基准测试
- 比较不同方案
- 迭代改进当前最佳实现
- 把当前最佳版本维护在 `optimized_lora.cu` 中

由于隐藏评测张量在正式测试期间不会暴露给你的 agent，你的 agent 应在公开尺寸范围内自行生成**合成测试张量**，用于本地搜索。

### 重要澄清

这个项目**不是**让你交一个静态 kernel。

课程组期望的是真正的 agentic workflow，而不是一次性写死的固定方案。

---

## 4. 官方评测环境

官方评测环境如下：

- **GPU：**NVIDIA GeForce RTX 3090
- **OS：**Ubuntu 22.04.4 LTS
- **Python：**3.10.12
- **PyTorch：**2.3.0a0+6ddf5cf85e.nv24.04
- **CUDA toolkit：**12.4
- **GCC：**11.4.0

`nvcc` 在 CUDA toolkit 安装中即可使用。
需要时也可依赖标准 PyTorch 扩展工具链。

---

## 5. 官方 Python Harness

课程组将使用如下 Python harness：

1. 加载隐藏的 `W.pt`、`X.pt`、`A.pt`、`B.pt`
2. 编译 `optimized_lora.cu`
3. 调用导出的 CUDA 实现
4. 将其输出与标准 PyTorch 参考实现比较
5. 基准测试运行时间
6. 计算相对标准 PyTorch 实现的加速比
7. 将结果写入 `result.out`

以下为简化参考版本：

```python
import torch
from pathlib import Path
from torch.utils.cpp_extension import load


def load_inputs(base_dir: str):
    base = Path(base_dir)
    W = torch.load(base / "W.pt", map_location="cpu").contiguous().cuda()
    X = torch.load(base / "X.pt", map_location="cpu").contiguous().cuda()
    A = torch.load(base / "A.pt", map_location="cpu").contiguous().cuda()
    B = torch.load(base / "B.pt", map_location="cpu").contiguous().cuda()
    return W, X, A, B


def reference_impl(W, X, A, B):
    with torch.no_grad():
        return W @ X + A @ (B.transpose(0, 1).contiguous() @ X)


def build_module(cu_path: str):
    module = load(
        name="optimized_lora_ext",
        sources=[cu_path],
        verbose=False,
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
    )
    return module


def check_correctness(y, y_ref):
    diff = (y - y_ref).float()
    max_abs_err = diff.abs().max().item()
    rel_l2_err = (diff.norm() / (y_ref.float().norm() + 1e-12)).item()
    passed = torch.allclose(y, y_ref, rtol=1e-4, atol=1e-4)
    return passed, max_abs_err, rel_l2_err


def benchmark(fn, W, X, A, B, warmup=10, iters=50):
    for _ in range(warmup):
        _ = fn(W, X, A, B)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = fn(W, X, A, B)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # milliseconds

    times.sort()
    return times[len(times) // 2]


def main():
    input_dir = "./hidden_inputs"
    cu_path = "./optimized_lora.cu"
    result_path = "./result.out"

    W, X, A, B = load_inputs(input_dir)
    module = build_module(cu_path)

    with torch.no_grad():
        y_student = module.forward(W, X, A, B)
        y_ref = reference_impl(W, X, A, B)

    passed, max_abs_err, rel_l2_err = check_correctness(y_student, y_ref)

    if passed:
        student_ms = benchmark(module.forward, W, X, A, B)
        torch_ms = benchmark(reference_impl, W, X, A, B)
        speedup = torch_ms / student_ms
    else:
        student_ms = None
        torch_ms = None
        speedup = 0.0

    with open(result_path, "w") as f:
        f.write(f"correct: {passed}\n")
        f.write(f"max_abs_err: {max_abs_err}\n")
        f.write(f"rel_l2_err: {rel_l2_err}\n")
        f.write(f"student_median_ms: {student_ms}\n")
        f.write(f"torch_median_ms: {torch_ms}\n")
        f.write(f"speedup: {speedup}\n")


if __name__ == "__main__":
    main()
```

建议你在自己的 agent workflow 中使用兼容的本地 harness，以减少环境差异。

---

## 6. `optimized_lora.cu` 的接口要求

你最终的 `optimized_lora.cu` 必须是：

- **单个文件**
- **自包含**
- 能被官方 Python harness 直接编译
- 能导出可调用的入口点

预期的可调用接口为：

```cpp
torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B);
```

且模块必须通过 `PYBIND11_MODULE(...)` 暴露，使 harness 可以调用：

```python
module.forward(W, X, A, B)
```

### 允许的依赖

可以使用：

- 标准 CUDA 头文件
- 标准 C/C++ 库头文件
- 系统环境已有的标准 PyTorch 扩展头文件

**不允许**依赖提交侧 `optimized_lora.cu` 之外的额外源文件。

这意味着不能有额外的：

- `.cu`
- `.cuh`
- `.h`
- `.cpp`

---

## 7. 必须仔细阅读的专项规则

### 7.1 输入输出格式

- 隐藏输入以 `.pt` 文件存储
- 通过 `torch.load` 加载
- 隐藏评测张量使用 `float32`
- 被测算子为：

\[Y = W X + A(B^T X)\]

- 隐藏维度 $d$ **不固定**
- 评测使用多个尺寸，$d \in [3584, 4608]$
- 低秩维度固定为 **16**
- 你生成的 CUDA 实现必须接受给定张量并返回输出张量

### 7.2 评分策略与标准

**正确性是硬性要求。**

未通过正确性检查的提交**不会**获得性能分数。

正确性对照 PyTorch 参考实现检查：

\[Y_{\text{ref}} = W X + A(B^T X)\]

使用：

```python
torch.allclose(Y_student, Y_ref, rtol=1e-4, atol=1e-4)
```

同时记录：

- `max_abs_err`
- `rel_l2_err`

对于通过正确性检查的提交，最终分数为：

- **70% 加速比**
- **30% Agent 实现 / 工程方法论**

#### 加速比

加速比计算为：

\[\text{speedup} = \frac{\text{标准 PyTorch 实现的运行时间中位数}}{\text{你的 CUDA 实现的运行时间中位数}}\]

运行时间测量使用：

- **先 warmup**
- CUDA events
- 多次重复运行取中位延迟

#### Agent 实现 / 工程方法论

这部分奖励真正实现了优化 agent 的提交，考察因素包括：

- 迭代改进 workflow
- 候选方案的生成与比较
- 使用 benchmark/profiling 来驱动决策
- 代码组织与可复现性
- agent 系统的整体工程质量

### 7.3 必须使用你自己的 API Key

如果你的 agent 依赖外部模型 API，你必须使用**你自己的 API key**。

课程组**不会**为你提供 API 额度。

你应该设计你的系统，使得你自己的 API key 可以安全、干净地提供——例如通过环境变量或提交中的本地配置。

### 7.4 严格禁止

以下行为被禁止：

1. **仅提交静态 kernel**——本项目要求的是一个 **agent**，而不是一个手工写的固定最终 kernel
2. **把最终 CUDA 代码硬编码在 agent 内部**——不要把预先写好的 `optimized_lora.cu` 作为固定字面量/模板/字符串嵌在 agent 里，运行时直接 dump 出来。你的 agent 应真正执行优化，而不是仅仅暴露出一个隐藏的最终答案
3. **依赖额外源文件作为最终评测实现**——最终评测实现必须是单文件 `optimized_lora.cu`
4. **破坏官方 I/O 规约**——评测系统在提交根目录执行 `bash run.sh`，并从同一目录读取 `./optimized_lora.cu`

---

## 8. 实用建议

一个强提交通常需要：

- 生成候选 CUDA 代码变体
- 自动编译和测试它们
- 对照本地 PyTorch 参考实现验证正确性
- warmup 后重复 benchmark
- 把当前最佳有效实现维护在 `optimized_lora.cu` 中
- 避免对某一个精确矩阵尺寸过拟合

---

## 9. 总结

在这个项目中，你要为一个 LoRA 算子构建一个 **agentic CUDA 优化系统**。

你的 agent 应该：

- 通过 `bash run.sh` 运行
- 迭代改进 CUDA 代码
- 始终保持一个有效的 `optimized_lora.cu`
- 产出一个单文件、自包含的最终 CUDA 实现
- 对算子

\[Y = W X + A(B^T X)\]

在 [3584, 4608] 区间内的多个隐藏测试尺寸上做优化。

只有正确的实现才会被排名。在正确的提交中，最终评测基于：

- **70% 加速比**
- **30% Agent 实现 / 工程方法论**
