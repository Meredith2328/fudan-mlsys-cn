import hashlib
import html
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = ROOT / ".translation_cache.json"

PAGES = [
    {
        "url": "https://memxlife.github.io/books/mlsys/chapter1.html",
        "output": ROOT / "chapter1.md",
        "chapter_label": "第 1 章",
    },
    {
        "url": "https://memxlife.github.io/books/mlsys/chapter2.html",
        "output": ROOT / "chapter2.md",
        "chapter_label": "第 2 章",
    },
]

HEADING_OVERRIDES = {
    "Chapter 1": "第 1 章",
    "Machine Learning Systems: Questions, Constraints, and Vertical Integration": "机器学习系统：问题、约束与垂直整合",
    "Part I": "第一部分",
    "Critical Questions: What Makes Machine Learning Systems Hard?": "关键问题：机器学习系统为什么难？",
    "1.1 Why Is Computer System Design Difficult?": "1.1 为什么计算机系统设计很困难？",
    "1.2 What Is the Workload of Intelligence?": "1.2 智能的工作负载是什么？",
    "1.3 If Intelligence Is Linear Algebra, What Is the Dominant Operation?": "1.3 如果智能可以归结为线性代数，主导操作是什么？",
    "Parameter Scale and Memory Reality": "参数规模与内存现实",
    "Arithmetic vs Data Movement": "算术与数据移动",
    "1.4 If Matrix Multiplication Is Simple, Why Is AI Hard?": "1.4 如果矩阵乘法很简单，为什么 AI 依然困难？",
    "Computation Versus Communication": "计算与通信",
    "Staying Local": "保持局部性",
    "1.5 Can Hardware Scaling Solve the Problem?": "1.5 硬件扩展能解决这个问题吗？",
    "1.6 Why Did GPUs Win?": "1.6 为什么 GPU 胜出？",
    "1.7 Why Is Hardware Improvement Not Enough?": "1.7 为什么仅靠硬件进步还不够？",
    "1.8 Where Is the Real Optimization Opportunity?": "1.8 真正的优化机会在哪里？",
    "1.9 What Is Machine Learning Systems?": "1.9 什么是机器学习系统？",
    "Part II": "第二部分",
    "The Intellectual Map of Machine Learning Systems": "机器学习系统的知识地图",
    "1.10 GPU Architecture: The Physical Substrate": "1.10 GPU 架构：物理基础",
    "1.11 CUDA Programming: Bridging Abstraction and Hardware": "1.11 CUDA 编程：连接抽象与硬件",
    "1.12 ML Compilers: Automating Optimization": "1.12 ML 编译器：自动化优化",
    "1.13 Parallel Training: Scaling Intelligence": "1.13 并行训练：扩展智能",
    "1.14 Inference Systems: Serving Intelligence": "1.14 推理系统：服务智能",
    "1.15 Model Optimization: Reshaping the Workload": "1.15 模型优化：重塑工作负载",
    "1.16 Data Engineering and Lifecycle": "1.16 数据工程与生命周期",
    "1.17 Beyond Performance": "1.17 超越性能",
    "1.18 Final Reflection: Why Machine Learning Systems Is Relevant": "1.18 最终思考：为什么机器学习系统重要",
    "Chapter 2": "第 2 章",
    "CPU Foundations, GPU Emergence, and the Logic of Throughput Computing": "CPU 基础、GPU 的兴起与吞吐计算的逻辑",
    "2.1 Machine Learning Systems Again: Why Architecture Matters": "2.1 再谈机器学习系统：为什么架构重要",
    "2.2 Performance as an Optimization Decomposition": "2.2 将性能分解为一个优化问题",
    "2.3 From Instruction Execution to Pipelining": "2.3 从指令执行到流水线",
    "2.4 Branch Prediction, Out-of-Order Execution, and Superscalar Design": "2.4 分支预测、乱序执行与超标量设计",
    "2.5 The Software Side: Loop Unrolling and Vectorization": "2.5 软件侧：循环展开与向量化",
    "2.6 Data Locality and the Logic of the Memory Hierarchy": "2.6 数据局部性与存储层次的逻辑",
    "2.7 Why the CPU Reaches a Limit": "2.7 为什么 CPU 会遇到极限",
    "2.8 GPU Was Born Differently": "2.8 GPU 的诞生路径不同",
    "2.9 From Graphics to General-Purpose Throughput Computing": "2.9 从图形处理到通用吞吐计算",
    "2.10 Programmability, Unified Shaders, and the Meaning of Flexibility": "2.10 可编程性、统一着色器与灵活性的含义",
    "2.11 CUDA as the Decisive Software Abstraction": "2.11 CUDA：决定性的软硬件抽象",
    "2.12 The Lesson of Chapter 2": "2.12 第 2 章的结论",
}

GLOSSARY = {
    "Machine Learning Systems": "机器学习系统",
    "machine learning systems": "机器学习系统",
    "Model FLOPs Utilization (MFU)": "模型 FLOPs 利用率（MFU）",
    "Model FLOPs Utilization": "模型 FLOPs 利用率",
    "Moore’s Law": "摩尔定律",
    "Moore's Law": "摩尔定律",
    "Amdahl’s Law": "阿姆达尔定律",
    "Amdahl's Law": "阿姆达尔定律",
    "Rayleigh–Ritz theorem": "Rayleigh-Ritz 定理",
    "Rayleigh-Ritz theorem": "Rayleigh-Ritz 定理",
    "Tensor Cores": "Tensor Core",
    "Tensor Core": "Tensor Core",
    "GEMM": "GEMM",
    "GPU": "GPU",
    "CPU": "CPU",
    "CUDA": "CUDA",
    "HBM": "HBM",
    "NVLink": "NVLink",
    "DRAM": "DRAM",
    "SRAM": "SRAM",
    "SIMD": "SIMD",
    "FLOP": "FLOP",
    "FLOPs": "FLOPs",
    "BF16": "BF16",
    "FP16": "FP16",
    "FP8": "FP8",
}

INLINE_PATTERNS = [
    re.compile(r"`[^`\n]+`"),
    re.compile(r"\[[^\]]+\]\([^)]+\)"),
    re.compile(r"https?://[^\s)]+"),
    re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)"),
]

POST_REPLACEMENTS = {
    "工作量": "工作负载",
    "工作负荷": "工作负载",
    "缓存记忆体": "缓存",
    "运算强度": "算术强度",
    "本地性": "局部性",
    "推理系统": "推理系统",
    "机器学习 系统": "机器学习系统",
    "矩阵乘法内核": "矩阵乘法内核",
    "通用矩阵乘法": "通用矩阵乘法",
}


def load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict) -> None:
    CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def fetch_markdown(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read().decode("utf-8", "ignore")

    match = re.search(
        r'<script id="md-source" type="text/markdown">(.*?)</script>',
        raw,
        re.S,
    )
    if not match:
        raise RuntimeError(f"Could not find embedded markdown in {url}")

    return html.unescape(match.group(1)).strip() + "\n"


def protect_literals(text: str):
    replacements = {}
    counter = 0

    def add_token(value: str, restore_value: str | None = None) -> str:
        nonlocal counter
        token = f"__TK{counter:04d}__"
        counter += 1
        replacements[token] = restore_value if restore_value is not None else value
        return token

    for src, dst in sorted(GLOSSARY.items(), key=lambda item: len(item[0]), reverse=True):
        if src in text:
            text = text.replace(src, add_token(src, dst))

    for pattern in INLINE_PATTERNS:
        while True:
            match = pattern.search(text)
            if not match:
                break
            text = text[: match.start()] + add_token(match.group(0)) + text[match.end() :]

    return text, replacements


def restore_literals(text: str, replacements: dict) -> str:
    for token, value in replacements.items():
        text = text.replace(token, value)
        match = re.fullmatch(r"__TK(\d+)__", token)
        if match:
            token_id = match.group(1)
            text = re.sub(rf"__\s*TK{token_id}\s*__", lambda _: value, text)
    return text


def split_for_translation(text: str, limit: int = 420) -> list[str]:
    if len(text) <= limit:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > limit:
            words = sentence.split()
            piece = ""
            for word in words:
                if not piece:
                    piece = word
                elif len(piece) + 1 + len(word) <= limit:
                    piece += " " + word
                else:
                    chunks.append(piece)
                    piece = word
            if piece:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.append(piece)
            continue

        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= limit:
            current += " " + sentence
        else:
            chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)
    return chunks


def google_translate(text: str) -> str:
    params = urllib.parse.urlencode(
        {
            "client": "gtx",
            "sl": "en",
            "tl": "zh-CN",
            "dt": "t",
            "q": text,
        }
    )
    url = f"https://translate.googleapis.com/translate_a/single?{params}"
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    parts = payload[0]
    translated = "".join(part[0] for part in parts if part and part[0])
    if not translated:
        raise RuntimeError("empty translation from Google endpoint")
    return html.unescape(translated.strip())


def mymemory_translate(text: str) -> str:
    params = urllib.parse.urlencode(
        {
            "q": text,
            "langpair": "en|zh-CN",
        }
    )
    url = f"https://api.mymemory.translated.net/get?{params}"
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    translated = html.unescape(payload["responseData"]["translatedText"]).strip()
    if not translated:
        raise RuntimeError("empty translation from MyMemory")
    return translated


def remote_translate(text: str, cache: dict) -> str:
    key = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if key in cache:
        return cache[key]

    last_error = None
    for translator in (google_translate, mymemory_translate):
        for attempt in range(3):
            try:
                translated = translator(text)
                cache[key] = translated
                save_cache(cache)
                time.sleep(0.15)
                return translated
            except Exception as error:
                last_error = error
                time.sleep(1.0 * (attempt + 1))

    raise RuntimeError(f"Translation failed for chunk: {text[:120]!r}") from last_error


def translate_text(text: str, cache: dict) -> str:
    stripped = text.strip()
    if not stripped:
        return text

    masked_text, replacements = protect_literals(stripped)
    chunks = split_for_translation(masked_text)
    translated_chunks = [remote_translate(chunk, cache) for chunk in chunks]
    translated = " ".join(chunk.strip() for chunk in translated_chunks if chunk.strip())
    translated = restore_literals(translated, replacements)
    return cleanup_translation(translated)


def cleanup_translation(text: str) -> str:
    text = text.replace(" ,", "，").replace(" .", "。")
    text = text.replace(" :", "：").replace(" ;", "；")
    text = text.replace(" ?", "？").replace(" !", "！")
    text = re.sub(r"\s+([，。！？；：])", r"\1", text)
    text = re.sub(r"([，。！？；：])\s+", r"\1", text)
    text = re.sub(r"([（(])\s+", r"\1", text)
    text = re.sub(r"\s+([）)])", r"\1", text)
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[A-Za-z0-9])", "", text)
    text = re.sub(r"(?<=[A-Za-z0-9])\s+(?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    for src, dst in POST_REPLACEMENTS.items():
        text = text.replace(src, dst)
    return text.strip()


def replace_norm_bars(math_text: str) -> str:
    count = 0

    def repl(_: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return r"\lVert" if count % 2 == 1 else r"\rVert"

    return re.sub(r"\\\|", repl, math_text)


def normalize_latex_delimiters(markdown: str) -> str:
    patterns = [
        re.compile(r"\$\$(.*?)\$\$", re.S),
        re.compile(r"\\\[(.*?)\\\]", re.S),
        re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", re.S),
    ]

    for pattern in patterns:
        markdown = pattern.sub(lambda match: replace_norm_bars(match.group(0)), markdown)

    return markdown


def is_list_item(line: str) -> bool:
    return bool(re.match(r"^\s*(?:[-*+]\s+|\d+\.\s+)", line))


def translate_table_row(row: str, cache: dict) -> str:
    stripped = row.strip()
    body = stripped.strip("|")
    if set(body.replace(":", "").replace("-", "").replace(" ", "")) == set():
        return stripped

    cells = [cell.strip() for cell in body.split("|")]
    translated = [translate_text(cell, cache) if cell else "" for cell in cells]
    return "| " + " | ".join(translated) + " |"


def translate_markdown(markdown: str, url: str, chapter_label: str, cache: dict) -> str:
    lines = markdown.splitlines()
    out = [
        f"> 原文：[{chapter_label}]({url})",
        "> 说明：内容由原网页整理翻译为中文 Markdown，公式与章节结构尽量保持不变。",
        "",
    ]

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            out.append("")
            i += 1
            continue

        if stripped == "---":
            out.append("---")
            i += 1
            continue

        if stripped.startswith("```"):
            fence_lines = [line]
            i += 1
            while i < len(lines):
                fence_lines.append(lines[i])
                if lines[i].strip().startswith("```"):
                    i += 1
                    break
                i += 1
            out.extend(fence_lines)
            continue

        if stripped in {"$$", "\\["}:
            delimiter = stripped
            closing = "$$" if delimiter == "$$" else "\\]"
            math_lines = [line]
            i += 1
            while i < len(lines):
                math_lines.append(lines[i])
                if lines[i].strip() == closing:
                    i += 1
                    break
                i += 1
            out.extend(math_lines)
            continue

        if line.startswith("#"):
            prefix, title = line.split(" ", 1)
            title = title.strip()
            translated_title = HEADING_OVERRIDES.get(title) or translate_text(title, cache)
            out.append(f"{prefix} {translated_title}")
            i += 1
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            table_lines = []
            while i < len(lines):
                row = lines[i].strip()
                if not (row.startswith("|") and row.endswith("|")):
                    break
                table_lines.append(lines[i])
                i += 1
            out.extend(translate_table_row(row, cache) for row in table_lines)
            continue

        if is_list_item(line):
            match = re.match(r"^(\s*(?:[-*+]\s+|\d+\.\s+))(.*)$", line)
            assert match is not None
            marker, content = match.groups()
            out.append(marker + translate_text(content, cache))
            i += 1
            continue

        paragraph_lines = []
        while i < len(lines):
            current = lines[i]
            current_stripped = current.strip()
            if (
                not current_stripped
                or current_stripped == "---"
                or current.startswith("#")
                or current_stripped.startswith("```")
                or current_stripped in {"$$", "\\["}
                or (current_stripped.startswith("|") and current_stripped.endswith("|"))
                or is_list_item(current)
            ):
                break
            paragraph_lines.append(current_stripped)
            i += 1

        paragraph = " ".join(paragraph_lines)
        out.append(translate_text(paragraph, cache))

    markdown = "\n".join(out).rstrip() + "\n"
    return normalize_latex_delimiters(markdown)


def main() -> None:
    cache = load_cache()
    for page in PAGES:
        markdown = fetch_markdown(page["url"])
        translated = translate_markdown(
            markdown=markdown,
            url=page["url"],
            chapter_label=page["chapter_label"],
            cache=cache,
        )
        page["output"].write_text(translated, encoding="utf-8")
        print(f"Wrote {page['output'].name}")


if __name__ == "__main__":
    main()
