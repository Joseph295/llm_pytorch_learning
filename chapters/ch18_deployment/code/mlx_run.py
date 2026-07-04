"""第 18 章 · 用 MLX 在 M4 上跑量化 LLM（端侧实战）

运行：
  uv pip install mlx-lm          # 先装（Apple Silicon 专用）
  uv run chapters/ch18_deployment/code/mlx_run.py

MLX 是 Apple 为自家芯片设计的框架，利用统一内存 + Metal，比 PyTorch MPS 更快更省。
首次运行会下载模型（4bit 量化的 7B 约 4.4GB）。你的 M4 变成本地 LLM 推理机。
"""

import sys
import time


def main():
    try:
        from mlx_lm import generate, load
    except ImportError:
        print("未安装 mlx-lm。安装：uv pip install mlx-lm")
        print("（MLX 仅支持 Apple Silicon；这是端侧部署最贴近你 M4 硬件的实战）")
        print("\n没装也没关系——下面讲清楚它在做什么：")
        print("""
  from mlx_lm import load, generate
  model, tokenizer = load("mlx-community/Qwen2.5-7B-Instruct-4bit")
  # ↑ 下载 4bit 量化的 7B（~4.4GB），加载到统一内存
  text = generate(model, tokenizer, prompt="解释注意力机制", max_tokens=200)
  # ↑ 在 M4 GPU 上生成，利用统一内存零拷贝（第 0 章）

对照第 16/18 章原理：
- 4bit 量化把 7B 从 14GB 压到 4.4GB（16.2-③）→ 装得进 24GB 统一内存
- MLX 惰性求值 + Metal kernel，比 PyTorch MPS 推理更快
- decode 是 memory-bound（16.2-②），统一内存的高带宽在这里发挥作用
""")
        return

    MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"   # 小模型演示（大的换 7B）
    print(f"加载 {MODEL} ...")
    t0 = time.time()
    model, tokenizer = load(MODEL)
    print(f"加载耗时 {time.time() - t0:.1f}s")

    prompt = "用一句话解释什么是注意力机制"
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    t0 = time.time()
    response = generate(model, tokenizer, prompt=formatted, max_tokens=150, verbose=True)
    dt = time.time() - t0
    print(f"\n生成耗时 {dt:.1f}s")
    print("→ 你的 M4 正在本地运行量化 LLM——端侧推理实战（隐私、离线、零 API 成本）")


if __name__ == "__main__":
    main()
