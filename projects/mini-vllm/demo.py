"""mini-vLLM · 多请求并发生成演示：观察 continuous batching 调度与分块

运行：uv run projects/mini-vllm/demo.py

用第 8 章的 GPT，服务多个不同长度的请求，观察：
- 短请求先完成退出、KV 块释放
- 新请求动态加入
- continuous batching vs 静态 batching 的效率差异
"""

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "chapters", "ch08_transformer", "code"))
from engine import MiniVLLM  # noqa: E402
from gpt_model import GPT, GPTConfig  # noqa: E402

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 用一个小 GPT（未训练也能演示调度逻辑；EOS 用 token 0）
torch.manual_seed(0)
cfg = GPTConfig(vocab_size=256, block_size=128, n_layer=4, n_head=4, n_embd=128)
model = GPT(cfg).to(device)

print("═══ mini-vLLM: continuous batching 演示 ═══")
print("提交 6 个请求，不同的 max_new_tokens（模拟长短不一的生成）\n")

engine = MiniVLLM(model, eos_token=0, num_kv_blocks=64, block_size=16, max_running=4)
requests = [
    (0, [1, 2, 3], 8),       # 短请求
    (1, [4, 5], 30),         # 长请求
    (2, [6, 7, 8, 9], 12),
    (3, [10], 45),           # 最长
    (4, [11, 12], 6),        # 最短
    (5, [13, 14, 15], 20),
]
for rid, prompt, maxnew in requests:
    engine.add_request(rid, prompt, maxnew)

print(f"max_running=4（同时最多 4 个请求），KV 块 64 个（每块 16 token）\n")
print("调度过程（每 10 步快照）：")
finished = engine.run(verbose=True)

print(f"\n全部 {len(finished)} 个请求完成，总步数 {engine.step_count}")
print(f"{'请求':>4} | {'prompt长':>8} | {'生成长':>8} | {'max_new':>8}")
for req in sorted(finished, key=lambda r: r.req_id):
    print(f"{req.req_id:>4} | {len(req.prompt_tokens):>8} | {len(req.output_tokens):>8} | {req.max_new_tokens:>8}")

print("""
观察到的 continuous batching 行为（17.2-①）：
- 短请求（0,4）先完成退出，KV 块立即释放回空闲池（看'空闲KV块'回升）
- 释放的位置让 waiting 队列的新请求（4,5）动态加入 running
- GPU 始终有 ≤4 个请求在跑，没有"等最慢请求"的空转
对比静态 batching：6 个请求分批跑，每批等最长的完成——短请求空转浪费。
真实 vLLM 在此之上：变长注意力 kernel（不 padding）、块表直接索引 KV、
抢占换出、prefix caching（挑战题）。你手写的抓住了核心结构（17.5-①）。
""")
