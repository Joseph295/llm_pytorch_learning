"""第 11 章 · 计算 miniGPT 的 MFU（Model FLOPs Utilization）

运行：uv run chapters/ch11_performance/code/mfu.py
"""

import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "ch08_transformer", "code"))
from gpt_model import GPT, GPTConfig  # noqa: E402

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def measure_peak_flops():
    """用大矩阵乘实测本机可达的 FLOPS 上限（当硬件规格不好查时的实用近似）。"""
    n = 4096
    a, b = torch.randn(n, n, device=device), torch.randn(n, n, device=device)
    for _ in range(3):
        a @ b
    if device.type == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    iters = 20
    for _ in range(iters):
        a @ b
    if device.type == "mps":
        torch.mps.synchronize()
    dt = (time.perf_counter() - t0) / iters
    return 2 * n**3 / dt


cfg = GPTConfig(vocab_size=4096, block_size=256, n_layer=6, n_head=6, n_embd=384)
model = GPT(cfg).to(device)
N = model.num_params()

batch_size = 16
x = torch.randint(0, cfg.vocab_size, (batch_size, cfg.block_size), device=device)
y = torch.randint(0, cfg.vocab_size, (batch_size, cfg.block_size), device=device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

# 稳态计时（warmup 后，第 0/11 章计时纪律）
for _ in range(5):
    _, loss = model(x, y)
    opt.zero_grad(); loss.backward(); opt.step()
if device.type == "mps":
    torch.mps.synchronize()

t0 = time.perf_counter()
STEPS = 20
for _ in range(STEPS):
    _, loss = model(x, y)
    opt.zero_grad(); loss.backward(); opt.step()
if device.type == "mps":
    torch.mps.synchronize()
step_time = (time.perf_counter() - t0) / STEPS

# MFU 计算
tokens_per_step = batch_size * cfg.block_size
flops_per_step = 6 * N * tokens_per_step            # 6N/token（前向2+反向4）
achieved_flops = flops_per_step / step_time
peak = measure_peak_flops()

print(f"模型: {N / 1e6:.1f}M 参数, batch={batch_size}, seq={cfg.block_size}")
print(f"每步耗时: {step_time * 1000:.1f} ms | 吞吐: {tokens_per_step / step_time:.0f} token/s")
print(f"每步计算量: 6 × {N / 1e6:.1f}M × {tokens_per_step} = {flops_per_step / 1e9:.1f} GFLOP")
print(f"达到算力: {achieved_flops / 1e9:.0f} GFLOPS")
print(f"本机峰值(矩阵乘实测): {peak / 1e9:.0f} GFLOPS")
print(f"\nMFU = {achieved_flops / peak:.1%}")
print("""
解读：
- 小模型 + 小 batch 在 M4 上 MFU 通常偏低——大量 memory-bound 胶水层、
  Python 调度开销、kernel 启动开销占比高（第 0 章小矩阵现象的放大版）。
- 提升 MFU 的方向：调大 batch/模型（摊薄固定开销）、bf16、torch.compile（融合）。
- 大模型云端训练 MFU 到 40-55% 是优秀工程（第 20 章面试硬指标）。
- 注意 6N 近似不含注意力的 O(T²) 部分，长序列时要修正（本例 seq 短，误差小）。
""")
