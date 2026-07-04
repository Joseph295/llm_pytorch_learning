"""第 16 章 · prefill vs decode：compute-bound vs memory-bound

运行：uv run chapters/ch16_inference/code/prefill_decode.py

两个阶段两种瓶颈（16.1）：prefill 处理整个 prompt（大矩阵乘，compute-bound）；
decode 逐 token（读全部权重算一个 token，memory-bound）。这决定了推理优化方向。
"""

import time

import torch
import torch.nn as nn

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 模拟一层的权重（推理时权重固定，读取它的成本是 decode 的瓶颈）
d = 2048
W = nn.Linear(d, 4 * d, bias=False).to(device)


def sync():
    if device.type == "mps":
        torch.mps.synchronize()


def bench(x):
    W(x); sync()
    t0 = time.perf_counter()
    for _ in range(20):
        W(x)
    sync()
    return (time.perf_counter() - t0) * 1000 / 20


print("═══ prefill（长序列）vs decode（单 token）的算术强度 ═══\n")
weight_bytes = d * 4 * d * 4                              # 权重字节数（fp32）

print(f"{'场景':<20} | {'序列长':>6} | {'耗时(ms)':>8} | {'算术强度':>10} | 瓶颈")
print("-" * 66)
for name, seq in [("decode（单token）", 1), ("prefill（短）", 128), ("prefill（长）", 2048)]:
    x = torch.randn(1, seq, d, device=device)
    t = bench(x)
    flops = 2 * seq * d * 4 * d                           # 矩阵乘 FLOPs
    # 访存：读权重(固定) + 读写激活
    mem = weight_bytes + 2 * seq * 4 * d * 4
    ai = flops / mem
    bound = "compute-bound" if ai > 20 else "memory-bound"
    print(f"{name:<20} | {seq:>6} | {t:>8.2f} | {ai:>9.1f} | {bound}")

print("""
读数（16.2-②）：
- decode（seq=1）：算术强度极低 → memory-bound。计算量小但要读全部权重，
  权重读取时间主导 → 单 token 生成的成本 ≈ 读一遍模型的时间
- prefill（长序列）：算术强度高 → compute-bound。大矩阵乘充分利用算力
- 关键推论：decode 时权重读一次可服务多个请求（batching），算术强度随 batch ×B 提升
  → 这是推理服务拼命 batching 的根本原因（第 17 章 continuous batching）
- 量化直接减权重字节数 → 直接加速 memory-bound 的 decode（16.2-③）
""")

# ═══ batching 让 decode 从 memory-bound 走向 compute-bound ═══
print("═══ batch decode：读一次权重服务 B 个请求 ═══")
print(f"{'batch':>6} | {'耗时(ms)':>8} | {'每请求(ms)':>10} | {'吞吐(req/s)':>12}")
for B in [1, 8, 32, 128]:
    x = torch.randn(B, 1, d, device=device)              # B 个请求各 1 个 token
    t = bench(x)
    print(f"{B:>6} | {t:>8.2f} | {t / B:>10.3f} | {B / t * 1000:>12.0f}")
print("→ batch 越大，每请求成本越低（权重读取被摊薄）——decode 的免费午餐（16.2-②）")
