"""第 11 章 · Roofline：算术强度判定 compute-bound vs memory-bound

运行：uv run chapters/ch11_performance/code/roofline.py
"""

import time

import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def sync():
    if device.type == "mps":
        torch.mps.synchronize()


def bench(fn, iters=50):
    fn(); sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - t0) / iters


print("═══ 算术强度（FLOP/byte）决定瓶颈类型 ═══\n")

# ── 操作 1：大矩阵乘（compute-bound 典型）──
n = 2048
a, b = torch.randn(n, n, device=device), torch.randn(n, n, device=device)
t = bench(lambda: a @ b)
flops = 2 * n**3                                    # n³ 乘 + n³ 加
bytes_moved = 3 * n * n * 4                         # 读 A,B 写 C，fp32
ai_mm = flops / bytes_moved
print(f"矩阵乘 {n}³:")
print(f"  算术强度 = {flops:.2e} FLOP / {bytes_moved:.2e} byte = {ai_mm:.0f} FLOP/byte（高）")
print(f"  实测 {t * 1000:.2f} ms → {flops / t / 1e9:.0f} GFLOPS")
print("  → 算术强度高 = compute-bound：优化靠更快的矩阵单元/低精度\n")

# ── 操作 2：逐元素加（memory-bound 典型）──
m = 16_000_000
x = torch.randn(m, device=device)
t = bench(lambda: x + 1.0)
flops_add = m                                       # m 次加法
bytes_add = 2 * m * 4                               # 读 x 写结果
ai_add = flops_add / bytes_add
print(f"逐元素 x+1（{m:,} 元素）:")
print(f"  算术强度 = {ai_add:.2f} FLOP/byte（极低）")
print(f"  实测 {t * 1000:.2f} ms → 带宽 {bytes_add / t / 1e9:.0f} GB/s")
print("  → 算术强度低 = memory-bound：优化靠融合减访存，换算力单元没用\n")

# ── 操作 3：融合的威力（多个逐元素 vs 一次)──
print("═══ 融合演示：5 个逐元素操作 分立 vs 合并 ═══")


def separate():
    y = x + 1.0
    y = y * 2.0
    y = y - 0.5
    y = torch.relu(y)
    return y * 0.1


def fused():                                        # 数学等价，一次遍历
    return torch.relu((x + 1.0) * 2.0 - 0.5) * 0.1


# torch 的 eager 模式下两者 kernel 数不同（fused 表达式仍是多 kernel，
# 真正融合要 torch.compile；这里对比"中间张量个数"的内存影响）
t_sep = bench(separate)
t_fus = bench(fused)
print(f"  分立 5 步（5 个中间张量）: {t_sep * 1000:.2f} ms")
print(f"  单表达式（编译器易融合）: {t_fus * 1000:.2f} ms")
print("  → memory-bound 操作的时间 ∝ 访存次数；torch.compile 把它们融成一个 kernel")
print("    （eager 模式收益有限，本演示看趋势；真加速在 compile/CUDA 上，第 11.2-③）")
