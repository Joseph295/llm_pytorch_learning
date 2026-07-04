"""第 13 章 · 流水线并行的气泡量化

运行：uv run chapters/ch13_parallelism/code/pipeline_bubble.py

气泡（bubble）= 流水线填充/排空阶段的空转。micro-batch 越多，气泡占比越小。
"""


def bubble_fraction(stages, micro_batches):
    """GPipe 式气泡占比 = (P-1) / (M + P - 1)，P=段数，M=micro-batch 数。

    直觉：填满流水线要 P-1 步，这段时间部分卡空转；总共 M+P-1 步。
    """
    return (stages - 1) / (micro_batches + stages - 1)


def speedup(stages, micro_batches):
    """相对单卡的有效加速比 = P × (1 - 气泡占比)。理想 P，气泡拉低它。"""
    return stages * (1 - bubble_fraction(stages, micro_batches))


P = 8
print(f"═══ {P} 段流水线：micro-batch 数 vs 气泡与加速 ═══\n")
print(f"{'micro-batch':>12} | {'气泡占比':>8} | {'有效加速':>10} | {'效率':>6}")
print("-" * 46)
for M in [1, 2, 4, 8, 16, 32, 64, 128]:
    b = bubble_fraction(P, M)
    s = speedup(P, M)
    print(f"{M:>12} | {b:>7.1%} | {s:>8.2f}× | {s / P:>5.1%}")

# 找气泡 < 10% 需要多少 micro-batch
target = 0.10
M = 1
while bubble_fraction(P, M) > target:
    M += 1
print(f"\n气泡降到 <{target:.0%} 需要 ≥ {M} 个 micro-batch（P={P}）")

print("""
结论（13.2-④）：
- micro-batch=1（朴素 PP）：气泡 88%，8 段只有 1.1× 加速——几乎白搭
- micro-batch 越多气泡越小，但受单卡显存限制（micro-batch 的激活要驻留）
- 1F1B 调度（Megatron）通过"算完一个 micro-batch 立即反向"减少激活驻留，
  允许更多 micro-batch，进一步压气泡
- 权衡：段数 P 越大模型切得越细（放得下更大模型），但气泡也越大——
  所以 PP 段数通常不会很大（8~16），配合 TP/DP 组成 3D 并行
""")
