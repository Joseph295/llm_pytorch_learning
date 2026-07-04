"""进阶 1 参考答案：寻找 CPU/MPS 的性能交叉点

运行：uv run chapters/ch00_environment/exercises/solutions/advanced_1.py

结论先行（M4 实测，你的数字会有波动）：
- 交叉点大约在 512~1024 之间：更小的矩阵 CPU 赢，更大的 MPS 赢
- 解释框架：总耗时 ≈ 固定开销 + 计算量/吞吐
    CPU：固定开销极小（函数调用），吞吐较低
    MPS：固定开销大（kernel 启动、命令队列提交，微秒级），吞吐高
  计算量随 n^3 增长，固定开销不变 → 必然存在交叉点。
  这与大数据的"小文件问题"同构：单条任务太小时，调度成本主导一切。

对推理的启示：小 batch、小模型的在线推理未必该用 GPU——
每次前向的计算量可能摊不平 kernel 启动开销。工业界的对策正是
"把小任务攒成大任务"：批量化（batching），第 17 章 continuous
batching 是它的极致形态。
"""

import time

import torch


def bench(n: int, device: torch.device, warmup: int = 5, iters: int = 50) -> float:
    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)

    def sync() -> None:
        if device.type == "mps":
            torch.mps.synchronize()

    for _ in range(warmup):
        a @ b
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        a @ b
    sync()
    return (time.perf_counter() - t0) * 1000 / iters


def main() -> None:
    cpu, mps = torch.device("cpu"), torch.device("mps")
    # 比题目要求更密的尺寸网格，把交叉点夹得更准
    sizes = [64, 128, 256, 384, 512, 640, 768, 896, 1024]
    crossover = None

    print(f"{'尺寸':>6} | {'CPU (ms)':>9} | {'MPS (ms)':>9} | {'胜者':>4}")
    print("-" * 42)
    for n in sizes:
        t_cpu, t_mps = bench(n, cpu), bench(n, mps)
        winner = "MPS" if t_mps < t_cpu else "CPU"
        if winner == "MPS" and crossover is None:
            crossover = n
        print(f"{n:>6} | {t_cpu:>9.4f} | {t_mps:>9.4f} | {winner:>4}")

    if crossover:
        print(f"\n本机交叉点：约 n={crossover}（首个 MPS 获胜的尺寸）")
    else:
        print("\n此网格内 MPS 未反超——把尺寸上限调大再试")


if __name__ == "__main__":
    main()
