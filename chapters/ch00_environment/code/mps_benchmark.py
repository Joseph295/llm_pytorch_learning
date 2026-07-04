"""第 0 章 · CPU vs MPS 矩阵乘基准测试

运行方式（在仓库根目录）：
    uv run chapters/ch00_environment/code/mps_benchmark.py

观察点（对应讲义 0.3 实验 3）：
1. 大矩阵上 MPS 数倍快于 CPU——GPU 的并行算力优势
2. 小矩阵上 MPS 可能反而更慢——kernel 启动固定开销摊不平
   （类比大数据的"小文件问题"：调度开销 > 任务本身）
3. 正确的 GPU 计时姿势：warmup + synchronize（讲义易错点⑥）
"""

import time

import torch


def bench_matmul(n: int, device: torch.device, warmup: int = 3, iters: int = 10) -> float:
    """返回 n×n 矩阵乘的平均耗时（毫秒）。

    计时协议（第 11 章 Profiler 会用更精细的工具，协议思想相同）：
    - warmup 轮不计时：首次运行含 kernel 编译/缓存预热等一次性开销
    - 多次迭代取平均：抹平单次抖动
    - 每段计时前后同步：GPU 异步执行，不同步测到的是"提交耗时"
    """
    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)

    def sync() -> None:
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

    for _ in range(warmup):
        a @ b
    sync()

    t0 = time.perf_counter()
    for _ in range(iters):
        a @ b
    sync()
    return (time.perf_counter() - t0) * 1000 / iters


def main() -> None:
    if not torch.backends.mps.is_available():
        raise SystemExit("此脚本需要 MPS（Apple Silicon）。CUDA 机器请自行把 'mps' 换成 'cuda'。")

    cpu, mps = torch.device("cpu"), torch.device("mps")
    sizes = [256, 512, 1024, 2048, 4096]

    print(f"{'尺寸':>6} | {'CPU (ms)':>10} | {'MPS (ms)':>10} | {'加速比':>7} | {'MPS GFLOPS':>10}")
    print("-" * 58)
    for n in sizes:
        t_cpu = bench_matmul(n, cpu)
        t_mps = bench_matmul(n, mps)
        # 矩阵乘浮点运算量 = 2*n^3（n^3 次乘法 + n^3 次加法）
        gflops = 2 * n**3 / (t_mps / 1000) / 1e9
        print(f"{n:>6} | {t_cpu:>10.3f} | {t_mps:>10.3f} | {t_cpu / t_mps:>6.1f}x | {gflops:>10.1f}")

    print(
        "\n思考：加速比随尺寸如何变化？在多小的矩阵上 MPS 会输给 CPU？"
        "\n     （进阶练习 1 会让你找出精确的交叉点）"
    )


if __name__ == "__main__":
    main()
