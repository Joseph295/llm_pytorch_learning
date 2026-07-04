"""挑战 1 参考答案：验证 MPS fallback 的真实代价

运行（注意环境变量必须在进程启动时设置，torch 导入后再设无效）：
    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run chapters/ch00_environment/exercises/solutions/challenge_1.py

torch 2.12.1 实测的算子覆盖情况（会随版本变化，探测方法比结论重要）：
- 未实现、需 fallback 才能用：linalg.eig / linalg.matrix_exp / take / special.airy_ai
- 未实现、但有"内置 fallback"（不设环境变量也自动回 CPU + UserWarning）：linalg.svd
- 本题选 linalg.matrix_exp：计算量足够大，适合计时

为什么 fallback 往往比"直接用 CPU"还慢？
    fallback 路径：MPS 显存里的数据 → 拷回 CPU 内存 → CPU 算 → 拷回 MPS
    直接 CPU 路径：数据本来就在 CPU 内存 → 算完即用
    差的就是两次跨设备搬运。若 fallback 的算子处在训练热路径上，
    每一步都白付两次搬运税——这时应换算子实现或把实验搬上 CUDA。
    （统一内存架构下"搬运"是内存内拷贝而非走 PCIe，代价比独显小，
     但同步点带来的流水线中断依然存在。）
"""

import os
import time

import torch

if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1":
    raise SystemExit(
        "请这样运行：PYTORCH_ENABLE_MPS_FALLBACK=1 uv run .../challenge_1.py\n"
        "（环境变量在 torch 导入前生效，脚本内 os.environ 赋值已太迟）"
    )


def bench(fn, warmup: int = 3, iters: int = 20) -> float:
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t0) * 1000 / iters


n = 256
x_mps = torch.randn(n, n, device="mps") * 0.1
x_cpu = x_mps.cpu()

# a) MPS 张量 + fallback：数据 MPS→CPU→计算→CPU→MPS 来回搬
t_fallback = bench(lambda: torch.linalg.matrix_exp(x_mps))
# b) 直接 CPU 张量：无搬运
t_cpu = bench(lambda: torch.linalg.matrix_exp(x_cpu))

print(f"matrix_exp({n}x{n})")
print(f"  MPS 张量走 fallback : {t_fallback:8.3f} ms")
print(f"  CPU 张量直接计算    : {t_cpu:8.3f} ms")
print(f"  fallback 额外代价   : {t_fallback / t_cpu:.2f}x  ← 两次跨设备搬运 + 同步的税")
