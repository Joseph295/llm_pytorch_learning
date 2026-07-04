"""第 0 章 · 环境体检脚本

运行方式（在仓库根目录）：
    uv run chapters/ch00_environment/code/check_env.py

作用：
1. 打印 Python / PyTorch 版本与平台架构（疑难排查案例 2 的第一步）
2. 检查各计算后端（CUDA / MPS / CPU）的可用性
3. 探测 MPS 对各数据类型的支持情况（对应讲义 0.2 节的限制说明）
4. 在最优设备上做一次真实计算并计时（演示"设备无关代码"的标准写法）

这也是全教程遇到环境问题时的第一排查工具（讲义 0.7 节）。
"""

import platform
import sys
import time

import torch


def section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


# ---------------------------------------------------------------
# 1. 版本与架构
# ---------------------------------------------------------------
section("1. 版本与架构")
print(f"Python     : {sys.version.split()[0]}")
# Apple Silicon 上必须是 arm64；若显示 x86_64 说明解释器跑在 Rosetta
# 转译下，装的 wheel 会架构错乱（讲义 0.7 案例 2）
print(f"机器架构   : {platform.machine()}")
print(f"操作系统   : {platform.system()} {platform.release()}")
print(f"PyTorch    : {torch.__version__}")

# ---------------------------------------------------------------
# 2. 计算后端可用性
# ---------------------------------------------------------------
section("2. 计算后端")
cuda_ok = torch.cuda.is_available()
mps_ok = torch.backends.mps.is_available()
# Mac 上 cuda=False 不是错误——CUDA 是 NVIDIA 专有生态（讲义易错点②）
print(f"CUDA 可用  : {cuda_ok}")
print(f"MPS  可用  : {mps_ok}")

# 设备无关代码的标准写法：三级回退，全教程统一使用这个模式。
# 写代码永远面向 device 变量，而不是硬编码 "cuda"/"mps"。
device = torch.device("cuda" if cuda_ok else "mps" if mps_ok else "cpu")
print(f"本机最优设备: {device}")

# ---------------------------------------------------------------
# 3. dtype 支持探测（在最优设备上逐一试创建张量）
# ---------------------------------------------------------------
section(f"3. dtype 支持情况（设备: {device}）")
dtypes = {
    "float32 (训练默认)": torch.float32,
    "float16 (半精度)": torch.float16,
    "bfloat16 (LLM 训练主流)": torch.bfloat16,
    "float64 (科学计算)": torch.float64,
}
for name, dt in dtypes.items():
    try:
        t = torch.ones(4, 4, dtype=dt, device=device)
        (t @ t).sum().item()  # 不只创建，还要真的算一下
        print(f"  {name:<28} ✓")
    except (TypeError, RuntimeError) as e:
        # MPS 预期在 float64 上失败——Metal 没有 fp64 算力（讲义 0.2 节）
        print(f"  {name:<28} ✗  ({type(e).__name__}: {str(e)[:60]}...)")

# ---------------------------------------------------------------
# 4. 真实计算 + 计时
# ---------------------------------------------------------------
section("4. 真实计算测试")


def sync(dev: torch.device) -> None:
    """按后端分派同步调用——GPU 是异步执行的，计时前必须同步（讲义易错点⑥）。"""
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()
    # CPU 是同步执行的，无需操作


n = 2048
for dev in [torch.device("cpu"), device]:
    if dev.type == "cpu" and device.type == "cpu":
        pass  # 最优设备就是 CPU 时不重复跑
    a = torch.randn(n, n, device=dev)
    b = torch.randn(n, n, device=dev)
    a @ b  # warmup：首次调用包含 kernel 编译等一次性开销，不计入
    sync(dev)
    t0 = time.perf_counter()
    c = a @ b
    sync(dev)
    ms = (time.perf_counter() - t0) * 1000
    print(f"  {n}x{n} 矩阵乘 @ {str(dev):<5}: {ms:8.2f} ms")

print("\n环境体检完成 ✓ 若上面 MPS 可用且 float32/bf16 打勾，即可开始全部本地章节。")
