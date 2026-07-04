"""基础 2 参考答案：亲手触发并修复 NumPy fp64 → MPS 的 dtype 错误

运行：uv run chapters/ch00_environment/exercises/solutions/basic_2.py
"""

import numpy as np
import torch

arr = np.random.randn(64, 64)
# 关键事实：NumPy 浮点数组默认 float64
print(f"NumPy 数组 dtype: {arr.dtype}")

# ---- 错误版本 ----
try:
    t = torch.from_numpy(arr).to("mps")  # fp64 张量搬上 MPS → 报错
except TypeError as e:
    print(f"\n[预期的报错] {e}\n")

# ---- 修复版本 ----
# 先降到 fp32 再上 MPS。注意顺序：在 CPU 上转 dtype，再搬设备。
t = torch.from_numpy(arr).float().to("mps")
result = t @ t
print(f"修复后: dtype={result.dtype}, device={result.device}, 结果和={result.sum().item():.4f}")

# 引申（第 2 章细讲）：torch.from_numpy 与源数组共享内存（零拷贝），
# 而 .float() 和 .to("mps") 各产生一次拷贝。数据管线里链式转换的
# 拷贝成本，是第 5 章 DataLoader 性能话题的伏笔。
