"""挑战 1 参考答案：as_strided 零拷贝滑动窗口

运行：uv run chapters/ch02_tensor/exercises/solutions/challenge_1.py

as_strided 直接指定 (shape, stride) 构造视图——最锋利也最危险的
张量工具（越界不检查！），理解它 = 彻底理解 stride。
"""

import torch

data = torch.arange(10.0)
W = 3
n_windows = len(data) - W + 1            # 8 个窗口

# 核心：窗口矩阵 (8, 3)，行间步长 1（窗口起点每次右移 1），行内步长 1
windows = data.as_strided(size=(n_windows, W), stride=(1, 1))
print("滑动窗口矩阵：")
print(windows)

# ── 零拷贝证明 ──
print(f"\n同 storage: {windows.untyped_storage().data_ptr() == data.untyped_storage().data_ptr()}")
data[4] = 99.0
hits = (windows == 99.0).sum().item()
print(f"改 data[4]=99 后，窗口矩阵中出现 99 的位置数 = {hits}")
print("→ 是 3 处：元素 4 同时出现在窗口 2/3/4（它们的 stride 映射都指向同一物理位置）")
print(windows)

# ── 对 WindowedDataset 的启示 ──
print("""
用于第 1 章 WindowedDataset：
    xs = data.as_strided((N, w), (1, 1))     # 全部输入窗口，一次性零拷贝
    ys = data[w:]                            # 全部标签，切片零拷贝
  好处：__getitem__ 退化为 xs[i]（一次索引），且整体可直接喂给张量化流水线。
代价 / 风险：
  1. as_strided 不做越界检查——shape/stride 算错会读到脏内存（静默错误数据！）
     生产代码优先用封装好的 tensor.unfold(0, w, 1)，语义相同且安全。
  2. 窗口间共享内存：任何窗口的原地写都会穿透到其他窗口——只读使用。
  3. 若下游 DataLoader 需要独立样本（如 pin_memory 传输），可能触发隐式拷贝，
     零拷贝的收益在边界处被吃掉——优化前先测（第 5/11 章的 profiling 习惯）。
""")

# 附：安全版等价写法
unfolded = torch.arange(10.0).unfold(0, W, 1)
print(f"安全版 unfold(0, {W}, 1) 结果一致: {torch.equal(unfolded, torch.arange(10.0).as_strided((8, 3), (1, 1)))}")
