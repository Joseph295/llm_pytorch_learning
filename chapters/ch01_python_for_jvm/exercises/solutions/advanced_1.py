"""进阶 1 参考答案：从零写符合 PyTorch 协议的 WindowedDataset

运行：uv run chapters/ch01_python_for_jvm/exercises/solutions/advanced_1.py

这就是第 9 章语言模型数据的形态：拿前 w 个 token，预测第 w+1 个。
"""

import torch
from torch.utils.data import DataLoader


class WindowedDataset:                      # 不继承任何类——鸭子类型
    def __init__(self, data: torch.Tensor, window: int):
        assert data.dim() == 1 and len(data) > window
        self.data, self.window = data, window

    def __len__(self):
        # 能切出的窗口数：最后一个窗口的"下一个元素"也要存在
        return len(self.data) - self.window

    def __getitem__(self, i):
        if i >= len(self):
            raise IndexError(i)             # 协议要求！否则 for 遍历不会停（第 1 章血案）
        x = self.data[i : i + self.window]  # 切片是视图，零拷贝（第 2 章）
        y = self.data[i + self.window]
        return x, y


ds = WindowedDataset(torch.arange(10.0), window=3)

# 三种访问方式验证
print(f"len(ds) = {len(ds)}")
print(f"ds[0]   = {ds[0]}")                 # (tensor([0,1,2]), tensor(3))
print(f"for 遍历: {[(x.tolist(), y.item()) for x, y in ds][:3]} ...")

# 为什么 DataLoader 能直接消费它：
# DataLoader 只按协议办事——用 len(ds) 确定采样范围，用 ds[i] 取样本，
# 再把一批样本用默认 collate 函数堆叠成 batch 张量。类型无关，协议满足即可。
loader = DataLoader(ds, batch_size=4, shuffle=True)
xb, yb = next(iter(loader))
print(f"\nDataLoader 直接消费: batch_x {tuple(xb.shape)}, batch_y {tuple(yb.shape)}")
print(f"batch_x[0]={xb[0].tolist()} -> y={yb[0].item()}  ← 窗口与标签对应关系保持正确")
