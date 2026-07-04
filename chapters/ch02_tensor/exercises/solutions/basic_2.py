"""基础 2 参考答案：data_ptr 指针取证

运行：uv run chapters/ch02_tensor/exercises/solutions/basic_2.py

data_ptr()               = 该张量第一个元素的内存地址（含 offset）
untyped_storage().data_ptr() = 底层 storage 起点的地址
两者配合可以取证：共享 storage？offset 多少？
"""

import torch

base = torch.arange(10.0)
sl = base[3:7]          # 切片：视图
cl = base.clone()       # 克隆：全新内存
fx = base[[3]]          # 花式索引：拷贝

sp = lambda t: t.untyped_storage().data_ptr()

print(f"切片   : 同 storage={sp(sl) == sp(base)}, "
      f"data_ptr 偏移={sl.data_ptr() - base.data_ptr()} 字节（=3 元素×4 字节，offset 的体现）")
print(f"clone  : 同 storage={sp(cl) == sp(base)}, data_ptr 也不同={cl.data_ptr() != base.data_ptr()}")
print(f"花式索引: 同 storage={sp(fx) == sp(base)} ← 无法用等距 stride 描述，只能物化")
