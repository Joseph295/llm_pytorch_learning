"""第 2 章 · 张量内部结构实证：storage / stride / 视图共享

运行：uv run chapters/ch02_tensor/code/tensor_internals.py
"""

import torch


def inspect(name: str, t: torch.Tensor):
    print(f"  {name:<22} shape={str(tuple(t.shape)):<12} stride={str(t.stride()):<12} "
          f"contig={str(t.is_contiguous()):<5} data_ptr={t.data_ptr() % 100000:>6}")
    # data_ptr 取模只是为了打印短一点；同尾数=同地址（本脚本范围内可这么读）


print("═══ 1. 定位公式：内存位置 = offset + Σ 索引×stride ═══")
t = torch.arange(6).reshape(2, 3)          # storage: [0,1,2,3,4,5]
print(f"t = {t.tolist()}, stride={t.stride()}")
i, j = 1, 2
pos = i * t.stride()[0] + j * t.stride()[1]
print(f"t[{i},{j}] 应在 storage 第 {pos} 位 → flatten()[{pos}] = {t.flatten()[pos].item()}，"
      f"与 t[{i},{j}] = {t[i, j].item()} 一致 ✓")

print("\n═══ 2. 转置只交换 stride，数据纹丝不动 ═══")
a = torch.arange(6.0).reshape(2, 3)
b = a.transpose(0, 1)
inspect("a (2,3)", a)
inspect("a.T (3,2)", b)
print(f"  同一 storage？ {a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()}")
print(f"  a 连续、b 不连续 ← stride (1,3) 不满足行优先递推")

print("\n═══ 3. 切片 = 改 offset/shape；花式索引 = 拷贝 ═══")
base = torch.arange(10.0)
sl = base[4:8]              # 基本索引：视图
fancy = base[[4, 5, 6, 7]]  # 花式索引：拷贝
inspect("base", base)
inspect("base[4:8]", sl)
inspect("base[[4,5,6,7]]", fancy)
print(f"  切片与 base 同 storage: {sl.untyped_storage().data_ptr() == base.untyped_storage().data_ptr()}"
      f"（data_ptr 差 {sl.data_ptr() - base.data_ptr()} 字节 = offset 4 元素 × 4 字节）")
print(f"  花式索引另起 storage : {fancy.untyped_storage().data_ptr() != base.untyped_storage().data_ptr()}")

print("\n═══ 4. 视图穿透：改视图 = 改本体 ═══")
sl.fill_(-1)
print(f"  对切片 fill_(-1) 后 base = {base.tolist()}   ← 4:8 被穿透")
fancy.fill_(-2)
print(f"  对花式索引 fill_(-2) 后 base = {base.tolist()}   ← 不受影响（它是拷贝）")

print("\n═══ 5. expand 的 stride=0 与写入灾难 ═══")
row = torch.zeros(3)
big = row.expand(4, 3)
inspect("row (3,)", row)
inspect("row.expand(4,3)", big)
try:
    big += 1                # 原地写入 stride=0 视图
except RuntimeError as e:
    print(f"  big += 1 → RuntimeError（torch 拦住了你）: {str(e)[:58]}...")
safe = row.expand(4, 3).clone()
safe += 1
print(f"  正确姿势 expand().clone() 后写入: row 仍为 {row.tolist()}")

print("\n═══ 6. contiguous()：非连续时物化，连续时 no-op ═══")
c = a.transpose(0, 1)
c2 = c.contiguous()
print(f"  转置后 contiguous(): 新 storage？ "
      f"{c2.untyped_storage().data_ptr() != c.untyped_storage().data_ptr()}，"
      f"stride {c.stride()} → {c2.stride()}")
d = a.contiguous()
print(f"  本就连续时 contiguous(): 返回自身？ {d is a}   ← no-op，零代价")
