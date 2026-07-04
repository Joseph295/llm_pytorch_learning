# 基础 1 参考答案：手算 stride

起点：`t = torch.arange(24).reshape(2, 3, 4)`，shape `(2,3,4)`，stride `(12,4,1)`。

**a) `t.transpose(0, 2)`** → shape `(4,3,2)`，stride `(1,4,12)`（交换第 0、2 维的 shape 和 stride）。**非 contiguous**（stride 不是递减到 1 的行优先形态）。

**b) `t[1]`** → shape `(3,4)`，stride `(4,1)`，offset 前移 12 个元素（1×stride₀）。**contiguous**（切掉最高维不破坏连续性）。

**c) `t[:, 1:, :]`** → shape `(2,2,4)`，stride **保持 `(12,4,1)`**，offset +4。**非 contiguous**！中间维被切后，行优先递推要求 stride₀ = 2×4=8，实际仍是 12（物理上每个"大块"仍隔 12 个元素）。这是"切片看着无害却破坏连续性"的典型。

**d) `t.permute(2, 0, 1)`** → shape `(4,2,3)`，stride `(1,12,4)`（按 (2,0,1) 重排原 stride）。**非 contiguous**。

**e) `t.unsqueeze(1)`** → shape `(2,1,3,4)`，stride `(12,12,4,1)`（新维度 stride 可为任意值，因为该维长度 1 坐标恒 0，torch 给 12）。**contiguous**（长度 1 的维不影响连续性判定）。

验证代码：

```python
import torch
t = torch.arange(24).reshape(2, 3, 4)
for name, u in {
    "transpose(0,2)": t.transpose(0, 2), "t[1]": t[1], "t[:,1:,:]": t[:, 1:, :],
    "permute(2,0,1)": t.permute(2, 0, 1), "unsqueeze(1)": t.unsqueeze(1),
}.items():
    print(f"{name:16s} shape={tuple(u.shape)} stride={u.stride()} contig={u.is_contiguous()}")
```

**心法**：transpose/permute = 重排 stride；切最高维 = 挪 offset；切中间维 = stride 不变但连续性存疑（用递推式 `stride[i] == shape[i+1]*stride[i+1]` 逐维检查）。
