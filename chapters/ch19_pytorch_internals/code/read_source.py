"""第 19 章 · 从 Python API 追到 ATen：读 PyTorch 源码的方法

运行：uv run chapters/ch19_pytorch_internals/code/read_source.py

演示"怎么读 PyTorch 源码"（19.2-② / 面试常考）：从一个算子出发，
用 torch 自省工具追踪它的实现位置。
"""

import torch

print("═══ 读 PyTorch 源码的方法（19.2-②）═══\n")

print("① 从 Python API 找到底层函数名")
print(f"  torch.softmax 是: {torch.softmax}")
print(f"  它的 __module__: {torch.softmax.__module__}")

print("\n② 用 __torch_dispatch__ 看它实际调用的 ATen 算子")
from torch.utils._python_dispatch import TorchDispatchMode


class WhichAten(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        print(f"  → ATen 算子: {func._schema}")     # 打印算子的完整 schema（签名）
        return func(*args, **(kwargs or {}))


x = torch.randn(3, 4)
with WhichAten():
    torch.softmax(x, dim=-1)

print("""
③ 有了 ATen 算子名（如 aten::_softmax），在源码里定位实现：
   - aten/src/ATen/native/native_functions.yaml  搜 '_softmax'
     → 找到定义 + 各 dispatch key 的实现函数名
   - aten/src/ATen/native/SoftMax.cpp  (CPU kernel)
   - aten/src/ATen/native/cuda/SoftMax.cu  (CUDA kernel)
   - tools/autograd/derivatives.yaml  搜 '_softmax'  (反向公式，第 3 章)

④ schema 告诉你一切：参数类型、别名标注（哪些参数会被原地修改）、
   返回值——这是算子的"契约"。

方法论价值：文档不清时，读源码比试错快且权威。面试问"你怎么读 PyTorch 源码"，
答案就是这条路径：Python API → __torch_dispatch__ 看 ATen 算子 → native_functions.yaml
→ native/ 的 kernel → derivatives.yaml 的反向。
""")

print("═══ 附：算子 schema 的信息量 ═══")
# schema 里的 '(a!)' 表示该参数会被原地修改（别名标注，第 2/3 章原地操作的底层）
print(f"  add_ 的 schema（原地版）: {torch.ops.aten.add_.Tensor._schema}")
print("  注意 self 参数的 '(a!)' 标注——表示它会被原地修改（第 3 章版本计数器的依据）")
