"""第 19 章 · 追踪算子的分发路径

运行：uv run chapters/ch19_pytorch_internals/code/dispatch_trace.py

展示同一个操作在不同张量属性（设备/是否需要梯度）下经过的 dispatch key。
让 19.2-① 的多维分发变成可见的。
"""

import torch

print("═══ 1. 张量携带的 dispatch key ═══")
cpu_t = torch.randn(3)
grad_t = torch.randn(3, requires_grad=True)
print(f"普通 CPU 张量的 key set: {torch._C._dispatch_key_set(cpu_t)}")
print(f"requires_grad 张量的 key set: {torch._C._dispatch_key_set(grad_t)}")
print("→ 两者的 key set 相同（都含 AutogradCPU 键）——Autograd 层始终在分发链上。")
print("  差别在运行时：requires_grad=True 时 Autograd 层才真正记录反向图，")
print("  否则该层直接 redispatch 透传（下一节看它对 grad_t 的实际效果）。")

print("\n═══ 2. autograd 是一个 dispatch 层（做完记录后 redispatch）═══")
a = torch.randn(3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
c = a + b
print(f"a+b 的 grad_fn: {c.grad_fn}")
print("分发过程（19.2-①）：")
print("  1. Autograd key（高优先级）：记录 AddBackward（第 3 章）→ redispatch")
print("  2. CPU key：调用真正的加法 kernel，执行")
print("→ autograd 不是硬编码在 add 里，而是 dispatcher 的可插拔层")

print("\n═══ 3. TorchDispatchMode：在 Python 层观察每个 ATen 调用 ═══")
from torch.utils._python_dispatch import TorchDispatchMode


class TraceMode(TorchDispatchMode):
    """拦截并打印所有经过的 ATen 算子（dispatcher 的 Python 探针）。"""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        print(f"  ATen 调用: {func}")
        return func(*args, **(kwargs or {}))


print("一个 x @ y + b 的完整 ATen 调用序列：")
x, y, bias = torch.randn(2, 3), torch.randn(3, 4), torch.randn(4)
with TraceMode():
    out = x @ y + bias
print("→ 你写的一行 Python，展开成多个 ATen 算子（每个都经过 dispatcher）")
print("  torch.compile（第 11/19 章）捕获的正是这个 ATen 算子图，再融合编译")

print("\n═══ 4. 不同 dtype/设备走不同 kernel ═══")
for t, desc in [(torch.randn(2, 2), "fp32 CPU"),
                (torch.randn(2, 2, dtype=torch.float16), "fp16 CPU")]:
    # 同一个 add，dispatcher 按 dtype 选对应 kernel
    r = t + t
    print(f"  {desc}: {t.dtype} + {t.dtype} → {r.dtype}（dispatcher 按 dtype 选 kernel）")
if torch.backends.mps.is_available():
    m = torch.randn(2, 2, device="mps")
    print(f"  MPS: 同样的 add，dispatcher 走 Metal kernel（{(m + m).device}）")
print("→ 同一个 torch.add，透明支持所有后端——这就是 dispatcher 的可扩展性价值")
