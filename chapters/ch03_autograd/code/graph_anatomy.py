"""第 3 章 · 计算图解剖：grad_fn 链、叶子行为、三种关图方式

运行：uv run chapters/ch03_autograd/code/graph_anatomy.py
"""

import torch

print("═══ 1. 图是前向的副产品 ═══")
x = torch.tensor([2.0], requires_grad=True)
y = x * 3
z = y**2
loss = z.sum()
print(f"x: is_leaf={x.is_leaf}, grad_fn={x.grad_fn}")
print(f"y: is_leaf={y.is_leaf}, grad_fn={type(y.grad_fn).__name__}")
print(f"z: grad_fn={type(z.grad_fn).__name__}")
print(f"loss.grad_fn.next_functions = {loss.grad_fn.next_functions}")
print("  ← 反向图就是这条 grad_fn 链，终点 AccumulateGrad = 往叶子 .grad 累加的节点")


def print_graph(fn, depth=0):
    """沿 next_functions 递归打印反向图（进阶练习 1 的雏形）。"""
    if fn is None:
        return
    print("  " * depth + f"└─ {type(fn).__name__}")
    for child, _ in getattr(fn, "next_functions", []):
        print_graph(child, depth + 1)


print("\n整张反向图：")
print_graph(loss.grad_fn)

print("\n═══ 2. 中间节点默认不留 .grad ═══")
loss.backward()
print(f"x.grad = {x.grad}   ← 叶子，dz/dx = 2·(3x)·3 = 18x = 36 ✓")
print(f"y.grad = {y.grad}   ← 中间节点默认 None（会有 UserWarning）")

x.grad = None                       # 清掉重来
y2 = x * 3
y2.retain_grad()                    # 显式要求保留
(y2**2).sum().backward()
print(f"retain_grad 后 y2.grad = {y2.grad}（= 2y = 12）")

print("\n═══ 3. 梯度累加：多路径求和 + 跨 backward 累积 ═══")
w = torch.tensor([1.0], requires_grad=True)
out = w * 2 + w * 3                 # w 有两条消费路径
out.backward()
print(f"双路径梯度 = {w.grad.item()}（2+3，多元链式法则的和）")
(w * 10).backward()                 # 不清零再来一次
print(f"再 backward(w*10) 不清零 → {w.grad.item()}（5+10 累积，梯度累积训练的基础）")

print("\n═══ 4. 三种关图方式 ═══")
a = torch.tensor([1.0], requires_grad=True)
with torch.no_grad():
    b = a * 2
print(f"no_grad 内运算: b.grad_fn={b.grad_fn}, requires_grad={b.requires_grad}")
c = (a * 2).detach()
print(f"detach: 共享数据={c.data_ptr() == (a * 2).data_ptr() or '（新运算新内存）'}, "
      f"requires_grad={c.requires_grad}")
with torch.inference_mode():
    d = a * 2
print(f"inference_mode: requires_grad={d.requires_grad}", end="")
try:
    d.requires_grad_(True)
except RuntimeError as e:
    print(f"，且永远进不了图: {str(e)[:46]}...")

print("\n═══ 5. 版本计数器：原地操作的哨兵 ═══")
e = torch.tensor([1.0, 2.0], requires_grad=True)
f = e.exp()                         # exp 的反向公式依赖输出 f
print(f"f._version = {f._version}")
f.add_(1)                           # 原地改 f
print(f"f.add_(1) 后 _version = {f._version}")
try:
    f.sum().backward()
except RuntimeError as err:
    print(f"backward → RuntimeError: {str(err)[:80]}...")
    print("  ← 宁可报错，不给错梯度（对比 dtype 提升的静默，体会设计取舍）")
