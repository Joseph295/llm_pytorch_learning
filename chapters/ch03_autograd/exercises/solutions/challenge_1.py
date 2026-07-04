"""挑战 1 参考答案：手写 micrograd 标量引擎 + XOR 训练 + 与 PyTorch 对拍

运行：uv run chapters/ch03_autograd/exercises/solutions/challenge_1.py

核心 60 行：Value 类 = 数值 + children + 局部导数回调。
backward = 拓扑排序 + 链式法则 + 梯度累加——PyTorch autograd 的全部骨架。
"""

import math

import torch


class Value:
    def __init__(self, data: float, children=()):
        self.data = data
        self.grad = 0.0                    # 累加语义：多路径求和
        self._children = children
        self._backward = lambda: None      # 本节点的局部反向：把梯度分发给 children

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))

        def _backward():                   # d(a+b)/da = 1, /db = 1
            self.grad += out.grad          # += 而不是 =：多消费者时求和！
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))

        def _backward():                   # d(ab)/da = b, /db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,))

        def _backward():                   # dtanh/dx = 1 - tanh²
            self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        """拓扑排序后逆序执行各节点的局部反向——loss.backward() 的骨架。"""
        topo, visited = [], set()

        def build(v):
            if id(v) not in visited:
                visited.add(id(v))
                for c in v._children:
                    build(c)
                topo.append(v)
        build(self)
        self.grad = 1.0                    # dL/dL = 1，反向的种子
        for node in reversed(topo):
            node._backward()

    # 让 -x、x-y、x+y 的反射形式都能用
    def __neg__(self):        return self * -1
    def __sub__(self, other): return self + (-other if isinstance(other, Value) else -Value(other))
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other


# ═══ 1. 与 PyTorch 对拍：同一表达式，两边求梯度 ═══
def expr_mine(a, b):
    return ((a * b + a).tanh() * b + a * a)          # 刻意让 a、b 多路复用

a_v, b_v = Value(0.7), Value(-1.3)
expr_mine(a_v, b_v).backward()

a_t = torch.tensor(0.7, requires_grad=True)
b_t = torch.tensor(-1.3, requires_grad=True)
((a_t * b_t + a_t).tanh() * b_t + a_t * a_t).backward()

da, db = abs(a_v.grad - a_t.grad.item()), abs(b_v.grad - b_t.grad.item())
print(f"梯度对拍: |Δa|={da:.2e} |Δb|={db:.2e}  {'✓ < 1e-6' if max(da, db) < 1e-6 else '✗'}")

# ═══ 2. 用自己的引擎训练 2-4-1 网络拟合 XOR ═══
import random

random.seed(7)


def make_param():
    return Value(random.uniform(-1, 1))


W1 = [[make_param() for _ in range(2)] for _ in range(4)]   # 4×2
B1 = [make_param() for _ in range(4)]
W2 = [make_param() for _ in range(4)]                       # 1×4
B2 = make_param()
params = [p for row in W1 for p in row] + B1 + W2 + [B2]


def forward(x1, x2):
    hidden = [(W1[j][0] * x1 + W1[j][1] * x2 + B1[j]).tanh() for j in range(4)]
    out = B2
    for j in range(4):
        out = out + W2[j] * hidden[j]
    return out.tanh()


data = [((0, 0), -1), ((0, 1), 1), ((1, 0), 1), ((1, 1), -1)]  # XOR，标签用 ±1 配 tanh

for epoch in range(300):
    loss = Value(0.0)
    for (x1, x2), y in data:
        diff = forward(x1, x2) - y
        loss = loss + diff * diff
    for p in params:
        p.grad = 0.0                       # zero_grad！累加语义的必然要求
    loss.backward()
    for p in params:
        p.data -= 0.1 * p.grad             # 裸 SGD（对照第 3 章易错点④：改本体不重绑定）
    if epoch % 100 == 0 or epoch == 299:
        print(f"epoch {epoch:>3}: loss = {loss.data:.4f}")

print("\nXOR 预测:")
for (x1, x2), y in data:
    print(f"  ({x1},{x2}) -> {forward(x1, x2).data:+.3f}  (目标 {y:+d})")

print("""
写完这 100 行你已经理解了 autograd 的全部机制要点：
  1. 前向时每个运算记录 children + 局部导数闭包（≈ grad_fn + saved_tensors）
  2. backward = 拓扑逆序执行（≈ 引擎的图遍历）
  3. grad 用 += 累加（多消费者求和 → 顺带支持梯度累积）
  4. 训练前手动清零（≈ optimizer.zero_grad）
  5. 参数更新改 .data 本体、不重新绑定（≈ 易错点④与 optimizer.step 的实现）
PyTorch 与此的差距是工程量而非思想：张量化、C++ 引擎、并行调度、版本计数等安全网。
""")
