"""基础 1 参考答案：手推两层网络梯度并验证

运行：uv run chapters/ch03_autograd/exercises/solutions/basic_1.py

目标函数：z = w2 · relu(w1 · x)（全标量）

手推（链式法则）：
  设 h = w1·x, a = relu(h), z = w2·a
  dz/dw2 = a = relu(w1·x)
  dz/dw1 = w2 · relu'(h) · x，其中 relu'(h) = 1[h>0]（h≤0 时梯度整条路熄灭）
"""

import torch


def check(x_val: float):
    w1 = torch.tensor(2.0, requires_grad=True)
    w2 = torch.tensor(3.0, requires_grad=True)
    x = torch.tensor(x_val)

    z = w2 * torch.relu(w1 * x)
    z.backward()

    h = w1.item() * x_val
    manual_dw2 = max(h, 0.0)
    manual_dw1 = w2.item() * (1.0 if h > 0 else 0.0) * x_val

    print(f"x={x_val:+.1f}: autograd dw1={w1.grad.item():+.1f} dw2={w2.grad.item():+.1f} | "
          f"手推 dw1={manual_dw1:+.1f} dw2={manual_dw2:+.1f} | "
          f"一致={abs(w1.grad.item() - manual_dw1) < 1e-6 and abs(w2.grad.item() - manual_dw2) < 1e-6}")


check(1.5)    # h=3>0，relu 导通，两个梯度都非零
check(-1.5)   # h=-3<0，relu 关断 → dz/dw1 = dz/dw2 = 0：整条支路的梯度熄灭

print("""
观察：x<0 时 relu 掩码切断了梯度流——'死 ReLU' 问题的微观机制。
一个神经元若对所有输入都输出负值，它的上游参数永远得不到梯度，
这就是 LLM 激活函数演进（ReLU → GELU/SwiGLU，处处有微小梯度）的动机之一（第 10 章）。
""")
