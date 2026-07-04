"""第 3 章 · 自定义 autograd.Function 全流程：实现 + gradcheck 验证

运行：uv run chapters/ch03_autograd/code/custom_function.py

演示两个有代表性的自定义 Function：
  1. MyGELU —— 正常可导函数：手写导数 + 数值对拍
  2. RoundSTE —— 不可导函数的"直通估计器"（量化训练 QAT 的核心技巧，第 16 章）
"""

import math

import torch

# ═══ 1. MyGELU：手写前向 + 反向 ═══
SQRT2 = math.sqrt(2.0)
SQRT2PI = math.sqrt(2.0 * math.pi)


class MyGELU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)                 # 反向要用的中间量自己保存
        return 0.5 * x * (1 + torch.erf(x / SQRT2))

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        # d/dx [x·Φ(x)] = Φ(x) + x·φ(x)，Φ 是标准正态 CDF，φ 是 PDF
        cdf = 0.5 * (1 + torch.erf(x / SQRT2))
        pdf = torch.exp(-0.5 * x * x) / SQRT2PI
        return grad_out * (cdf + x * pdf)


x = torch.randn(20, dtype=torch.float64, requires_grad=True)  # gradcheck 必须用 float64！
ok = torch.autograd.gradcheck(MyGELU.apply, (x,), eps=1e-6, atol=1e-4)
print(f"MyGELU gradcheck（数值微分对拍，金标准）: {ok}")

y_mine = MyGELU.apply(x)
y_ref = torch.nn.functional.gelu(x)
print(f"与官方 F.gelu 前向一致: {torch.allclose(y_mine, y_ref, atol=1e-6)}")

g_mine = torch.autograd.grad(y_mine.sum(), x, retain_graph=True)[0]
x2 = x.detach().requires_grad_(True)
g_ref = torch.autograd.grad(torch.nn.functional.gelu(x2).sum(), x2)[0]
print(f"与官方 F.gelu 反向一致: {torch.allclose(g_mine, g_ref, atol=1e-6)}")

# ═══ 2. RoundSTE：给不可导运算"编"一个有用的梯度 ═══
class RoundSTE(torch.autograd.Function):
    """round() 的导数几乎处处为 0——真用它梯度全灭，网络学不动。

    直通估计器（Straight-Through Estimator）：前向真 round，
    反向假装它是恒等映射（梯度原样通过）。数学上"不对"，
    工程上让量化感知训练（QAT）成为可能——第 16 章的主角之一。
    """

    @staticmethod
    def forward(ctx, x):
        return torch.round(x)                    # 不用保存任何东西

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out                          # 直通！


w = torch.tensor([0.7, 1.2, 2.4], requires_grad=True)

naive = torch.round(w).sum()
naive.backward()
print(f"\n真 round 的梯度  : {w.grad.tolist()}   ← 全 0，参数永远不更新")

w.grad = None
ste = RoundSTE.apply(w).sum()
ste.backward()
print(f"STE round 的梯度 : {w.grad.tolist()}   ← 直通，量化训练得以进行")
