"""第 3 章 · 梯度累积的数学等价性验证

运行：uv run chapters/ch03_autograd/code/grad_accumulation.py

命题：batch=8 一次算的梯度 == batch=2 分 4 次累积（每次 loss/4）的梯度。
这是"小显存模拟大 batch"的正确性依据（第 6 章训练循环正式采用）。
"""

import torch

torch.manual_seed(42)

# 一个小线性模型 + 同一批数据
w = torch.randn(4, 1, requires_grad=True)
X = torch.randn(8, 4)
Y = torch.randn(8, 1)


def mse(pred, target):
    return ((pred - target) ** 2).mean()


# ── 方式 A：整批一次 ──
loss_full = mse(X @ w, Y)
loss_full.backward()
grad_full = w.grad.clone()          # clone！否则拿到的是会被后续覆盖的引用（第 2 章）
w.grad = None

# ── 方式 B：4 个 micro-batch 累积 ──
K = 4
for i in range(K):
    xb, yb = X[i * 2 : (i + 1) * 2], Y[i * 2 : (i + 1) * 2]
    loss_micro = mse(xb @ w, yb) / K      # 除以 K 是关键！
    loss_micro.backward()                 # 不清零 → .grad 自动累加
grad_accum = w.grad.clone()

print(f"整批梯度        : {grad_full.flatten().tolist()}")
print(f"micro-batch 累积: {[round(v, 6) for v in grad_accum.flatten().tolist()]}")
print(f"allclose: {torch.allclose(grad_full, grad_accum, atol=1e-6)} ✓")

print("""
为什么除以 K：
  整批 loss = mean(全部 8 个样本) = (1/8)Σℓᵢ
  micro loss 之和 = Σₖ mean(2 个样本) = Σₖ (1/2)Σℓ = (1/2)Σℓᵢ  ← 是整批的 4 倍
  每个 micro loss 除以 K=4 后，梯度之和才等于整批梯度。
  忘除 K = 等效学习率放大 K 倍，大 K 时直接发散——梯度累积的第一大坑。

注意：对 mean 型 loss 且 micro-batch 等大时上述成立；
     若样本数不均（如按 token 数聚合的 LM loss），要按加权比例折算——
     第 9 章 miniGPT 训练会遇到真实版本。
""")
