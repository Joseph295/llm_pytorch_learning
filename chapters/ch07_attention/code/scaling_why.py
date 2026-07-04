"""第 7 章 · √d 缩放的方差推导数字验证 + softmax 饱和/上溢现场

运行：uv run chapters/ch07_attention/code/scaling_why.py
"""

import torch

torch.manual_seed(0)

print("═══ 1. 点积方差 = d_k（推导的数字验证）═══")
print(f"{'d_k':>6} | {'q·k 实测方差':>12} | {'缩放后方差':>10}")
for d in [16, 64, 256, 1024]:
    q = torch.randn(10000, d)
    k = torch.randn(10000, d)
    dots = (q * k).sum(-1)
    print(f"{d:>6} | {dots.var().item():>12.1f} | {(dots / d**0.5).var().item():>10.3f}")
print("→ 方差恰等于 d_k；除以 √d_k 后回到 1（推导成立）")

print("\n═══ 2. softmax 饱和：大 logits 杀死梯度 ═══")
# 实验设计：softmax 输出接一个通用下游损失（随机线性组合），
# 测"穿过 softmax 回到 logits 的梯度"的平均范数。多次采样取均值。
T, TRIALS = 8, 2000
for scale, label in [(1.0, "缩放后 (std≈1)"), (16.0, "未缩放 (std≈16，d=256 的现实)")]:
    total, sat = 0.0, 0.0
    for _ in range(TRIALS):
        logits = (torch.randn(T) * scale).requires_grad_(True)
        p = logits.softmax(-1)
        (p * torch.randn(T)).sum().backward()      # 通用下游：随机方向的加权和
        total += logits.grad.norm().item()
        sat += p.max().item()
    print(f"  {label:<28} 平均最大权重={sat / TRIALS:.3f} 平均梯度范数={total / TRIALS:.2e}")
print("→ 饱和 softmax（输出≈one-hot）的雅可比 p(1-p)≈0：std=16 时梯度已衰减约 5 倍，")
print("  d 越大衰减越狠（试试把 scale 改成 32）。√d 缩放守住的就是这条梯度通路")

print("\n═══ 3. 手写 softmax 的上溢现场（减 max 的必要性）═══")
big = torch.tensor([30.0, 20.0, 10.0], dtype=torch.float16)
naive = big.exp() / big.exp().sum()                     # exp(30) 上溢 fp16（max 65504）
stable = (big - big.max()).exp() / (big - big.max()).exp().sum()
print(f"  朴素 softmax(fp16): {naive.tolist()}   ← exp(30)≈1e13 上溢成 inf，inf/inf=nan")
print(f"  减 max 后        : {[round(v, 4) for v in stable.tolist()]}   ← 数学等价，数值安全")
print("→ F.softmax 内部帮你减了 max；任何手写实现必须自己减（易错点/面试题）")

print("\n═══ 4. -inf mask 经过 softmax 变成精确的 0 ═══")
scores = torch.tensor([1.0, 2.0, float("-inf"), 0.5])
print(f"  logits {scores.tolist()}")
print(f"  softmax {[round(v, 4) for v in scores.softmax(-1).tolist()]} ← exp(-inf)=0，权重精确为零")
