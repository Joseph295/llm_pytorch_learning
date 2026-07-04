"""第 14 章 · LoRA 秩 r 的权衡扫描：参数量 vs 拟合能力

运行：uv run chapters/ch14_finetuning/code/lora_rank_sweep.py

秩 r 越大：可训练参数越多、拟合能力越强，但越接近全参微调（失去省参优势）。
本脚本扫描 r，量化这个权衡。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

D = 512


class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha=None):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        alpha = alpha or 2 * r
        self.A = nn.Parameter(torch.randn(r, D) * 0.01)
        self.B = nn.Parameter(torch.zeros(D, r))
        self.scaling = alpha / r

    def forward(self, x):
        return self.base(x) + self.scaling * F.linear(F.linear(x, self.A), self.B)


# 目标任务：拟合一个"真实" ΔW（本身是某个秩，测 LoRA 能否捕捉）
base = nn.Linear(D, D, bias=False)
true_rank = 16
U = torch.randn(D, true_rank) * 0.1
V = torch.randn(true_rank, D) * 0.1
target_weight = base.weight.detach() + U @ V             # 真实改变是秩 16
X = torch.randn(512, D)
Y = (X @ target_weight.T).detach()

print(f"任务：拟合一个秩={true_rank} 的权重改变 ΔW\n")
print(f"{'秩 r':>5} | {'可训练参数':>10} | {'占全参':>7} | {'最终 loss':>10}")
print("-" * 44)
full = D * D
for r in [1, 4, 8, 16, 32, 64]:
    torch.manual_seed(1)
    b = nn.Linear(D, D, bias=False)
    with torch.no_grad():
        b.weight.copy_(base.weight)
    lora = LoRALinear(b, r=r)
    opt = torch.optim.AdamW([p for p in lora.parameters() if p.requires_grad], lr=5e-3)
    for _ in range(400):
        loss = F.mse_loss(lora(X), Y)
        opt.zero_grad(); loss.backward(); opt.step()
    n = sum(p.numel() for p in lora.parameters() if p.requires_grad)
    print(f"{r:>5} | {n:>10,} | {n / full:>6.1%} | {loss.item():>10.5f}")

print(f"""
读数（14.2-② / 易错点③）：
- r < 真实秩({true_rank})：欠拟合，loss 高（低秩旁路容量不足以捕捉 ΔW）
- r ≥ 真实秩：loss 骤降到接近 0（旁路容量足够）
- r 继续增大：loss 不再改善，但参数量线性增长（浪费）
- 实践启示：r 要匹配任务"内在复杂度"。从 r=8/16 起步，欠拟合再加大。
  真实任务的 ΔW 秩未知，靠验证集扫描选 r（本图的家庭作坊版）。
LoRA 的甜点通常 r=8~64；q/v 投影加 LoRA 是最省的起点（易错点③）。
""")
