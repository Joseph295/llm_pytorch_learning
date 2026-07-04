"""第 14 章 · 从零手写 LoRA，验证低秩适配 + 合并等价

运行：uv run chapters/ch14_finetuning/code/lora_from_scratch.py

LoRA 的全部魔法就在这个 LoRALinear 类里：冻结主干 W₀，只训低秩旁路 B·A。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


class LoRALinear(nn.Module):
    """h = W₀·x + (α/r)·B·A·x。W₀ 冻结，只训 A、B。"""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)                    # 冻结主干（不产生梯度/优化器状态）
        d_out, d_in = base.weight.shape
        self.A = nn.Parameter(torch.randn(r, d_in) * 0.01)   # A 随机小值
        self.B = nn.Parameter(torch.zeros(d_out, r))         # B 全 0 → 初始 ΔW=0（易错点②）
        self.scaling = alpha / r

    def forward(self, x):
        return self.base(x) + self.scaling * F.linear(F.linear(x, self.A), self.B)

    def merged_weight(self):
        """合并 B·A 进 W₀，得到单一等价权重（推理零开销，易错点⑥）。"""
        return self.base.weight + self.scaling * (self.B @ self.A)


# ═══ 1. 参数量对比 ═══
d = 512
base = nn.Linear(d, d, bias=False)
lora = LoRALinear(base, r=8, alpha=16)
full_params = d * d
trainable = sum(p.numel() for p in lora.parameters() if p.requires_grad)
print(f"全参微调可训练: {full_params:,} | LoRA(r=8) 可训练: {trainable:,} "
      f"（{trainable / full_params:.1%}，省 {1 - trainable / full_params:.1%}）")

# ═══ 2. 低秩适配能学会新任务吗 ═══
# 任务设计：目标 = 主干 + 一个"低秩"的权重改变（模拟微调的 ΔW 本身低秩，14.2-②）。
# 关键：ΔW 的秩(6) ≤ LoRA 的秩(8)，所以旁路容量足够捕捉它。
delta_rank = 6
U = torch.randn(d, delta_rank) * 0.1
V = torch.randn(delta_rank, d) * 0.1
X = torch.randn(256, d)
Y = (X @ (base.weight.detach() + U @ V).T).detach()      # 目标含一个秩=6 的 ΔW

opt = torch.optim.AdamW([p for p in lora.parameters() if p.requires_grad], lr=1e-2)
init_loss = F.mse_loss(lora(X), Y).item()
for _ in range(400):
    loss = F.mse_loss(lora(X), Y)
    opt.zero_grad(); loss.backward(); opt.step()
print(f"\n低秩适配任务(ΔW 秩={delta_rank} ≤ LoRA 秩=8): 初始 loss {init_loss:.3f} → 训练后 {loss.item():.5f}")
print("→ 只训 2rd 个参数（冻结 d² 主干），旁路捕捉到了低秩的 ΔW（低秩假设成立）")
print("  （若 ΔW 是满秩，r=8 旁路容量不足会欠拟合——见 lora_rank_sweep.py 的 r<真实秩 行）")

# ═══ 3. 合并等价性验证（推理零开销的依据）═══
x_test = torch.randn(10, d)
out_lora = lora(x_test)                                # 主干 + 旁路
merged = nn.Linear(d, d, bias=False)
with torch.no_grad():
    merged.weight.copy_(lora.merged_weight())
out_merged = merged(x_test)                            # 单一合并权重
print(f"\n合并后单权重前向 == 主干+旁路: {torch.allclose(out_lora, out_merged, atol=1e-5)} ✓")
print("→ 部署时合并 B·A 进 W₀，推理和原模型一样快（不像 adapter 增加延迟）")

# ═══ 4. B 初始化为 0 的重要性（易错点②）═══
lora2 = LoRALinear(nn.Linear(d, d, bias=False), r=8)
x = torch.randn(4, d)
diff = (lora2(x) - lora2.base(x)).abs().max().item()
print(f"\n初始时 LoRA 输出与原模型的差异: {diff:.2e}（B=0 → ΔW=0 → 不破坏预训练）")
