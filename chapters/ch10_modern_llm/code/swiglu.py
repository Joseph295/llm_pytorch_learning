"""第 10 章 · SwiGLU 替换 GELU FFN，参数量对齐验证

运行：uv run chapters/ch10_modern_llm/code/swiglu.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


class GELU_FFN(nn.Module):
    """miniGPT 的 FFN：两矩阵，中间 4d。"""

    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, 4 * d, bias=False)
        self.proj = nn.Linear(4 * d, d, bias=False)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))


class SwiGLU_FFN(nn.Module):
    """LLaMA 的 FFN：三矩阵（gate/up/down），中间 8/3·d 对齐参数量。"""

    def __init__(self, d):
        super().__init__()
        hidden = int(8 / 3 * d)
        hidden = 32 * ((hidden + 31) // 32)                # 向上取整到 32 的倍数（硬件友好）
        self.gate = nn.Linear(d, hidden, bias=False)
        self.up = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))   # 门控 ⊙ 内容（10.2-②）


d = 512
gelu, swiglu = GELU_FFN(d), SwiGLU_FFN(d)
n_gelu = sum(p.numel() for p in gelu.parameters())
n_swiglu = sum(p.numel() for p in swiglu.parameters())

print("═══ 参数量对齐验证 ═══")
print(f"GELU  FFN（2矩阵×4d）  : {n_gelu:>9,} 参数  = 2 × d × 4d = 8d²")
print(f"SwiGLU FFN（3矩阵×8/3d）: {n_swiglu:>9,} 参数  ≈ 3 × d × 8/3·d = 8d²")
print(f"比值: {n_swiglu / n_gelu:.2f}（应 ≈1.0，8/3 宽度正是为了对齐；取整造成微小偏差）")
print("→ LLaMA-7B 的 11008 = round(8/3 × 4096) 向上取整到硬件友好倍数")

print("\n═══ 门控在做什么 ═══")
x = torch.randn(2, 8, d)
gate_vals = F.silu(swiglu.gate(x))
print(f"门控值范围: [{gate_vals.min():.2f}, {gate_vals.max():.2f}]，均值 {gate_vals.mean():.2f}")
print("→ SiLU 门控是数据依赖的'软开关'：每个通道开多大由输入决定，")
print("  而 GELU 是固定曲线——这个乘法交互就是 SwiGLU 表达力的来源")

print("\n═══ 小任务收敛对比 ═══")
for name, ffn_cls in [("GELU", GELU_FFN), ("SwiGLU", SwiGLU_FFN)]:
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(64, d), ffn_cls(d), nn.Linear(d, 10))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    X, Y = torch.randn(512, 64), torch.randint(0, 10, (512,))
    for _ in range(300):
        loss = F.cross_entropy(model(X), Y)
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"  {name:<7} 300 步后 loss={loss.item():.4f}")
print("→ 同参数量下 SwiGLU 通常略优（小任务差距不大，规模化后更明显）")
