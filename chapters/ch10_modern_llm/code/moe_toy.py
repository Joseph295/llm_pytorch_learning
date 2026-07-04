"""第 10 章 · 玩具 MoE：路由、负载失衡、均衡损失

运行：uv run chapters/ch10_modern_llm/code/moe_toy.py

亲眼看 MoE 的核心张力：不加约束 router 会"赢者通吃"，均衡损失把它掰平。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


class MoE(nn.Module):
    def __init__(self, d, n_experts=8, top_k=2):
        super().__init__()
        self.n_experts, self.top_k = n_experts, top_k
        self.router = nn.Linear(d, n_experts, bias=False)
        self.experts = nn.ModuleList(
            nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
            for _ in range(n_experts)
        )

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.view(-1, D)                              # (B*T, D)，每个 token 独立路由
        logits = self.router(x_flat)                       # (N, n_experts)
        probs = logits.softmax(-1)
        topk_p, topk_i = probs.topk(self.top_k, dim=-1)    # 每 token 选 top-k 专家
        topk_p = topk_p / topk_p.sum(-1, keepdim=True)     # 重新归一化

        out = torch.zeros_like(x_flat)
        # 统计每个专家的负载（token 计数）
        load = torch.zeros(self.n_experts)
        for k in range(self.top_k):
            for e in range(self.n_experts):
                mask = topk_i[:, k] == e
                if mask.any():
                    out[mask] += topk_p[mask, k : k + 1] * self.experts[e](x_flat[mask])
                    load[e] += mask.sum().item()

        # 负载均衡辅助损失（Switch Transformer 式）：
        # 鼓励"路由概率质量"和"实际分派比例"都均匀
        frac_tokens = load / load.sum()                    # 各专家实际 token 占比
        frac_prob = probs.mean(0)                          # 各专家平均路由概率
        aux_loss = self.n_experts * (frac_tokens * frac_prob).sum()
        return out.view(B, T, D), load, aux_loss


d, N = 64, 8
moe = MoE(d, n_experts=N, top_k=2)
x = torch.randn(4, 32, d)

print("═══ 初始路由的负载分布（未训练）═══")
_, load, aux = moe(x)
frac = (load / load.sum() * 100).round().int().tolist()
print(f"各专家 token 占比: {frac} %（理想均匀 = {100 // N}% 左右）")
print(f"辅助均衡损失: {aux.item():.3f}（越接近 1.0 越均衡，越高越失衡）")

print("\n═══ 对比训练：有/无均衡损失，各专家最终负载 ═══")


def train(use_aux: bool, steps=200):
    torch.manual_seed(1)
    m = MoE(d, n_experts=N, top_k=2)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    X = torch.randn(8, 32, d)
    Y = torch.randn(8, 32, d)
    for _ in range(steps):
        out, load, aux = m(X)
        loss = F.mse_loss(out, Y) + (0.1 * aux if use_aux else 0.0)
        opt.zero_grad(); loss.backward(); opt.step()
    _, final_load, _ = m(X)
    return (final_load / final_load.sum() * 100).round().int().tolist()


no_aux = train(use_aux=False)
with_aux = train(use_aux=True)
print(f"无均衡损失: {no_aux} %  ← 分布开始倾斜（易错点③）")
print(f"有均衡损失: {with_aux} %  ← 明显更均匀，全部专家都被利用")
print(f"\n失衡度（max-min）: 无 {max(no_aux) - min(no_aux)}% vs 有 {max(with_aux) - min(with_aux)}%")
print("→ 本玩具任务（随机回归、8 专家）失衡温和，但方向清楚：均衡损失把分布掰平。")
print("  真实预训练里，router 对语言模式高度敏感，不加约束会演化成严重的赢者通吃")
print("  （少数专家占 80%+ token），等效参数量在训练中悄悄缩水")
print("  （注：本玩具用 Python 循环遍历专家，真实 MoE 用分组 gather/scatter；")
print("   多卡时 token 路由产生 all-to-all 通信，是第 13 章专家并行的主题）")
