"""第 6 章 · 优化器解剖：手写 SGD/AdamW 对拍官方 + state 显存实测

运行：uv run chapters/ch06_training_loop/code/optimizer_anatomy.py
"""

import copy

import torch
import torch.nn as nn


def clone_model():
    torch.manual_seed(7)
    return nn.Sequential(nn.Linear(16, 32), nn.Tanh(), nn.Linear(32, 4))


def run_steps(model, opt_step, n=10):
    """跑 n 步固定数据的训练，opt_step(model) 负责更新。"""
    torch.manual_seed(123)
    for _ in range(n):
        x, y = torch.randn(8, 16), torch.randn(8, 4)
        loss = ((model(x) - y) ** 2).mean()
        model.zero_grad(set_to_none=True)
        loss.backward()
        opt_step(model)
    return [p.detach().clone() for p in model.parameters()]


# ═══ 1. 手写 AdamW 对拍官方 ═══
LR, B1, B2, EPS, WD = 1e-3, 0.9, 0.95, 1e-8, 0.1
my_state: dict = {}


def my_adamw_step(model):
    with torch.no_grad():                                    # 更新不建图（第 3 章易错点④）
        for i, p in enumerate(model.parameters()):
            if p.grad is None:
                continue
            st = my_state.setdefault(i, {"t": 0, "m": torch.zeros_like(p), "v": torch.zeros_like(p)})
            st["t"] += 1
            # ① 解耦 weight decay：直接作用于权重，不进梯度（AdamW 的全部秘密）
            p.mul_(1 - LR * WD)
            # ② Adam 主体
            st["m"].mul_(B1).add_(p.grad, alpha=1 - B1)      # m = β1·m + (1-β1)·g
            st["v"].mul_(B2).addcmul_(p.grad, p.grad, value=1 - B2)
            m_hat = st["m"] / (1 - B1 ** st["t"])            # 偏差修正
            v_hat = st["v"] / (1 - B2 ** st["t"])
            p.addcdiv_(m_hat, v_hat.sqrt().add_(EPS), value=-LR)


m1 = clone_model()
mine = run_steps(m1, my_adamw_step)

m2 = clone_model()
official = torch.optim.AdamW(m2.parameters(), lr=LR, betas=(B1, B2), eps=EPS, weight_decay=WD)
ref = run_steps(m2, lambda _: official.step())

ok = all(torch.allclose(a, b, atol=1e-6) for a, b in zip(mine, ref))
print(f"手写 AdamW 10 步后与官方参数 allclose: {ok} ✓")

# ═══ 2. state 的惰性创建与显存账 ═══
if torch.backends.mps.is_available():
    dev = "mps"
    big = nn.Linear(2048, 2048, bias=False).to(dev)          # 4M 参数
    opt = torch.optim.AdamW(big.parameters(), lr=1e-3)
    before = torch.mps.current_allocated_memory()
    loss = big(torch.randn(8, 2048, device=dev)).sum()
    loss.backward()
    after_bwd = torch.mps.current_allocated_memory()
    opt.step()                                               # ← state 在这一刻诞生
    after_step = torch.mps.current_allocated_memory()
    n = 2048 * 2048
    print(f"\nbackward 后新增: {(after_bwd - before) / 1024**2:6.1f} MB（梯度 ≈ {n * 4 / 1024**2:.0f} MB）")
    print(f"第一次 step 新增: {(after_step - after_bwd) / 1024**2:6.1f} MB"
          f"（m+v = {2 * n * 4 / 1024**2:.0f} MB —— 8 字节/参数的账，第 2 章的理论在此落地）")
    print("→ OOM 若发生在'第一次 step'而非前向，嫌疑人就是优化器状态")

# ═══ 3. param_groups：decay/no_decay 分组（nanoGPT 模式）═══
model = nn.Sequential(nn.Linear(16, 32), nn.LayerNorm(32), nn.Linear(32, 4))
decay = [p for p in model.parameters() if p.dim() >= 2]      # 矩阵/embedding
no_decay = [p for p in model.parameters() if p.dim() < 2]    # bias、norm 的 scale/shift
opt = torch.optim.AdamW(
    [{"params": decay, "weight_decay": 0.1},
     {"params": no_decay, "weight_decay": 0.0}],
    lr=3e-4,
)
print(f"\ndecay 组 {sum(p.numel() for p in decay)} 参数 | no_decay 组 {sum(p.numel() for p in no_decay)} 参数")
print("→ bias/norm 不做 weight decay 是 LLM 惯例（易错点①）；分组同时是分层 lr 的机制")
