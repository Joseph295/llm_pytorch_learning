"""第 6 章 · 学习率调度曲线 + 梯度裁剪救 spike 演示

运行：uv run chapters/ch06_training_loop/code/lr_and_clip.py
"""

import math

import torch
import torch.nn as nn

# ═══ 1. warmup + cosine：LLM 标配形状 ═══


def get_lr(step: int, warmup: int, total: int, peak: float, floor_ratio: float = 0.1) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup                       # 线性热身
    progress = (step - warmup) / max(1, total - warmup)         # 0→1
    cos = 0.5 * (1 + math.cos(math.pi * progress))              # 1→0
    floor = peak * floor_ratio
    return floor + (peak - floor) * cos


WARMUP, TOTAL, PEAK = 200, 2000, 3e-4
print("warmup+cosine 曲线（ASCII 采样）：")
for s in [0, 50, 100, 199, 200, 500, 1000, 1500, 1999]:
    lr = get_lr(s, WARMUP, TOTAL, PEAK)
    bar = "█" * int(lr / PEAK * 40)
    stage = "warmup" if s < WARMUP else "cosine"
    print(f"  step {s:>4} [{stage:>6}] lr={lr:.2e} {bar}")
print(f"  终点值 = 峰值×10% = {PEAK * 0.1:.1e}（不衰到 0 是惯例）")

# ═══ 2. 梯度裁剪：单步事故的保险丝 ═══
print("\n═══ 裁剪对'毒 batch'的防护实验 ═══")


def train(clip: bool, seed: int = 0):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(8, 32), nn.Tanh(), nn.Linear(32, 1))
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    history = []
    for step in range(60):
        x, y = torch.randn(16, 8), torch.randn(16, 1)
        if step == 30:                                   # 毒 batch：目标值放大 300 倍
            y = y * 300
        loss = ((model(x) - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0 if clip else float("inf"))
        opt.step()
        history.append((loss.item(), gnorm.item()))
    return history


for clip in [False, True]:
    h = train(clip)
    tail = [round(l, 2) for l, _ in h[-3:]]
    spike_norm = h[30][1]
    print(f"  clip={str(clip):<5} 毒 batch 处梯度范数={spike_norm:8.1f} → 最后 3 步 loss={tail}")
print("→ 不裁剪：一个毒 batch 的大梯度把参数踢飞，之后 loss 回不来（甚至 nan）")
print("→ 裁剪  ：保方向限步长，单步事故不扩大；同时 gnorm 是免费的健康监控信号")
