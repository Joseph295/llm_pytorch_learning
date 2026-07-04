"""进阶 1 题目：这个训练循环埋了 5 个 bug。它能跑、不报错，但训练质量被暗中破坏。

运行：uv run chapters/ch06_training_loop/exercises/buggy_loop.py
任务：找齐 5 个 bug，修复，并写出每个 bug 的预期症状。答案在 solutions/advanced_1.md。
"""

import torch
import torch.nn as nn

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = nn.Sequential(nn.Linear(16, 64), nn.Dropout(0.2), nn.GELU(), nn.Linear(64, 1))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
model.to(device)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)


def get_batch():
    x = torch.randn(32, 16, device=device)
    return x, (x[:, :1] * 3).detach()


@torch.no_grad()
def evaluate():
    model.eval()
    total = 0.0
    for _ in range(5):
        x, y = get_batch()
        total += ((model(x) - y) ** 2).mean().item()
    return total / 5


for step in range(100):
    x, y = get_batch()
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()
    optimizer.step()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scheduler.step()
    if step % 20 == 0:
        val = evaluate()
        print(f"step {step:>3}: train={loss.item():.4f} val={val:.4f} "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

print("训练'完成'。它看起来在工作——但有 5 处暗伤。")
