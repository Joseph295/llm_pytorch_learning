"""挑战 1 参考答案：优化器收敛对比（特征尺度不均 = Adam 系的主场）

运行：uv run chapters/ch06_training_loop/exercises/solutions/challenge_1.py

实验设计：二分类小 MLP，输入特征尺度故意做成 [1, 1, 100, 100]——
重现讲义 6.1"病 2"（不同参数需要的步长天差地别）。
四种配置 × 3 种子，比较固定步数后的 loss。
"""

import math

import torch
import torch.nn as nn


def make_data(n=512, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 4, generator=g)
    x[:, 2:] *= 100.0                              # 尺度不均：后两维放大 100 倍
    w_true = torch.tensor([1.5, -2.0, 0.02, -0.01])
    y = ((x @ w_true) > 0).float().unsqueeze(1)
    return x, y


def train(opt_name: str, seed: int, steps=300) -> float:
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(4, 32), nn.Tanh(), nn.Linear(32, 1))
    x, y = make_data(seed=seed)
    loss_fn = nn.BCEWithLogitsLoss()

    if opt_name == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=1e-3)
        sched = None
    elif opt_name == "sgd_momentum":
        opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
        sched = None
    elif opt_name == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sched = None
    else:  # adamw_sched
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        warmup = 30

        def lam(s):
            if s < warmup:
                return (s + 1) / warmup
            p = (s - warmup) / (steps - warmup)
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lam)

    for _ in range(steps):
        loss = loss_fn(model(x), y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if sched:
            sched.step()
    return loss_fn(model(x), y).item()


print(f"{'配置':<16} | {'3 种子平均 loss':>14} | 各种子")
print("-" * 58)
for name in ["sgd", "sgd_momentum", "adamw", "adamw_sched"]:
    losses = [train(name, seed) for seed in [0, 1, 2]]
    avg = sum(losses) / 3
    print(f"{name:<16} | {avg:>14.4f} | {[round(l, 4) for l in losses]}")

print("""
预期结论（你的数字会有波动，但排序应稳定）：
1. 裸 SGD 在尺度不均的输入上几乎训不动：大尺度特征对应的权重梯度大，
   lr 迁就它们就照顾不了小尺度特征——"病 2"的临床表现。
2. momentum 有帮助但治不了本（它平滑方向，不解决逐参数步长）。
3. AdamW 的每参数自适应缩放直接对症，快一个数量级。
4. warmup+cosine 再叠加：允许更高的峰值 lr（3e-3 vs 1e-3）而不炸，
   末期小步精修，最终 loss 最低——这就是 LLM 全家桶配置的缩影。
附注：若把特征归一化（尺度均匀），SGD 与 Adam 的差距会大幅缩小——
"优化器的先进性"很多时候是在替糟糕的数据条件买单；两头都要做好。
""")
