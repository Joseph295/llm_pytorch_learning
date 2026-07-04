"""第 6 章 · bf16 混合精度实测 + 完整六步训练模板

运行：uv run chapters/ch06_training_loop/code/amp_training.py

三个实验：
  1. MPS 上 bf16 autocast 的吞吐对比
  2. "没有 fp32 主参数时小更新被吞"的数字演示
  3. 完整六步训练循环模板（第 9 章的骨架）跑通
"""

import time

import torch
import torch.nn as nn

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ═══ 1. bf16 autocast 吞吐实测 ═══
model = nn.Sequential(*[nn.Linear(1024, 1024) for _ in range(8)]).to(device)
x = torch.randn(256, 1024, device=device)


def bench(use_amp: bool, iters: int = 30) -> float:
    def step():
        if use_amp:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                out = model(x)
        else:
            out = model(x)
        return out
    step()
    torch.mps.synchronize() if device.type == "mps" else None
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    torch.mps.synchronize() if device.type == "mps" else None
    return (time.perf_counter() - t0) * 1000 / iters


t32, t16 = bench(False), bench(True)
print(f"前向耗时: fp32 {t32:.2f} ms | bf16 autocast {t16:.2f} ms | 比值 {t32 / t16:.2f}x")
print("""诚实解读（M4 实测常在 1.0x 上下，别惊讶）：
  - MPS 上 autocast 有 cast 开销，且 Apple GPU 对 bf16 matmul 没有专用加速单元，
    小中型负载常打平甚至略慢——bf16 在本机的主要收益是"激活显存减半"。
  - 云端 A100/H100 的 Tensor Core 对 bf16 有数倍吞吐，同一段代码收益立现（第 11 章实测）。
  - 教训：AMP 的收益是硬件×负载的函数，上生产前永远实测，不背结论。""")

# ═══ 2. 更新被吞：为什么需要 fp32 主参数 ═══
print("\n═══ 小更新在 half 精度下被吞 ═══")
w32 = torch.tensor(1.0, dtype=torch.float32)
w16 = torch.tensor(1.0, dtype=torch.bfloat16)
tiny = 1e-4                                     # 训练后期的典型更新量级
for _ in range(100):
    w32 += tiny
    w16 += tiny
print(f"fp32 累加 100 次 1e-4: {w32.item():.4f}（正确 1.01）")
print(f"bf16 累加 100 次 1e-4: {w16.item():.4f} ← 1.0+0.0001 每次都被舍回 1.0，更新全吞")
print("→ 这就是 fp32 主参数存在的理由：更新累积在 fp32，前向才转 half")

# ═══ 3. 完整六步模板（第 9 章 miniGPT 的骨架）═══
print("\n═══ 六步模板实跑（合成回归任务）═══")
torch.manual_seed(0)
net = nn.Sequential(nn.Linear(32, 128), nn.GELU(), nn.Linear(128, 1)).to(device)
decay = [p for p in net.parameters() if p.dim() >= 2]
no_decay = [p for p in net.parameters() if p.dim() < 2]
opt = torch.optim.AdamW(
    [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
    lr=3e-3, betas=(0.9, 0.95),
)

import math

WARMUP_OPT, TOTAL_OPT = 25, 300                # 以 optimizer step 计（调度按 opt step 走！）


def lr_lambda(s: int) -> float:                # 返回相对峰值的倍率
    if s < WARMUP_OPT:
        return (s + 1) / WARMUP_OPT
    progress = (s - WARMUP_OPT) / (TOTAL_OPT - WARMUP_OPT)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
ACCUM = 2

net.train()
gnorm = torch.tensor(0.0)          # 第一个累积边界前日志会引用它
for step in range(600):                        # 600 micro-step = 300 optimizer step
    x = torch.randn(64, 32, device=device)
    y = (x[:, :1] * 2 + x[:, 1:2]).detach()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):   # ① 前向
        loss = ((net(x) - y) ** 2).mean() / ACCUM
    loss.backward()                                                       # ② 反向
    if (step + 1) % ACCUM == 0:
        gnorm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)     # ③ 裁剪
        opt.step()                                                        # ④ 更新
        sched.step()                                                      # ⑤ 调度
        opt.zero_grad(set_to_none=True)                                   # ⑥ 清零
    if step % 150 == 0 or step == 599:
        print(f"  step {step:>3}: loss={loss.item() * ACCUM:.4f} lr={sched.get_last_lr()[0]:.2e} "
              f"gnorm={gnorm:.2f}")
print("→ loss 收敛 + lr 按计划走 + gnorm 平稳 = 三个健康指标同框，这就是要监控的最小集")
