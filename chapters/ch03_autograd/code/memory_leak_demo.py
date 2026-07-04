"""第 3 章 · 亲眼看内存泄漏：评估循环里 total += loss 的代价曲线

运行：uv run chapters/ch03_autograd/code/memory_leak_demo.py

一个重要的精确化（本脚本初版设计错了，实测后修正——见输出末尾的说明）：
  - 训练循环里 total += loss（backward 之后）：泄漏很小——backward 默认
    释放 saved_tensors，残留的只是轻量的图结构对象。是坏习惯，不是灾难。
  - 评估/指标循环里 total += loss（从不 backward）：灾难——每步的整张图
    连同全部激活值被 total 引住，内存线性上涨直到 OOM。
本演示复现的是后者，也给出两种正确写法。
"""

import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = torch.nn.Sequential(
    torch.nn.Linear(2048, 2048), torch.nn.ReLU(),
    torch.nn.Linear(2048, 2048), torch.nn.ReLU(),
    torch.nn.Linear(2048, 2048),
).to(device)

mem_mb = (lambda: torch.mps.current_allocated_memory() / 1024**2) if device.type == "mps" \
    else (lambda: 0.0)


def eval_loop(steps: int, mode: str) -> list:
    """模拟验证集评估：只算 loss 不反向。mode 决定累加方式。"""
    total = torch.zeros(1, device=device)
    curve = []
    for _ in range(steps):
        x = torch.randn(512, 2048, device=device)
        if mode == "no_grad":                       # 正确姿势 A：压根不建图
            with torch.no_grad():
                loss = model(x).pow(2).mean()
            total = total + loss
        else:
            loss = model(x).pow(2).mean()           # 建了图，且永远不 backward
            total = total + (loss.detach() if mode == "detach" else loss)
        curve.append(mem_mb())
    return curve


torch.manual_seed(0)
results = {}
for mode in ["leak", "detach", "no_grad"]:
    if device.type == "mps":
        torch.mps.empty_cache()
    results[mode] = eval_loop(24, mode)

if device.type == "mps":
    print("步数 |   泄漏版(MB) | detach版(MB) | no_grad版(MB)")
    print("-" * 52)
    for i in [0, 5, 11, 17, 23]:
        print(f"{i + 1:>4} | {results['leak'][i]:>11.0f} | {results['detach'][i]:>11.0f} "
              f"| {results['no_grad'][i]:>12.0f}")
    growth = results["leak"][-1] - results["leak"][0]
    print(f"\n泄漏版 24 步净涨 {growth:.0f} MB：total 带 grad_fn → 引住每步整张图")
    print("→ 图引住 saved_tensors（激活值）→ 从不 backward 所以永不释放。")
    print("detach 版只留数值；no_grad 版更进一步——图根本没建（推理场景首选，还更快）。")
    print("\n附注：训练循环里 backward() 会释放 saved_tensors，所以'训练时忘 detach'")
    print("泄漏很小（只剩图结构对象）；真正的重灾区是评估/指标累加循环——本演示的场景。")
else:
    print("CPU 模式无分配器计数，逻辑同上。")
