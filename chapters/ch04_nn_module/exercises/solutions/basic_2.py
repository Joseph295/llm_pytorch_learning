"""基础 2 参考答案：参数账本

运行：uv run chapters/ch04_nn_module/exercises/solutions/basic_2.py
"""

import torch.nn as nn


def param_report(model: nn.Module, expected: int | None = None):
    total = 0
    print(f"{'子模块':<12} | {'参数量':>10} | 占比")
    print("-" * 38)
    rows = []
    for name, child in model.named_children():
        n = sum(p.numel() for p in child.parameters())
        rows.append((name, n))
        total += n
    # 挂在根节点自身的直接参数（不属于任何子模块）
    own = sum(p.numel() for p in model.parameters(recurse=False))
    if own:
        rows.append(("(root)", own))
        total += own
    for name, n in rows:
        print(f"{name:<12} | {n:>10,} | {n / max(total, 1):>5.1%}")
    print(f"{'合计':<12} | {total:>10,}")
    if expected is not None:
        status = "✓ 对账通过" if total == expected else f"✗ 缺 {expected - total:,} ——查易错点①！"
        print(f"理论值 {expected:,} → {status}")
    return total


class Broken(nn.Module):
    """故意用普通 list 藏 4 层，每层 8*8+8=72 参数。"""

    def __init__(self):
        super().__init__()
        self.head = nn.Linear(8, 2)                          # 18 参数，正常注册
        self.hidden = [nn.Linear(8, 8) for _ in range(4)]    # 288 参数，全部失踪


print("── 故障模型 ──")
param_report(Broken(), expected=4 * 72 + 18)

print("\n── 修复后 ──")


class FixedM(Broken):
    def __init__(self):
        super().__init__()
        self.hidden = nn.ModuleList(nn.Linear(8, 8) for _ in range(4))


param_report(FixedM(), expected=4 * 72 + 18)
