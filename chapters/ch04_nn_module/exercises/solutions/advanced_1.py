"""进阶 1 参考答案：激活统计器（上下文管理器 + hook，第 15 章排查工具雏形）

运行：uv run chapters/ch04_nn_module/exercises/solutions/advanced_1.py
"""

import torch
import torch.nn as nn


class ActivationStats:
    """with ActivationStats(model) as stats: model(x)
    退出后 stats.table() 打印每个叶子模块输出的 mean/std/absmax。

    设计要求的落点：
    - hook 内当场 detach 并算成 Python 标量 → 不持有任何张量（4.7 案例 3）
    - __exit__ 保证摘除（异常路径也摘）→ 不留 hook 残骸（易错点⑥）
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.records: list[tuple] = []
        self._handles = []

    def __enter__(self):
        for name, m in self.model.named_modules():
            if next(m.children(), None) is None:            # 叶子模块
                self._handles.append(m.register_forward_hook(self._make_hook(name)))
        return self

    def _make_hook(self, name):
        def hook(module, args, output):
            if isinstance(output, torch.Tensor):
                o = output.detach().float()
                self.records.append(
                    (name, module.__class__.__name__,
                     o.mean().item(), o.std().item(), o.abs().max().item())
                )
        return hook

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def table(self):
        print(f"{'层':<8} {'类型':<10} | {'mean':>8} | {'std':>7} | {'absmax':>8}")
        for name, cls, mean, std, amax in self.records:
            print(f"{name:<8} {cls:<10} | {mean:>8.4f} | {std:>7.4f} | {amax:>8.4f}")


model = nn.Sequential(
    nn.Linear(32, 128), nn.ReLU(),
    nn.Linear(128, 128), nn.ReLU(),
    nn.Linear(128, 10),
)

with ActivationStats(model) as stats:
    model(torch.randn(64, 32))

stats.table()
assert all(len(m._forward_hooks) == 0 for m in model.modules()), "hook 应已全部摘除"
print("\nhook 已全部摘除 ✓（重复 with 使用也不会叠加）")
