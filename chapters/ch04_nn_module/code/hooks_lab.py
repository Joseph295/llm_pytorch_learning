"""第 4 章 · hooks 实战：激活探针 + NaN 哨兵

运行：uv run chapters/ch04_nn_module/code/hooks_lab.py

两个工业级 hook 用法：
  1. 激活探针——不改模型代码，旁路观察每层输出分布
  2. NaN 哨兵——输出出现 NaN/Inf 时立刻报出是哪一层（第 15 章排查 loss 爆炸复用）
"""

import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(16, 64), nn.ReLU(),
    nn.Linear(64, 64), nn.ReLU(),
    nn.Linear(64, 4),
)

print("═══ 1. 激活探针：规范写法（当场算标量，不存张量，用完摘除）═══")
stats, handles = [], []


def make_probe(name):
    def probe(module, args, output):
        # 当场 detach + 算标量——绝不把带图的激活存进闭包（4.7 案例 3）
        o = output.detach()
        stats.append((name, o.mean().item(), o.std().item(), o.abs().max().item()))
    return probe


for name, m in model.named_modules():
    if len(list(m.children())) == 0:                    # 只挂叶子模块
        handles.append(m.register_forward_hook(make_probe(name)))

model(torch.randn(32, 16))
for h in handles:
    h.remove()                                          # 摘除！

print(f"{'层':<4} | {'mean':>8} | {'std':>7} | {'absmax':>7}")
for name, mean, std, amax in stats:
    print(f"{name:<4} | {mean:>8.4f} | {std:>7.4f} | {amax:>7.4f}")
print("→ 健康网络的激活 std 应保持在稳定量级；逐层放大/缩小是初始化或结构问题（第 8 章）")

print("\n═══ 2. NaN 哨兵：污染源定位器 ═══")


class NaNSentry:
    """with NaNSentry(model): ... —— 前向中任何一层输出 NaN/Inf 立刻定位。"""

    def __init__(self, model: nn.Module):
        self.model = model
        self.handles = []

    def __enter__(self):
        for name, m in self.model.named_modules():
            if len(list(m.children())) == 0:
                self.handles.append(m.register_forward_hook(self._check(name)))
        return self

    @staticmethod
    def _check(name):
        def hook(module, args, output):
            bad = (~torch.isfinite(output)).sum().item()
            if bad:
                raise RuntimeError(
                    f"NaN/Inf 哨兵触发！层 [{name}] ({module.__class__.__name__}) "
                    f"输出中有 {bad} 个非有限值"
                )
        return hook

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()          # 异常路径也保证摘除（第 1 章 with 语义）


# 制造一个会产生 NaN 的模型：中间权重埋一个 inf
sick = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8), nn.Linear(8, 8))
with torch.no_grad():
    sick[1].weight[0, 0] = float("inf")

try:
    with NaNSentry(sick):
        sick(torch.randn(4, 8))
except RuntimeError as e:
    print(f"捕获: {e}")
print(f"哨兵退出后 hook 已摘干净: {all(len(m._forward_hooks) == 0 for m in sick.modules())}")
print("→ 没有哨兵时你只能看到最终 loss=nan；有哨兵，污染源直接指到层。")
