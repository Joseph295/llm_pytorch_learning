"""挑战 1 参考答案：手写 RMSNorm vs LayerNorm 的 500 步实测

运行：uv run chapters/ch08_transformer/exercises/solutions/challenge_1.py
"""

import time

import torch
import torch.nn as nn

torch.manual_seed(0)


class MyRMSNorm(nn.Module):
    """工业写法三要素：fp32 内部计算、rsqrt（比 1/sqrt 快）、无 bias。"""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dt)


def build(norm_cls):
    def norm():
        return norm_cls(128)
    layers = []
    for _ in range(4):
        layers += [norm(), nn.Linear(128, 128), nn.GELU()]
    return nn.Sequential(*layers, norm(), nn.Linear(128, 8))


def train(norm_cls, steps=500):
    torch.manual_seed(42)
    model = build(norm_cls)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(256, 128)
    y = torch.randint(0, 8, (256,))
    t0 = time.perf_counter()
    for _ in range(steps):
        loss = nn.functional.cross_entropy(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    dt = (time.perf_counter() - t0) * 1000 / steps
    n_params = sum(p.numel() for m in model.modules() if isinstance(m, (norm_cls,))
                   for p in m.parameters())
    return loss.item(), dt, n_params


for cls, name in [(nn.LayerNorm, "LayerNorm"), (MyRMSNorm, "RMSNorm")]:
    loss, ms, np_ = train(cls)
    print(f"{name:<10} 500 步后 loss={loss:.4f} | 每步 {ms:.2f} ms | norm 参数量 {np_}")

print("""
预期读数：两者最终 loss 相当（RMSNorm '不掉点'的小规模复现），
RMSNorm 参数量减半（没有 β/bias）。
速度上手写版反而略慢——official LayerNorm 是单个融合 kernel，我们的
Python RMSNorm 是 pow/mean/rsqrt/mul 四次分立调用（每次都有启动开销与
中间张量）。工业界 RMSNorm 更快的前提是它也被写成融合 kernel。
这是第 11 章 kernel fusion / torch.compile 的活教材：数学上更简单 ≠
跑得更快，算子粒度决定一切。

为什么"减均值"可以删：
1. 数学上：后续 Linear 层的权重可以吸收输入的常数偏移（W(x+c) = Wx + Wc，
   Wc 并入有效偏置），均值分量并非不可替代的信息；
2. 注意力中：softmax 对 logits 的整体平移不变（softmax(z+c)=softmax(z)），
   均值方向的贡献部分被归一化吸收；
3. 实证上：Zhang & Sennrich (2019) 与其后所有主流 LLM 的选择一致——
   re-scaling 是 LayerNorm 起效的主成分，re-centering 可弃。
LLM 时代的设计品味：每个组件都要为自己的算力/参数买单，证不出必要性就删。
""")
