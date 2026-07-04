"""第 4 章 · 注册机制 / 模块树 / state_dict 解剖

运行：uv run chapters/ch04_nn_module/code/module_registry.py
"""

import torch
import torch.nn as nn

print("═══ 1. __setattr__ 的三路分流 ═══")


class Demo(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(3, 3))       # → _parameters
        self.sub = nn.Linear(3, 3)                     # → _modules
        self.register_buffer("mask", torch.ones(3))    # → _buffers
        self.temp = torch.zeros(3)                     # → 普通属性，三本账都不进


d = Demo()
print(f"_parameters: {list(d._parameters)}")
print(f"_modules   : {list(d._modules)}")
print(f"_buffers   : {list(d._buffers)}")
print(f"state_dict : {list(d.state_dict())}   ← 参数+buffer，没有 temp")
print(f"parameters : {[n for n, _ in d.named_parameters()]}   ← buffer 不在（优化器不碰它）")

print("\n═══ 2. 易错点①现场：list 藏层 = 静默丢参数 ═══")


class Broken(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(8, 8) for _ in range(4)]      # ✗ 普通 list

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


class Fixed(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(8, 8) for _ in range(4))   # ✓

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


n_broken = sum(p.numel() for p in Broken().parameters())
n_fixed = sum(p.numel() for p in Fixed().parameters())
theory = 4 * (8 * 8 + 8)
print(f"理论参数量 {theory} | list 版注册到 {n_broken} | ModuleList 版 {n_fixed}")
print("→ list 版 forward 能跑、不报错，但优化器拿到 0 个参数——静默型事故")
print("→ 排查手段：参数量对账（一行代码，每次搭完模型都做）")

print("\n═══ 3. 模块树的命名寻址 ═══")
model = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Sequential(nn.Linear(8, 8), nn.ReLU()),      # 嵌套：名字会带路径
    nn.Linear(8, 2),
)
for name, p in model.named_parameters():
    print(f"  {name:<16} {tuple(p.shape)}")
print("→ 点分路径 = 树上寻址；LLM checkpoint 里 layers.3.attn.q_proj.weight 同理")

print("\n═══ 4. state_dict 是引用不是快照（易错点④）═══")
lin = nn.Linear(2, 2)
sd_ref = lin.state_dict()
w0 = sd_ref["weight"][0, 0].item()
with torch.no_grad():
    lin.weight[0, 0] = 777.0
print(f"改模型后，之前拿的 state_dict['weight'][0,0]: {w0:.3f} → {sd_ref['weight'][0, 0].item():.3f}")
print("→ 跟着变了！保存最优权重要 copy.deepcopy(model.state_dict())")

print("\n═══ 5. train/eval 与 no_grad 正交 ═══")
net = nn.Sequential(nn.Linear(4, 4), nn.Dropout(p=0.5))
x = torch.ones(1, 4)
net.train()
o1, o2 = net(x), net(x)
print(f"train 模式两次前向相同? {torch.equal(o1, o2)}   ← dropout 在随机丢")
net.eval()
o3, o4 = net(x), net(x)
print(f"eval  模式两次前向相同? {torch.equal(o3, o4)}，但建图了吗? grad_fn={o3.grad_fn is not None}")
print("→ eval 管行为、no_grad 管图，推理时两个都要")
