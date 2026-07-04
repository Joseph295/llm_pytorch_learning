"""基础 1 参考答案：手写 MySequential

运行：uv run chapters/ch04_nn_module/exercises/solutions/basic_1.py
"""

import torch
import torch.nn as nn


class MySequential(nn.Module):
    def __init__(self, *modules):
        super().__init__()
        for i, m in enumerate(modules):
            self.add_module(str(i), m)     # 等价于 setattr(self, "0", m) 的注册效果，
                                           # 但属性名是数字字符串，setattr 语法写不出来

    def forward(self, x):
        for m in self.children():          # children() 按注册顺序遍历直接子模块
            x = m(x)
        return x


mine = MySequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
ref = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))

print("MySequential 参数名:", [n for n, _ in mine.named_parameters()])
print("nn.Sequential 参数名:", [n for n, _ in ref.named_parameters()])
assert [n for n, _ in mine.named_parameters()] == [n for n, _ in ref.named_parameters()]

# 权重对齐后验证前向一致
mine.load_state_dict(ref.state_dict())
x = torch.randn(3, 4)
print(f"前向一致: {torch.allclose(mine(x), ref(x))} ✓")
print("\n要点：数字命名的子模块用 add_module 注册（这就是 nn.Sequential 的做法），")
print("     命名规则一致 → state_dict 可与官方版互换。")
