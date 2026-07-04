"""第 4 章 · 保存/加载全流程：strict 语义、前缀修复、部分加载

运行：uv run chapters/ch04_nn_module/code/save_load.py
"""

import os
import tempfile

import torch
import torch.nn as nn


def build():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


print("═══ 1. 标准往返：存 CPU、加载显式 map_location + weights_only ═══")
model = build()
path = os.path.join(tempfile.gettempdir(), "demo_ckpt.pt")
sd_cpu = {k: v.cpu() for k, v in model.state_dict().items()}   # 保存前搬 CPU（生产惯例）
torch.save(sd_cpu, path)

model2 = build()
# weights_only=True：只允许张量/基础容器，堵住 pickle 任意代码执行（torch 2.6+ 已默认）
sd = torch.load(path, map_location="cpu", weights_only=True)
model2.load_state_dict(sd)                                     # strict=True 默认：全对上才过
x = torch.randn(3, 4)
print(f"往返后输出一致: {torch.allclose(model(x), model2(x))}")

print("\n═══ 2. DDP 前缀事故与修复 ═══")
ddp_style_sd = {f"module.{k}": v for k, v in sd.items()}       # 模拟 DDP 保存的权重
model3 = build()
try:
    model3.load_state_dict(ddp_style_sd)
except RuntimeError as e:
    print(f"直接加载 → RuntimeError: {str(e)[:70]}...")

# 修复法 A：torch 自带工具
fixed = dict(ddp_style_sd)
torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(fixed, "module.")
model3.load_state_dict(fixed)
print(f"consume_prefix 修复后加载成功，输出一致: {torch.allclose(model(x), model3(x))}")

print("\n═══ 3. strict=False 的正确姿势：必须检查返回值 ═══")
bigger = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2), nn.Linear(2, 5))
missing, unexpected = bigger.load_state_dict(sd, strict=False)
print(f"missing_keys（模型有、ckpt 没有）  : {missing}")
print(f"unexpected_keys（ckpt 有、模型没有）: {unexpected}")
print("→ 新加的分类头 3.weight/3.bias 缺失是预期的（保持随机初始化，之后微调）")
print("→ 若 missing 里出现主干层名，说明键错位——必须停下排查，不能带病继续（4.7 案例 1）")

print("\n═══ 4. 快照 vs 引用（易错点④的正确写法）═══")
import copy

net = build()
best = copy.deepcopy(net.state_dict())         # 真快照
with torch.no_grad():
    for p in net.parameters():
        p.add_(100.0)                          # 模拟继续训练
net.load_state_dict(best)                      # 回滚到快照
print(f"回滚后第一个权重均值恢复正常: {net[0].weight.mean().item():.4f}（若是引用会是 100+）")

os.remove(path)
