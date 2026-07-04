"""第 10 章 · RoPE 位置插值：把短上下文模型扩展到长上下文

运行：uv run chapters/ch10_modern_llm/code/rope_scaling.py

演示 PI（位置插值）与 NTK-aware 如何把训练过的角度几何"拉伸"到新长度。
"""

import torch

torch.manual_seed(0)

D = 64
TRAIN_LEN = 256
TARGET_LEN = 1024


def rope_freqs(head_dim, base=10000.0):
    return 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))


print("═══ 1. 问题：超出训练长度，位置角度进入未见过的区域 ═══")
inv_freq = rope_freqs(D)
# 最慢的维度在训练长度末尾转了多少
slow_angle_train = TRAIN_LEN * inv_freq[-1].item()
slow_angle_target = TARGET_LEN * inv_freq[-1].item()
print(f"训练时最慢维度在末位置(256)累计角度: {slow_angle_train:.4f} rad")
print(f"外推到 1024 时该角度: {slow_angle_target:.4f} rad ← 模型从没见过这么大的角度")
print("→ 直接外推 = 让模型处理没学过的旋转角，输出崩坏（第 8 章 RoPE 外推衰减）")

print("\n═══ 2. PI（位置插值）：把新位置线性压回训练范围 ═══")
scale = TRAIN_LEN / TARGET_LEN                             # 1/4
# 位置 m 在 PI 下等效为 m·scale，落回 [0, 256] 训练过的范围
pi_angle = (TARGET_LEN * scale) * inv_freq[-1].item()
print(f"PI 缩放因子: {scale:.3f}")
print(f"PI 后位置 1024 的最慢维度角度: {pi_angle:.4f} rad（= 训练末位置的角度，安全）")
print("→ 代价：相邻位置的角度差也被压缩 1/4，近距离分辨率下降（高频维度受损最重）")

print("\n═══ 3. NTK-aware：按频率差异化缩放，保护高频 ═══")
# NTK 的思路：调整 base，使高频维度几乎不变、低频维度多压缩
alpha = TARGET_LEN / TRAIN_LEN
ntk_base = 10000.0 * (alpha ** (D / (D - 2)))
ntk_freq = rope_freqs(D, base=ntk_base)
print(f"原始 base=10000 → NTK base={ntk_base:.0f}")
hi_change = abs(ntk_freq[0].item() / inv_freq[0].item() - 1)
lo_change = abs(ntk_freq[-1].item() / inv_freq[-1].item() - 1)
print(f"最高频维度频率变化: {hi_change:.1%}（几乎不变，保住近距离分辨率）")
print(f"最低频维度频率变化: {lo_change:.1%}（大幅压缩，容纳长距离）")
print("→ NTK 比 PI 更聪明：近处清晰、远处也能编码，YaRN 是它 + 温度调整 + 微调的精修版")

print("\n═══ 4. NTK 的差异化：高频维度几乎不动，低频维度才压缩 ═══")
print("对比位置 512（超训练范围）在最高频维度 vs 最低频维度上的角度：")
print(f"{'维度':>8} | {'原始(外推)':>12} | {'PI':>10} | {'NTK':>10}")
for label, idx in [("最高频(0)", 0), ("最低频(-1)", -1)]:
    raw = 512 * inv_freq[idx].item()
    pi = 512 * scale * inv_freq[idx].item()
    ntk = 512 * ntk_freq[idx].item()
    print(f"{label:>8} | {raw:>12.4f} | {pi:>10.4f} | {ntk:>10.4f}")
print("→ 看最高频维度：PI 把它也压了 1/4（近距离分辨率受损），")
print("  NTK 几乎保留原值（近距离清晰）——这就是 NTK 优于 PI 的地方。")
print("  最低频维度上两者接近（都需要压缩以容纳长距离）。")
print("  YaRN = NTK + 温度调整 + 少量长文本微调，让模型适应压缩后的几何")
