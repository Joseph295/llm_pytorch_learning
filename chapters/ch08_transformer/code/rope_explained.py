"""第 8 章 · RoPE 的旋转直觉与相对性验证

运行：uv run chapters/ch08_transformer/code/rope_explained.py
"""

import torch

from gpt_model import apply_rope, precompute_rope

torch.manual_seed(0)

D = 64                                   # head_dim
cos, sin = precompute_rope(D, max_len=512)

print("═══ 1. 多尺度时钟指针：不同维度的旋转速度 ═══")
inv_freq = 1.0 / (10000 ** (torch.arange(0, D, 2).float() / D))
print(f"最快的指针（维度 0）  : 每个位置转 {inv_freq[0].item():.4f} rad（~57°）")
print(f"中速的指针（维度 {D // 2}）: 每个位置转 {inv_freq[D // 4].item():.4f} rad")
print(f"最慢的指针（维度 {D - 2}）: 每个位置转 {inv_freq[-1].item():.6f} rad（转一圈要 ~{int(2 * 3.14159 / inv_freq[-1]):,} 个位置）")
print("→ 快指针分辨近距离，慢指针编码远距离——和钟表的时分秒针同构")

print("\n═══ 2. 相对性验证：q_m·k_n 只依赖 m−n ═══")
q = torch.randn(1, 1, 1, D)              # 同一对 q, k 内容
k = torch.randn(1, 1, 1, D)


def rope_dot(m: int, n: int) -> float:
    """把 q 放在位置 m、k 放在位置 n，计算旋转后的点积。"""
    qm, _ = apply_rope(q, q, cos[m : m + 1], sin[m : m + 1])
    kn, _ = apply_rope(k, k, cos[n : n + 1], sin[n : n + 1])
    return (qm * kn).sum().item()


pairs = [(5, 3), (105, 103), (400, 398)]          # 三对，相对位移都是 2
print(f"相对位移=2 的三对位置的点积: {[round(rope_dot(m, n), 5) for m, n in pairs]}")
print(f"相对位移=7 的三对位置的点积: {[round(rope_dot(m, n), 5) for m, n in [(9, 2), (109, 102), (409, 402)]]}")
print("→ 同位移点积全等：注意力分数天然只看相对距离（绝对位置被消掉）")

print("\n═══ 3. 距离衰减的注意力偏置（统计包络，1000 对随机 q,k 取平均）═══")
# 单对 q,k 的点积随距离剧烈波动（试试就知道），衰减是"期望包络"性质：
# 对相似方向的 q,k（有关联的内容），平均相关度随距离下降。
for gap in [0, 1, 4, 16, 64, 256]:
    total = 0.0
    for i in range(1000):
        g = torch.Generator().manual_seed(i)
        qq = torch.randn(1, 1, 1, D, generator=g)
        qm, _ = apply_rope(qq, qq, cos[gap : gap + 1], sin[gap : gap + 1])
        k0, _ = apply_rope(qq, qq, cos[0:1], sin[0:1])       # k 用相同内容（自相关）
        total += (qm * k0).sum().item()
    print(f"  |m−n|={gap:>3}: 平均自相关 = {total / 1000:8.2f}")
print("→ 相同内容的 q,k 随距离拉开，点积从满值单调滑落：")
print("  RoPE 隐式引入'近处更相关'的软先验（可被内容差异覆盖，不是硬约束）")
