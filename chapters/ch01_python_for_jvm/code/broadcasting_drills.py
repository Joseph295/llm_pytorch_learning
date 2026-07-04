"""第 1 章 · 广播规则演练

运行：uv run chapters/ch01_python_for_jvm/code/broadcasting_drills.py

两条规则（讲义 1.2-⑦）：从右往左逐维对齐；每维上 相等/是1/不存在 皆可。
最后演示 LLM 里最重要的广播实战：因果掩码。
"""

import torch


def try_broadcast(shape_a, shape_b):
    """演示工具：报告两个 shape 相加的广播结果。"""
    a, b = torch.zeros(shape_a), torch.zeros(shape_b)
    try:
        out = (a + b).shape
        print(f"  {str(shape_a):>12} + {str(shape_b):<9} -> {tuple(out)}")
    except RuntimeError:
        print(f"  {str(shape_a):>12} + {str(shape_b):<9} -> ✗ 报错（从右对齐后有维度既不等也没有 1）")


print("═══ 规则演练 ═══")
try_broadcast((3, 4), (4,))        # 右对齐: 4=4 ✓, 3 vs 无→1 ✓
try_broadcast((8, 1, 6), (7, 6))   # 6=6, 1vs7→7, 8vs无→8
try_broadcast((3, 4), (3,))        # 4 vs 3 ✗
try_broadcast((8, 1), (1, 8))      # 双向扩展 → (8, 8)
try_broadcast((2, 3), (3, 2))      # 3vs2 ✗
try_broadcast((5, 1, 4), (3, 4))   # → (5, 3, 4)

print("\n═══ 广播不复制数据：expand 的证据 ═══")
row = torch.arange(3.0)                 # shape (3,)，3 个元素
big = row.expand(1000, 3)               # 逻辑上 1000 份拷贝
print(f"row 存储 {row.untyped_storage().nbytes()} 字节; "
      f"expand 成 {tuple(big.shape)} 后仍是 {big.untyped_storage().nbytes()} 字节")
print(f"秘密在 stride: big.stride() = {big.stride()}  ← 第 0 维步长为 0（第 2 章揭底）")

print("\n═══ LLM 实战：因果掩码的广播 ═══")
B, H, T = 2, 4, 6                       # batch、注意力头数、序列长
scores = torch.randn(B, H, T, T)        # 每个头的注意力分数矩阵
causal = torch.tril(torch.ones(T, T))   # 下三角: 位置 i 只能看 ≤i
# (T,T) 广播到 (B,H,T,T)：一份掩码服务所有 batch/head，零拷贝
masked = scores.masked_fill(causal == 0, float("-inf"))
print(f"scores {tuple(scores.shape)} 被 mask {tuple(causal.shape)} 广播覆盖")
print("看第 0 个样本第 0 个头，未来位置(-inf)已被屏蔽：")
print(masked[0, 0].round(decimals=1))
print("\n第 7 章手写注意力时，这段代码会原样出现在核心路径上。")
