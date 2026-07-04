"""第 16 章 · 手写 int8 量化：精度 vs 访存权衡

运行：uv run chapters/ch16_inference/code/quantization.py

量化把 fp32/fp16 权重压到 int8，直接减访存加速 memory-bound 的 decode。
本脚本实现 per-tensor / per-channel 对称量化，量化一个线性层，测精度损失。
"""

import torch
import torch.nn.functional as F

torch.manual_seed(0)


def quantize_per_tensor(W, bits=8):
    """整个矩阵共享一个 scale（简单但精度差）。"""
    qmax = 2 ** (bits - 1) - 1                            # int8: 127
    scale = W.abs().max() / qmax
    Wq = torch.clamp(torch.round(W / scale), -qmax, qmax)
    return Wq.to(torch.int8), scale


def quantize_per_channel(W, bits=8):
    """每列（输出通道）一个 scale（精度高，元数据略多）。"""
    qmax = 2 ** (bits - 1) - 1
    scale = W.abs().amax(dim=0, keepdim=True) / qmax      # (1, out) 每列一个 scale
    Wq = torch.clamp(torch.round(W / scale), -qmax, qmax)
    return Wq.to(torch.int8), scale


def dequantize(Wq, scale):
    return Wq.to(torch.float32) * scale


d = 1024
W = torch.randn(d, d) * 0.05                              # 模拟一个权重矩阵
x = torch.randn(64, d)
y_ref = x @ W.T                                           # fp32 参照

print("═══ int8 量化的精度 vs 访存 ═══\n")
print(f"{'方法':<16} | {'输出相对误差':>12} | {'权重字节':>10} | {'访存倍数':>8}")
print("-" * 54)

fp32_bytes = W.numel() * 4
for name, quant_fn in [("per-tensor", quantize_per_tensor), ("per-channel", quantize_per_channel)]:
    Wq, scale = quant_fn(W)
    W_deq = dequantize(Wq, scale)
    y_q = x @ W_deq.T
    rel_err = (y_q - y_ref).norm() / y_ref.norm()
    int8_bytes = Wq.numel() * 1 + scale.numel() * 4      # int8 权重 + fp32 scale
    print(f"{name:<16} | {rel_err.item():>12.2e} | {int8_bytes:>9}B | {fp32_bytes / int8_bytes:>6.1f}×")

print(f"""
读数（16.2-③）：
- int8 量化把权重从 4 字节压到 1 字节 → 访存减 ~4× → decode 直接快 ~4×
- per-channel 比 per-tensor 精度更好（每列独立 scale，适应各列不同数值范围）；
  差距在权重均匀时不大，但有 outlier 时拉开到一个数量级（见下方 outlier 实验）
- 权衡：位数越低越快越省，但精度损失越大；per-channel 用少量元数据换精度
真实的 GPTQ/AWQ 更精细：
- GPTQ 用 Hessian 二阶信息 + 误差补偿逐列量化
- AWQ 保护"激活大的重要通道"（outlier 是精度杀手）
- 都能把 7B 压到 4-bit（14GB→3.5GB）精度损失很小
- group 量化（每 128 个权重一个 scale）是精度/元数据的折中（易错点③）
""")

# ═══ 量化误差的分布：outlier 通道是精度杀手 ═══
print("═══ 为什么 outlier 是精度杀手 ═══")
W_outlier = W.clone()
W_outlier[:, 0] *= 50                                     # 制造一个 outlier 列
Wq, scale = quantize_per_tensor(W_outlier)
err_with = ((dequantize(Wq, scale) - W_outlier).abs().mean()).item()
Wq2, scale2 = quantize_per_channel(W_outlier)
err_pc = ((dequantize(Wq2, scale2) - W_outlier).abs().mean()).item()
print(f"有 outlier 列时 per-tensor 平均量化误差: {err_with:.4f}（一个大值撑大 scale，其余全损精度）")
print(f"           per-channel 平均量化误差: {err_pc:.4f}（outlier 隔离在自己的列，不影响其他）")
print("→ 这就是 AWQ/GPTQ 都在处理 outlier 的原因：少数异常通道决定量化质量")
