"""第 2 章 · LLM 内存账：理论计算器 + M4 实测对账

运行：uv run chapters/ch02_tensor/code/memory_math.py

讲义 2.2-④ 的内存账写成代码，并用一个真实小模型在本机对账，
验证"理论 ≈ 实测"——学会算这本账，第 13 章 ZeRO 的动机不言自明。
"""

import torch


def model_state_memory(n_params: float, training: bool = True) -> dict:
    """AdamW + 混合精度（bf16/fp16 计算、fp32 主参数）的模型状态内存，单位 GB。

    只算"模型状态"（参数/梯度/优化器），不含激活值——激活值和
    batch、序列长相关，第 11 章单独算。
    """
    GB = 1024**3
    if not training:
        return {"参数(half)": n_params * 2 / GB}
    return {
        "参数(half)": n_params * 2 / GB,
        "梯度(half)": n_params * 2 / GB,
        "Adam m(fp32)": n_params * 4 / GB,
        "Adam v(fp32)": n_params * 4 / GB,
        "fp32 主参数": n_params * 4 / GB,
    }


print("═══ 理论账本 ═══")
for name, n in [("0.5B", 0.5e9), ("7B", 7e9), ("70B", 70e9)]:
    infer = sum(model_state_memory(n, training=False).values())
    train_detail = model_state_memory(n, training=True)
    train = sum(train_detail.values())
    bytes_per_param = train * 1024**3 / n
    print(f"{name:>5}: 推理 {infer:7.1f} GiB | 训练模型状态 {train:7.1f} GiB "
          f"(= {bytes_per_param:.0f} 字节/参数)")
print("→ 7B 训练 ≈104 GiB（1.12×10¹¹ 字节），远超 A100 的 80GB：")
print("  单卡装不下，这就是 ZeRO/FSDP 的存在理由（第 13 章）")

print("\n═══ M4 实测对账：0.1B 参数的真实张量 ═══")
n = 100_000_000
before = torch.mps.current_allocated_memory()

params = torch.randn(n, dtype=torch.float16, device="mps")     # 参数
grads = torch.zeros_like(params)                               # 梯度
m = torch.zeros(n, dtype=torch.float32, device="mps")          # Adam m
v = torch.zeros(n, dtype=torch.float32, device="mps")          # Adam v
master = params.float()                                        # fp32 主参数

measured = (torch.mps.current_allocated_memory() - before) / 1024**3
theory = sum(model_state_memory(n).values())
print(f"理论: {theory:.3f} GB   实测: {measured:.3f} GB   偏差: {abs(measured - theory) / theory:.1%}")
print("（分配器有对齐/缓存开销，个位数百分比偏差正常）")

# 确定性析构（第 1 章 Q4）：引用归零，内存立即回到分配器
del params, grads, m, v, master
after_free = (torch.mps.current_allocated_memory() - before) / 1024**3
print(f"del 五个张量后占用回落到: {after_free:.3f} GB ← 引用计数的确定性释放，不等 GC")
