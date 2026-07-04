"""进阶 1 参考答案：显存计算器（0.5B ~ 671B 四档）

运行：uv run chapters/ch02_tensor/exercises/solutions/advanced_1.py
"""

GiB = 1024**3


def estimate_memory(n_params: float, dtype_bytes: int = 2,
                    training: bool = True, optimizer: str = "adamw") -> dict:
    """模型状态显存明细（GiB）。不含激活值与 KV cache。"""
    detail = {"参数": n_params * dtype_bytes / GiB}
    if not training:
        return detail
    detail["梯度"] = n_params * dtype_bytes / GiB
    if optimizer == "adamw":
        detail["Adam m (fp32)"] = n_params * 4 / GiB
        detail["Adam v (fp32)"] = n_params * 4 / GiB
        detail["fp32 主参数"] = n_params * 4 / GiB     # 混合精度需要（第 6 章）
    elif optimizer == "sgd":
        detail["动量 (fp32)"] = n_params * 4 / GiB
        detail["fp32 主参数"] = n_params * 4 / GiB
    return detail


models = [("0.5B", 0.5e9), ("7B", 7e9), ("70B", 70e9), ("671B (DeepSeek-V3)", 671e9)]

print(f"{'模型':<20} | {'推理(half)':>12} | {'训练模型状态':>14} | {'A100-80G 至少':>12}")
print("-" * 70)
for name, n in models:
    infer = sum(estimate_memory(n, training=False).values())
    train = sum(estimate_memory(n, training=True).values())
    cards = -(-train // 80)  # 向上取整
    print(f"{name:<20} | {infer:>10.1f} G | {train:>12.1f} G | {cards:>10.0f} 卡")

print("""
结论与讨论：
- 70B 训练模型状态 ≈ 1043 GiB → 至少 14 张 A100-80G（这还没算激活值，
  实际配置远多于此——第 13 章 Megatron 的 TP×PP×DP 布局就是在解这道题）
- 671B 的 MoE 有省钱点：每 token 只激活部分专家，但**训练时所有专家的
  参数/梯度/优化器状态都要存**——模型状态账不打折，10TB 级，必须大规模分片
- 推理账里 half 参数只是起点：还要加 KV cache（第 16 章）——长上下文时
  KV cache 可能超过参数本身
""")
