"""第 13 章 · ZeRO 各阶段显存计算器

运行：uv run chapters/ch13_parallelism/code/zero_memory_calc.py

把 13.2-① 的表变成可算的数字：直观看到"为什么训 70B 必须 ZeRO-3"。
"""

GiB = 1024**3


def model_state_gib(n_params, n_gpus, stage):
    """混合精度 AdamW 下，单卡的模型状态显存（GiB）。不含激活。

    每参数字节数（第 2 章）：
      参数 fp16=2, 梯度 fp16=2, 优化器(fp32 m/v/主参)=12  →  共 16 字节/参数
    ZeRO 按阶段把这些切到 N 卡：
      DDP    : 全冗余  = 16
      stage1 : 切优化器 = 2 + 2 + 12/N
      stage2 : 切+梯度  = 2 + (2+12)/N
      stage3 : 全切     = 16/N
    """
    p = n_params
    if stage == "DDP":
        per = 16
    elif stage == "ZeRO-1":
        per = 2 + 2 + 12 / n_gpus
    elif stage == "ZeRO-2":
        per = 2 + 14 / n_gpus
    elif stage == "ZeRO-3":
        per = 16 / n_gpus
    return p * per / GiB


print("单卡模型状态显存（GiB，混合精度 AdamW，不含激活）\n")
for name, n in [("7B", 7e9), ("70B", 70e9), ("175B", 175e9)]:
    print(f"═══ {name} 模型 ═══")
    print(f"{'卡数':>6} | {'DDP':>8} | {'ZeRO-1':>8} | {'ZeRO-2':>8} | {'ZeRO-3':>8}")
    for g in [1, 8, 64, 1024]:
        row = [model_state_gib(n, g, s) for s in ["DDP", "ZeRO-1", "ZeRO-2", "ZeRO-3"]]
        mark = lambda v: f"{v:.0f}" if v <= 80 else f"{v:.0f}✗"     # 80GB 卡放不下标 ✗
        print(f"{g:>6} | {mark(row[0]):>8} | {mark(row[1]):>8} | {mark(row[2]):>8} | {mark(row[3]):>8}")
    print()

print("""读表结论（✗ = 超 80GB 单卡放不下，只算模型状态还没算激活）：
- 7B: DDP 单卡就要 112GB ✗；ZeRO-1 8卡即可塞进 80GB
- 70B: 必须 ZeRO-3 且要足够多卡（8 卡 ZeRO-3 = 140GB 仍✗，要 16+ 卡）
- 175B: ZeRO-3 也要上千卡才放得下模型状态——这就是为什么还要叠加 TP/PP
- 通信代价递增（ZeRO-1≈DDP，ZeRO-3≈1.5×DDP），能用低 stage 就用低的
提醒：这只是模型状态。激活值（第 11 章）另算，长序列/大 batch 时可能更大。
""")
