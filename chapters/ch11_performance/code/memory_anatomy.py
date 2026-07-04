"""第 11 章 · 训练显存逐项拆解 + gradient checkpointing 实测

运行：uv run chapters/ch11_performance/code/memory_anatomy.py

把第 2 章（模型状态）和第 3 章（激活值/checkpointing）的账在 M4 上验证。
"""

import os
import sys
import time

import torch
import torch.utils.checkpoint as cp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "ch08_transformer", "code"))
from gpt_model import GPT, GPTConfig  # noqa: E402

if not torch.backends.mps.is_available():
    raise SystemExit("本脚本用 MPS 内存计数演示；CPU 上换 CUDA 计数思路相同")

device = torch.device("mps")


def mb():
    return torch.mps.current_allocated_memory() / 1024**2


def measure(use_checkpoint: bool, batch_size=32):
    torch.mps.empty_cache()
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=4096, block_size=256, n_layer=8, n_head=8, n_embd=512)
    model = GPT(cfg).to(device)

    if use_checkpoint:                          # 给每个 Block 包 checkpoint（第 3 章）
        orig_forward = model.forward

        def ckpt_forward(idx, targets=None):
            b, t = idx.shape
            h = model.drop(model.embed(idx))
            for block in model.blocks:
                # 重算换显存：前向不存 block 内部激活，反向时重跑
                h = cp.checkpoint(block, h, model.rope_cos, model.rope_sin, use_reentrant=False)
            h = model.norm_f(h)
            logits = model.lm_head(h)
            loss = None
            if targets is not None:
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        model.forward = ckpt_forward

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, cfg.vocab_size, (batch_size, cfg.block_size), device=device)
    yb = torch.randint(0, cfg.vocab_size, (batch_size, cfg.block_size), device=device)

    base = mb()
    _, loss = model(x, yb)
    after_fwd = mb()                            # 前向后 = 模型状态 + 激活
    loss.backward()
    opt.step()                                  # 优化器 state 诞生

    # 稳态计时
    for _ in range(3):
        _, loss = model(x, yb)
        opt.zero_grad(); loss.backward(); opt.step()
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        _, loss = model(x, yb)
        opt.zero_grad(); loss.backward(); opt.step()
    torch.mps.synchronize()
    step_ms = (time.perf_counter() - t0) * 1000 / 5

    activation_mb = after_fwd - base
    return activation_mb, step_ms


print("═══ gradient checkpointing 的时间-显存权衡（8 层, d=512, batch=32）═══\n")
act_off, t_off = measure(use_checkpoint=False)
act_on, t_on = measure(use_checkpoint=True)

print(f"{'配置':<16} | {'前向激活显存':>12} | {'每步耗时':>10}")
print("-" * 44)
print(f"{'不 checkpoint':<16} | {act_off:>10.0f} MB | {t_off:>8.1f} ms")
print(f"{'checkpoint':<16} | {act_on:>10.0f} MB | {t_on:>8.1f} ms")
print(f"\n激活显存降低: {(1 - act_on / act_off):.0%}（重算换来的）")
print(f"时间开销增加: {(t_on / t_off - 1):.0%}（多跑一次前向，理论 ~33%）")
print("""
第 2/3 章的账在此落地：
- 激活值是训练显存的最大变量（模型状态固定，激活随 batch/seq/层数涨）
- checkpointing 用 ~33% 额外算力换掉大部分激活显存
- 第 15 章微调 7B 时，这是"单卡放不下 → 能放下"的关键开关之一
- 只在显存吃紧时用（易错点⑥：不缺显存时纯亏算力）
""")
