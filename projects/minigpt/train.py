"""miniGPT · 预训练主脚本（第 6 章训练循环 + 第 8 章 GPT + 第 5 章数据）

运行：uv run projects/minigpt/train.py
快速冒烟：uv run projects/minigpt/train.py --smoke   （几十步，验证管线通）

这是前八章的总交卷：设备无关代码(0/2)、GPT(8)、六步训练循环(6)、
memmap 数据(5)、bf16 AMP(6)、梯度裁剪与调度(6)、checkpoint 四件套(6)。
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                "chapters", "ch08_transformer", "code"))
from gpt_model import GPT, GPTConfig  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")


def get_batch(split: str, block_size: int, batch_size: int, device):
    """从 memmap 随机采样一批定长块。每次重新 memmap 避免 fork 后的句柄问题（第 5 章）。"""
    data = np.memmap(os.path.join(DATA_DIR, f"{split}.bin"), dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    # 异步搬运（CUDA 上 pin_memory 才生效；MPS 直接 to）
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def estimate_loss(model, block_size, batch_size, device, iters=20):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(iters)
        for k in range(iters):
            x, y = get_batch(split, block_size, batch_size, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def get_lr(step, warmup, total, peak, floor_ratio=0.1):
    if step < warmup:
        return peak * (step + 1) / warmup
    if step > total:
        return peak * floor_ratio
    ratio = (step - warmup) / (total - warmup)
    return peak * floor_ratio + peak * (1 - floor_ratio) * 0.5 * (1 + math.cos(math.pi * ratio))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="几十步冒烟测试")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1337)

    # 读词表大小（prepare_data 产出）
    from tokenizer import BPETokenizer
    tok = BPETokenizer.load(os.path.join(DATA_DIR, "tokenizer.json"))

    # 模型配置：~10M 参数，M4 可在几分钟内看到明显收敛
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=256,
                    n_layer=6, n_head=6, n_embd=384, dropout=0.0)
    model = GPT(cfg).to(device)
    print(f"设备={device} | 模型 {model.num_params() / 1e6:.2f}M 参数 | vocab={cfg.vocab_size}")

    # 优化器：decay/no_decay 分组（第 6 章）
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
        lr=1e-3, betas=(0.9, 0.95),
    )

    total_steps = 60 if args.smoke else args.steps
    warmup = 10 if args.smoke else 150
    batch_size = 16 if device.type != "cpu" else 8
    peak_lr = 1e-3
    ckpt_path = os.path.join(DATA_DIR, "ckpt.pt")

    start_step = 0
    if args.resume and os.path.exists(ckpt_path):
        # weights_only=True：只反序列化张量与基础容器（第 4 章安全教训）。
        # 我们的 ckpt 只存 state_dict + 原始类型的 config dict，完全兼容。
        ck = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])       # 优化器状态！（第 6 章易错点④）
        start_step = ck["step"]
        print(f"从 step {start_step} 续训")

    model.train()
    t0 = time.time()
    for step in range(start_step, total_steps):
        lr = get_lr(step, warmup, total_steps, peak_lr)
        for g in opt.param_groups:
            g["lr"] = lr

        x, y = get_batch("train", cfg.block_size, batch_size, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):   # ① 前向
            _, loss = model(x, y)
        loss.backward()                                                       # ② 反向
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)       # ③ 裁剪
        opt.step()                                                            # ④ 更新
        opt.zero_grad(set_to_none=True)                                       # ⑥ 清零

        if step % (10 if args.smoke else 250) == 0 or step == total_steps - 1:
            dt = time.time() - t0
            msg = f"step {step:>4}/{total_steps} | loss {loss.item():.3f} | lr {lr:.2e} | gnorm {gnorm:.2f} | {dt:.0f}s"
            if not args.smoke and step > 0:
                ev = estimate_loss(model, cfg.block_size, batch_size, device)
                msg += f" | train {ev['train']:.3f} val {ev['val']:.3f}"
            print(msg)

    # checkpoint 四件套（第 6 章）
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                "step": total_steps, "config": cfg.__dict__}, ckpt_path)
    print(f"已保存 checkpoint → {ckpt_path}")

    # 生成一段看看学到了什么
    print("\n════ 生成样本 ════")
    start = tok.encode("话说")
    idx = torch.tensor([start], device=device)
    out = model.generate(idx, max_new_tokens=200, temperature=0.8, top_k=40)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
