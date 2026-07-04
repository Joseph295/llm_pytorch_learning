"""第 5 章 · DataLoader 机制实测：多进程加速、collate 定制

运行：uv run chapters/ch05_data_pipeline/code/dataloader_mechanics.py

注意整个入口都包在 if __name__ == "__main__" 里——macOS 用 spawn 启动
worker，子进程会重新 import 本文件，顶层代码裸奔 = 无限递归（易错点①）。
"""

import time

import torch
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader, Dataset


class SlowDataset(Dataset):
    """模拟'解码/tokenize 有开销'的数据集：每个样本人为耗时 2ms。"""

    def __len__(self):
        return 512

    def __getitem__(self, i):
        time.sleep(0.002)                     # 模拟 CPU 预处理
        return torch.randn(64), i % 10


class VarLenDataset(Dataset):
    """变长序列数据集：长度 5~40 不等的 token id 序列。"""

    def __len__(self):
        return 64

    def __getitem__(self, i):
        n = 5 + (i * 7) % 36
        return torch.arange(n), i % 2


def pad_collate(batch):
    """变长 → padded batch + lengths + attention mask（基础练习 1 的官方版）。"""
    seqs = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch])
    lens = torch.tensor([len(s) for s in seqs])
    padded = rnn_utils.pad_sequence(seqs, batch_first=True, padding_value=0)
    mask = torch.arange(padded.size(1))[None, :] < lens[:, None]   # 广播（第 1 章）
    return padded, lens, mask, labels


def bench(loader, name):
    t0 = time.perf_counter()
    for _ in loader:
        pass
    dt = time.perf_counter() - t0
    print(f"  {name:<38} {dt:6.2f}s")
    return dt


if __name__ == "__main__":
    print("═══ 1. 多进程加速实测（512 样本 × 2ms 预处理）═══")
    ds = SlowDataset()
    t0 = bench(DataLoader(ds, batch_size=32, num_workers=0), "num_workers=0（主进程串行）")
    t2 = bench(
        DataLoader(ds, batch_size=32, num_workers=2, persistent_workers=True),
        "num_workers=2 + persistent",
    )
    t4 = bench(
        DataLoader(ds, batch_size=32, num_workers=4, persistent_workers=True),
        "num_workers=4 + persistent",
    )
    print(f"  → 加速 {t0 / t4:.1f}x。注意 mac 上 spawn 启动 worker 本身要 1~2 秒，")
    print("    数据集小的时候启动开销可能吃掉全部收益——persistent_workers 的价值所在")

    print("\n═══ 2. 变长序列的自定义 collate ═══")
    loader = DataLoader(VarLenDataset(), batch_size=4, collate_fn=pad_collate)
    padded, lens, mask, labels = next(iter(loader))
    print(f"  batch 内各序列长度: {lens.tolist()} → padded 到 {tuple(padded.shape)}")
    print(f"  attention mask（1=真实 token）:\n{mask.int()}")
    waste = 1 - lens.sum().item() / padded.numel()
    print(f"  本 batch 的 padding 浪费率: {waste:.0%} ← bucketing/packing 要消灭的就是它")
