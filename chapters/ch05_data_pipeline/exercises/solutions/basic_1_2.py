"""基础 1+2 参考答案：变长 collate_fn + 吞吐压测

运行：uv run chapters/ch05_data_pipeline/exercises/solutions/basic_1_2.py

（基础 1 的 pad_collate 与本章 code/dataloader_mechanics.py 的官方版一致，
 此处独立重写并加满测试；基础 2 是压测函数。）
"""

import time

import torch
from torch.utils.data import DataLoader, Dataset

# ═══ 基础 1：pad_collate ═══


def pad_collate(batch):
    seqs, labels = zip(*batch)
    lens = torch.tensor([len(s) for s in seqs])
    T = int(lens.max())
    padded = torch.zeros(len(seqs), T, dtype=seqs[0].dtype)
    for i, s in enumerate(seqs):
        padded[i, : len(s)] = s
    mask = torch.arange(T)[None, :] < lens[:, None]
    return padded, lens, mask, torch.tensor(labels)


# 注意：pad_collate/SlowDS 必须留在模块层（spawn 平台要求可 pickle），
# 但测试/打印代码必须进 __main__ 守卫——本文件初版把基础 1 的测试写在
# 顶层，spawn 的 worker 重新 import 时把它执行了 4 遍（易错点①现行犯）。

# ═══ 基础 2：吞吐压测 ═══


class SlowDS(Dataset):
    def __len__(self):
        return 256

    def __getitem__(self, i):
        time.sleep(0.01)                    # 模拟解码开销
        return torch.randn(32)


def measure(loader, n_batches=None) -> float:
    """返回 batches/s。先空跑一遍热身 worker（persistent 时启动开销不算进去）。"""
    it = iter(loader)
    next(it)                                # 热身：触发 worker 启动/预取
    t0 = time.perf_counter()
    n = 0
    for _ in it:
        n += 1
        if n_batches and n >= n_batches:
            break
    return n / (time.perf_counter() - t0)


if __name__ == "__main__":
    batch = [(torch.arange(3), 0), (torch.arange(6), 1), (torch.arange(1), 0), (torch.arange(4), 1)]
    padded, lens, mask, labels = pad_collate(batch)
    assert padded.shape == (4, 6) and mask.sum() == lens.sum()
    assert torch.equal(padded[0, :3], torch.arange(3)) and padded[0, 3:].sum() == 0
    assert mask[2].int().tolist() == [1, 0, 0, 0, 0, 0]
    print(f"基础1: padded{tuple(padded.shape)} mask 校验 ✓，padding 率 {1 - mask.float().mean():.0%}")

    print("\n基础2: 吞吐 vs num_workers（每样本 10ms 解码，batch=16）")
    for w in [0, 2, 4]:
        kw = dict(num_workers=w, persistent_workers=(w > 0))
        bps = measure(DataLoader(SlowDS(), batch_size=16, **kw))
        ideal = 1 / (0.01 * 16) * max(w, 1)
        print(f"  workers={w}: {bps:5.2f} batches/s（理想上限 ≈ {ideal:.1f}）")
    print("""
曲线解读：
- 每 batch 纯预处理 = 16 样本 × 10ms = 160ms → 串行上限 1/0.16 = 6.25 batch/s
  一般公式：吞吐上限 ≈ workers / (每样本耗时 × batch_size)
- workers 翻倍吞吐近似翻倍，直到撞上其它瓶颈（CPU 核数/共享内存/主进程消费速度）
- 若翻倍不涨：瓶颈已不在 worker 数——去查 IO、collate 或主进程侧
""")
