"""第 5 章 · worker 随机性 bug 复现：NumPy 种子在 worker 间重复

运行：uv run chapters/ch05_data_pipeline/code/worker_pitfalls.py

经典 bug：DataLoader 只管理 torch 的 worker 种子，NumPy/random 不归它管。
本脚本复现"所有 worker 吐出相同随机数"，再演示 worker_init_fn 修法。
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class AugmentDataset(Dataset):
    """模拟带随机增强的数据集：返回 (worker_id, numpy 随机数, torch 随机数)。"""

    def __len__(self):
        return 8

    def __getitem__(self, i):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else -1
        return wid, float(np.random.rand()), float(torch.rand(1))


def fix_seed(worker_id):
    """修复：从 torch 的 worker 种子派生 numpy/random 的种子。

    torch.initial_seed() 在每个 worker 里已经是 base_seed+worker_id（DataLoader 保证），
    拿它喂 numpy 即可让各 worker 的 numpy 流互不相同。
    """
    import random
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    ds = AugmentDataset()

    print("═══ 有 bug 的版本（4 worker，无 worker_init_fn）═══")
    loader = DataLoader(ds, batch_size=1, num_workers=4)
    rows = [(w.item(), n.item(), t.item()) for w, n, t in loader]
    print(f"{'worker':>6} | {'np.random':>10} | {'torch.rand':>10}")
    for w, n, t in sorted(rows)[:8]:
        print(f"{w:>6} | {n:>10.6f} | {t:>10.6f}")
    np_vals = [round(n, 6) for _, n, _ in rows]
    dup = len(np_vals) - len(set(np_vals))
    print(f"→ numpy 随机数重复条数: {dup}")
    if dup:
        print("  不同 worker 撞出了相同的'随机'增强——bug 现形（你在 Linux/fork 平台）")
    else:
        print("  本平台（macOS/spawn）侥幸逃过：spawn 的 worker 重新初始化了 numpy 熵源。")
        print("  同一份代码在 Linux/fork 集群上是必然 100% 重复——跨平台不一致让这坑更隐蔽，")
        print("  云端训练（Linux）中招而本地（mac）测不出来。修法见下，两个平台都该用。")

    print("\n═══ 修复版（worker_init_fn=fix_seed）═══")
    loader2 = DataLoader(ds, batch_size=1, num_workers=4, worker_init_fn=fix_seed)
    rows2 = [(w.item(), n.item(), t.item()) for w, n, t in loader2]
    np_vals2 = [round(n, 6) for _, n, _ in rows2]
    print(f"numpy 随机数重复条数: {len(np_vals2) - len(set(np_vals2))} ✓")
    print("→ 修复原理：torch.initial_seed()（每 worker 不同）派生 numpy/random 种子")
