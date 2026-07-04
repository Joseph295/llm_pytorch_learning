"""进阶 1 参考答案：IterableDataset 多 worker 正确分片（不重不漏）

运行：uv run chapters/ch05_data_pipeline/exercises/solutions/advanced_1.py
"""

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

# 模拟 8 个 shard 文件，每个含 10 条样本
SHARDS = {f"shard_{s}": list(range(s * 10, s * 10 + 10)) for s in range(8)}


class BrokenStream(IterableDataset):
    """错误示范：不分片——每个 worker 都遍历全量。"""

    def __iter__(self):
        for shard in SHARDS.values():
            yield from shard


class ShardedStream(IterableDataset):
    """正确版：按 worker 取模分 shard（shard 级分片，与工业实践一致——
    样本级取模会破坏 shard 内顺序读的 IO 友好性）。"""

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info else 0
        nw = info.num_workers if info else 1
        for idx, shard in enumerate(SHARDS.values()):
            if idx % nw == wid:               # 本 worker 只认领自己的 shard
                yield from shard


def collect(ds, workers):
    loader = DataLoader(ds, batch_size=None, num_workers=workers)  # batch_size=None: 逐样本
    return [x.item() if torch.is_tensor(x) else x for x in loader]


if __name__ == "__main__":
    full = sorted(v for shard in SHARDS.values() for v in shard)

    got_broken = collect(BrokenStream(), workers=4)
    print(f"错误版 4 workers: 取到 {len(got_broken)} 条（应为 {len(full)}）"
          f"—— 每条重复 {len(got_broken) // len(full)} 次 ✗")

    got = collect(ShardedStream(), workers=4)
    assert sorted(got) == full, "应不重不漏"
    print(f"分片版 4 workers: 取到 {len(got)} 条，排序后与全量一致（不重不漏）✓")

    got1 = collect(ShardedStream(), workers=0)          # 无 worker 也要能跑
    assert sorted(got1) == full
    print("workers=0 退化路径也正确 ✓")

    print("""
要点：
1. get_worker_info() 只在 worker 进程内非 None——写分片逻辑必须处理 None（主进程直迭代）
2. 多机分布式还要叠加外层 rank 分片：shard_idx % (world_size*nw) == rank*nw + wid
   （或先按 rank 切 shard 列表再按 worker 切——第 12 章 DistributedSampler 一节）
3. shard 数应远大于 worker 总数，否则分配不均 → 某 worker 提前吐完，
   吞吐被最慢 worker 拖住（和 Spark task 数 vs executor 数的道理相同）
""")
