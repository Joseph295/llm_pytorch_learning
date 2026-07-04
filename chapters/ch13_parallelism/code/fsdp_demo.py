"""第 13 章 · 手动模拟 FSDP/ZeRO-3 的分片—聚合语义（gloo，M4 可跑）

运行：uv run chapters/ch13_parallelism/code/fsdp_demo.py

真正的 FSDP 类强依赖 CUDA（在 MPS/CPU 上初始化会失败），所以这里用
AllGather 手动复刻它的核心机制——比调用黑盒更能看清"分片—按层聚合—释放"：
  - 参数平时分片：每 rank 只存每层参数的 1/N
  - 前向到某层：AllGather 拼出完整参数 → 计算 → 释放（显存立即回收）
理解了这个循环，就理解了 FSDP 为什么能在有限显存上训大模型（13.2-②）。
"""

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F


def all_gather_param(shard, world):
    """把各 rank 的参数分片 AllGather 成完整参数（FSDP 前向 pre-hook 做的事）。"""
    gathered = [torch.zeros_like(shard) for _ in range(world)]
    dist.all_gather(gathered, shard)
    return torch.cat(gathered, dim=0)


def worker(rank, world):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29505"
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.manual_seed(0)

    # 3 层 MLP 的完整权重（所有 rank 用同种子生成相同的"真值"作对照）
    D = 256
    full_weights = [torch.randn(D, D) * 0.05 for _ in range(3)]

    # ── 分片：每 rank 只保留每层权重的 1/world 行 ──
    rows = D // world
    my_shards = [w[rank * rows:(rank + 1) * rows].clone() for w in full_weights]
    local_params = sum(s.numel() for s in my_shards)
    full_params = sum(w.numel() for w in full_weights)
    if rank == 0:
        print(f"完整模型: {full_params:,} 参数")
        print(f"每 rank 分片后本地: {local_params:,}（= 全量/{world}）\n")
        print("前向逐层演示（AllGather 聚合 → 计算 → 释放）：")

    # ── 前向：逐层 AllGather 完整参数、算完释放 ──
    x = torch.randn(8, D)
    peak_extra = 0
    for layer, shard in enumerate(my_shards):
        full_w = all_gather_param(shard, world)            # pre-hook: 聚合
        peak_extra = max(peak_extra, full_w.numel())       # 峰值只多一层的完整参数
        x = F.relu(x @ full_w.T)
        correct = torch.allclose(full_w, full_weights[layer], atol=1e-6)
        if rank == 0:
            print(f"  层{layer}: AllGather 得完整 {tuple(full_w.shape)} 参数"
                  f"（与真值一致={correct}）→ 计算 → 释放")
        del full_w                                          # forward-hook: 释放，显存回收

    if rank == 0:
        print(f"\n峰值额外显存 = 单层完整参数 = {peak_extra:,}（而非全部 {full_params:,}）")
        print(f"→ 任意时刻显存 ≈ 分片({local_params:,}) + 当前层完整({peak_extra:,})")
        print("  这就是 FSDP 用通信换显存的原理。真实 FSDP 用 hook 自动做这套聚合/释放，")
        print("  所以直接调 model.forward 会拿到未聚合的空分片（第 4 章 / 易错点①）。")

    dist.destroy_process_group()


if __name__ == "__main__":
    WORLD = 4
    print(f"手动模拟 ZeRO-3/FSDP 分片—聚合（gloo {WORLD} 进程，M4 可跑）\n")
    mp.spawn(worker, args=(WORLD,), nprocs=WORLD, join=True)
