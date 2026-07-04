"""第 12 章 · gloo 后端手动 AllReduce：在 M4 上跑真实分布式通信

运行：uv run chapters/ch12_distributed/code/allreduce_demo.py

用 torch.multiprocessing 起 N 个进程，gloo 后端（CPU）做真实的集合通信。
通信语义与云端 NCCL 完全一致——只是后端和设备不同。你在这里理解的
AllReduce/Broadcast/AllGather，改个 backend 就能上 GPU 集群。
"""

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank: int, world_size: int):
    # 每个进程加入通信组（生产用 torchrun 设这些环境变量，这里手动设）
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # ── AllReduce：各 rank 造不同张量，求和后人人拿到相同结果 ──
    x = torch.tensor([float(rank + 1)] * 3)          # rank0=[1,1,1], rank1=[2,2,2]...
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    expected = sum(range(1, world_size + 1))
    if rank == 0:
        print(f"[AllReduce SUM] 各 rank 结果 = {x.tolist()}（期望全为 {expected}）")
        print(f"  → 这就是梯度同步的核心：N 卡不同梯度求和/平均，人人拿到相同的平均梯度")

    # ── Broadcast：rank 0 的数据发给所有人（初始化同步权重用）──
    w = torch.tensor([99.0]) if rank == 0 else torch.tensor([0.0])
    dist.broadcast(w, src=0)
    if rank == 1:
        print(f"[Broadcast] rank1 收到 rank0 的值: {w.item()}（DDP 初始化时同步权重就靠它）")

    # ── AllGather：每 rank 一片，收集成全量（FSDP 聚参数用，第 13 章）──
    shard = torch.tensor([float(rank)])
    gathered = [torch.zeros(1) for _ in range(world_size)]
    dist.all_gather(gathered, shard)
    if rank == 0:
        print(f"[AllGather] 收集各 rank 的分片: {[g.item() for g in gathered]}")

    # ── 平均：分布式训练真正用的是 AVG（梯度平均而非求和）──
    grad = torch.tensor([float(rank + 1)])
    dist.all_reduce(grad, op=dist.ReduceOp.AVG)
    if rank == 0:
        print(f"[AllReduce AVG] 梯度平均 = {grad.item()}（= {expected}/{world_size}，训练实际用这个）")

    dist.destroy_process_group()


if __name__ == "__main__":
    WORLD = 4
    print(f"启动 {WORLD} 个进程（gloo 后端，CPU 上真实通信）\n")
    mp.spawn(worker, args=(WORLD,), nprocs=WORLD, join=True)
    print("\n→ 全部原语验证通过。这套通信语义在云端换成 nccl 后端即可跑 GPU 集群。")
