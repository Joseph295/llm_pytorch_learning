"""第 12 章 · 手写 Ring AllReduce：验证通信量与进程数无关

运行：uv run chapters/ch12_distributed/code/ring_allreduce.py

用点对点 send/recv 实现 Ring AllReduce（ReduceScatter + AllGather 两阶段），
与内置 dist.all_reduce 对拍，并统计单进程收发的数据量。
"""

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def ring_all_reduce(tensor: torch.Tensor, rank: int, world: int) -> int:
    """环形归约。返回本进程单向发送的元素总数（验证通信量用）。

    张量切成 world 块。两阶段各 world-1 步，每步只和左右邻居通信。
    只计发送量（收发对称，算一个方向即可，与网络带宽讨论惯例一致）。
    """
    chunks = list(tensor.chunk(world))
    left = (rank - 1) % world
    right = (rank + 1) % world
    sent = 0

    # ── 阶段一：ReduceScatter（world-1 步）──
    # 每步：把某块发给右邻居，从左邻居收一块累加。
    for step in range(world - 1):
        send_idx = (rank - step) % world
        recv_idx = (rank - step - 1) % world
        recv_buf = torch.zeros_like(chunks[recv_idx])
        # 用 isend/irecv 避免死锁（同时收发）
        reqs = [dist.isend(chunks[send_idx].contiguous(), right),
                dist.irecv(recv_buf, left)]
        for r in reqs:
            r.wait()
        chunks[recv_idx] += recv_buf
        sent += chunks[send_idx].numel()             # 只计发送方向

    # ── 阶段二：AllGather（world-1 步）──
    # 每步：把已完整的块沿环传，人人集齐。
    for step in range(world - 1):
        send_idx = (rank - step + 1) % world
        recv_idx = (rank - step) % world
        recv_buf = torch.zeros_like(chunks[recv_idx])
        reqs = [dist.isend(chunks[send_idx].contiguous(), right),
                dist.irecv(recv_buf, left)]
        for r in reqs:
            r.wait()
        chunks[recv_idx] = recv_buf
        sent += chunks[send_idx].numel()             # 只计发送方向

    tensor.copy_(torch.cat(chunks))
    return sent


def worker(rank, world, size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29502"
    dist.init_process_group("gloo", rank=rank, world_size=world)

    torch.manual_seed(rank)
    x = torch.randn(size)
    ref = x.clone()
    dist.all_reduce(ref, op=dist.ReduceOp.SUM)       # 内置版做参照

    sent = ring_all_reduce(x, rank, world)
    ok = torch.allclose(x, ref, atol=1e-5)
    if rank == 0:
        theory = 2 * (world - 1) / world * size
        print(f"[world={world}] Ring AllReduce 与内置 all_reduce 对拍: {ok}")
        print(f"  单进程发送元素数 = {sent}（理论 2·(N-1)/N·size = {theory:.0f}）")
        print(f"  占模型大小的倍数 = {sent / size:.2f}×（N→∞ 时趋于 2×，永远不超过 2×）")

    dist.destroy_process_group()


if __name__ == "__main__":
    SIZE = 1200                                       # 能被 2/3/4/6 整除
    print("验证 Ring AllReduce 的通信量与进程数无关（12.2-②）：\n")
    for world in [2, 3, 4]:
        mp.spawn(worker, args=(world, SIZE), nprocs=world, join=True)
    print("\n→ 单进程收发量始终 ≈ 2×模型大小，与卡数无关——数据并行可扩展到千卡的根基")
