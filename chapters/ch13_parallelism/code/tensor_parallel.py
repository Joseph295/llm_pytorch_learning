"""第 13 章 · 手写 MLP 张量并行（Megatron 式），验证与完整 MLP 等价

运行：uv run chapters/ch13_parallelism/code/tensor_parallel.py

Megatron MLP TP：Y = GeLU(X·A)·B
  - A 按列切：各卡算 GeLU(X·Aᵢ)（列并行，无需通信）
  - B 按行切：各卡算部分和，最后 AllReduce 求和（行并行，一次通信）
"""

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F


def worker(rank, world, d, hidden, X, A, B, out_ref):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29504"
    dist.init_process_group("gloo", rank=rank, world_size=world)

    # 每卡分到 A 的一段列、B 的对应段行
    h_per = hidden // world
    A_shard = A[:, rank * h_per:(rank + 1) * h_per]      # (d, hidden/world)  列切
    B_shard = B[rank * h_per:(rank + 1) * h_per, :]      # (hidden/world, d)  行切

    # 列并行：本卡算自己那段隐藏单元（无通信）
    h_local = F.gelu(X @ A_shard)                        # (N, hidden/world)
    # 行并行：本卡算部分输出和
    y_partial = h_local @ B_shard                        # (N, d)
    # AllReduce 求和得到完整输出（每层一次通信——13.2-③ 的关键代价）
    dist.all_reduce(y_partial, op=dist.ReduceOp.SUM)

    if rank == 0:
        ok = torch.allclose(y_partial, out_ref, atol=1e-4)
        print(f"[world={world}] TP MLP 与完整 MLP 对拍: {ok}")
        comm = X.size(0) * d                             # AllReduce 一个 (N,d) 张量
        print(f"  每次前向通信量 = 1 次 AllReduce({X.size(0)}×{d}) = {comm} 元素")
        print(f"  → 每层前向都要通信同步激活，所以 TP 必须机内 NVLink（易错点②）")
    dist.destroy_process_group()


if __name__ == "__main__":
    torch.manual_seed(0)
    N, d, hidden = 8, 64, 256
    X = torch.randn(N, d)
    A = torch.randn(d, hidden) * 0.1
    B = torch.randn(hidden, d) * 0.1
    out_ref = F.gelu(X @ A) @ B                          # 单卡完整 MLP 作参照

    print("验证张量并行 MLP 的正确性与通信代价：\n")
    for world in [2, 4]:
        mp.spawn(worker, args=(world, d, hidden, X, A, B, out_ref), nprocs=world, join=True)
    print("\n→ TP 把矩阵乘切到多卡并行算，代价是每层一次 AllReduce（在关键路径上）")
