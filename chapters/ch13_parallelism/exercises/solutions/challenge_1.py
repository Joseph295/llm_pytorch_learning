"""挑战 1 参考答案：3D 并行的 rank 分组（TP=2 × PP=2 × DP=2 = 8 进程）

运行：uv run chapters/ch13_parallelism/exercises/solutions/challenge_1.py

3D 并行的核心是：每个 rank 同时属于三个通信子组（tp/pp/dp），
每个维度的集合通信只在对应子组内进行。搞错分组 = 通信在错误 rank 集合
上做 = 结果错或死锁（13.4 易错点④）。理解这个映射是配置 3D 并行的基础。
"""

import os

import torch.distributed as dist
import torch.multiprocessing as mp

TP, PP, DP = 2, 2, 2                                  # 三个维度的并行度，乘积 = world_size


def rank_to_coord(rank):
    """把全局 rank 映射到 (dp, pp, tp) 三维坐标。

    约定布局（与 Megatron 一致）：tp 变化最快（相邻 rank 在同一机内做 TP），
    然后 pp，最后 dp。rank = dp*(PP*TP) + pp*TP + tp
    """
    tp = rank % TP
    pp = (rank // TP) % PP
    dp = rank // (TP * PP)
    return dp, pp, tp


def worker(rank, world):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29506"
    dist.init_process_group("gloo", rank=rank, world_size=world)
    dp, pp, tp = rank_to_coord(rank)

    # 构造三套子组：同一子组内的 rank 才互相集合通信
    # tp_group: 固定 (dp,pp)，tp 变化 —— 同组做 TP 的 AllReduce
    # pp_group: 固定 (dp,tp)，pp 变化 —— 同组传流水线激活
    # dp_group: 固定 (pp,tp)，dp 变化 —— 同组做数据并行的梯度 AllReduce
    def build_groups(axis):
        groups = {}
        for r in range(world):
            d, p, t = rank_to_coord(r)
            key = {"tp": (d, p), "pp": (d, t), "dp": (p, t)}[axis]
            groups.setdefault(key, []).append(r)
        return groups

    my_groups = {}
    for axis in ["tp", "pp", "dp"]:
        groups = build_groups(axis)
        for members in sorted(groups.values()):
            g = dist.new_group(members)              # 所有 rank 必须一起创建所有组
            key = {"tp": (dp, pp), "pp": (dp, tp), "dp": (pp, tp)}[axis]
            d0, p0, t0 = rank_to_coord(members[0])
            member_key = {"tp": (d0, p0), "pp": (d0, t0), "dp": (p0, t0)}[axis]
            if member_key == key:
                my_groups[axis] = (g, members)

    if rank == 0:
        print(f"世界大小 {world} = TP{TP} × PP{PP} × DP{DP}\n")
    dist.barrier()
    # 每个 rank 报告自己的三维坐标和三个组
    info = (f"rank {rank}: 坐标(dp={dp},pp={pp},tp={tp}) | "
            f"tp组={my_groups['tp'][1]} pp组={my_groups['pp'][1]} dp组={my_groups['dp'][1]}")
    # 顺序打印（用 barrier 粗略串行化）
    for r in range(world):
        if r == rank:
            print(info)
        dist.barrier()

    # 验证：同一 tp 组内能 AllReduce
    import torch
    t = torch.tensor([float(rank)])
    dist.all_reduce(t, group=my_groups["tp"][0], op=dist.ReduceOp.SUM)
    if rank == 0:
        print(f"\nrank0 的 tp 组 AllReduce 结果 = {t.item()}"
              f"（= tp组成员 rank 之和 {sum(my_groups['tp'][1])}）✓")
        print("→ 每个维度的通信严格限制在对应子组内，这是 3D 并行正确性的基础")

    dist.destroy_process_group()


if __name__ == "__main__":
    WORLD = TP * PP * DP
    mp.spawn(worker, args=(WORLD,), nprocs=WORLD, join=True)
