"""第 12 章 · 用 DDP 训练小模型（gloo 后端，M4 多进程）

运行：uv run chapters/ch12_distributed/code/ddp_minigpt.py

在 CPU 上用 gloo 后端跑真实的 DistributedDataParallel。演示：
  - 进程组初始化、DDP 包装、DistributedSampler 数据分片
  - 只在 rank 0 记日志（易错点④）
  - 跨 rank 聚合 loss（易错点⑤）
  - DDP 数学上等价于大 batch 单卡（基础练习 2）

改 backend='nccl' + device='cuda' 即可上云——通信逻辑一字不改。
"""

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset


def make_dataset():
    # 合成任务：y = sign(x·w)，固定种子保证各进程数据集一致（再由 sampler 切分）
    g = torch.Generator().manual_seed(0)
    X = torch.randn(2048, 16, generator=g)
    w = torch.randn(16, 1, generator=g)
    Y = (X @ w > 0).float()
    return TensorDataset(X, Y)


def worker(rank, world):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29503"
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.manual_seed(1337)                          # 各 rank 同种子 → 初始权重一致

    model = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 1))
    model = DDP(model)                               # 包装：注册 grad hook 做通信/计算重叠
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)

    ds = make_dataset()
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=32, sampler=sampler)   # 注意：用 sampler，不设 shuffle

    for epoch in range(20):
        sampler.set_epoch(epoch)                     # 每 epoch 变 shuffle（易错点②）
        for x, y in loader:
            loss = nn.functional.binary_cross_entropy_with_logits(model(x), y)
            opt.zero_grad()
            loss.backward()                          # 反向中 DDP 自动 AllReduce 梯度
            opt.step()

        # 跨 rank 聚合 loss 才是全局指标（易错点⑤）
        global_loss = loss.detach().clone()
        dist.all_reduce(global_loss, op=dist.ReduceOp.AVG)
        if rank == 0 and epoch % 5 == 0:             # 只在 rank 0 记日志（易错点④）
            print(f"[rank0] epoch {epoch:>2}: 全局平均 loss = {global_loss.item():.4f}")

    if rank == 0:
        # 存 checkpoint 要剥 DDP 的 module. 前缀（第 4 章 Q3）
        print(f"[rank0] 保存权重（model.module.state_dict()，剥 DDP 前缀）")
        print(f"        world_size={world}，有效 batch = 32 × {world} = {32 * world}")
    dist.destroy_process_group()


if __name__ == "__main__":
    WORLD = 4
    print(f"启动 {WORLD} 进程 DDP（gloo/CPU）。DistributedSampler 把 2048 样本切成 4 份不重叠。\n")
    mp.spawn(worker, args=(WORLD,), nprocs=WORLD, join=True)
    print("\n→ 4 进程各训练 1/4 数据，梯度 AllReduce 平均。数学上等价于 batch=128 的单卡训练。")
    print("  上云：backend='nccl'、模型和数据 .to(f'cuda:{local_rank}')、torchrun 启动。")
