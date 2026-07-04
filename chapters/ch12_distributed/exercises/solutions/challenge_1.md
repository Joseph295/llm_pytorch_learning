# 挑战 1 参考答案：云端 NCCL 实战 + scaling 分析

目标：把本章 gloo 代码搬到真实 GPU 集群，测量分布式训练的 scaling 效率。

## 从 gloo 到 NCCL：改动清单

本章 `ddp_minigpt.py` 上云只需 4 处改动（通信逻辑一字不变）：

```python
# 1. backend: gloo → nccl
dist.init_process_group("nccl", rank=rank, world_size=world)

# 2. 每进程绑定一张卡
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

# 3. 模型和数据搬到对应卡
model = DDP(model.to(device), device_ids=[local_rank])
x, y = x.to(device), y.to(device)

# 4. 启动用 torchrun（自动设 RANK/LOCAL_RANK/WORLD_SIZE 等环境变量）
#    torchrun --nproc_per_node=2 ddp_minigpt.py
```

## 云端操作步骤（AutoDL 为例，预估 ¥10~20）

```bash
# 1. 租 2×RTX 4090 实例，选官方 PyTorch 镜像（第 0 章纪律）
# 2. 传代码 + 数据（gpt_model.py, train.py, data/）
# 3. 单卡基线
torchrun --nproc_per_node=1 train_ddp.py --steps 500   # 记录 token/s
# 4. 双卡
torchrun --nproc_per_node=2 train_ddp.py --steps 500   # 记录 token/s
# 5. 实验完立即关机！（第 0 章纪律：按小时计费不看你在不在用）
```

## Scaling 效率分析

```
scaling efficiency = (N卡吞吐) / (N × 单卡吞吐)
理想 = 100%（2 卡吞吐 = 2× 单卡）
实际 < 100%，损失来自：AllReduce 通信开销、负载不均、数据加载瓶颈
```

小模型上 scaling 效率往往较低（通信占比高，模型小算得快、通信来不及藏）；
大模型效率高（计算时间长，通信被充分掩盖）。这解释了为什么数据并行更适合
"大模型 + 大 batch"。

## 用 profiler 看通信/计算重叠（第 11 章）

```python
with torch.profiler.profile(activities=[CPU, CUDA]) as prof:
    for _ in range(10): train_step()
# 在 trace 时间线上找 nccl AllReduce kernel：
# - 好的重叠：AllReduce 与反向的 compute kernel 在时间上并行（藏住了）
# - 差的重叠：AllReduce 独占一段时间（气泡）
```

## 实验清单（写进你的报告）

1. 单卡 vs 双卡吞吐（token/s）与 scaling efficiency
2. loss 曲线对比：双卡（有效 batch 2×）与单卡是否需要调 lr
3. 关掉 DDP bucketing（`bucket_cap_mb` 调到很小）观察吞吐下降 → 验证分桶的价值
4. profiler 时间线截图：AllReduce 是否被计算掩盖
5. `NCCL_DEBUG=INFO` 输出：确认走了哪条通信路径（4090 无 NVLink 走 PCIe）

## 加分观察

- 2×4090 无 NVLink，通信走 PCIe（带宽有限），小模型 scaling 可能只有 1.5~1.7×
- 加 `--gradient-accumulation` 用 `no_sync()`（12.5-①）减通信，看 scaling 改善
- 这就是为什么大厂训练用 NVLink/NVSwitch/InfiniBand——通信带宽直接决定 scaling 上限
