# 第 12 章 · 分布式训练原理：AllReduce 就是你熟悉的 shuffle

> **本章目标**：理解数据并行的通信机制，并在 M4 上用 gloo 后端亲手跑通多进程分布式。学完你应该能回答：
> 1. DDP 每步到底在通信什么？为什么是 AllReduce 而不是别的？
> 2. Ring AllReduce 的通信量为什么与 GPU 数无关？（这题你的分布式背景会秒懂）
> 3. DDP 怎么把通信和反向计算重叠起来藏住延迟？
> 4. `DistributedSampler` 不加会怎样？为什么每个 rank 要看不同数据？

**前置**：第 6 章（训练循环）、第 3 章（梯度）、第 5 章（数据分片）。 **硬件路径**：gloo 后端本地多进程（免费，真实通信语义）+ 云端 NCCL 实战（标注费用）。 **预计用时**：6~7 小时。
**你的主场**：AllReduce、Ring 算法、通信/计算重叠、数据分片——这些概念你在 Spark shuffle、参数服务器、MPI 里都见过。本章几乎是"把已知概念翻译到 PyTorch API"。

---

## 12.1 来龙去脉：单卡装不下，或者单卡太慢

两个独立的动机把训练推向多卡：

1. **速度**：数据太多，单卡训一遍要几个月。多卡**数据并行**——每卡拿一份数据分片、各算各的梯度、同步后一起更新，理想情况下 N 卡快 N 倍。
2. **容量**：模型太大，单卡显存装不下（第 2 章：7B 训练要 112GB）。这需要**模型并行**（第 13 章），本章先讲数据并行。

**数据并行的核心问题**：每卡用不同数据算出不同的梯度，但它们要更新**同一个模型**。怎么让 N 份梯度合并成一份？答案是求平均（数学上等价于用大 batch 算的梯度，第 3 章梯度累积的分布式版）。"把 N 台机器上的数组按元素求和再分发回去"——这正是 **AllReduce**，你在 MPI/参数服务器/Spark 里的老朋友。

**心智映射表**（把 PyTorch 分布式翻译成你已知的概念）：

| PyTorch 分布式 | 你已经懂的对应物 |
|---|---|
| AllReduce 梯度 | Spark 的 reduce/aggregate，MPI_Allreduce |
| Ring AllReduce | 环形拓扑的带宽最优归约 |
| rank / world_size | executor id / 总 executor 数 |
| DistributedSampler | 数据分片（每 partition 归一个 worker） |
| 通信/计算重叠 | 流水线，隐藏 IO 延迟 |
| NCCL / gloo | 通信后端（≈ 网络传输层实现） |
| 集合通信卡住 | 某个 executor 掉队导致 barrier 死锁 |

---

## 12.2 核心原理

### ① 集合通信原语：分布式的"指令集"

分布式训练建立在几个**集合通信（collective）**原语上（NCCL/gloo 实现它们）：

- **AllReduce**：所有 rank 各有一个数组，归约（求和）后每个 rank 都拿到结果。梯度同步的主力。
- **Broadcast**：一个 rank 的数据发给所有 rank。初始化时同步模型权重（保证各卡起点一致）。
- **AllGather**：每个 rank 有一片，收集后每个 rank 拿到全部拼接。FSDP 聚参数用（第 13 章）。
- **ReduceScatter**：归约后每个 rank 只拿结果的一片（AllReduce = ReduceScatter + AllGather）。ZeRO 用。
- **All-to-All**：每个 rank 给每个 rank 发不同的数据。MoE 的 token 路由用（第 10/13 章）。

这套原语就是分布式训练的"指令集"，所有并行策略都是它们的组合。你会发现它和 MPI 的原语几乎一一对应——因为思想同源。

### ② Ring AllReduce：为什么通信量与卡数无关

朴素 AllReduce（所有卡把梯度发给一个中心节点求和再广播）有个致命问题：中心节点的带宽成为瓶颈，且通信量随卡数线性增长。**Ring AllReduce**（Baidu 2017 引入深度学习）是带宽最优解，你的分布式背景会秒懂它的精妙：

N 个 GPU 组成环，每个只和左右邻居通信。梯度数组切成 N 块，两个阶段：
- **ReduceScatter 阶段**（N-1 步）：每步每个 GPU 把一块发给下家、从上家收一块累加。N-1 步后，每个 GPU 持有"某一块的完整求和结果"。
- **AllGather 阶段**（N-1 步）：把这些完整块沿环传一圈，人人集齐。

**关键结论**：每个 GPU 单向发送的数据量 = `2 × (N-1)/N × 模型大小`——**N→∞ 时趋于 2×模型大小，且永远不超过 2×**（N=2 时是 1×，N 越大越接近但不超 2×）。对比朴素参数服务器方案：中心节点要收 N×模型大小（线性爆炸）。Ring 的通信量**被 2× 封顶、不随 N 线性增长**，这就是数据并行能扩展到成千上万卡的根基。（注意区分：通信**量**有上界，但**步数** 2(N-1) 随 N 增长，所以超大规模用分层/树形拓扑减步数。）本章实验用 gloo 后端实测，验证正确性与这个通信量规律。

### ③ DDP 的两把利器：桶化 + 计算通信重叠

朴素实现：反向算完所有梯度 → AllReduce 全部 → 更新。问题：反向计算时通信带宽闲着，通信时算力闲着——串行浪费。DDP（DistributedDataParallel）的优化：

**通信/计算重叠**：反向传播是从输出层往输入层算的，**先算完的是靠近输出的层的梯度**。DDP 用 hook（第 4 章！每个参数注册 grad hook）监听——某层梯度一算完，立刻异步 AllReduce 它，同时反向继续算前面的层。等最后一层梯度算完时，前面层的通信早已完成。**通信被计算藏住了**——这就是你熟悉的"流水线隐藏延迟"。

**梯度分桶（bucketing）**：一个参数一个 AllReduce 太碎（每次通信有固定开销，第 0 章小 kernel 现象的通信版）。DDP 把多个参数的梯度攒进一个"桶"（默认 25MB）再一次 AllReduce——摊薄通信固定开销。桶的划分和触发时机是 DDP 性能的关键调优点。

### ④ 数据分片：DistributedSampler 的必要性

每个 rank 必须看**不同的数据**，否则 N 卡算的是同一批数据的相同梯度，AllReduce 求平均等于没并行（还白白 N 倍算力）。`DistributedSampler` 按 rank 把数据集切成不重叠的 N 份（第 5 章分片的分布式版）。三个必须记住的点：

1. **每 epoch 要 `sampler.set_epoch(epoch)`**——否则每个 epoch 的 shuffle 顺序相同（数据顺序固定，收敛变差，易错点②）；
2. **各 rank 的数据量要一致**——不齐会导致某 rank 提前跑完、在下一次 AllReduce 处死等其他 rank（集合通信要求所有 rank 都参与，第 12.7 的经典死锁）；
3. **有效 batch size = 单卡 batch × world_size**——学习率可能要相应缩放（linear scaling rule）。

### ⑤ 分布式的启动与进程模型

分布式训练是**多进程**（每 GPU 一个进程，不是多线程——GIL，第 1 章）。`torchrun --nproc_per_node=4 train.py` 启动 4 个进程，每个进程通过环境变量拿到自己的 `rank`/`local_rank`/`world_size`，用 `dist.init_process_group(backend="nccl")` 加入通信组。进程间通过 NCCL（GPU）或 gloo（CPU）通信。

关键概念：**rank**（全局进程号）、**local_rank**（本机内的 GPU 号，用于 `torch.cuda.set_device`）、**world_size**（总进程数）。这套模型和你启动 Spark executor、MPI rank 几乎一样。本章实验在 M4 上用 gloo 后端 + 多进程模拟这整套流程——没有 GPU 也能理解通信语义。

---

## 12.3 动手实验

```bash
# gloo 后端本地多进程（免费，真实通信语义）
uv run chapters/ch12_distributed/code/allreduce_demo.py       # 手动 AllReduce，验证梯度平均
uv run chapters/ch12_distributed/code/ddp_minigpt.py          # 用 DDP 训练小模型，4 进程 gloo
uv run chapters/ch12_distributed/code/ring_allreduce.py       # 手写 Ring AllReduce，验证与卡数无关
```

云端 NCCL 实战（第 12.8 挑战题）：租 2×GPU 实例，`torchrun --nproc_per_node=2` 跑同样的 DDP 脚本，对比 gloo/NCCL 的吞吐——预估费用 ¥10~20（2 小时双卡）。

`ddp_minigpt.py` 用 `torch.multiprocessing.spawn` 起 4 个进程，gloo 后端在 CPU 上做真实的 AllReduce——**通信逻辑与云端 NCCL 完全一致**，只是后端和设备不同。你在这里调通的代码，改个 backend 就能上云。

---

## 12.4 易错点清单

**① 忘了用 DistributedSampler，每个 rank 看相同数据**
→ **现象**：4 卡训练效果 = 1 卡（甚至更差，因为有效 batch 虚高但数据没变多），但速度没提升的收益。
→ **修正**：train loader 传 `sampler=DistributedSampler(dataset)`，且 `shuffle` 交给 sampler（不要同时设 `shuffle=True`，冲突报错）。

**② 忘了 `sampler.set_epoch(epoch)`**
→ **现象**：每个 epoch 数据顺序完全相同，等效于没 shuffle，收敛质量下降。
→ **修正**：每个 epoch 开头 `train_sampler.set_epoch(epoch)`。容易漏因为单卡训练没这个概念。

**③ 各 rank 数据量不齐导致死锁**
→ **现象**：训练卡在某一步永久 hang，无报错。
→ **原因**：某 rank 数据少，先跑完退出，其他 rank 在下一个 AllReduce 处等一个永远不来的参与者（集合通信要求全员到齐）。
→ **修正**：DistributedSampler 默认 `drop_last` 或补齐使各 rank 等长；`join()` 上下文处理不均。第 12.7 案例 1 详解。

**④ 日志/checkpoint 每个 rank 都写**
→ **现象**：4 份重复日志、checkpoint 互相覆盖/损坏、指标重复计数。
→ **修正**：只在 `rank == 0` 写日志和存 checkpoint（`if dist.get_rank() == 0:`）；存的是 `model.module.state_dict()`（剥 DDP 前缀，第 4 章 Q3）。

**⑤ 指标聚合忘了跨 rank 求平均**
→ **现象**：打印的 loss 只是 rank 0 那一片数据的 loss，不代表全局。
→ **修正**：`dist.all_reduce(loss, op=ReduceOp.AVG)` 后再记录（评估指标同理）。

**⑥ 随机性不同步导致模型分叉**
→ **现象**：各 rank 的模型悄悄变得不一样（dropout mask 不同 + 某些操作未同步），训练不稳定。
→ **修正**：DDP 初始化时会 broadcast rank 0 的权重保证起点一致；但要注意数据增强的随机性（第 5 章 worker seed）、以及不要在 rank 间做依赖随机数的分支。

---

## 12.5 开源项目的最佳实践

**① nanoGPT 的 DDP 集成（极简范本）**
nanoGPT 的 `train.py` 用 ~15 行加上 DDP：环境变量判断是否分布式、`init_process_group`、包 `DDP(model)`、只在 master rank 记日志、梯度累积时用 `no_sync()` 避免中间 micro-step 的无谓通信（只在最后一个 micro-step 同步）。**`no_sync()` 是个重要优化**：梯度累积的前 K-1 步不需要 AllReduce（反正还要继续累加），只在最后一步同步——省 K-1 次通信。

**② `DistributedDataParallel` 源码里的 Reducer**
[torch/nn/parallel/distributed.py](https://github.com/pytorch/pytorch/blob/main/torch/nn/parallel/distributed.py) + C++ 的 `reducer.cpp`：分桶逻辑、autograd hook 注册、通信/计算重叠的调度。读它 = 看到第 4 章 hook 机制在工业级系统里的核心应用。

**③ accelerate / lightning 的抽象**
HF `accelerate` 把"单卡 vs 多卡 vs 多机"抽象成同一份代码（`accelerator.prepare(model, opt, loader)` 自动处理 DDP 包装、sampler、设备放置）。生产代码常用它避免手写样板。但**理解底层再用抽象**——否则出问题（死锁、指标错）无从排查。本章先手写，第 15 章用 accelerate 微调。

---

## 12.6 典型面试题

**Q1：数据并行每步通信什么？Ring AllReduce 为什么高效？**

> **参考答案**：通信各卡的梯度，做 AllReduce（求和/平均）使所有卡用相同的平均梯度更新，保持模型一致。Ring AllReduce 把梯度分块沿环传递（ReduceScatter + AllGather 两阶段各 N-1 步），每卡单向通信量 2(N-1)/N×模型大小——被 2×模型大小封顶、不随 N 线性增长（对比朴素参数服务器中心节点的 N×爆炸），避免带宽瓶颈，可扩展到超大规模。**加分点**：区分通信量（有 2× 上界）和延迟（步数 ∝ N，故超大规模用分层/树形拓扑减步数）；NCCL 会根据拓扑（NVLink/PCIe/网络）自动选最优算法。

**Q2：DDP 如何隐藏通信开销？梯度累积时怎么进一步优化通信？**

> **参考答案**：通信/计算重叠——反向从输出层向输入层算，DDP 用 autograd hook 监听，某层梯度就绪即异步 AllReduce，同时反向继续算前层，通信被计算掩盖。梯度分桶把多参数梯度合并成一次通信摊薄固定开销。梯度累积时用 `no_sync()`：前 K-1 个 micro-step 不通信（梯度还要继续累加），只在第 K 步 AllReduce 一次，省 K-1 次通信。**加分点**：桶大小是延迟/带宽权衡的调优点；`find_unused_parameters` 的开销（动态图有未用参数时才需要，默认关）。

**Q3：DistributedSampler 的作用？不用会怎样？多机时还要注意什么？**

> **参考答案**：把数据集按 rank 切成不重叠的 N 份，保证每卡看不同数据——否则 N 卡算相同梯度，AllReduce 后等于单卡效果却花 N 倍算力。要点：每 epoch `set_epoch` 保证 shuffle 变化；各 rank 数据量须一致（不齐会死锁）；有效 batch = 单卡 batch × world_size（lr 可能要 linear scaling）。多机还要叠加节点级分片，且要保证各节点数据可访问（共享存储或各自本地副本）。**加分点**：IterableDataset 的分片要手动做（第 5 章 get_worker_info + rank）；数据不均时的 Join 上下文。

**Q4：分布式训练卡住（hang）不报错，如何排查？**

> **参考答案**：最常见是集合通信的 rank 不齐——某 rank 因数据量不同/异常提前退出/走了不同代码分支，导致其他 rank 在 AllReduce/barrier 处死等。排查：① 看各 rank 是否都到了同一个通信点（加 rank 日志）；② 检查数据分片是否等长（DistributedSampler 的 drop_last）；③ 是否有 rank 特有的条件分支跳过了某次集合通信（如 `if rank==0: extra_allreduce()`——致命）；④ `TORCH_DISTRIBUTED_DEBUG=DETAIL` 打印集合通信调用；⑤ NCCL 超时设置（`NCCL_TIMEOUT`）让 hang 变成报错。**加分点**：py-spy dump 各进程栈定位卡在哪个 collective；网络/NCCL 环境问题（`NCCL_DEBUG=INFO`）。

---

## 12.7 疑难杂症排查

**案例 1：多卡训练永久 hang，GPU 利用率 100% 但不前进**

分布式第一杀手，几乎都是**集合通信 rank 不齐**（易错点③）。排查树：① 每个 rank 在关键点打印 `[rank X] reached step N`，看谁没到——没到的那个 rank 就是病灶；② 数据分片不等长（最常见）→ DistributedSampler 确认 drop_last 或补齐；③ 条件分支导致某 rank 少调用了一次 collective（如 loss 为 nan 时某 rank `continue` 跳过了 backward+AllReduce）——**任何 rank-dependent 的控制流都可能让通信不齐**；④ `TORCH_DISTRIBUTED_DEBUG=DETAIL` + NCCL 超时把 hang 转成带栈报错。**方法论**：hang 找"谁没到齐"，不是找"谁错了"。

**案例 2：多卡 loss 正常但比单卡差**

① 忘了 DistributedSampler（易错点①）——各卡看相同数据；② 有效 batch 变大但 lr 没调（大 batch 需要 scale lr 或加 warmup，第 6 章）；③ 指标聚合错（易错点⑤）——其实模型没问题只是 loss 显示错；④ BN 类跨卡统计问题（LLM 用 LayerNorm/RMSNorm 无此问题，但其他模型要 SyncBN）。

**案例 3：NCCL 报错 / 初始化超时**

云端多卡高发：① `NCCL_DEBUG=INFO` 看它选了哪条通信路径（NVLink/PCIe/IB/socket）；② 网络接口选错（多网卡时 `NCCL_SOCKET_IFNAME` 指定）；③ 防火墙挡了 rendezvous 端口（`MASTER_ADDR`/`MASTER_PORT`）；④ 容器共享内存不足（第 5 章 `/dev/shm`，NCCL 也用它）。**方法论**：NCCL 问题先开 `NCCL_DEBUG=INFO`，它话很多但答案通常在里面。

---

## 12.8 练习题

### 基础 1：手动 AllReduce 验证梯度平均
用 gloo 后端起 2 个进程，每个进程造一个不同的张量，`dist.all_reduce` 求和，验证结果 = 两个张量之和 / 各进程拿到相同结果。（`allreduce_demo.py` 是起点。）

### 基础 2：DDP vs 手动数据并行
用 `ddp_minigpt.py` 跑 2 进程 DDP 训练，与单进程训练对比：相同总数据量下 loss 曲线是否一致？（应该一致——DDP 数学上等价于大 batch 单卡。）

### 进阶 1：手写 Ring AllReduce
用点对点通信（`dist.send`/`dist.recv`）实现 Ring AllReduce，与 `dist.all_reduce` 对拍结果。测量并验证：单进程收发的数据量与进程数近似无关（12.2-②）。

### 挑战 1：云端 NCCL 实战 + 通信分析
租 2×GPU 云实例，`torchrun --nproc_per_node=2` 跑 DDP 训练。用 profiler（第 11 章）看通信/计算重叠：AllReduce 是否被反向计算掩盖？测量 DDP 的 scaling efficiency（2 卡吞吐 / 单卡吞吐，理想 2.0，实际因通信 <2）。故意关掉 bucketing 或调小桶，观察吞吐变化。写一份分布式性能报告。（预估 ¥10~20。）

---

## 本章小结与下一章预告

数据并行 = 各卡不同数据算梯度 → AllReduce 平均 → 一起更新。Ring AllReduce 让通信量与卡数无关（可扩展的根基）；DDP 用 hook 做通信/计算重叠 + 分桶藏住延迟；DistributedSampler 保证数据分片。这些机制对你不陌生——它们是分布式系统思想在深度学习里的复用。你在 M4 上用 gloo 跑通的代码，改个 backend 就能上云千卡。

**下一章（第 13 章）**：大模型并行策略。数据并行解决"训得快"，但模型本身装不下单卡（7B 训练 112GB）时怎么办？ZeRO 把优化器状态/梯度/参数切片分到多卡，FSDP 是它的 PyTorch 原生实现；张量并行（TP）、流水线并行（PP）、专家并行（EP）各切一个维度。Megatron 和 DeepSpeed 的设计取舍，是训练千亿模型的工程核心。
