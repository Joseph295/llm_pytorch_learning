# 第 13 章 · 大模型并行策略：ZeRO / FSDP / TP / PP / EP

> **本章目标**：当模型本身装不下单卡时，怎么把它切开。学完你应该能回答：
> 1. ZeRO 三个阶段各切什么？为什么 stage-1 几乎免费、stage-3 通信最贵？
> 2. FSDP 和 DDP 的本质区别是什么？FSDP 每层前向在做什么通信？
> 3. 张量并行（TP）和流水线并行（PP）各切模型的哪个维度？为什么 TP 只能在机内？
> 4. 训练一个 175B 模型，几种并行怎么组合（3D 并行）？
> 5. Megatron 和 DeepSpeed 的设计哲学差异？

**前置**：第 12 章（数据并行、集合通信）、第 2 章（显存账）。 **硬件路径**：概念 + FSDP 在 gloo 本地演示切分语义；真实多卡切分上云。 **预计用时**：6~7 小时。
**你的主场延续**：这一整章是"如何把一个大状态切分到多台机器并协调"——分区、副本、通信开销权衡，全是你的分布式老本行。

---

## 13.1 来龙去脉：数据并行的墙

第 12 章的数据并行有个前提没说破：**每张卡都得放下完整的模型状态**。第 2 章算过，7B 训练要 112GB，单卡放不下——数据并行在这里撞墙。DDP 复制 N 份模型，纯属浪费：N 张卡存了 N 份一模一样的参数/梯度/优化器状态。

**核心洞察**：既然 N 张卡存了 N 份相同的东西，为什么不各存 1/N，用时再互相要？这就是 **ZeRO（Zero Redundancy Optimizer）** 的思想——**消除数据并行中的冗余**。它不改变数据并行的计算逻辑，只是把"每卡存全量"改成"每卡存一片，用时 AllGather 拼出来"。用通信换显存。

而当单**层**都放不下、或者想要更极致的效率时，就要真正把模型的计算切开——**张量并行**（切矩阵乘）、**流水线并行**（切层）、**专家并行**（切 MoE 专家）。这些统称**模型并行**。

一句话理清这一章的地图：**数据并行切数据，模型并行切模型**；ZeRO/FSDP 是"切模型状态存储但不切计算"的中间形态；TP/PP/EP 是"连计算一起切"。真实的千亿模型训练是它们的组合（3D/4D 并行）。

---

## 13.2 核心原理

### ① ZeRO 三阶段：渐进式消除冗余

数据并行里每卡冗余存储三样东西（第 2 章的账，混合精度 AdamW）：优化器状态（12 字节/参数，fp32 的 m/v/主参数）、梯度（2 字节）、参数（2 字节）。ZeRO 按"通信代价从小到大"依次切分：

- **ZeRO-1（切优化器状态）**：每卡只存 1/N 的优化器状态。省最大头（12→12/N 字节/参数），**几乎免费**——优化器更新本来就各管各的参数分片，只在更新后 AllGather 参数。7B 从 112GB 降到约 112 - 12×7×(1-1/N) GB。
- **ZeRO-2（再切梯度）**：梯度也只存 1/N。反向时梯度算完立即 ReduceScatter（每卡只保留自己那片的归约结果），而非 AllReduce 全量。通信量与 DDP 相当，显存再降。
- **ZeRO-3（再切参数）**：参数本身也只存 1/N。**代价最大**：前向和反向每用到一层，都要先 AllGather 那层的完整参数、用完立即释放。通信量增加约 50%，但显存降到极致（参数/梯度/优化器全部 1/N）——这才能训单卡完全放不下的模型。

**关键权衡表**（记住这张，面试常问）：

| | 切什么 | 单卡显存(model state) | 额外通信 |
|---|---|---|---|
| DDP | 无（全冗余） | 16 字节/参数 | AllReduce 梯度 |
| ZeRO-1 | 优化器状态 | ~4 + 12/N | ≈DDP |
| ZeRO-2 | +梯度 | ~2 + 14/N | ≈DDP |
| ZeRO-3 | +参数 | 16/N | ≈1.5×DDP |

### ② FSDP：PyTorch 原生的 ZeRO-3

**FSDP（Fully Sharded Data Parallel）**是 PyTorch 官方内置的 ZeRO-3 实现（DeepSpeed 是微软的第三方实现）。它的工作方式，正好把第 4 章的 hook 和第 12 章的 AllGather 串起来：

模型按"包裹单元"（通常每个 Transformer 层一个 FSDP unit）分片，每卡只存每层参数的 1/N。运行时：

```
前向到第 k 层 → forward_pre_hook: AllGather 第 k 层完整参数（从各卡凑齐）
             → 计算这一层
             → forward_hook: 释放非本卡那部分参数（显存立即回收）
反向到第 k 层 → 同样 AllGather 参数 + 计算梯度 + ReduceScatter 梯度
```

**任意时刻，显存里只有"全部参数的分片 + 当前这一层的完整参数"**——峰值显存 ≈ 参数/N + 单层参数。这就是 FSDP 能在有限显存上训大模型的原理。第 4 章"为什么 FSDP 模型不能直接调 `model.forward()`"的答案在此：直接调跳过 pre-hook，参数还是分片状态（空的），直接崩。

FSDP 的调优点：**包裹粒度**（每层一个 unit 还是几层一个——粒度细省显存但通信频繁）、**prefetch**（提前 AllGather 下一层，藏住通信，又是流水线思想）、**混合分片**（机内分片 + 机间复制，`HYBRID_SHARD`）。

### ③ 张量并行（TP）：把单个矩阵乘切开

ZeRO/FSDP 切的是"存储"，计算还是每卡完整跑一遍（只是参数临时凑齐）。**张量并行**切的是"计算本身"——把一个大矩阵乘分到多卡并行算。

Megatron 的经典做法，切 Transformer 的两个部分：
- **MLP**：`Y = GeLU(X·A)·B`。把 A 按列切（A = [A₁, A₂]），各卡算 `GeLU(X·Aᵢ)`；B 按行切，各卡算部分和，最后 AllReduce 求和。一次前向一次 AllReduce。
- **注意力**：按注意力头切（每卡算一部分头），天然并行，输出投影时 AllReduce。

**关键约束**：TP 每层前向/反向都要 AllReduce（同步激活），通信量大且频繁 → **只能在机内高速互联（NVLink）上做**，跨机器网络太慢会拖垮。所以 TP 的并行度通常 = 单机 GPU 数（如 8）。这是 TP 和数据并行/PP 的本质区别：**TP 通信在关键路径上，必须极快**。

### ④ 流水线并行（PP）：把层切到不同卡

**流水线并行**把模型的层分段放到不同卡：卡0 放 1-8 层，卡1 放 9-16 层……前向时激活像流水线一样从卡0传到卡1传到卡2。问题是**流水线气泡**：卡1 要等卡0 算完第一个 micro-batch 才能开工，最后一卡最后才启动——首尾有空转。

解法是 **micro-batch 流水**（GPipe/1F1B）：把一个 batch 切成多个 micro-batch 连续送入，让各卡的流水线填满。气泡占比 ≈ (stages-1)/(micro-batches + stages - 1)——micro-batch 越多气泡越小。PP 的通信量小（只在段边界传激活）、可跨机器，但气泡和负载均衡（每段计算量要相等）是调优难点。这套"流水线填充"你在 CPU 指令流水、Spark stage 流水里都见过。

### ⑤ 3D 并行：组合起来训千亿模型

真实的超大模型训练是三者组合（DeepSpeed/Megatron-LM 的 3D 并行）：

```
TP（机内 8 卡，切矩阵）× PP（跨节点分段，切层）× DP（多副本，切数据）
```

例如 175B 模型在 1024 卡上：TP=8（机内）× PP=8（8 段）× DP=16（16 个副本），8×8×16=1024。每个维度解决一个问题：TP 让单层放得下、PP 让整个模型放得下、DP 加速吞吐。再叠加 ZeRO 切 DP 维度的冗余、EP 切 MoE 专家——就是 4D/5D 并行。**配置这些并行度的组合是大模型训练工程师的核心技能**（也是第 20 章面试重点）。

### ⑥ 专家并行（EP）：MoE 的专属维度

MoE（第 10 章）的专家可以分到不同卡：卡0 放专家 0-7，卡1 放专家 8-15……token 经过 router 后要 **All-to-All** 通信发到对应专家所在的卡，算完再 All-to-All 收回。这是 MoE 特有的通信模式（第 12 章的 All-to-All 原语在此落地）。EP 让 MoE 的海量专家参数分摊到多卡，代价是 All-to-All 通信（对网络要求高）。

---

## 13.3 动手实验

```bash
uv run chapters/ch13_parallelism/code/zero_memory_calc.py    # ZeRO 各阶段显存账计算器
uv run chapters/ch13_parallelism/code/fsdp_demo.py           # gloo 本地 FSDP，观察参数分片与聚合
uv run chapters/ch13_parallelism/code/pipeline_bubble.py     # 流水线气泡的量化模拟
uv run chapters/ch13_parallelism/code/tensor_parallel.py     # 手写 MLP 张量并行，验证等价性
```

`zero_memory_calc.py` 把 13.2-① 的表做成交互计算器：输入模型大小和卡数，算出 DDP/ZeRO-1/2/3 各自的单卡显存——直观看到"为什么训 70B 必须 ZeRO-3"。`fsdp_demo.py` 用 AllGather 在 M4 上**手动复刻** FSDP 的分片—聚合—释放循环，打印每层前向的聚合过程——比调用黑盒更能看清机制。

> **为什么手动复刻而不用 `FSDP` 类**：PyTorch 的 `FullyShardedDataParallel` 强依赖 CUDA（在 MPS 上初始化会因缺 `torch.mps.current_device` 而崩——本章开发时实测）。FSDP 本质是 CUDA 生态的技术，M4 上跑不了真身。但它的**核心机制就是 AllGather 参数 + 用完释放**，我们用第 12 章验证过的 gloo AllGather 手动实现同一逻辑，语义完全一致。上云换成真 `FSDP(model, device_id=...)` 即可，机制你已经懂了。

---

## 13.4 易错点清单

**① FSDP 模型直接调 forward 或访问 .weight**
→ **现象**：拿到空/分片的参数，输出错误或崩溃。
→ **原因**：参数平时是分片的，只在 forward 的 pre-hook 里临时 AllGather（13.2-②）。绕过 `model(x)` 就绕过了聚合。
→ **修正**：永远 `model(x)`；要访问完整参数用 `FSDP.summon_full_params(model)` 上下文。

**② 把 TP 跨机器部署**
→ **现象**：训练极慢，通信成为瓶颈。
→ **原因**：TP 每层 AllReduce 在关键路径上，跨机网络（几十 GB/s）比 NVLink（几百 GB/s）慢一个量级（13.2-③）。
→ **修正**：TP 并行度 ≤ 单机 GPU 数；跨机用 PP/DP。

**③ PP 的 micro-batch 太少，气泡吃掉收益**
→ **现象**：8 段流水线，加速远不到 8×。
→ **原因**：micro-batch 少时气泡占比大（13.2-④）。
→ **修正**：增加 micro-batch 数（受显存限制）；用 1F1B 调度减少激活驻留。

**④ 各并行维度的 rank 分组搞错**
→ **现象**：3D 并行下通信组配置错，AllReduce 在错误的 rank 集合上做，结果错误或死锁。
→ **原因**：TP/PP/DP 各有自己的通信子组（process group），rank 到组的映射复杂。
→ **修正**：用成熟框架（Megatron/DeepSpeed）管理 process group，别手搓；理解 `torch.distributed` 的 subgroup 机制。

**⑤ FSDP + 梯度累积 + 混合精度的配置冲突**
→ FSDP 有自己的混合精度策略（`MixedPrecision` 配置分片/通信/计算各用什么精度），和 autocast 叠加要小心；梯度累积时 `no_sync` 的语义与 DDP 不同（FSDP 累积期间参数仍要 AllGather）。跟随官方 recipe，别自由组合。

**⑥ checkpoint 的分片保存/加载**
→ FSDP 的 state_dict 默认是分片的（每卡存自己那片），保存/加载要用 `FSDP.state_dict_type` 指定 full/sharded/local。存成 full 会 OOM（要把全参数聚到一张卡），存 sharded 又不能直接给单卡模型加载。第 15 章微调实战会踩这个。

---

## 13.5 开源项目的最佳实践

**① DeepSpeed ZeRO：配置驱动的并行**
DeepSpeed 用一个 JSON 配置（`zero_optimization: {stage: 3, offload_optimizer: {device: cpu}}`）就切换 ZeRO 阶段和 CPU offload（把优化器状态卸载到内存，用带宽换显存，能在单卡训超大模型）。读它的配置文档 = 看到"显存不够时的完整武器菜单"。ZeRO-Infinity 进一步 offload 到 NVMe。

**② PyTorch FSDP2：原生分片的演进**
新版 `torch.distributed.fsdp` 的 `fully_shard` API（FSDP2）用 DTensor（分布式张量）重构，分片语义更清晰。读官方 FSDP tutorial 的包裹策略（`auto_wrap_policy` 按 Transformer 层自动分 unit）和 prefetch 配置。

**③ Megatron-LM：TP+PP 的工业标杆**
Megatron 的 `ColumnParallelLinear`/`RowParallelLinear` 是张量并行的教科书实现（13.2-③ 的 A 按列切、B 按行切）。读它如何用 `all_reduce`/`all_gather` 在恰当的位置同步激活，以及 PP 的 1F1B 调度实现。NVIDIA 训所有大模型都用它。

**④ 组合的现实**：Llama-3 405B 用 TP×PP×DP×CP（上下文并行，切超长序列的注意力）多维组合，配置这些的比例是一门经验科学（受硬件拓扑、模型结构、batch 大小共同约束）。

---

## 13.6 典型面试题

**Q1：ZeRO 三个阶段分别切什么？通信开销如何递增？什么时候用哪个？**

> **参考答案**：Stage-1 切优化器状态（省最大头 12 字节/参数，通信≈DDP，几乎免费，优先用）；Stage-2 再切梯度（反向用 ReduceScatter 代替 AllReduce，通信仍≈DDP）；Stage-3 再切参数（前向/反向按层 AllGather 参数用完释放，通信增约 50%，但显存降到 16/N，训单卡放不下的模型才用）。选择：能用低 stage 就用低的（通信少）；显存不够逐级往上。**加分点**：ZeRO-Offload/Infinity 把状态卸载到 CPU/NVMe 进一步省显存（带宽换容量）；ZeRO-3 = FSDP。

**Q2：FSDP 和 DDP 的本质区别？FSDP 每层前向做什么？**

> **参考答案**：DDP 每卡存完整模型（冗余），只 AllReduce 梯度；FSDP（=ZeRO-3）每卡只存参数的 1/N，前向到每层时 AllGather 该层完整参数、算完释放，反向 AllGather + ReduceScatter 梯度。FSDP 用通信换显存，能训大得多的模型。前向每层：pre-hook AllGather 参数 → 计算 → 释放非本卡分片。**加分点**：FSDP 的 prefetch 藏通信、包裹粒度权衡、直接调 forward 会崩（参数是分片的）；峰值显存 ≈ 参数/N + 单层完整参数。

**Q3：张量并行为什么只能机内，流水线并行为什么能跨机？**

> **参考答案**：TP 把单个矩阵乘切开，每层前向/反向都要 AllReduce 同步激活——通信在关键路径上且频繁，必须极快，只有机内 NVLink（几百 GB/s）扛得住，跨机网络会拖垮。PP 只在层段边界传激活，通信量小、频率低，跨机网络（几十 GB/s）够用。所以典型配置 TP=机内卡数、PP 跨节点。**加分点**：TP 通信量 ∝ 激活大小 × 层数，PP ∝ 激活大小 × 段数；PP 的代价是流水线气泡（要 micro-batch 填充）；CP（上下文并行）切序列维处理长上下文。

**Q4：训练一个 175B 模型在 1024 张 A100 上，你会怎么组合并行策略？**

> **参考答案**：3D 并行。TP=8（机内 8 卡切矩阵，让单层放得下，用 NVLink）；PP=8（跨节点分 8 段，让整个模型放得下）；DP=16（16 个副本加速吞吐，8×8×16=1024）。DP 维度叠加 ZeRO-1 切优化器状态冗余。再配 gradient checkpointing（第 11 章）省激活、bf16 混合精度。调优目标是 MFU 最大化（第 11 章），受气泡、通信、负载均衡制约。**加分点**：具体比例取决于模型结构（层数决定 PP 上限）、单层大小（决定 TP 需求）、全局 batch（决定 DP）、网络拓扑；MoE 模型再加 EP。

---

## 13.7 疑难杂症排查

**案例 1：FSDP 训练显存比预期高，或时不时 OOM**

① 包裹粒度太粗（整个模型一个 unit）→ AllGather 时聚了太多参数，改成按层包裹（`transformer_auto_wrap_policy`）；② prefetch 太激进（同时聚了太多层的参数）→ 调 `backward_prefetch`；③ full state_dict 保存时把全参数聚到一张卡 OOM → 用 sharded state_dict（易错点⑥）；④ 激活没 checkpoint → 叠加 gradient checkpointing。**方法论**：FSDP 显存问题先看"同时聚齐了多少参数"。

**案例 2：3D 并行下结果错误但不报错**

几乎都是 process group 配置错（易错点④）——某个集合通信在错误的 rank 子集上做。排查：打印每个 rank 的 (dp_rank, tp_rank, pp_rank) 确认分组正确；用小模型 + 已知输出做端到端对拍；逐个关闭并行维度（先纯 DP 对拍，再加 TP，再加 PP）定位哪一维引入错误。**方法论**：多维并行出错，逐维度二分定位。

**案例 3：PP 加速远不及理论**

① micro-batch 太少，气泡大（易错点③）——用 `pipeline_bubble.py` 算理论气泡占比对照；② 各段计算量不均（层数平分但 embedding/lm_head 那段更重）→ 手动调整分段边界均衡负载；③ 段间通信没和计算重叠。**方法论**：PP 效率问题先算气泡占比，再查负载均衡。

---

## 13.8 练习题

### 基础 1：ZeRO 显存计算器
用 `zero_memory_calc.py` 算：7B/70B/175B 模型在 8/64/1024 卡上，DDP vs ZeRO-1/2/3 的单卡显存。回答：70B 用 8 卡 ZeRO-3 够 80GB 卡吗？需要多少卡 ZeRO-3 才放得下 175B？

### 基础 2：流水线气泡
用 `pipeline_bubble.py` 算：8 段流水线，micro-batch 数 = 1/8/32/128 时的气泡占比和有效加速比。画出"micro-batch 数 vs 加速比"曲线，找到"气泡 < 10%"需要的 micro-batch 数。

### 进阶 1：手写 MLP 张量并行
用 gloo 起 2 进程，实现 Megatron 式 MLP 张量并行（A 列切、B 行切、AllReduce 求和），与单进程完整 MLP 对拍 allclose。测量通信量，理解为什么每层都要通信。

### 挑战 1：3D 并行的 rank 分组
不用框架，手动为 TP=2×PP=2×DP=2（8 进程）构造三套 process group（每个 rank 属于一个 tp_group、一个 pp_group、一个 dp_group）。打印每个 rank 的三维坐标和它所属的三个组，验证分组正确（同一 tp_group 的 rank 应能互相 AllReduce）。这是理解 3D 并行通信结构的核心练习。

---

## 本章小结与下一章预告

模型装不下单卡的解法分两层：ZeRO/FSDP 切模型状态存储（不切计算，通信换显存，渐进消除冗余）；TP/PP/EP 切计算本身（TP 切矩阵必须机内、PP 切层可跨机、EP 切专家）。千亿模型训练是它们的多维组合（3D/4D 并行），配置比例是核心工程技能。这一整章都是你的分布式老本行——分区、副本、通信权衡的深度学习版。

**下一章（第 14 章）**：微调工程。预训练太贵（第 9-13 章都在为它铺路），大多数人的实战是**微调**——在预训练模型上做 SFT、用 LoRA/QLoRA 低成本适配、用 DPO 对齐人类偏好。参数高效微调（PEFT）为什么能用 1% 的参数达到接近全参微调的效果？第 15 章你会用这些技术在云端真正微调一个 7B 模型。
