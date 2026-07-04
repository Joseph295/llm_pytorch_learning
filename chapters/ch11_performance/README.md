# 第 11 章 · 单卡性能优化：让 GPU 忙起来

> **本章目标**：把"能训"变成"训得快"。学完你应该能回答：
> 1. 你的 miniGPT 在 M4 上慢，慢在哪？怎么用 profiler 看见？
> 2. 一个操作是被算力卡住（compute-bound）还是被带宽卡住（memory-bound）？怎么判断？
> 3. `torch.compile` 到底做了什么能加速？FlashAttention 为什么又快又省显存？
> 4. 显存到底被谁吃了？逐字节能不能算清楚？

**前置**：第 6 章（训练循环）、第 9 章（miniGPT）、第 2 章（显存账）。 **硬件路径**：本地测 profiler/compile/显存；Tensor Core 类加速云端更明显。 **预计用时**：6~7 小时。
**给你的定位**：性能分析是你的老本行（Spark stage、火焰图、shuffle 瓶颈）。这章把那套方法论迁移到 GPU——工具换了，"先测量、找主导项、再优化"的纪律不变。

---

## 11.1 来龙去脉：性能优化的第一性原理——Roofline

优化的第一步永远是**判断瓶颈类型**，否则就是瞎猜。GPU 上有一个比大数据世界更硬的物理约束框架：**Roofline 模型**。

任何计算的性能被两堵墙夹住：
- **算力墙**：芯片每秒能做多少浮点运算（FLOPS）。A100 bf16 ≈ 312 TFLOPS。
- **带宽墙**：芯片每秒能从显存搬多少字节（GB/s）。A100 HBM ≈ 2 TB/s。

一个操作到底撞哪堵墙，取决于它的**算术强度**（Arithmetic Intensity）= 计算量 / 访存量（FLOP per byte）：

```
算术强度低（如逐元素加法：1 次加法要读写 3 个数）→ 带宽墙 → memory-bound
算术强度高（如大矩阵乘：N³ 计算只需 N² 访存）    → 算力墙 → compute-bound
```

**这个判断决定了优化方向**：memory-bound 的操作，增加算力没用（芯片在等数据），要减少访存——**算子融合**（kernel fusion）把多个逐元素操作合并成一次读写；compute-bound 的操作才该上更快的矩阵单元（Tensor Core、更低精度）。

**LLM 的关键事实**：训练时大矩阵乘是 compute-bound（Tensor Core 的主场），但**大量胶水操作（norm、激活、残差加、dropout、bias）是 memory-bound**——它们计算量小却各自要完整读写激活张量。这些操作单独看不起眼，加起来能占相当比例的时间，且全是 kernel 启动开销 + 访存。`torch.compile` 和 FlashAttention 的核心价值就是**融合这些 memory-bound 操作**，减少 HBM 往返。你在第 8 章练习里实测过"手写 RMSNorm 比融合 kernel 慢"——那就是这个道理的预告。

---

## 11.2 核心原理

### ① Profiler：GPU 的火焰图

`torch.profiler` 是你的火焰图工具。它记录每个算子的 CPU 时间、GPU（CUDA/MPS）时间、调用次数、张量 shape，导出 Chrome trace 可视化时间线。三个必看信号：

1. **GPU 时间 vs 挂钟时间的比值**——GPU 忙碌占比（低 = 有气泡，被数据/同步/Python 卡住）；
2. **算子耗时排名**——时间花在哪几个 kernel（帕累托，优化前 20% 的热点）；
3. **kernel 数量**——大量微小 kernel = 融合机会（memory-bound 胶水操作的信号）。

方法论和你读 Spark UI 找慢 stage 完全一致：**先看全局占比，再钻热点，别优化不在关键路径上的东西**。

### ② 三种"同步气泡"的来源

GPU 忙碌占比低，气泡来自三处（第 0/2 章埋的伏笔在此收口）：
- **数据饥饿**（第 5 章）：DataLoader 喂不上；
- **强制同步**：`.item()`/`.cpu()`/`print(tensor)`/`if tensor>0` 逼 CPU 等 GPU（第 2 章易错点⑤）；
- **CPU 瓶颈**：Python 调度速度跟不上 GPU（小模型、小 batch 时 Python 每个算子的调度开销占比高——第 0 章的"小矩阵 GPU 反而慢"是同源）。`torch.compile` 和 CUDA Graph 能消掉 Python 调度开销。

### ③ torch.compile：把动态图"编译"掉

第 3 章说 PyTorch 的动态图灵活但优化空间小（看不到全局）。`torch.compile`（PyTorch 2.0 的旗舰）用 TorchDynamo 在运行时**捕获**计算图，用 TorchInductor **编译**成融合后的高效 kernel（GPU 上生成 Triton 代码）。带来三重收益：

1. **算子融合**：多个 memory-bound 操作合并成一个 kernel，一次读写完成（11.1 的核心）；
2. **消除 Python 开销**：编译后的图不再逐算子走 Python 调度；
3. **kernel 特化**：针对具体 shape/dtype 生成最优代码。

用法一行：`model = torch.compile(model)`。代价：首次运行要编译（几秒到几十秒，"预热"）；shape 频繁变化会触发重编译（LLM 训练定长 batch 所以无碍，动态 shape 要 `dynamic=True`）。**注意 MPS 后端对 compile 的支持仍不完整**，本章 compile 实验主要讲原理，加速在云端 CUDA 上兑现。

### ④ FlashAttention：memory-bound 的教科书胜利

第 7 章算过：注意力矩阵 (B,H,T,T) 的显存是 O(T²)，T=4K 时单层就 8.6GB。更糟的是它 **memory-bound**——把整个 T² 矩阵写到 HBM 再读回来做 softmax，访存量巨大而计算量相对小。

FlashAttention 的洞察：**永远不物化完整的 T² 矩阵**。它分块（tile）遍历 K/V，用**在线 softmax**（增量维护 running max 和 running sum，数学上等价于标准 softmax）在 SRAM（片上高速缓存）里完成计算，只把最终 O(T·d) 的结果写回 HBM。收益双杀：

- **显存** O(T²) → O(T)（不存注意力矩阵，反向时重算——第 3 章 gradient checkpointing 哲学的注意力专版）；
- **速度**更快（虽然多了重算的 FLOPs，但省下的 HBM 往返更值——因为它 memory-bound）。

这是"理解 Roofline 就能理解为什么它快"的最佳案例：省访存 > 多算几次。PyTorch 的 `F.scaled_dot_product_attention`（第 7 章你一直在用）在 CUDA 上自动调用 FlashAttention-2。

### ⑤ 显存逐字节解剖

第 2 章给了模型状态的账（16 字节/参数），这里补上完整的训练显存构成：

```
总显存 = 模型状态（参数+梯度+优化器，第 2 章）
       + 激活值（前向存下来给反向用，第 3 章，∝ batch×seq×layers×width）
       + 临时缓冲（kernel workspace、通信 buffer）
       + 分配器碎片（PyTorch 缓存分配器保留但未用的块）
```

**激活值常是训练显存的最大变量**（模型状态固定，激活随 batch/seq 涨）。三个压激活的武器：
- **gradient checkpointing**（第 3 章）：不存中间激活，反向重算，~33% 额外算力换大幅显存；
- **更小的 batch + 梯度累积**（第 6 章）：拿时间换显存；
- **FlashAttention**（④）：消掉注意力激活的 O(T²)。

工具：`torch.cuda.memory_summary()`/`memory_allocated()`（MPS 上 `torch.mps.current_allocated_memory()`）。排查 OOM 先看这本账——第 15 章的 OOM 实战全靠它。

### ⑥ MFU：性能的终极标尺

**GPU 利用率（utilization）会骗人**——它只表示"有 kernel 在跑"，不管跑得好不好（一个 memory-bound 的低效 kernel 也让利用率 100%）。真正的标尺是 **MFU（Model FLOPs Utilization）**：实际达到的 FLOPS / 硬件峰值 FLOPS。

```
每步理论计算量 ≈ 6 × 参数量 × token 数    （6 = 前向2 + 反向4 的经验系数）
MFU = (6 × N × tokens / 每步耗时) / 硬件峰值FLOPS
```

大模型训练的 MFU 能到 40-55% 就是很好的工程（Llama 训练报告在这个区间）。低于 30% 说明有明显浪费（数据、通信、气泡）。**MFU 是评价训练效率的行业硬指标**，面试高频（第 20 章）。本章实验会让你算出 miniGPT 在 M4 上的 MFU。

---

## 11.3 动手实验

```bash
uv run chapters/ch11_performance/code/profile_minigpt.py   # profiler 剖析 miniGPT，找热点与气泡
uv run chapters/ch11_performance/code/roofline.py          # 算术强度判定 compute/memory-bound
uv run chapters/ch11_performance/code/memory_anatomy.py    # 训练显存逐项拆解 + checkpointing 对比
uv run chapters/ch11_performance/code/mfu.py               # 算 miniGPT 的 MFU
```

`memory_anatomy.py` 会实测 gradient checkpointing 省了多少激活显存、多花了多少时间——第 3 章的理论这里变成 M4 上的数字。

---

## 11.4 易错点清单

**① 用 GPU 利用率当性能指标**
→ 利用率高 ≠ 高效（⑥）。memory-bound 的低效 kernel 也能打满利用率。用 MFU 或吞吐（token/s）。

**② profiling 不 warmup**
→ **现象**：第一次迭代包含 CUDA 上下文初始化、cudnn benchmark、compile 预热、内存分配等一次性开销，把它计进去数据全错。
→ **修正**：profile 前跑几步 warmup，只测稳态（第 0 章 benchmark 就是这么做的）。

**③ 计时忘同步**（第 0 章易错点⑥的重申）
→ GPU 异步，`time.perf_counter()` 前后要 `synchronize()`，否则测的是"提交"不是"执行"。profiler 内部帮你处理，手动计时必须自己同步。

**④ torch.compile 后 shape 频繁变化，反复重编译**
→ **现象**：明明编译了却更慢，每个不同 shape 都触发一次编译。
→ **修正**：LLM 用定长序列（packing，第 5 章）天然规避；变长场景用 `dynamic=True` 或分桶到固定几个 shape。

**⑤ 把 memory-bound 操作当 compute-bound 优化**
→ 给一个逐元素操作换更低精度/更快矩阵单元没用（它卡在带宽不是算力）。要融合减访存。判断错方向 = 白忙。先用 roofline 定性（11.2-①）。

**⑥ gradient checkpointing 到处乱加**
→ 它是拿算力换显存，不缺显存时加了纯亏（多 33% 算力）。只在显存吃紧时用，且优先 checkpoint 激活大的段落（Transformer 块级别，不是每个算子）。

---

## 11.5 开源项目的最佳实践

**① nanoGPT 的 compile + AMP + flash 三连**
nanoGPT 的 `train.py`：`model = torch.compile(model)` + bf16 autocast + SDPA（自动 flash）。三个开关叠起来，A100 上 MFU 可达 40%+。读它如何用 `--compile` flag 控制、如何处理 compile 与 checkpoint 保存的交互（compile 会加 `_orig_mod.` 前缀，第 4 章 Q3）。

**② HF `Trainer` 的显存优化菜单**
`gradient_checkpointing=True`、`bf16=True`、`optim="adamw_bnb_8bit"`（8-bit 优化器，把优化器状态从 8 字节/参数压到 2——第 2 章账的直接优化）、`gradient_accumulation_steps`。这些是一张"显存不够时依次打开"的菜单，第 15 章微调 7B 会逐个用上。

**③ PyTorch Profiler + TensorBoard / HTA**
生产训练用 `torch.profiler` 导出 trace，配合 HolisticTraceAnalysis 或 Chrome trace viewer 分析。学会看时间线上的"空隙"（气泡）和 kernel 排布——这是 GPU 性能工程师的核心技能，和你读 Spark DAG/火焰图是同一种肌肉。

---

## 11.6 典型面试题

**Q1：如何判断一个操作是 compute-bound 还是 memory-bound？各自怎么优化？**

> **参考答案**：算术强度（FLOP/byte）对比硬件的 FLOPS/带宽比（ridge point）——高于则 compute-bound（受算力限），低于则 memory-bound（受带宽限）。大矩阵乘 compute-bound（优化：Tensor Core、低精度、更大矩阵摊薄）；逐元素操作（norm/激活/加法）memory-bound（优化：算子融合减少 HBM 往返、就地操作）。**加分点**：LLM 训练的矩阵乘 compute-bound 但大量胶水层 memory-bound，torch.compile/FlashAttention 主攻后者；推理的 decode 阶段是 memory-bound（每步只算一个 token 但要读全部权重，第 16 章）。

**Q2：FlashAttention 为什么又快又省显存？它省显存的代价是什么？**

> **参考答案**：不物化完整 O(T²) 注意力矩阵，而是分块在 SRAM 里用在线 softmax（增量维护 max/sum）完成计算，只写回 O(T·d) 结果。省显存：不存 T² 矩阵，反向时重算（激活重算哲学）。快：注意力是 memory-bound，省下的 HBM 往返（读写 T² 矩阵）远超重算多出的 FLOPs。代价：反向多算一次前向的注意力部分（额外算力），以及实现复杂（要手写 CUDA kernel 精细管理 SRAM）。**加分点**：这是"内存墙"时代的典型设计——为省访存不惜多算；FlashAttention-2/3 进一步优化了并行度和对新硬件的适配。

**Q3：什么是 MFU？为什么它比 GPU 利用率更有意义？典型值多少？**

> **参考答案**：MFU = 实际 FLOPS / 硬件峰值 FLOPS，衡量算力真正用于有效计算的比例。GPU 利用率只表示"有 kernel 在执行"，无法区分高效与低效 kernel（memory-bound 低效 kernel 也 100%）。MFU 直接反映训练效率。大模型训练 MFU 40-55% 为优秀，<30% 有明显浪费（数据/通信/气泡）。估算：每步 FLOPs ≈ 6×参数量×token 数，除以耗时和峰值。**加分点**：6 的来源（前向 2N + 反向 4N per token）；MFU 受序列长影响（注意力的 O(T²) 部分不计入 6N 近似，长序列要修正）；HFU（Hardware FLOPs Utilization）含重算，MFU 不含。

**Q4：torch.compile 做了什么？为什么能加速？有什么坑？**

> **参考答案**：TorchDynamo 捕获 Python 字节码层的计算图，TorchInductor 编译成融合 kernel（GPU 上生成 Triton）。加速来源：算子融合（减 memory-bound 操作的 HBM 往返）、消除逐算子 Python 调度开销、shape/dtype 特化。坑：首次编译耗时（预热）、动态 shape 触发重编译（需 dynamic=True 或固定 shape）、部分算子不支持会 graph break（回退 eager，收益打折）、与 checkpoint 的前缀交互（`_orig_mod.`）。**加分点**：graph break 的排查（`torch._dynamo.explain`）；compile 与 DDP/FSDP 的组合顺序；不同 mode（default/reduce-overhead/max-autotune）的权衡。

---

## 11.7 疑难杂症排查

**案例 1：GPU 利用率高但训练就是慢**

利用率高是假象（易错点①）。① 算 MFU——低说明 kernel 低效或大量 memory-bound 胶水；② profiler 看 kernel 排名，大量微小 kernel = 融合机会（上 torch.compile）；③ 看有没有意外的低精度回退或 shape 不对齐导致走了慢路径。**方法论**：利用率高而慢，一定是"忙着做低效的事"，用 MFU + profiler 定位。

**案例 2：加了 torch.compile 反而更慢**

① 只测了含编译的第一次迭代（易错点②）——warmup 后再测；② shape 频繁变导致反复重编译（易错点④）——`TORCH_LOGS=recompiles` 查；③ 大量 graph break 让编译收益归零——`torch._dynamo.explain(model)(x)` 看断点；④ 模型太小，Python 开销本就不是瓶颈，编译收益有限。

**案例 3：显存莫名不够，或时涨时降**

① 先 `memory_summary()` 看构成（模型状态/激活/碎片各多少）；② 激活占大头 → 上 gradient checkpointing / 减 batch；③ "碎片"大 → 分配器碎片化（变长 shape 反复分配），试 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；④ 缓慢上涨 → 泄漏（第 3 章 total+=loss 类，或 hook 存张量，第 4 章）。第 15 章有完整 OOM 排查树，本章先建立"显存分四类"的账本视角。

---

## 11.8 练习题

### 基础 1：profile 你的 miniGPT
用 `profile_minigpt.py` 剖析第 9 章的 miniGPT 训练一步，找出 GPU 时间占比最高的 3 个算子。它们是 compute-bound 还是 memory-bound？（提示：matmul 类 vs 逐元素类。）

### 基础 2：算术强度计算
手算三个操作的算术强度并判断瓶颈：a) `(4096,4096)@(4096,4096)` 矩阵乘；b) `(1M,)` 张量的 `x+1`；c) `(4096,4096)` 的 RMSNorm。用你机器的 FLOPS/带宽比验证判断。

### 进阶 1：gradient checkpointing 的时间-显存权衡
用 `torch.utils.checkpoint` 给 miniGPT 的 Block 加上 checkpointing，实测：激活显存降了多少？每步耗时增加多少？画出"batch size vs 峰值显存"在开/关 checkpointing 下的两条曲线，找出"不 checkpoint 会 OOM 但 checkpoint 能跑"的 batch 区间。

### 挑战 1：算 miniGPT 的 MFU 并优化
用 `mfu.py` 算出 miniGPT 在 M4 上的 MFU（需要 M4 GPU 的峰值 FLOPS，查 Apple 规格或用大矩阵乘实测上限）。然后尝试提升：调大 batch、bf16、（云端可试 compile）。记录每个改动对 MFU 的影响，写一份"优化前后"报告。讨论：为什么小模型在 M4 上 MFU 往往偏低？

---

## 本章小结与下一章预告

性能优化的第一性原理是 Roofline——先判断 compute/memory-bound，再选对应武器。profiler 是 GPU 火焰图；torch.compile 融合 memory-bound 胶水；FlashAttention 是"省访存 > 多算"的教科书；显存分四类（模型状态/激活/缓冲/碎片），激活是最大变量；MFU 是比利用率诚实的效率标尺。你的性能分析老本行，工具换 GPU，方法论不变。

**下一章（第 12 章）**：分布式训练原理。单卡榨干了，下一步是多卡。通信原语（AllReduce ≈ 你熟悉的 shuffle）、DDP 的源码级剖析、以及在你的 M4 上用 gloo 后端多进程模拟分布式语义——不上云也能理解通信机制。第三篇从"单卡快"迈向"多卡并行"。
