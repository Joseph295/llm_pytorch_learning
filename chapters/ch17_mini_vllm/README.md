# 第 17 章 · 🏆 里程碑三：手写 mini-vLLM

> **第三次交卷，推理侧的总决战**。你将把 KV Cache + continuous batching + 简化版 PagedAttention 组装成一个**能跑的推理引擎**，然后对照 vLLM 真实源码。第 16 章的推理原理在这里变成一个你亲手写的服务系统。
>
> 学完你将拥有：一个自己写的推理引擎、对 vLLM 核心设计的源码级理解、以及"我知道推理引擎内部怎么工作"的底气。

**前置**：第 16 章（KV cache/batching）、第 8 章（GPT）、第 7 章（注意力）。 **硬件路径**：逻辑层 M4 可跑；性能验证上云。 **项目位置**：`projects/mini-vllm/`。

---

## 17.1 来龙去脉：朴素推理的三个浪费

第 16 章你给单个请求加了 KV cache。但真实的推理服务同时处理**成百上千个请求**，朴素的"一个 batch 一起生成"有三个致命浪费：

1. **长度不齐的等待**：batch 里有的请求生成 10 个 token 就结束，有的要生成 500 个。朴素静态 batching 要等**最长的**那个完成，短请求生成完还占着位置空转——GPU 利用率暴跌。
2. **KV cache 的显存碎片**：每个请求的 KV cache 要预留"最大可能长度"的连续显存，但实际用不了那么多——大量预留显存闲置，能同时服务的请求数被浪费的显存限制。
3. **无法动态插入新请求**：静态 batch 一旦开始就固定，新来的请求要等整个 batch 结束——延迟高。

vLLM 的两个核心创新分别解决这些：**continuous batching**（动态调度，请求完成就退出、新请求随时加入）解决 1 和 3；**PagedAttention**（像操作系统分页管理 KV cache）解决 2。这一章我们手写它们的简化版。

**你的分布式/系统背景在这里是巨大优势**：continuous batching 是任务调度，PagedAttention 是内存分页——都是操作系统/分布式系统的经典思想搬到推理引擎里。

---

## 17.2 核心原理

### ① Continuous Batching：请求级的动态调度

朴素静态 batching：`[请求1, 请求2, 请求3]` 一起进，一起出，等最慢的。

Continuous batching（也叫 in-flight batching）：把调度粒度从"整个 batch"降到"单个 token 步"。每一步：
- 所有活跃请求各生成一个 token；
- 生成了结束符的请求**立即退出**，释放它的 KV cache；
- 等待队列里的新请求**立即填入**空出的位置；
- 下一步继续。

效果：GPU 始终满载（没有请求在等别人完成），吞吐大幅提升（实测比静态 batching 高数倍）。这本质是**你熟悉的动态任务调度**——把固定的批处理换成流式的、请求随到随处理的调度器。

关键数据结构：一个**调度器**维护 `running`（活跃请求）、`waiting`（等待队列）、`finished`（已完成）三个队列，每步循环推进。本章实验实现这个调度循环。

### ② PagedAttention：KV Cache 的虚拟内存

问题：每个请求的 KV cache 长度不同且动态增长。朴素做法给每个请求预留"最大长度"的**连续**显存——比如 max_len=2048，但请求实际只用了 100，剩下 1948 的显存被预留却闲置。高并发时这种碎片让能服务的请求数远低于显存理论上限。

PagedAttention 的洞察（直接借用操作系统虚拟内存）：**把 KV cache 切成固定大小的块（block/page，如 16 个 token 一块），按需分配，不要求连续**。每个请求维护一个"块表"（block table，就是**页表**），记录它的 KV cache 用了哪些物理块。请求增长时按需分配新块，结束时释放所有块回到空闲池。

收益：
- **几乎零碎片**：块粒度分配，浪费最多一个块（<16 个 token）；
- **显存利用率从 ~20-40% 提升到 >90%**（vLLM 论文数据），能服务的并发请求数翻几倍；
- **前缀共享（copy-on-write）**：多个请求共享相同前缀（如同一个 system prompt）时，可以共享物理块，进一步省显存——这就是**写时复制**，你的老朋友。

块表 = 页表，物理块 = 物理页，逻辑 token 位置 = 虚拟地址——**PagedAttention 就是把虚拟内存搬到了 KV cache 管理上**。本章实验实现块分配器和块表。

### ③ 把两者组装成引擎

```
mini-vLLM 主循环（每步）：
  1. 调度器：从 waiting 队列取新请求（如果有空闲 KV 块），加入 running
  2. 对所有 running 请求，用它们的块表收集 KV cache
  3. batch 前向一步，每个请求生成一个 token
  4. 新 token 的 K/V 写入各请求的 KV 块（不够就分配新块）
  5. 检查结束条件（EOS 或 max_len），完成的请求退出、释放块
  6. 回到 1
```

这个循环把第 16 章的 KV cache、continuous batching、PagedAttention 串起来。本章的 mini-vLLM 用第 8 章的 GPT 做模型，实现这个引擎，服务多个并发"请求"。

---

## 17.3 动手：构建 mini-vLLM

```bash
# 逻辑层演示（M4 可跑，用第 8 章的小 GPT）
uv run projects/mini-vllm/demo.py            # 多请求并发生成，观察调度与分块
uv run projects/mini-vllm/test_paging.py     # 块分配器与块表的单元测试
```

项目结构：
```
projects/mini-vllm/
├── block_manager.py    # PagedAttention 的块分配器 + 块表（②）
├── scheduler.py        # continuous batching 调度器（①）
├── engine.py           # 组装：主循环（③）
└── demo.py             # 多请求并发生成演示
```

`engine.py` 是核心——它把调度、分块、前向串成一个循环。你会看到不同长度的请求如何动态进出、KV cache 如何按块分配释放——第 16 章的原理变成可见的系统行为。

---

## 17.4 易错点清单

**① 块表映射错误**
→ **现象**：请求生成乱码或读到别的请求的 KV。
→ **原因**：逻辑 token 位置到物理块的映射算错（`block_id = pos // block_size`，`offset = pos % block_size`）——和页表计算一样，off-by-one 就串数据。
→ **修正**：块表索引严格用整除/取模；写单元测试覆盖块边界（第 15/16 个 token 跨块的时刻）。

**② 块释放时机错误**
→ **现象**：显存泄漏（块没释放）或提前释放（还在用的块被复用）。
→ **原因**：请求结束时忘了释放块，或前缀共享时引用计数没管好（copy-on-write 的引用计数，第 3 章 Python 引用计数的系统版）。
→ **修正**：块用引用计数管理，归零才真正释放；请求退出时递减它所有块的引用。

**③ continuous batching 的位置索引混乱**
→ **现象**：不同请求处于不同生成位置（有的第 5 个 token、有的第 100 个），RoPE 位置索引要各算各的（第 16 章易错点①的批量版）。
→ **修正**：每个请求维护自己的当前位置；batch 前向时位置是一个向量而非标量。

**④ 调度器饥饿/死锁**
→ **现象**：某些请求永远得不到调度（显存一直被长请求占着）。
→ **原因**：调度策略没考虑公平性；或显存不足时没有抢占（preemption）机制。
→ **修正**：vLLM 用 FCFS + 抢占（显存不够时把某些请求的 KV cache 换出到 CPU 或重算）。mini 版可以先不做抢占，但要意识到这个问题。

**⑤ batch 内 padding 又回来了**
→ **现象**：continuous batching 里不同请求当前长度不同，naive 实现又要 padding。
→ PagedAttention 配合变长注意力 kernel（如 flash-attn 的 varlen 接口）避免 padding——不同长度的请求在同一个 batch 里用块表各取所需，不 padding。

---

## 17.5 开源项目的最佳实践

**① vLLM 源码对照（本章的终极目标）**
读 [vllm/core/scheduler.py](https://github.com/vllm-project/vllm)（你的 scheduler.py 对照）、`vllm/core/block_manager.py`（你的 block_manager.py 对照）、`vllm/attention`（PagedAttention kernel）。你会发现你手写的简化版抓住了核心结构，vLLM 多的是：抢占与换出、prefix caching、chunked prefill、各种量化/并行的集成、CUDA kernel 优化。**能读懂 vLLM = 你在推理引擎领域有了源码级理解。**

**② PagedAttention 论文（Kwon et al. 2023）**
理解块大小的权衡（块大省元数据但碎片多、块小反之）、copy-on-write 的实现、抢占策略。这是"操作系统思想解决 ML 系统问题"的典范论文。

**③ SGLang / TensorRT-LLM 的不同取舍**
SGLang 的 RadixAttention（前缀树共享 KV，比 vLLM 的 prefix caching 更激进）、TensorRT-LLM 的 kernel 融合。看不同引擎在同一问题上的不同设计，理解权衡空间。

---

## 17.6 典型面试题

**Q1：什么是 continuous batching？相比静态 batching 好在哪？**

> **参考答案**：静态 batching 一批请求一起进出、等最慢的完成，短请求生成完仍占位空转，GPU 利用率低。continuous batching 把调度粒度降到 token 级：每步所有活跃请求各生成一个 token，完成的立即退出释放资源、等待的新请求立即填入。GPU 始终满载，吞吐提升数倍，且新请求无需等整批结束（低延迟）。**加分点**：本质是动态任务调度；配合 PagedAttention 管理变长 KV cache；调度器维护 running/waiting/finished 队列；需要处理不同请求位置不同的问题（RoPE 位置向量化）。

**Q2：PagedAttention 解决什么问题？和操作系统的什么概念对应？**

> **参考答案**：解决 KV cache 的显存碎片——朴素做法给每个请求预留最大长度的连续显存，实际用不满造成大量浪费。PagedAttention 把 KV cache 切成固定大小的块按需分配（不要求连续），每个请求用块表（=页表）记录用了哪些物理块（=物理页），逻辑位置=虚拟地址。碎片降到最多一个块，显存利用率从 ~20-40% 提到 >90%，并发数翻几倍。对应操作系统的虚拟内存/分页。**加分点**：copy-on-write 实现前缀共享（多请求共享 system prompt 的块）；块大小的权衡；块的引用计数管理。

**Q3：LLM 推理服务的 GPU 利用率为什么容易低？如何提升？**

> **参考答案**：① decode 是 memory-bound（第 16 章），单请求浪费带宽——靠 batching 提升；② 静态 batching 等最慢请求，短请求空转——靠 continuous batching；③ KV cache 碎片限制并发数——靠 PagedAttention；④ prefill 和 decode 混在一起相互干扰——靠 chunked prefill / 分离部署；⑤ 序列长度差异大 padding 浪费——靠变长 kernel。综合手段：vLLM 类引擎把这些都做了，MFU 显著提升。**加分点**：prefill/decode 分离（P/D disaggregation）是前沿方向；投机解码利用空闲算力（第 16 章）。

**Q4：手写一个 continuous batching 调度器的核心循环。**

> **参考答案**要点：维护 running/waiting 队列；主循环每步：① 若有空闲 KV 块且 waiting 非空，取新请求加入 running（并分配初始块）；② 对 running 批量前向一步生成 token；③ 新 token 的 KV 写入各请求的块（不够则分配新块）；④ 检查每个请求是否 EOS/超长，完成的移出 running、释放块；⑤ 循环。关键：块的分配/释放、请求位置各自维护、显存不足时的处理（拒绝新请求或抢占）。**加分点**：抢占与换出、公平性、prefill/decode 的调度差异。

---

## 17.7 疑难杂症排查

**案例 1：多请求并发生成，某个请求的输出串到了另一个请求**

块表映射 bug（易错点①）——某请求读到了不属于它的物理块。排查：打印每个请求的块表，确认物理块不重叠（除非是有意的前缀共享）；检查 `pos // block_size` 和 `pos % block_size` 的计算；单元测试块边界（跨块时刻）。

**案例 2：跑一会儿就 OOM，但请求数不多**

块泄漏（易错点②）——请求结束时块没释放回空闲池。排查：监控空闲块数量，应随请求完成而回升；检查请求退出路径是否释放了所有块；前缀共享时引用计数是否正确。

**案例 3：吞吐没有比静态 batching 高多少**

① 调度粒度还是太粗（没真正做到 token 级动态）；② 显存碎片没解决（没用分块）导致并发数上不去；③ batch 内 padding 浪费（易错点⑤）；④ 模型太小/序列太短，调度开销占比高。方法论：先确认 continuous batching 和 PagedAttention 都真正生效，再看 kernel。

---

## 17.8 练习题

### 基础 1：块分配器
用 `block_manager.py` 的框架，实现块分配器：`allocate(n_blocks)` 返回空闲块 id、`free(block_ids)` 归还。测试：分配到耗尽、释放后重新分配、碎片场景。

### 基础 2：块表读写
实现逻辑位置到物理块的映射：给定请求的块表和 token 位置，返回该 token 的 KV 存在哪个物理块的哪个 offset。覆盖跨块边界的测试（第 15、16、17 个 token）。

### 进阶 1：完整调度循环
用 `engine.py` 的框架，实现 continuous batching 主循环：多个不同长度的请求并发生成，验证短请求先退出、新请求动态加入、KV 块正确分配释放。对比静态 batching 的 GPU"利用率"（用总步数/理论最优步数近似）。

### 挑战 1：前缀共享（copy-on-write）
实现前缀 KV cache 共享：多个请求有相同前缀（如同一 system prompt）时，共享前缀的物理块（引用计数 >1），只在写入分叉处 copy-on-write。验证：共享后总块数减少；某请求写入不影响其他请求。这是 vLLM prefix caching 的核心，也是"写时复制"在 ML 系统的应用。

---

## 本章小结与下一章预告

mini-vLLM = continuous batching（token 级动态调度，请求随到随处理）+ PagedAttention（KV cache 分页管理，消除碎片）+ KV cache（第 16 章）。这些是你的系统/分布式背景在推理引擎里的直接应用——任务调度和虚拟内存。你手写的简化版抓住了 vLLM 的核心结构，现在能读懂它的源码。

**下一章（第 18 章）**：服务化与端侧部署。mini-vLLM 是教学引擎，生产要用 vLLM/SGLang（怎么部署、调优、监控）；而在你的 M4 上，llama.cpp 和 MLX 让 7B 模型本地流畅运行——量化推理的极致工程。第四篇从"引擎原理"走向"生产部署"。
