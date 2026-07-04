# 第 5 章 · 数据管线：别让 GPU 饿着

> **本章目标**：把数据高效喂进模型。学完你应该能回答：
> 1. `num_workers` 背后是怎样一条多进程管道？数据怎么从 worker 回到主进程？
> 2. `pin_memory` 到底加速了什么？
> 3. LLM 预训练的数据管线为什么不用"读文件 + shuffle=True"的朴素方案？packing 是什么？
> 4. GPU 利用率只有 30%，怎么定位是不是数据管线的锅？

**前置**：第 1 章（生成器、GIL）、第 4 章。 **硬件路径**：本地。 **预计用时**：4~5 小时。
**给你的定位说明**：这章你带着主场优势——生产者-消费者、背压、分片、shuffle 代价这些概念不用教。讲义把力气花在 PyTorch 的**具体实现机制**和 **LLM 数据的特殊形态**上。

---

## 5.1 来龙去脉：GPU 饥饿问题

训练循环的每一步：取 batch → 前向 → 反向 → 更新。GPU 做后三件事的速度是每秒几十步，而"取 batch"涉及磁盘 IO、解码、tokenize、组装——纯 Python 的 CPU 工作。如果串行执行，GPU 大部分时间在**等数据**：万元/小时的算力晒太阳，这就是 GPU 饥饿（GPU starvation）。

解法你闭着眼睛都能说出来：**生产者-消费者 + 预取**。CPU 上多个 worker 并行准备数据，队列缓冲，GPU 消费。PyTorch 的 `DataLoader` 就是这套模式的实现。但有一个 Python 特色的前提你已经在第 1 章见过：**GIL 逼着我们用多进程而不是多线程**——数据预处理是纯 Python CPU 密集工作，线程会被 GIL 串行化。多进程带来了新问题（序列化、内存复制、启动开销），本章一半的坑源于此。

与你熟悉的大数据管线相比，三个关键差异值得先立住：

| | 大数据批处理管线 | 训练数据管线 |
|---|---|---|
| 优化目标 | 总吞吐 | **每步延迟稳定**（GPU 不能空转等长尾） |
| 消费模式 | 一次扫过 | **反复多个 epoch**，每个 epoch 要重新 shuffle |
| 随机性 | 无关紧要 | **是正确性的一部分**（shuffle 影响收敛质量） |

---

## 5.2 核心原理

### ① 两种 Dataset：有索引的表 vs 流

**map-style**（`__getitem__` + `__len__`，第 1 章写过）：随机访问，像一张有索引的表。Sampler 决定访问顺序，shuffle = 打乱索引序列——代价是 O(1) 随机读，**前提是数据支持高效随机访问**。

**IterableDataset**（`__iter__`）：只能顺序消费，像 Kafka 流。适合：数据是流（TB 级预训练语料没法建全量索引）、顺序读远快于随机读（对象存储上的大文件）、数据本身在线生成。代价：没有全局 shuffle（只能局部 buffer 打乱）、**分片去重要自己做**（多 worker、多节点都会拿到同一条流的副本——易错点里有它，面试也爱考）。

选型口诀：装得进本地磁盘且可索引 → map-style；TB 级流式语料 → IterableDataset（第 9 章 miniGPT 数据量小用前者，讲义会演示后者的正确写法为大规模做准备）。

### ② DataLoader 的多进程管道解剖

`DataLoader(ds, batch_size=32, num_workers=4, pin_memory=True, prefetch_factor=2)` 启动后的完整机器：

```
主进程                          worker 进程 ×4
  │ 把"索引批"发给 worker ────→  ds[i] 逐个取样 → collate_fn 组装 batch
  │                              │ 结果张量写入共享内存（不走 pickle 大数据体！）
  │ ←──── 队列返回 batch 引用 ───┘
  │ pin_memory 线程: 拷入页锁定内存
  ▼ 训练循环拿到 batch
```

要点逐个说：

**共享内存传输**：worker 产出的张量放进 `/dev/shm` 共享内存段，主进程零拷贝拿引用——大 batch 不走进程间 pickle 的慢路径。这是"多进程方案能活"的关键工程细节（也是坑：容器里 `/dev/shm` 默认 64MB，不够就崩，见 5.7）。

**pin_memory（页锁定内存）**：GPU 通过 DMA 从主机内存拉数据，要求物理地址固定；普通内存页可能被 OS 换出，CUDA 得先拷到内部的锁定缓冲再 DMA——多一跳。`pin_memory=True` 让 batch 直接落在页锁定内存，`.to(device, non_blocking=True)` 即可**异步** H2D 传输，与计算重叠。**MPS 注**：统一内存架构没有独立显存和 PCIe 传输，此参数无意义；这是 CUDA 语境的知识，上云实战（第 12 章）时生效。

**prefetch_factor**：每个 worker 预取的 batch 数（默认 2）——队列深度，你熟悉的背压参数。**persistent_workers=True**：epoch 之间不销毁 worker（默认会销毁重建！macOS spawn 启动贵，这个开关很值）。

**num_workers 怎么定**：经验起点 = 物理核数的一半到全数，然后**实测吞吐**调整。太少喂不饱，太多则内存×N、上下文切换、共享内存压力。判断"够不够"的方法见 5.7 案例 1 的饥饿诊断。

### ③ collate_fn：样本堆成 batch 的规则

默认 collate 把 N 个同 shape 样本 `stack` 成 `(N, ...)`。变长序列（NLP 常态）直接 stack 会报错，需要自定义：

```python
def pad_collate(batch):                      # batch: List[(ids, label)]
    ids = [b[0] for b in batch]
    lens = torch.tensor([len(x) for x in ids])
    padded = nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=0)
    return padded, lens, torch.stack([b[1] for b in batch])
```

进阶技巧**按长度分桶（bucketing）**：把长度相近的样本组一个 batch，padding 浪费从 50%+ 降到个位数——SFT 微调（第 14 章）的标配优化。

### ④ LLM 预训练的数据形态：packing

预训练样本是变长文档，但 GPU 喜欢定长稠密张量。**padding 方案**把每条文档补齐到 max_len——短文档多时，一半算力在算 pad token。预训练的标准解法是 **packing**：

```
文档A(300 tok) 文档B(800) 文档C(1200) ...
    → 全部 tokenize 后首尾拼接成一条超长 token 流（文档间插 <eos>）
    → 按 seq_len=1024 硬切成块，每块都是满的，零 padding
```

代价：一个块可能跨两篇文档（用 `<eos>` 告诉模型边界，朴素做法连注意力都不隔离，GPT-2/3 就这么训的；讲究的做法配块内文档级 attention mask）。本章实验实现完整 packing 管线，**第 9 章 miniGPT 直接复用**。

### ⑤ 随机性：worker 是随机 bug 的温床

每个 worker 是独立进程，各有各的随机状态。PyTorch 会给每个 worker 的 **torch** 种子做区分（base_seed + worker_id），但 **NumPy/Python random 不归它管**——Linux/fork 平台上 worker 继承同一份 NumPy 状态，**所有 worker 产生相同的"随机"增强**（经典 bug，2021 年一篇博客扫描 GitHub 发现上百个项目中招）。阴险之处在于跨平台不一致：macOS/spawn 的 worker 会重新初始化熵源、往往测不出来，而云端 Linux 集群必然中招——本地验证通过、上云才发病。修法：`worker_init_fn` 里用 `torch.initial_seed()` 派生重设 numpy/random 种子（实验演示，两个平台都应加上）。

shuffle 的实现也值得一说：map-style 的 shuffle 是 epoch 开始时生成全量索引排列（`generator` 控制可复现）；IterableDataset 只能 **shuffle buffer**——蓄水池里随机抽（你会认出这是水库抽样的近亲），buffer 越大随机性越好、内存越贵。预训练语料的标准做法是**离线全局 shuffle 一次 +在线 buffer 局部打乱**的组合。

---

## 5.3 动手实验

```bash
uv run chapters/ch05_data_pipeline/code/dataloader_mechanics.py   # 多进程加速实测 + collate 定制
uv run chapters/ch05_data_pipeline/code/worker_pitfalls.py        # NumPy 种子重复 bug 复现与修复
uv run chapters/ch05_data_pipeline/code/llm_packing.py            # packing 管线（第 9 章直接复用）
```

注意第一个脚本里的 `if __name__ == "__main__":` 保护——在 macOS 上这不是风格问题而是生死问题（易错点①）。

---

## 5.4 易错点清单

**① macOS/Windows 上 `num_workers>0` 直接崩或无限递归**
→ **现象**：`RuntimeError: An attempt has been made to start a new process before the current process has finished its bootstrapping phase`。
→ **原因**：macOS/Windows 用 **spawn** 启动 worker（Linux 默认 fork）：子进程重新 import 主脚本——顶层代码会被再执行一遍，若 DataLoader 创建也在顶层，就无限递归。同因：spawn 需要 pickle Dataset 和 collate_fn，**lambda/闭包/本地函数不可 pickle**，也会崩。
→ **修正**：入口代码全部包进 `if __name__ == "__main__":`；collate_fn 用模块级函数或函数对象。**Linux 上没事的代码上 mac 就崩**，跨平台复现问题先查这里。

**② 所有 worker 产出相同的"随机"数据**
→ **现象**：数据增强/负采样看起来没起作用；同 batch 里样本诡异地相似。
→ **原因**：NumPy/random 的种子不被 DataLoader 管理（5.2-⑤）。
→ **修正**：`worker_init_fn=lambda wid: np.random.seed(torch.initial_seed() % 2**32)`（示意；lambda 在 spawn 平台要换成模块级函数）。

**③ 内存随 worker 数线性膨胀，甚至随 epoch 增长**
→ **原因一**：每个 worker 持有一份 Dataset 副本——`__init__` 里加载的 10GB list × 8 workers = 80GB。
→ **原因二（隐蔽）**：Linux fork 的写时复制本该省内存，但 **Python 对象的引用计数写操作会触碰每个对象头**，把只读页也"写脏"逐步物化——内存随访问缓慢上涨，像泄漏。
→ **修正**：大数据放 NumPy 数组/memmap/Arrow（无 Python 对象头，COW 真正生效）；或惰性加载（`__getitem__` 时才读磁盘）。HF datasets 的 Arrow mmap 方案正是为此（5.5）。

**④ 变长样本直接用默认 collate**
→ **现象**：`RuntimeError: stack expects each tensor to be equal size`。
→ **修正**：自定义 pad_collate（5.2-③）；或数据侧 packing 消灭变长。

**⑤ 评估用了 shuffle / 训练忘了 shuffle**
→ **现象**：前者让评估结果不可复现比对；后者更糟——数据有序（比如按类别排序的文件）时模型先学 A 类再学 B 类，收敛显著变差且难察觉。
→ **修正**：train loader `shuffle=True`（或 DistributedSampler，第 12 章），eval loader 永远 False。

**⑥ `persistent_workers=False`（默认）+ 大启动开销**
→ **现象**：每个 epoch 开头卡几秒到几十秒（mac 上尤其明显——spawn + re-import）。
→ **修正**：`persistent_workers=True`，代价是 worker 常驻内存。多数训练场景应该开。

---

## 5.5 开源项目的最佳实践

**① HF `datasets`：Arrow + mmap 的零拷贝哲学**
数据集下载后转成 Apache Arrow 格式落盘，访问时 **mmap 不进堆**——百 GB 数据集的"加载"瞬间完成，物理读发生在访问页时，多进程共享同一份页缓存（COW 问题也顺带消灭：Arrow 是无对象头的连续内存）。你从 Parquet/Arrow 生态来，会认出这是同一套哲学在训练侧的应用。`dataset.map(tokenize, num_proc=8)` 的离线预处理 + 缓存指纹机制也值得学：**tokenize 一次，训练 N 次**。

**② Mosaic StreamingDataset / WebDataset：预训练的流式标准**
TB 级语料的工业方案：数据切成 ~100MB 分片（shard）存对象存储，训练时按节点/worker 分配 shard 流式拉取，shard 内顺序读 + shuffle buffer。WebDataset 用 tar 包做 shard（顺序读友好）；Mosaic 的 MDS 格式加了断点续训的精确定位（记录消费到第几个样本——exactly-once 语义，你的老朋友）。第 13 章多机训练会再遇到它们。

**③ tokenize 离线化：预训练管线的铁律**
在线 tokenize 是 CPU 大户，大规模预训练一律离线：语料 → tokenize → 拼接 packing → 存成定长块的二进制（`np.memmap` 可直读）。训练时的"数据加载"退化为"按索引读定长块"——快到不需要 worker。nanoGPT 的 `data/openwebtext/prepare.py` 是这个模式的最小可读实现，第 9 章我们照此办理。

---

## 5.6 典型面试题

**Q1：`pin_memory=True` 做了什么？什么时候没用？**

> **参考答案**：让 DataLoader 产出的 batch 落在页锁定（page-locked/pinned）内存。GPU 的 DMA 引擎只能从物理地址固定的内存读数据；普通可分页内存要先经 CUDA 内部staging 缓冲多拷一次。pinned 内存允许直接 DMA，且配合 `.to(device, non_blocking=True)` 实现传输与计算重叠。没用的场景：数据本来就小/传输不是瓶颈；MPS/CPU 训练（无独立显存）；过量 pin 会挤压系统可分页内存反而拖慢整体。**加分点**：non_blocking 只有源在 pinned 内存时才真异步，否则静默退化为同步——常见的"加了参数没变快"之谜。

**Q2：IterableDataset 配多个 worker，每个 worker 都吐出全量数据怎么办？多机分布式还要注意什么？**

> **参考答案**：IterableDataset 的 `__iter__` 会在每个 worker 里独立执行，默认各自遍历完整数据流 → 数据重复 N 遍。单机修法：`__iter__` 里用 `torch.utils.data.get_worker_info()` 拿 worker_id/num_workers，按取模跳片分工。多机还要叠加节点分片：`rank/world_size`（第 12 章）做外层切分，worker 做内层切分——两层都要做，漏一层就重复。**加分点**：分片后各 worker 数据量可能不均，epoch 边界要么 drop 尾部要么 padding 循环，分布式下不对齐会导致某 rank 提前退出而挂起集合通信（第 15 章真实事故）。

**Q3：训练吞吐上不去，如何系统性判断瓶颈在数据侧还是计算侧？**

> **参考答案**：三板斧——① **合成数据对照**：把真实 loader 换成常驻 GPU 的随机张量，吞吐大涨 → 数据侧是瓶颈；② **计时分解**：记录每步 `data_time`（取 batch 耗时）与 `compute_time`，健康管线 data_time 应接近 0（预取命中）；③ profiler 时间线看 GPU 空隙（第 11 章）。数据侧确诊后再细分：IO（换 SSD/预读）、解码/tokenize（离线化）、collate（packing/bucketing）、传输（pin_memory）。**加分点**：报出"GPU 利用率高"不等于没问题——utilization 只表示有 kernel 在跑，低效 kernel 也算（第 11 章 MFU 才是真指标）。

**Q4：为什么 DataLoader 用多进程而不是多线程？Python 3.13 的 free-threading 会改变这个设计吗？**

> **参考答案**：GIL 使纯 Python 的预处理无法多线程并行（第 1 章）。多进程绕开 GIL，代价是进程启动、序列化约束（spawn 平台）、内存复制、共享内存传输这一整套复杂度。free-threaded CPython（PEP 703）原则上允许多线程 DataLoader（省掉进程开销与内存复制），社区已有实验性支持，但生态库的线程安全和无 GIL 下的性能回归仍在演进——短期内多进程仍是默认。**加分点**：提到很多重预处理已被"离线化 + 零拷贝格式"消解（5.5-③），架构演进比并发模型演进更早解决了问题。

---

## 5.7 疑难杂症排查

**案例 1：GPU 利用率 30%，训练慢——数据饥饿确诊流程**

① 快速嫌疑排除：`num_workers=0` 对比 `num_workers=8` 的每步耗时，差异大 → 数据侧；
② 用 Q3 的合成数据对照法定量确认上限；
③ 细分定位：在 `__getitem__` 里埋计时（IO/解码/tokenize 各多少），或对 loader 单独压测 `for batch in loader: pass` 的裸吞吐；
④ 常见修复顺位：离线 tokenize（一次性根治）> persistent_workers + 调 workers 数 > packing/bucketing 减少浪费 > 换存储格式（Arrow/memmap）。

**案例 2：训练在某一步永久卡住，无报错**

高发病因：① worker 进程被 OOM killer 杀了（dmesg 查证，第 1 章 exit 137 的亲戚）主进程傻等队列；② 容器 `/dev/shm` 太小（docker 默认 64MB），共享内存写满死锁——`--shm-size=8g` 解决，**这是容器化训练最著名的坑之一**；③ 某条数据本身触发 `__getitem__` 死循环。排查：`py-spy dump --pid <worker_pid>` 直接看每个 worker 卡在哪一行（不用改代码，生产可用——强烈推荐进入你的工具箱）。

**案例 3：设置了所有种子，两次训练数据顺序仍不同**

检查清单：① DataLoader 要传显式 `generator=torch.Generator().manual_seed(...)`（否则用全局状态，任何库的消费都会移动它）；② `num_workers>0` 时增强的随机性要靠 worker_init_fn 固定；③ 字典序陷阱：`os.listdir` 返回顺序不保证，文件列表要显式 `sorted()`——跨机器复现差异的常见来源（这条在大数据世界你八成踩过 HDFS listStatus 的同款）。

---

## 5.8 练习题

### 基础 1：变长 collate_fn
实现 `pad_collate`：输入变长 token 序列列表，输出 padded batch + 长度张量 + attention mask（1=真实 token，0=padding）。测试奇偶长度混合的 batch。

### 基础 2：吞吐测量
写一个 loader 压测函数：`measure(loader, n_batches)` 返回 batches/s。对一个人为加了 `time.sleep(0.01)`（模拟解码开销）的 Dataset，实测 num_workers=0/2/4 的吞吐并解释曲线。

### 进阶 1：IterableDataset 正确分片
实现一个流式 Dataset（模拟读取 8 个 shard 文件），要求 num_workers=4 时数据不重不漏。用 `get_worker_info()` 实现，写断言验证：全部 worker 产出的并集 = 全量数据且无重复。

### 挑战 1：shuffle buffer 与随机性质量
实现流式 shuffle buffer（大小 B：先填满，之后每来一条随机换出一条）。用统计手段评估随机性：对有序输入 [0..9999]，B=10/100/1000 时输出序列的"位置漂移"分布如何变化？给出"buffer 大小怎么选"的定量结论，并说明它和水库抽样的联系与区别。

---

## 本章小结与下一章预告

DataLoader = 多进程生产者-消费者（共享内存传数据、pin_memory 加速 H2D、prefetch 做队列深度）。LLM 数据的关键形态是 packing（零 padding 的定长块），预训练管线的铁律是离线 tokenize。随机性是正确性的一部分：worker 种子、shuffle、文件序都要显式控制。

**下一章（第 6 章）**：训练循环完全解剖。模型（4）、数据（5）、梯度（3）都齐了，现在把它们拼成工业级训练循环：优化器从 SGD 到 AdamW 的因果演进、学习率调度、梯度裁剪、混合精度 AMP——第 9 章 miniGPT 预训练的全部零件在这一章配齐。
