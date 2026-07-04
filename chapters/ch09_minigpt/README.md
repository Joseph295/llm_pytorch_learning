# 第 9 章 · 🏆 里程碑一：从零预训练 miniGPT

> **这是第一次交卷**。前八章的每一个零件——设备无关代码、张量、autograd、nn.Module、数据管线、训练循环、注意力、Transformer——在这里组装成一个**真正能训练、能续写文本**的语言模型，全程在你的 M4 上完成。
>
> 学完你将拥有：一个自己写的 BPE tokenizer、一套离线数据管线、一个从随机权重训练到能续写古文的 ~10M 参数 GPT，以及——最重要的——**"我完整地做出来过一个"的底气**。

**前置**：第 0~8 章全部。 **硬件路径**：本地 M4（约 10~20 分钟训练）。 **项目位置**：`projects/minigpt/`。

---

## 9.1 来龙去脉：为什么"从零训一个小的"值得做

你可以直接 `AutoModel.from_pretrained` 拿一个 7B 模型开始 RAG/微调，永远不碰预训练。但亲手从随机权重训一个小 GPT 的价值，恰恰在于**把抽象变成肌肉记忆**：

- loss 从 `ln(vocab)` 开始下降的第一手体感（第 8 章算过的理论起点，这次亲眼看到）；
- "过拟合单 batch → 小数据 → 全量"的调试节奏（第 6 章讲过，这次真用上）；
- tokenizer、数据、模型、训练四者如何咬合（哪个环节错了会怎样）；
- 生成质量随训练步数肉眼可见地改善——从乱码到成词到成句。

工业预训练与此的差别是**规模而非原理**：把 10M 换成 100B、把《红楼梦》换成万亿 token、把单卡换成千卡（第 12/13 章）——循环还是这个循环。**小的先跑通，大的才有资格谈。**

---

## 9.2 项目结构与数据流

```
projects/minigpt/
├── tokenizer.py       # 从零 BPE：train / encode / decode / save / load
├── prepare_data.py    # 下载语料 → 训 BPE → tokenize → 90/10 划分 → memmap 二进制
├── train.py           # 六步训练循环 + AMP + 调度 + checkpoint + 生成
├── generate.py        # 加载 checkpoint 续写
└── data/              # (git 忽略) raw.txt / tokenizer.json / train.bin / val.bin / ckpt.pt
```

数据流（第 5 章的离线管线落地）：

```
《红楼梦》原文 → clean() 去页眉页脚
  → BPE 训练（4096 词表）→ tokenizer.json
  → encode 全文（按行分块，避免 O(n²) 卡死）
  → uint16 数组 → train.bin(90%) / val.bin(10%)   ← memmap 直读，无需 DataLoader worker
```

模型：复用第 8 章的 `gpt_model.GPT`（RoPE + RMSNorm + Pre-Norm + tie weights），配置 L=6/d=384/H=6 ≈ 10M 参数，block_size=256。

---

## 9.3 动手：四步跑通里程碑

```bash
# ① 准备数据（首次约 5~7 分钟，大头是纯 Python BPE 训练；产物会缓存）
uv run projects/minigpt/prepare_data.py

# ② 冒烟测试（几十步，确认管线通、loss 在动、不报错——第 6 章调试节奏第一档）
uv run projects/minigpt/train.py --smoke

# ③ 正式训练（M4 约 10~20 分钟，3000 步）
uv run projects/minigpt/train.py

# ④ 用训好的模型续写
uv run projects/minigpt/generate.py --prompt "话说" --tokens 300
```

**看什么**（训练时盯三条线，第 6 章的健康监控最小集）：
- **loss**：应从 ~8.3（=ln 4096，完全随机的理论值）开始，快速降到 3~4 区间；
- **gnorm**：梯度范数应平稳在个位数，持续飙高预示不稳定；
- **val vs train**：小语料上 val 会先降后升（过拟合），train 持续降——这个 gap 就是"背课文"的量化体现，正是第 14 章要用微调对抗的东西。

**生成质量的演进**（你会亲眼看到）：
- 冒烟后（几十步）：纯乱码/高频字堆砌；
- 500 步：开始出现成词、标点位置合理；
- 3000 步：短句成形、有《红楼梦》的文白腔调（虽然逻辑仍会漂移——10M 参数 + 几十万 token 的能力上限，别期待 GPT-4）。

---

## 9.4 易错点清单（本项目实战踩点）

**① 全语料一次性 encode 卡死**
→ **现象**：`prepare_data` 在 tokenize 阶段挂起十几分钟无输出。
→ **原因**：朴素 BPE encode 是 O(n·merges)，对 90 万 token 的整篇文档一次编码 ≈ 数十亿次 Python 操作。**这是本项目开发时真实踩的坑**（教程作者也中招了）。
→ **修正**：按行分块 encode（`tokenizer.encode` 已内置 `chunk_on="\n"`）——GPT-2 用正则按词切分是同一个道理（第 5 章）。

**② vocab_size 配置与 tokenizer 不一致**
→ **现象**：`IndexError` 或 embedding 越界，或 loss 起点不是 ln(vocab)。
→ **修正**：`GPTConfig.vocab_size` 必须取 `tokenizer.vocab_size`（train.py 已从加载的 tokenizer 读取，不硬编码）。

**③ block_size 与数据切块不匹配 / 采样越界**
→ `get_batch` 里 `randint(len(data) - block_size)` 的减法不能少，否则末尾采样越界。改 block_size 后无需重新准备数据（采样是在线切的），但改 vocab 必须重跑 prepare。

**④ resume 只存了模型没存优化器**（第 6 章易错点④的实战）
→ train.py 的 checkpoint 存了 model + optimizer + step 三件套。删掉 optimizer 那行再 resume，你会看到 loss 明显跳升——亲手验证优化器状态的价值。

**⑤ 生成时忘了 eval() / 忘了截断到 block_size**
→ `model.generate` 内部已 `eval()` 且对超长上下文做 `idx[:, -block_size:]` 截断（RoPE 虽支持外推但训练长度外质量下降，第 8 章）。自己写生成循环时这两点最易漏。

---

## 9.5 开源对照：nanoGPT

我们的 miniGPT 是 [nanoGPT](https://github.com/karpathy/nanoGPT) 的教学精简版，差异一目了然：

| | 我们的 miniGPT | nanoGPT |
|---|---|---|
| tokenizer | 自写 BPE（教学） | tiktoken（GPT-2 词表）或字符级 |
| 模型 | RoPE + RMSNorm（现代） | learned pos + LayerNorm（GPT-2 原味） |
| 数据 | 《红楼梦》，行分块 encode | OpenWebText，多进程 encode |
| 分布式 | 单卡（第 12 章再上多卡） | DDP 内置 |
| 规模 | 10M / M4 / 20 分钟 | 124M+ / 多 GPU / 数天 |

读 nanoGPT 的 `train.py` 你会发现结构与我们高度一致——因为训练循环的骨架就这一种。它多出来的是 DDP 包装（第 12 章）、`torch.compile`（第 11 章）、梯度累积到大 batch（我们的 train.py 留作练习）、和更完整的 checkpoint/日志。**这些正是第三篇要逐个补上的工业化拼图。**

---

## 9.6 典型面试题

**Q1：从零训练一个语言模型，端到端有哪些步骤？每步的常见坑？**

> **参考答案**：① 数据收集清洗（去重、质量过滤——脏数据直接决定上限）；② tokenizer 训练（词表大小权衡：大词表序列短但 embedding 大、稀有 token 训不充分）；③ 离线 tokenize + packing（O(n²) encode 坑、packing 提效）；④ 模型定义（初始化缩放、位置编码）；⑤ 训练循环（六步次序、AMP、调度、裁剪）；⑥ checkpoint 与续训（四件套）；⑦ 评估与生成（eval 模式、采样策略）。**加分点**：能说出每步在小规模和大规模下的不同侧重（小规模防过拟合，大规模防数据/通信瓶颈）。

**Q2：训练刚开始 loss 应该是多少？如果一开始就远低于这个值说明什么？**

> **参考答案**：均匀分布下 loss = ln(vocab_size)（如 vocab=4096 → ≈8.3；GPT-2 的 50257 → ≈10.8）——模型对下一个词毫无信息时的理论值。一开始就远低于它：要么标签泄漏（模型能看到答案，第 7/8 章因果泄漏），要么评估用了训练见过的数据，要么 loss 计算错误（如 target 没错位、ignore_index 用错）。**加分点**：loss 的理论下界是数据的条件熵，不可能降到 0（除非过拟合背下来）；用 bits-per-byte 或 perplexity 做跨词表的可比指标。

**Q3：小模型在小语料上 val loss 先降后升，为什么？工业大模型为什么较少见到这个现象？**

> **参考答案**：先降是在学真实规律，后升是开始死记训练集特例（过拟合）——容量相对数据过剩。大模型预训练很少见明显过拟合，因为数据量极大（万亿 token）、通常只训不到一个 epoch（每条数据基本只见一次），模型没有机会"背"；反而常处于欠拟合区（加数据/加算力还能继续降）。**加分点**：这解释了为什么预训练常不用 dropout（第 8 章），以及 Chinchilla 缩放律关心的"给定算力，参数与数据如何配比"。

**Q4：tokenizer 的词表大小如何影响模型？调大调小各有什么后果？**

> **参考答案**：词表大 → 序列变短（同样文本更少 token，算力省、上下文装更多内容），但 embedding/lm_head 参数随 V 线性增长、稀有 token 见得少训不透；词表小 → 序列长（O(T²) 注意力吃亏）、但每个 token 训练充分。中文/多语言倾向更大词表（覆盖字符多）。**加分点**：字节级 BPE 无 OOV（任何文本都能编码，我们的实现就是）；tie weights 让大词表的 embedding 成本减半（第 4 章）；实际是"压缩率 vs 参数量 vs 训练充分度"的三方权衡。

---

## 9.7 疑难杂症排查（本项目专属）

**问题 1：loss 不降/降得很慢**
→ 先做过拟合单 batch 测试（第 6 章黄金测试，gpt_model.py 自检里已内置）：抓一个 batch 反复训，几十步内应→0。不行 → 模型/数据连接 bug；行 → 回到全量，查 lr（试 3e-4/1e-3/3e-3）、查数据加载是否正确错位。

**问题 2：生成全是重复的高频字（"的的的的"/乱码）**
→ 训练不足（步数太少）或采样问题。检查：① 训练 loss 是否真降下来了；② 生成用了 temperature 和 top_k 没有（纯 argmax 贪心会陷入高频词循环，temperature=0.8 + top_k=40 缓解）；③ 是否忘了 eval()；④ **是不是保存了 loss spike 时刻的 checkpoint**（见问题 5 的真实事故）。

**问题 5：loss 反复震荡 + 最终生成乱码（本项目开发时的真实事故，完整复盘）**

这是教程作者训练本项目时**真实踩到并排查的坑**，值得完整讲一遍——它示范了 systematic-debugging（第 6 章）在实战中的样子。

**症状**：miniGPT 训练时 loss 在 3 和 11 之间反复震荡（而不是平滑下降），gnorm 高达 8~15（健康应 <1），最终保存的 checkpoint 恰好落在一个尖峰上，`generate` 出来全是 `喯喯喯` 乱码。

**排查经过（每步只改一个变量）**：
1. 怀疑学习率过高：1e-3 → 6e-4 → 3e-4 逐级降——**仍震荡**（排除单纯 lr）；
2. 怀疑 MPS 的 bf16 autocast：bf16 vs fp32 各跑 150/600 步对照——**两者都稳定**（排除混合精度）；
3. 隔离每个配置差异：param groups、warmup 长度、cosine 调度、autocast wrapper、non_blocking——**逐个测试全部稳定**；
4. 验证数据与初始化：检查 token id 越界（无）、逐位对比 batch/init 校验和（与稳定版**完全相同**）；
5. **关键突破**：注意到 gnorm 异常高（5~13），而 Adam 的 β₂=0.95 会让二阶矩快速适应——遇到偶发大梯度时，分母 √v 骤降会**放大**下一步。换回 AdamW 默认的 **β₂=0.999**（二阶矩更平滑），跑 3 个随机种子——**全部稳定，gnorm 稳定在 ~1**。

**根因**：β₂=0.95 是 LLaMA 等**大模型 + 大 batch**的配置（第 6 章 6.5-②），它对偶发大梯度更敏感。在这个 12M 小模型 + batch=16 的设置下，梯度噪声大，0.95 的快速二阶矩适应放大了噪声，把模型推入失稳。**同一个超参，大模型的良药是小模型的毒药**——这是"照抄大模型配置"的经典陷阱。

**修复（多管齐下，通用于任何 loss spike）**：① **β₂ 0.95 → 0.999**（根因）；② 梯度裁剪 1.0 → 0.5（限单步伤害）；③ 更长 warmup；④ **按 val 存最优 checkpoint 而非存最后一步**——这是最关键的工程保险：即便训练偶有尖峰，最终拿到的永远是好模型。

**三条教训**：① loss spike 的根因可能藏在你最不怀疑的超参里（谁会想到 β₂？）；② 高 gnorm 是失稳的早期信号，监控它（第 6 章三线监控）；③ 最可靠的防御不是消除所有尖峰，而是 best-checkpoint 这类"让偶发故障不影响最终产物"的设计——这个思想在第 15 章的大规模训练手册里会再次出现。

**问题 3：MPS 上训练比预期慢**
→ ① 确认真在 MPS 上（打印 device）；② batch_size 太小 kernel 启动开销占比高（第 0 章的小矩阵现象）——适当调大；③ 频繁 `.item()` 同步（第 2 章）——日志降频。第 11 章的 profiler 会教你精确定位。

**问题 4：`prepare_data` 下载失败**
→ Gutenberg 偶尔不稳。脚本会缓存 `data/raw.txt`，可手动下载放进去；或换任意 UTF-8 文本语料（改 `URL` 或直接替换 raw.txt）——这个管线对任何文本都通用。

---

## 9.8 练习题（把里程碑做深）

### 基础 1：换语料
把《红楼梦》换成你自己的语料（任意 UTF-8 文本：另一本书、你的聊天记录、代码）。重跑 prepare + train，观察 tokenizer 学到的 merge 和生成风格如何随语料改变。

### 基础 2：resume 验证
训练 1500 步存 checkpoint，然后 `--resume` 再训 1500 步，确认 loss 平滑衔接。再故意在 checkpoint 里删掉 optimizer state 重载，观察 loss 跳升——亲手验证第 6 章易错点④。

### 进阶 1：加梯度累积
train.py 目前每步一个 batch。加入梯度累积（第 3/6 章），用 accum_steps=4 模拟 4 倍 batch，对比收敛曲线与显存占用。验证 loss 要除以 accum_steps（第 3 章基础 2）。

### 进阶 2：scaling 迷你实验
在固定语料上训练三个尺寸（如 d=192/384/768，其余不变），画出"参数量 vs 最终 val loss"。你会看到一条粗糙的缩放律——Chinchilla 论文的家庭作坊版。讨论：什么时候加参数不再降 loss（数据成为瓶颈）？

### 挑战 1：实现 KV cache 加速生成（第 16 章预习）
当前 `generate` 每生成一个 token 都对整个前缀重新前向（O(T²) 浪费）。实现 KV cache：缓存历史的 K/V，每步只算新 token 的 Q 并 append K/V。测量生成 500 token 的加速比。这是第 16/17 章的核心技术，在这里先尝一口。

---

## 本章小结与下一章预告

你从零训出了一个能续写古文的 GPT——**第一篇到第二篇的所有知识在这里闭环**。tokenizer、数据、模型、训练循环第一次作为一个整体运转，你也建立了"小的先跑通"的工程直觉和调试节奏。

**下一章（第 10 章）**：现代 LLM 架构演进。我们的 miniGPT 用了 RoPE + RMSNorm，但真实的 LLaMA/Qwen/DeepSeek 还有 GQA（第 7 章挑战题已尝）、SwiGLU、MoE 等一系列升级。我们会逐行读它们的源码，把"为什么这样改"讲清楚——读懂现代 LLM 源码，你就摸到了第二篇的终点。
