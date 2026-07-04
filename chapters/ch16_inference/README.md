# 第 16 章 · 推理优化原理：KV Cache、量化、投机解码

> **本章目标**：理解 LLM 推理为什么慢、怎么加速。学完你应该能回答：
> 1. 自回归生成的 O(T²) 重复计算是什么？KV Cache 怎么消除它？
> 2. 为什么说 decode 阶段是 memory-bound？这决定了什么优化方向？
> 3. GPTQ/AWQ 怎么把模型量化到 4-bit 还能用？和 QLoRA 的量化有何不同？
> 4. 投机解码为什么能"用小模型加速大模型"而不损质量？

**前置**：第 7 章（注意力）、第 11 章（Roofline/memory-bound）、第 2 章（KV cache 账）。 **硬件路径**：本地实现 KV cache 与量化原理；vLLM 级性能上云。 **预计用时**：6~7 小时。
**视角转变**：前面都在讲训练（一次性成本），本章起讲推理（每天亿万次，成本大头）。优化目标从"MFU"变成"延迟 + 吞吐 + 成本"。

---

## 16.1 来龙去脉：推理才是成本大头

一个大模型训练一次，推理服务几年。ChatGPT 每天处理数十亿次请求——**推理的累计成本远超训练**。而推理有它独特的性能特征，和训练完全不同：

**自回归生成的本质**：LLM 一次只生成一个 token，然后把它拼回输入，再生成下一个。生成 100 个 token 要前向 100 次。每次前向都要处理"当前的完整序列"——这里藏着巨大的浪费（16.2-①）。

**两个阶段，两种瓶颈**：
- **Prefill（预填充）**：处理输入 prompt，一次性前向整个 prompt。计算密集（大矩阵乘），**compute-bound**。
- **Decode（解码）**：逐 token 生成。每步只算一个新 token，但要读取全部模型权重和 KV cache——计算量小、访存量大，**memory-bound**（第 11 章 Roofline）。

这个区分决定了一切推理优化：decode 是 memory-bound，所以优化方向是**减少访存**（量化压缩权重、KV cache 管理）和**提高并行度**（batching 摊薄权重读取）。这和训练（compute-bound，优化算力）截然不同。

---

## 16.2 核心原理

### ① KV Cache：消除自回归的重复计算

生成第 t 个 token 时，注意力要算 query_t 对所有历史 key/value 的注意力。朴素实现每步都重新计算所有位置的 K、V——但**历史 token 的 K、V 不变**（因果注意力，位置 i 的 K/V 只依赖位置 i 及之前）！重复计算它们是纯浪费。

**KV Cache**：把每个位置算过的 K、V 缓存起来，生成新 token 时只算**新位置**的 Q、K、V，K/V 追加到缓存，Q 对整个缓存做注意力：

```
朴素：生成第 t 步，重新算 t 个位置的 K/V  → 总计算 O(T²)
KV cache：第 t 步只算 1 个新位置的 K/V     → 总计算 O(T)
```

代价是显存：缓存要存 `2 × L × n_kv_heads × T × d_head × 字节`（第 2 章算过，也是第 10 章 GQA 要减的东西）。KV cache 是**用显存换计算**——而这正好把 decode 从"重复计算"变成"读取缓存"，强化了它的 memory-bound 特性。本章实验给第 8 章的 miniGPT 加 KV cache，实测生成加速。

### ② decode 是 memory-bound 的推论：batching 是免费的午餐

decode 每步的计算量 = 一个 token 过一遍模型 ≈ 2N FLOPs（N=参数量）。访存量 = 读取全部权重 ≈ 2N 字节（fp16）。算术强度 ≈ 1 FLOP/byte——**极度 memory-bound**（第 11 章）。

关键推论：**读一次权重可以服务多个请求**。把 B 个请求的 token 攒成一个 batch，权重只读一次却算了 B 个 token——计算量 ×B，访存量不变，算术强度 ×B。所以 **batch 越大，吞吐越高，直到算术强度撞上 compute-bound**。这就是为什么推理服务拼命做 batching（第 17 章 continuous batching 是它的极致）。单请求推理浪费带宽，是最贵的用法。

### ③ 量化：把权重压小，直接对症 memory-bound

既然 decode 卡在"读权重"，那**把权重变小**就直接加速。量化把 fp16 权重（2 字节）压到 int8（1 字节）甚至 int4（0.5 字节）——访存量减半/减四分之三，decode 直接快对应倍数。

**训练后量化（PTQ）的两个主流方法**（和 QLoRA 的量化不同——QLoRA 量化是为了微调时省显存，这里是为了推理加速）：

- **GPTQ**：逐层量化，用少量校准数据，基于二阶信息（Hessian）逐列量化并补偿误差。精度高，需要校准。
- **AWQ（Activation-aware）**：观察到不是所有权重同等重要——激活值大的通道对应的权重更关键。AWQ 保护这些"重要通道"（不量化或用更高精度），其余激进量化。

两者都能把 7B 压到 4-bit（14GB→3.5GB）而精度损失很小。**量化的核心权衡**：位数越低越快越省，但精度损失越大；关键是"在哪损失"——异常值（outlier）通道是精度杀手，各方法都在处理它。

**KV Cache 量化**：KV cache 也能量化（int8/int4），长上下文时 KV cache 比权重还大，量化它收益显著。

### ④ 投机解码（Speculative Decoding）：用小模型赌大模型

decode 是串行的（生成第 t 个才能生成第 t+1 个）——这是延迟的根源。**投机解码**打破串行：用一个小的"草稿模型"快速生成 K 个候选 token，然后用大模型**一次前向并行验证**这 K 个（大模型的一次前向能同时算 K 个位置，因为它们已经有草稿了）：

```
草稿模型（小、快）：连续生成 5 个候选 token（便宜）
大模型（大、准）：一次前向并行验证这 5 个
  - 接受草稿正确的前缀（可能全接受、可能接受前 3 个）
  - 从第一个错误处用大模型的输出纠正
```

**为什么不损质量**：验证用的是大模型的真实分布，接受/拒绝的规则（拒绝采样）保证最终输出分布和"纯大模型生成"完全一致——这是数学保证，不是近似。加速来源：大模型一次前向验证多个 token（利用它 memory-bound 时的空闲算力，②的推论）。草稿命中率越高加速越大（通常 2-3 倍）。变体：Medusa（多头自投机）、EAGLE 等。

### ⑤ 其他推理优化拼图

- **PagedAttention**（第 17 章主角）：像操作系统分页管理 KV cache，消除碎片，是 vLLM 的核心。
- **算子融合 / FlashAttention**（第 11 章）：推理同样受益。
- **连续批处理**（第 17 章）：动态调度不同长度的请求。
- **张量并行推理**（第 13 章）：大模型跨卡推理。

本章讲原理，第 17 章手写 mini-vLLM 把 KV cache + continuous batching + PagedAttention 组装成一个推理引擎。

---

## 16.3 动手实验

```bash
uv run chapters/ch16_inference/code/kv_cache.py            # 给 miniGPT 加 KV cache，实测加速
uv run chapters/ch16_inference/code/prefill_decode.py      # prefill vs decode 的 compute/memory-bound 对比
uv run chapters/ch16_inference/code/quantization.py        # 手写 int8 量化，验证精度/速度权衡
uv run chapters/ch16_inference/code/speculative.py         # 投机解码模拟，验证分布一致 + 加速
```

`kv_cache.py` 是第 9 章挑战题的完整实现——给 GPT 的注意力加缓存，生成从 O(T²) 降到 O(T)，序列越长加速越明显。

---

## 16.4 易错点清单

**① KV cache 的位置索引错误**
→ **现象**：加了 cache 后生成质量下降/乱码。
→ **原因**：RoPE 的位置索引要用"缓存中的绝对位置"，不是从 0 数（第 8 章 RoPE + 第 7 章易错点⑥）。新 token 的位置是 `cache_len`，不是 0。
→ **修正**：维护 cache 长度，RoPE 用它算新位置的旋转角。

**② KV cache 显存预估不足导致 OOM**
→ **现象**：长上下文或高并发时 OOM。
→ **原因**：KV cache 随序列长和并发数线性增长（第 2 章账），长上下文时可能超过权重本身。
→ **修正**：预估 KV cache 峰值（第 17 章 PagedAttention 就是为管理它）；用 GQA（第 10 章）、KV 量化（16.2-③）。

**③ 量化 group size 与精度权衡搞错**
→ 量化通常按 group（如每 128 个权重共享一个 scale）。group 越小精度越高但元数据越多。用错 group size 精度崩或省不了多少。跟随成熟配置（GPTQ/AWQ 默认 group 128）。

**④ 单请求推理还纳闷为什么慢/贵**
→ decode memory-bound，单请求浪费带宽（②）。这不是 bug 是特性——推理要 batching 才划算。个人本地跑单请求慢是正常的。

**⑤ 投机解码的草稿模型选错**
→ 草稿模型要和大模型分布接近（同系列小模型最好）且足够快。差太多则命中率低、加速有限甚至变慢（草稿开销 > 收益）。

**⑥ 忘了 eval 模式 / use_cache 配置**
→ 推理要 `model.eval()` + `use_cache=True`（第 4 章）；训练要 `use_cache=False`（省显存）。混了会行为异常或浪费显存。

---

## 16.5 开源项目的最佳实践

**① vLLM：推理引擎的事实标准**
[vllm-project/vllm](https://github.com/vllm-project/vllm)：PagedAttention + continuous batching + 各种量化支持。它把本章所有优化 + 第 17 章的调度组装成生产级引擎。读它的 `attention` 和 `scheduler` 模块——第 17 章我们会手写简化版再对照它。

**② llama.cpp：量化推理的极致**
[ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)：GGUF 格式 + 各种量化（Q4_K_M 等）+ CPU/Metal 优化。它让 7B 模型在你的 M4 上流畅跑（第 18 章实战）。看它的量化格式设计——为消费级硬件优化的典范。

**③ HF `generate` 的 KV cache 与采样**
transformers 的 `generate()` 内置 KV cache（`use_cache=True`）、各种采样策略（temperature/top-k/top-p/beam）、以及投机解码（`assistant_model` 参数）。读它的 `GenerationMixin` 理解生产级生成循环——比你手写的多了 cache 管理、停止条件、批处理等工程细节。

---

## 16.6 典型面试题

**Q1：KV Cache 是什么？为什么能加速？代价是什么？**

> **参考答案**：自回归生成中，历史 token 的 K/V 在因果注意力下不变，缓存它们避免每步重算。朴素生成 O(T²)（每步重算所有位置 K/V），KV cache 降到 O(T)（每步只算新位置）。代价是显存：2×L×n_kv_heads×T×d_head×字节，随序列长和并发线性增长，长上下文时可能超过权重。**加分点**：KV cache 强化了 decode 的 memory-bound 特性；GQA/MQA 减 KV 头数、KV 量化、PagedAttention 都是管理它的手段；prefill 阶段一次性填充整个 prompt 的 cache。

**Q2：为什么说 LLM 推理的 decode 阶段是 memory-bound？这有什么优化含义？**

> **参考答案**：decode 每步只生成一个 token，计算量 ≈ 2N FLOPs，但要读取全部权重 ≈ 2N 字节，算术强度 ≈ 1 FLOP/byte，远低于 GPU 的 ridge point（第 11 章）——卡在带宽不是算力。含义：① batching 免费提升吞吐（读一次权重服务多个请求，算术强度 ×B）；② 量化直接加速（权重变小减访存）；③ 增加算力（更强 GPU）对单请求 decode 帮助有限。而 prefill 是 compute-bound（处理整个 prompt 的大矩阵乘）。**加分点**：这解释了推理服务为什么拼命 batching（第 17 章 continuous batching）、为什么量化对推理如此重要、以及投机解码利用 decode 的空闲算力。

**Q3：GPTQ、AWQ、QLoRA 的量化有什么区别？**

> **参考答案**：目的不同——GPTQ/AWQ 是**推理**量化（PTQ，压缩已训练模型加速推理），QLoRA 的 4-bit 是**训练**量化（微调时省显存，主干只读）。方法：GPTQ 逐层基于 Hessian 二阶信息量化+误差补偿（需校准数据）；AWQ 激活感知，保护重要通道（激活大的通道对应权重更关键）；QLoRA 用 NF4（针对正态分布权重的格式）+ 双重量化，主干量化后冻结、LoRA 旁路高精度训练。**加分点**：都在处理 outlier 通道（精度杀手）；量化的 group size 权衡；KV cache 也可量化；推理量化追求精度-速度平衡，训练量化追求省显存不损梯度。

**Q4：投机解码为什么能加速且不损失质量？**

> **参考答案**：串行 decode 是延迟瓶颈。投机解码用小草稿模型快速生成 K 个候选，大模型一次前向并行验证（利用 decode memory-bound 时的空闲算力，验证 K 个 token 的成本≈验证 1 个）。用拒绝采样规则接受/纠正，数学上保证输出分布与纯大模型一致——不损质量。加速来自命中的草稿 token（一次大模型前向产出多个 token），通常 2-3 倍。**加分点**：草稿模型要分布接近且快（同系列小模型）；命中率决定加速比；变体 Medusa（自投机多头）、EAGLE（特征级投机）省掉独立草稿模型；对 memory-bound 的 decode 特别有效。

---

## 16.7 疑难杂症排查

**案例 1：加 KV cache 后生成结果和不加时不一致**

正确的 KV cache 应该产出**完全相同**的结果（只是更快）。不一致说明实现有 bug：① 位置索引错（易错点①，RoPE 用错位置）——最常见；② cache 追加顺序错或漏了某些位置；③ 注意力 mask 在 cache 场景下没正确处理（decode 时 query 长度 1、key 长度 =cache_len）。排查：对同一输入，对比有/无 cache 的 logits，应 allclose；不一致则二分定位哪一步 cache 出错。

**案例 2：量化后模型精度大幅下降**

① group size 太大（易错点③）——减小试试；② 没处理 outlier 通道——用 AWQ 类方法或混合精度；③ 量化了不该量化的层（如 embedding/lm_head 对量化敏感）——排除它们；④ 校准数据不代表实际分布（GPTQ）——用更有代表性的校准集。评估：用困惑度或下游任务对比量化前后，别只看能否运行。

**案例 3：推理吞吐远低于预期**

① 没做 batching（易错点④）——单请求 memory-bound 浪费带宽；② batch 内序列长度差异大，padding 浪费（第 17 章 continuous batching 解决）；③ KV cache 管理差、碎片多（第 17 章 PagedAttention）；④ 没用优化的注意力（FlashAttention）。方法论：推理性能先看 batching 是否充分，再看 KV cache 管理，最后看 kernel。

---

## 16.8 练习题

### 基础 1：给 miniGPT 加 KV cache
用 `kv_cache.py` 的框架（或完成第 9 章挑战题），给第 8 章的 GPT 注意力加 KV cache。验证：有/无 cache 生成结果 allclose（正确性），测量生成 500 token 的加速比（性能），画出"序列长度 vs 加速比"。

### 基础 2：prefill/decode 特征测量
用 `prefill_decode.py`，测量 prefill（处理长 prompt）和 decode（逐 token）的算术强度和瓶颈类型。验证 prefill compute-bound、decode memory-bound。回答：为什么 batch decode 能提升吞吐而 batch prefill 收益小？

### 进阶 1：手写 int8 量化
实现 per-channel int8 对称量化（每列一个 scale）+ 反量化，对一个线性层量化，测量：精度损失（量化前后输出的相对误差）、理论访存减少。对比 per-tensor（整个矩阵一个 scale）和 per-channel 的精度差异。

### 挑战 1：投机解码模拟
用两个不同大小的模型（如 miniGPT 的大小两版），实现投机解码：小模型生成 K 个草稿，大模型并行验证 + 拒绝采样。验证：输出分布与纯大模型一致（统计多次生成的分布）；测量加速比与草稿命中率的关系。讨论：什么情况投机解码反而变慢？

---

## 本章小结与下一章预告

推理是成本大头，且 memory-bound（decode 阶段）主导优化方向。KV Cache 消除自回归的 O(T²) 重复计算（用显存换计算）；batching 免费提升吞吐（读一次权重服务多请求）；量化（GPTQ/AWQ）直接减访存加速；投机解码用小模型草稿 + 大模型并行验证，不损质量地打破串行。这些是把训好的模型高效服务出去的核心武器。

**下一章（第 17 章）**：🏆 里程碑三。把 KV Cache + continuous batching + 简化版 PagedAttention 组装成一个**能跑的推理引擎 mini-vLLM**，然后对照 vLLM 真实源码。推理优化的原理在这里变成一个你亲手写的服务系统。
