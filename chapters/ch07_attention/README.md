# 第 7 章 · 注意力机制：从 RNN 的困境推导出必然的形状

> **本章目标**：不背公式，推导公式。学完你应该能回答：
> 1. RNN 到底败在哪三件事上？注意力分别怎么解的？
> 2. Q、K、V 三个投影为什么缺一不可？`softmax(QKᵀ/√d)V` 每个符号为什么必须在那？
> 3. 多头注意力的"头"是什么？实现时那串 view/transpose 在倒腾什么？
> 4. 注意力的 O(T²) 在哪里？它如何决定了后面三章的技术议程？

**前置**：第 1 章（广播掩码）、第 2 章（view/transpose）、第 6 章（数值稳定性直觉）。 **硬件路径**：本地。 **预计用时**：6~8 小时（本章值得慢读）。

---

## 7.1 来龙去脉：RNN 的三宗罪

2017 年之前，序列建模的统治者是 RNN/LSTM：按时间步吃 token，把"迄今为止的一切"压缩进一个固定大小的 hidden state。三个结构性缺陷最终杀死了它：

**罪一：顺序依赖，不能并行**。h_t 依赖 h_{t-1}，1000 个 token 必须串行算 1000 步。GPU 是吞吐机器（第 0 章你实测过它靠并行摊薄开销），串行负载暴殄天物——训练速度被序列长度锁死。**这条罪在"用海量数据训大模型"的时代是死刑**：Transformer 能吞下万亿 token，RNN 在工程上做不到。

**罪二：信息瓶颈**。整段前文压进固定维度的 hidden state——像要求你只用一张便签纸做整本书的笔记。第 500 个 token 想用第 3 个 token 的信息？它得在便签纸上活过 497 次改写。

**罪三：梯度长征**。反向传播要沿时间步链式回传（第 3 章的图，串成 1000 节），梯度逐步衰减/爆炸。LSTM 的门控是缓解不是治愈。

**注意力的起点是一个补丁**：2014 年 Bahdanau 给机器翻译的 RNN 加了个机制——解码每个词时，回头**直接看**编码器的所有位置，按相关性加权取用，绕过便签纸。效果拔群。2017 年《Attention is All You Need》的激进之处在于**把补丁变成全部**：扔掉 RNN，序列内每个位置直接与所有位置连线（self-attention）。三宗罪同时消解：所有位置并行计算（矩阵乘，GPU 最爱）；任意两位置一步直达（信息不过便签纸）；梯度路径长度 O(1)（第 3 章残差图分析的加强版）。

代价当场记下：**每个位置看所有位置 = O(T²)**。这笔账贯穿后面所有章节——FlashAttention（第 11 章）、KV cache 与 PagedAttention（第 16/17 章）都是在给这个 O(T²) 打工。

---

## 7.2 核心原理

### ① 从"软检索"推导出 QKV

需求：位置 t 的表示应该按相关性聚合全序列的信息。把它形式化成一次字典查询：

- 每个位置发出一个**查询（Query）**："我在找什么"
- 每个位置挂出一个**键（Key）**："我能被什么找到"
- 每个位置备好一个**值（Value）**："找到我后取走什么"

相似度 = q·k（点积，最便宜的相似度）；聚合 = 按相似度的 softmax 权重对 V 加权求和（"软"检索——不是取 top-1 而是全体加权，可导！硬选择不可导，第 3 章的知识告诉你那没法训练）。

**为什么必须是三个不同的投影**（`Q=XW_q, K=XW_k, V=XW_v`）而不是直接用 X 自己点积自己？因为一个词的三种角色语义不同："bank" 作为查询者想找 "river/money" 来消歧；作为被查者要能被 "deposit" 找到；被找到后要交出的是它的语义内容。三个可学习矩阵让模型自己学会三种角色的表达。**去掉投影的注意力（X·Xᵀ）退化成"相似词互看"，学不出语法这类非对称关系**。

### ② 为什么除以 √d_k：一次方差推导

设 q、k 的分量独立、均值 0、方差 1（初始化时近似成立）。点积 `q·k = Σᵢ qᵢkᵢ` 有 d_k 个独立项，**方差 = d_k**——d_k=128 时标准差 ≈ 11.3，logits 动辄 ±30。softmax 对这种量级的输入**饱和**：最大项拿走全部概率，其余归零。饱和的 softmax 梯度 ≈ 0（可以从 softmax 的雅可比看出：p(1-p) 在 p→0/1 时消失）——注意力学不动。除以 √d_k 把方差拉回 1，softmax 工作在灵敏区。**本章实验用数字验证这条推导**（不缩放时梯度范数塌缩几个数量级）。

同族的数值稳定性细节：softmax 实现要**先减去行最大值**（`exp(30)` 就上溢 fp16 了，减 max 数学等价、数值安全）——`F.softmax` 帮你做了，手写时必须记得（本章实验演示上溢现场）。

### ③ 因果掩码：自回归的信息屏障

语言模型的训练目标是"预测下一个 token"（第 5 章 packing 的 x/y 错位）。如果位置 t 的表示能看到位置 t+1，预测就是抄答案。因果掩码在 softmax **之前**把"未来位置"的 logits 设为 -inf：

```python
scores = scores.masked_fill(causal_mask == 0, float("-inf"))   # 第 1 章的广播实战
attn = scores.softmax(dim=-1)     # exp(-inf)=0 → 未来位置的权重精确为 0
```

**为什么必须在 softmax 之前**：softmax 之后置零会破坏"权重和为 1"（分母里已经计入了未来位置），信息仍然泄漏在归一化常数里。这是易错点②，也是面试题。

### ④ 多头：同一序列的多路视角

单头注意力每个位置只能输出**一种**加权模式。但"下一个词"可能同时需要语法视角（主语是谁）、指代视角（it 指什么）、语义视角（话题是什么）。多头的做法：把 d_model 切成 H 份，每份 d_head = d_model/H，各自独立做注意力，最后拼接再过一个输出投影 W_o 融合。

实现上的关键舞步（第 2 章 view/transpose 的实战考试）：

```python
# x: (B, T, C)   C = d_model
q = self.wq(x)                                   # (B, T, C) 一次大矩阵乘算全部头
q = q.view(B, T, H, D).transpose(1, 2)           # (B, H, T, D) ← 把头调到 batch 维旁边
...                                              # 每个头独立算注意力（广播并行）
out = out.transpose(1, 2).contiguous().view(B, T, C)   # 拼回 (B, T, C)
```

为什么要 transpose 到 `(B,H,T,D)`：让矩阵乘 `q @ k.transpose(-2,-1)` 的最后两维恰好是 (T,D)×(D,T)→(T,T)，B 和 H 都成为广播出去的"批"维。为什么回程要 `contiguous()`：transpose 后非连续，view 会报错——第 2 章易错点①在真实代码里的登场。**`(B,T,H,D)` 和 `(B,H,T,D)` 搞混不报错但全错**（shape 恰好都合法），这是注意力实现的第一大静默杀手（易错点⑤）。

### ⑤ 复杂度：这个机制的价格标签

- 时间：QKᵀ 是 (T,d)×(d,T) → **O(T²·d)**；序列翻倍，注意力计算×4。
- 内存：注意力矩阵 (B,H,T,T) → **O(T²)**。T=4096、H=32、B=8、fp16：8×32×4096²×2B = **8.6 GB**，只是一层的注意力矩阵！

这两个数字决定了技术演进的方向：训练侧 FlashAttention 用"分块重算"消掉 O(T²) 内存（第 11 章）；推理侧 KV cache 把逐 token 生成的重复计算摊掉、但换来显存包袱（第 16 章）。现在你知道这些技术"为什么存在"了。

---

## 7.3 动手实验

```bash
uv run chapters/ch07_attention/code/scaling_why.py            # √d 缩放的方差推导数字验证 + softmax 饱和/上溢
uv run chapters/ch07_attention/code/attention_from_scratch.py # 单头→因果→多头逐步构建，与官方 SDPA 对拍
uv run chapters/ch07_attention/code/attention_patterns.py     # 训练一个玩具注意力，看它学出的模式
```

第二个脚本的多头实现（`MultiHeadAttention` 类）就是第 8 章 Transformer 的核心组件——**逐行读，每行的 shape 注释都是纪律**。

---

## 7.4 易错点清单

**① 忘了除 √d_k**
→ **现象**：小模型勉强能训（d 小方差没那么夸张），d_model 一大 loss 不降。梯度范数监控（第 6 章）会显示注意力层梯度异常小。
→ **修正**：`scores = q @ k.transpose(-2,-1) / math.sqrt(d_head)`——注意除的是**每头维度** d_head，不是 d_model（用错的后果是缩放力度差 √H 倍）。

**② 掩码加在 softmax 之后**
→ **现象**：训练 loss 好得可疑、val 也好，一到自回归生成就胡言乱语——训练时模型一直在"偷看未来"，生成时没有未来可偷。
→ **修正**：softmax 前 masked_fill(-inf)（7.2-③ 的原理）。**识别信号**：训练指标好到不合理 = 先查信息泄漏。

**③ 全 -inf 行产生 NaN**
→ **现象**：softmax 输出 NaN（一整行都被 mask 时，exp 和为 0，0/0）。常见于 padding mask 与 causal mask 组合后某些行全遮。
→ **修正**：保证每行至少一个可见位置（自己看自己永远合法——对角线不 mask）；或 softmax 后 `nan_to_num`。第 15 章 NaN 排查的常客。

**④ `(B,T,H,D)` vs `(B,H,T,D)` 搞混**
→ **现象**：不报错！shape 全程合法，注意力在错误的维度上计算（token 之间变成头之间），模型能训但效果差一截——**静默错误的天花板**。
→ **修正**：铁律——每行张量操作后面写 shape 注释；实现完与 `F.scaled_dot_product_attention` 对拍（本章实验示范），allclose 过了才算实现完成。

**⑤ padding mask 与 causal mask 混淆**
→ 两个语义不同的 mask：causal 屏蔽"未来"（下三角，(T,T)，训练推理都要）；padding 屏蔽"填充位"（(B,1,1,T)，batch 内变长时要）。组合用**逻辑与**。packing 数据（第 5 章）没有 padding，只需 causal——第 9 章我们享受这个简化。

**⑥ 推理时忘换 mask 尺寸 / KV cache 下 mask 错位**
→ 生成阶段逐 token 前向，query 长度 1、key 长度 t——mask 形状与训练时不同，硬套训练 mask 会越界或错屏蔽。第 16 章 KV cache 实现时详解，这里先立牌子。

---

## 7.5 开源项目的最佳实践

**① `F.scaled_dot_product_attention`（SDPA）：官方统一入口**
PyTorch 2.x 把注意力收编成一个函数，内部自动分派最优后端：FlashAttention-2（CUDA）、memory-efficient、或朴素数学实现（MPS/CPU 回退）。**生产代码永远用它而不是手写循环**——同样的数学，显存 O(T²)→O(T)、速度数倍（后端支持时）。`is_causal=True` 参数让它内部生成因果掩码，比显式传 mask 更快。本章手写是为了理解，第 8 章起我们的 Transformer 直接用 SDPA（但保留手写版做对拍）。

**② HF `LlamaAttention` 的结构**
读 [modeling_llama.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) 的注意力类：同一套 QKV 投影 + 拆头 + SDPA 调用，多出来的是 RoPE 施加在 q/k 上（第 8 章）和 GQA 的 KV 头复用（第 2 章 expand 伏笔 + 本章挑战题）。你会发现读它毫无障碍——这就是本章的目的。

**③ shape 注释纪律**
nanoGPT/LLaMA/vLLM 的注意力代码全都带 shape 注释（`# (B, nh, T, hs)`）。这不是给初学者的怜悯，是给未来 debug 的自己留的路标——易错点④那种静默错误，靠注释与对拍双保险防。

---

## 7.6 典型面试题

**Q1：为什么注意力要除以 √d_k？推导给我看。**

> **参考答案**：设 q、k 分量独立同分布（均值 0 方差 1），q·k = Σqᵢkᵢ 的方差是 d_k（独立项方差相加），标准差 √d_k。d_k 大时 logits 量级大，softmax 进入饱和区——输出接近 one-hot，雅可比 p(1-p) 趋零，梯度消失。除以 √d_k 把 logits 方差归一到 1。**加分点**：这个假设在训练后会漂移，有些工作（如 μP）据此调整；温度参数与它同型（T 越小越尖锐）；缩放用 d_head 不是 d_model。

**Q2：多头注意力为什么比单头好？head 数怎么选？**

> **参考答案**：单头每位置只能表达一种注意力分布，多头让不同子空间学习不同关系模式（句法/指代/位置等，可视化研究证实分工存在）。参数量与单头相同（切分而非复制），计算量也几乎相同——"免费"的表达力提升。head 数选择：d_head = d_model/H 通常保持 64~128（太小每头容量不足，太大头数少视角贫乏）；GPT-3 96 头×128、LLaMA-7B 32 头×128。**加分点**：头有冗余（可剪枝研究），推理时 KV 头是显存瓶颈 → MQA/GQA 减 KV 头数（第 10 章）。

**Q3：self-attention 和 cross-attention 的区别？decoder-only 的 LLM 为什么没有 cross-attention？**

> **参考答案**：self-attention 的 QKV 同源（序列自己看自己）；cross-attention 的 Q 来自一个序列、KV 来自另一个（翻译里解码器查编码器）。decoder-only（GPT 系）把"条件"和"生成"拼成一条序列，用因果 self-attention 统一处理——prompt 就是前缀，不需要独立编码器。**加分点**：多模态模型里 cross-attention 回归（视觉特征做 KV）；encoder-decoder（T5）与 decoder-only 的取舍是训练效率与架构简洁性的权衡，规模化时代简洁胜出。

**Q4：注意力矩阵的显存是多少？训练 4K 上下文的 7B 模型，朴素实现单层注意力矩阵要多大？怎么优化？**

> **参考答案**：(B,H,T,T)×每元素字节数。B=8/H=32/T=4096/fp16 ≈ 8.6GB 一层——32 层朴素实现根本不可行。优化：FlashAttention 分块计算，从不物化整个 T² 矩阵（在线 softmax + 反向重算），显存 O(T²)→O(T)，同时因为省了 HBM 读写反而更快。**加分点**：说清"不物化"的机制（分块遍历 KV，维护 running max/sum 增量归一化）；推理时的对应问题是 KV cache 的 O(T) 显存随并发放大（第 16/17 章）。

---

## 7.7 疑难杂症排查

**案例 1：训练指标异常地好，生成一塌糊涂**

第一嫌疑：因果泄漏（易错点②的现象学）。核查手段：**泄漏测试**——取一个 batch，把位置 t 之后的输入 token 全部替换成随机值，若位置 t 的 logits 变了，就是在偷看未来。这个 5 行的测试写进第 8 章的单元测试里（对 Transformer 全模型做）。其他泄漏源：数据里 x/y 错位错了（用同一位置预测自己）、packing 时 target 没右移。

**案例 2：注意力层输出 NaN**

按序排查：① 全 -inf 行（易错点③：打印 `mask.all(dim=-1)` 找全遮行）；② logits 上溢（手写 softmax 没减 max？fp16 下 QK 值域检查）；③ 上游污染（第 4 章 NaN 哨兵定位第一枚 NaN 的层——如果第一枚在注意力之前，别冤枉注意力）。

**案例 3：换了序列长度，速度/显存的变化远超预期**

这是 O(T²) 的现象学：T 从 1K→4K，注意力算力×16、注意力矩阵内存×16（其余部分只×4）。判断当前瓶颈在注意力还是 FFN：粗算 FLOPs 占比（注意力 ≈ 4T²d·H... 简化：T > d 时注意力主导）。此时的选项：SDPA/FlashAttention 后端确认真的启用了（`torch.backends.cuda.sdp_kernel` 上下文可查/强制）、滑动窗口注意力、或接受并去调 batch。**方法论**：性能问题先定位主导项，再谈优化。

---

## 7.8 练习题

### 基础 1：手算注意力
给定 2 个 token 的 Q/K/V（2×2 小矩阵，数值给定），手算 scaled dot-product attention 的输出（含 softmax），再用 torch 验证。体感一遍"加权检索"。

### 基础 2：padding mask
给 `attention_from_scratch.py` 的多头实现加上 padding mask 支持（`(B, T)` 的布尔张量 → 广播成 `(B,1,1,T)` 与 causal mask 合并）。构造一个含 padding 的 batch，验证：padding 位置的注意力权重为 0，且非 padding 位置的输出与"不含 padding 的等价 batch"一致。

### 进阶 1：cross-attention
把自注意力改成 cross-attention：`forward(x_q, x_kv)`，Q 来自 x_q，KV 来自 x_kv（长度可以不同）。验证 shape 正确性，并回答：cross-attention 需要因果掩码吗？什么情况需要？

### 挑战 1：实现 GQA（第 10 章预热）
实现分组查询注意力：H 个 query 头共享 G 组 KV 头（H % G == 0）。要求用第 2 章的 `expand`（零拷贝）而不是 `repeat_interleave` 做 KV 头复制，与"KV 头显式复制的朴素实现"对拍 allclose，并算一笔账：H=32, G=8, T=4096, d_head=128 时 KV cache 比 MHA 省多少？

---

## 本章小结与下一章预告

注意力 = 可导的软检索：QKV 三投影表达三种角色，√d_k 缩放守住 softmax 灵敏区，因果掩码保住自回归语义，多头提供子空间多视角，代价是 O(T²)。你已经手写并对拍了工业级正确的多头注意力。

**下一章（第 8 章）**：完整 Transformer。注意力只是积木之一——还需要 FFN（存储知识的地方）、残差流（梯度高速公路，第 3 章图分析的兑现）、归一化（Pre-Norm vs Post-Norm 的稳定性之争）、位置编码（RoPE 为什么赢）。拼完这些，第 9 章就能开训了。
