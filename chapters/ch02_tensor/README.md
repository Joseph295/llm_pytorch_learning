# 第 2 章 · 张量的本质：storage、stride 与那本内存账

> **本章目标**：打开张量的"物理层"。学完你应该能回答：
> 1. `transpose` 为什么零拷贝？`contiguous()` 到底在做什么？
> 2. `view` 和 `reshape` 的区别是什么？什么时候 `view` 会报错？
> 3. 广播"不复制数据"是怎么实现的？
> 4. 一个 7B 模型，推理要多少内存？训练要多少？——这笔账算明白，第 13 章 ZeRO/FSDP 的动机就自动成立。

**前置**：第 1 章（引用语义、广播规则）。 **硬件路径**：本地。 **预计用时**：4~5 小时。

---

## 2.1 来龙去脉：一维内存如何伪装成 N 维数组

物理内存是一维的字节序列，而神经网络要的是 N 维数组。所有张量库都面临同一道设计题：**如何在一维内存上表达 N 维语义，并且让"变形"操作尽可能不搬数据？**

PyTorch 的答案（继承自 NumPy）是把张量拆成两层：

```
Tensor = Storage（一维物理内存） + 元数据（shape / stride / offset / dtype / device）
```

这个"物理与逻辑分离"的设计你应该觉得眼熟——**Parquet 的列式存储 + 逻辑 schema**、**数据库的堆表 + 索引视图**都是同一个思想：数据本体不动，靠元数据定义不同的读取方式。推论也一样：改元数据是 O(1)（建视图），改数据本体是 O(n)（物化拷贝）。PyTorch 大量操作快，就快在"只改元数据"。

**为什么必须懂这一层？** 三个理由：
1. **性能**：不懂 view/copy 的边界，就会写出满地隐式拷贝的代码（在 24GB 的机器上训练，一次多余拷贝可能就是 OOM）；
2. **正确性**：视图共享底层数据，一处修改多处可见——不懂这个，bug 会以"数据被神秘篡改"的形式出现；
3. **LLM 的内存账**：显存规划是大模型工程的第一课，而算账的基础是"张量到底占多少字节、什么时候产生副本"。

---

## 2.2 核心原理

### ① stride：从 N 维坐标到一维偏移的映射公式

stride（步长）是每个维度上"坐标 +1 时，在一维内存里跳几个**元素**"。定位公式：

```
内存位置(元素) = offset + Σ (索引ᵢ × strideᵢ)
```

一个 `(2, 3)` 的行优先张量，`stride = (3, 1)`：行号 +1 跳 3 个元素，列号 +1 跳 1 个。

```
逻辑视图              物理内存（storage）
[[a, b, c],    ←→    [a, b, c, d, e, f]
 [d, e, f]]           0  1  2  3  4  5
元素[1,2]=f 的位置 = 0 + 1×3 + 2×1 = 5 ✓
```

**行优先（row-major）连续**的定义：`stride[i] = shape[i+1] × stride[i+1]`，最后一维 stride=1。满足这个条件的张量就是 **contiguous**。

### ② 零拷贝变形的全部秘密：只改元数据

理解了 stride，一批"看似神奇"的操作全部祛魅：

**`transpose` / `permute`：交换 stride**。`(2,3)`、stride `(3,1)` 转置后 shape `(3,2)`、stride `(1,3)`——数据一个字节没动，只是"读取方向"变了。代价：转置结果**不再 contiguous**（stride 不满足行优先递推）。

**切片：改 offset 和 shape**。`t[1:, :]` 就是 offset 前移一行、shape 减一行。所以切片是视图——第 1 章 `WindowedDataset` 里"切片零拷贝"的原理在此。

**`expand`（广播的实现）：stride 置 0**。把 `(3,)` expand 成 `(1000, 3)`，新增维度的 stride 设为 0——第 0 维坐标怎么加，内存位置都不动，1000 行"逻辑复制"共享同 3 个元素。第 1 章看到的 `big.stride() = (0, 1)` 就是它。**广播运算的底层就是先 expand 再逐元素运算**，这就是"广播不复制数据"的答案。

**`view` / `reshape`：重算 stride，但有前提**。`view` 要求新 shape 能在**现有内存布局**上用一套合法 stride 表达。contiguous 张量总是可以；转置过的张量往往不行——此时 `view` 报错，`reshape` 则退而求其次**静默拷贝**。这就是两者的全部区别：

> `reshape` ≈ `view` 尽量视图，不行就 `contiguous().view()`（物化拷贝）。
> 想要"确定零拷贝、不行就大声报错"用 `view`；想要"一定成功、代价我认"用 `reshape`。

**`contiguous()`：按当前逻辑顺序物化一份行优先的新内存**。已经 contiguous 时是 no-op（返回自身）。何时必须调它：某些 kernel 要求输入连续（如 `view` 前、部分自定义算子前）；转置后要做大量逐元素读取时，物化一次反而更快（内存局部性——你在列式存储里对行式访问做物化的直觉完全适用）。

### ③ view 家族 vs copy 家族：一张必背的表

| 零拷贝（视图，共享 storage） | 产生新内存（副本） |
|---|---|
| `view` `transpose` `permute` `expand` | `reshape`（仅在无法视图时）`repeat` |
| 切片 `t[a:b]` `narrow` `select` `squeeze` `unsqueeze` | `clone` `contiguous`（非连续时）`.to(dtype/device)`（跨类型/设备时） |
| `detach`（共享数据，切断梯度） | 花式索引 `t[mask]` `t[[1,3]]` `index_select` |

两个高频陷阱藏在这张表里：
- **`expand` vs `repeat`**：语义都是"复制扩展"，前者零拷贝（stride=0 的视图，**只读场景专用**），后者真复制。LLaMA 的 GQA 实现选 expand 就是为了省显存（2.5 节）。
- **花式索引是拷贝，切片是视图**——`t[2:4]` 共享内存，`t[[2,3]]` 是新张量。改哪个会影响原张量？答错的人写出静默 bug。

### ④ dtype 与那本内存账

每元素字节数：fp32=4，fp16/bf16=2，int8/fp8=1，int64=8。两个立即有用的推论：

**推论一：整数张量的默认类型是 int64**——`torch.tensor([1,2,3])` 每个元素 8 字节。token id 用 int64 存无妨（数量少），但大规模索引结构要显式用 int32/int16。

**推论二（重要）：LLM 的内存账**。以 7B（70 亿参数）模型为例：

```
推理（half 精度）:  7B × 2 字节                    = 14 GB
                  + KV cache（随序列长度增长，第 16 章算细账）

训练（AdamW, 混合精度典型配置）:
  参数     fp16   7B × 2  = 14 GB
  梯度     fp16   7B × 2  = 14 GB
  Adam m   fp32   7B × 4  = 28 GB
  Adam v   fp32   7B × 4  = 28 GB
  参数主副本 fp32  7B × 4  = 28 GB    ← 混合精度需要 fp32 主参数（第 6 章）
  ──────────────────────────────
  仅"模型状态"       ≈ 16 字节/参数 = 112 GB   （还没算激活值！）
```

**一张 80GB 的 A100 装不下 7B 模型的训练状态**——这个结论请反复咀嚼。它是第 13 章全部内容（ZeRO 把这 16 字节/参数切分到 N 张卡）的存在理由。你现在就能理解 ZeRO 论文标题里 "Memory Optimization" 的分量。

### ⑤ 设备与同步：`.item()` 的隐藏代价

`.to(device)` 跨设备是拷贝 + 可能的同步点。更隐蔽的是 `.item()`、`.cpu()`、`print(tensor)`、`if tensor > 0`——它们要求**拿到数值**，会强制等待 GPU 算完（第 0 章讲的异步执行在此收口）。训练循环里每步 `loss.item()` 记日志没问题，但在热路径里频繁取值会把异步流水打成同步挤牙膏——第 11 章 profiler 会让你亲眼看到这种"同步气泡"。

---

## 2.3 动手实验

```bash
uv run chapters/ch02_tensor/code/tensor_internals.py   # storage/stride/视图共享的实证
uv run chapters/ch02_tensor/code/view_vs_copy.py       # view 家族边界 + 报错复现
uv run chapters/ch02_tensor/code/memory_math.py        # 内存账：理论计算 + 实测验证
```

第一个脚本用 `data_ptr()`（张量数据的内存地址）做"指针取证"，亲手验证视图共享与拷贝独立；第三个脚本把 ④ 的 LLM 内存账写成可复用的计算器，并用一个 0.1B 模型在 M4 上实测对账。

---

## 2.4 易错点清单

**① 转置后 `view` 报错**
```python
t = torch.randn(2, 3).transpose(0, 1)   # 非 contiguous
t.view(6)                               # ✗ RuntimeError
```
→ **现象**：`view size is not compatible with input tensor's size and stride`。
→ **原因**：转置后的内存布局无法用一套 stride 表达目标 shape。
→ **修正**：`t.contiguous().view(6)` 或 `t.reshape(6)`（知道自己在付拷贝代价）。

**② 在 expand 出来的视图上原地写入**
```python
bias = torch.zeros(3).expand(100, 3)
bias += 1        # ✗ 灾难：100 行共享 3 个元素，一次 += 叠加 100 遍
```
→ **现象**：新版本 torch 对明显情况直接报错；但绕过检查的路径会得到脏数据。
→ **原因**：stride=0 意味着"100 行"是同 3 个元素的镜像，原地写入互相踩踏。
→ **修正**：expand 的结果**只读**。要写就 `repeat`（真复制）或 `expand(...).clone()`。

**③ 视图篡改原张量（切片赋值穿透）**
```python
first_row = data[0]      # 视图！
first_row.fill_(0)       # data[0] 也被清零了
```
→ **修正**：需要独立副本时 `data[0].clone()`。反过来这也是特性：`param.data[:10] = 0` 这种"定向手术"正是靠视图穿透实现的。**判断口诀**：带下划线后缀的方法（`fill_`/`add_`/`mul_`）都是原地操作，作用在视图上必穿透。

**④ NaN 的比较陷阱**
```python
torch.tensor(float("nan")) == float("nan")   # False！NaN 不等于任何东西包括自己
```
→ **修正**：判断 NaN 用 `torch.isnan()`。第 15 章排查 loss NaN 时，`(x != x)` 是老手判断 NaN 的惯用法（利用 NaN≠自己）。

**⑤ 累积 `.item()`/CPU 取值打断异步流水**
```python
for step in range(1000):
    loss = train_step()
    total += loss.item()      # 每步强制同步一次
```
→ **现象**：GPU 利用率上不去，profiler 里满是空隙。
→ **修正**：日志类取值降频（每 50 步一次）；或先 `total += loss.detach()`（留在 GPU 上累加），最后一次性 `.item()`。

**⑥ dtype 静默提升吃掉一半显存**
```python
x = torch.randn(1024, 1024, dtype=torch.float16)
y = x * 0.1                        # Python 标量 → 仍 fp16 ✓
z = x * torch.tensor(0.1)          # fp32 零维张量 → 仍 fp16 ✓（零维按标量对待！）
w = x * torch.tensor([0.1])        # fp32 一维张量 → 提升为 fp32！显存×2
```
→ **原因**：类型提升有优先级三档：带维度张量 > 零维张量 > Python 标量。低档不改变高档的 dtype；同档才比 dtype 高低。所以 fp32 **带维度**张量会把 fp16 拉升成 fp32，而标量和零维张量不会。
→ **修正**：混合精度代码里对中间结果 `assert x.dtype == ...` 或打印抽查；从配置/数据里读进来的系数张量（往往 fp32）参与运算前显式 `.to(x.dtype)`。
→ **注**：本条的零维例外是实跑发现后修正的——教程初稿也写错了，可见这规则多容易记岔。

---

## 2.5 开源项目的最佳实践

**① safetensors：为什么新一代权重格式按 storage 序列化**
HuggingFace 的 [safetensors](https://github.com/huggingface/safetensors) 取代 pickle 格式的 `.bin`，文件结构 = JSON 头（每个张量的 shape/dtype/字节偏移）+ 原始字节。两个直接受益都源于本章知识：**零拷贝加载**——`mmap` 文件后按偏移直接构造张量视图，不经过反序列化（大模型加载从分钟级到秒级）；**安全**——pickle 能执行任意代码，纯字节 + 元数据不能。你在第 10 章加载 LLaMA 权重时会直接受益。

**② LLaMA 的 GQA：`expand` 省显存的教科书应用**
分组查询注意力（GQA）中 8 个 query 头共享 1 个 KV 头，计算时要把 KV "复制" 8 份对齐。看 transformers 的 [`repeat_kv`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)：用 `expand` + `reshape` 而不是 `repeat_interleave`——KV cache 是推理显存大头，expand 的零拷贝在这里是真金白银（第 16 章算 KV cache 的账时回收这个伏笔）。

**③ 训练框架的 `.detach()` 惯例**
到处可见的 `loss.detach()`、`hidden.detach()`：共享数据但切断梯度图（视图家族的特殊成员）。日志、指标累加、EMA 更新全都必须 detach，否则计算图越挂越长、内存泄漏（第 3 章讲计算图后你会彻底理解为什么）。

---

## 2.6 典型面试题

**Q1：`view` 和 `reshape` 的区别？什么时候 `view` 会失败？**

> **参考答案**：`view` 只做元数据变换，要求新 shape 与现有 stride 布局兼容，不兼容直接报错、绝不拷贝；`reshape` 先尝试 view，不行就隐式 `contiguous()` 拷贝，永远成功。典型失败场景：转置/permute 后的非 contiguous 张量。**加分点**：说出工程取向——性能敏感路径用 `view`（把隐式拷贝暴露成错误），通用工具代码用 `reshape`；以及 `view` 兼容性的本质是"新 shape 每一维能否映射为原 storage 上的等距步进"。

**Q2：`tensor.contiguous()` 做了什么？什么时候需要？调用它的代价？**

> **参考答案**：若张量已 contiguous（stride 满足行优先递推式）则 no-op 返回自身；否则分配新内存，按逻辑顺序拷贝，返回行优先的新张量。需要的场景：`view` 之前、要求连续输入的算子（部分 CUDA kernel/外部库）、以及对非连续张量做密集逐元素访问前的性能优化。代价：一次 O(n) 拷贝 + 峰值双份内存。**加分点**：channels-last 等特殊 memory format 下"contiguous"有不同含义（`is_contiguous(memory_format=...)`）。

**Q3：训练一个 7B 模型（AdamW + 混合精度），模型状态要多少显存？逐项拆。**

> **参考答案**：16 字节/参数 ≈ 112GB。拆账：fp16 参数 2 + fp16 梯度 2 + fp32 Adam 一阶矩 4 + 二阶矩 4 + fp32 主参数 4。另有激活值（与 batch/序列长相关，可用激活重算压缩，第 11 章）。**加分点**：主动引申——单卡放不下所以要 ZeRO/FSDP 分片（stage1 切优化器状态省 8 字节/参数，stage2 再切梯度，stage3 连参数一起切）；纯 fp32 训练是 16 字节/参数（4+4+4+4），混合精度不省模型状态、省的是激活和算力。

**Q4：`a = b[0]` 之后修改 `a`，`b` 会变吗？`a = b[[0]]` 呢？为什么设计成这样？**

> **参考答案**：`b[0]`（基本索引/切片）返回视图，原地修改穿透到 `b`；`b[[0]]`（花式索引）返回拷贝，互不影响。设计逻辑：基本索引的结果总能用 (offset, shape, stride) 表达（O(1) 视图），花式索引的元素集合在内存中不规则，无法用等距 stride 描述，只能物化。**加分点**：花式索引的**赋值**形态 `b[[0,2]] = x` 是原地写入原张量（不产生拷贝），读和写行为不对称。

---

## 2.7 疑难杂症排查

**案例 1：`RuntimeError: view size is not compatible ...`**

直接病因 90% 是上游有 `transpose`/`permute`。排查：报错行打印 `t.is_contiguous()` 和 `t.stride()`；沿调用链上溯找转置点。修复选项：① 改 `reshape`（接受可能的拷贝）；② 上游转置后就地 `.contiguous()`（如果后面多处使用，物化一次摊销）；③ 重新设计维度顺序避免转置（最优但改动大）。注意力实现里 `(B,T,H,D) ↔ (B,H,T,D)` 的反复转置是这个错误的高发区，第 7 章手写注意力时你会亲手处理。

**案例 2：显存/内存缓慢上涨最终 OOM，代码里"没创建新张量"**

嫌疑清单（按命中率排序）：① 日志累加没 `detach`——`total_loss += loss` 把整张计算图挂在了累加器上（第 3 章解释机制）；② 列表里存了带梯度的张量（`history.append(output)`）；③ 花式索引/`cat` 在循环里反复物化。排查工具：`gc` 模块数张量个数；MPS 上 `torch.mps.current_allocated_memory()` 打点（CUDA 对应 `torch.cuda.memory_allocated()`），二分定位增长点。

**案例 3：同一份代码，换了输入形状后性能掉一个数量级**

场景：`(B, T, D)` 换成 `(T, B, D)` 布局后某些算子明显变慢。
原因：算子对最后一维连续（stride=1）有快路径；转置后失去连续性，走 gather 式慢路径，缓存命中率骤降——与"列存表做行扫描"慢是同一物理原因（内存局部性）。
排查：`t.stride()` 看热点算子输入是否 stride=1 结尾；`contiguous()` 物化后对比耗时。**方法论**：形状/布局变更后性能异常，先查 stride 再查算法。

---

## 2.8 练习题

### 基础 1：手算 stride
不运行代码，写出下列每步之后的 shape 和 stride（起点：`t = torch.arange(24).reshape(2, 3, 4)`，stride `(12, 4, 1)`）：
a) `t.transpose(0, 2)`  b) `t[1]`  c) `t[:, 1:, :]`  d) `t.permute(2, 0, 1)`  e) `t.unsqueeze(1)`
再用代码验证，并指出哪几个结果不再 contiguous。

### 基础 2：指针取证
用 `data_ptr()` 和 `untyped_storage().data_ptr()` 验证：切片视图与原张量共享 storage 但 `data_ptr` 不同（offset 的体现）；`clone` 后两者都不同；`t[[1]]` 花式索引后 storage 不同。写 10 行代码输出三组对比。

### 进阶 1：显存计算器
写一个函数 `estimate_memory(n_params, dtype_bytes=2, training=True, optimizer="adamw")`，返回模型状态内存明细（参数/梯度/优化器状态/fp32 主参数）。用它算：0.5B、7B、70B、671B（DeepSeek-V3 规模）四档的推理与训练内存，输出成表。训练 70B 至少要几张 80GB 卡（只算模型状态）？

### 挑战 1：用 `as_strided` 手写滑动窗口
`torch.as_strided` 允许直接指定 (shape, stride) 构造视图。用它把一维张量 `[0..9]` 变成 shape `(8, 3)` 的滑动窗口矩阵（第 i 行 = `[i, i+1, i+2]`），**零拷贝**。验证零拷贝（改原张量一个元素，看窗口矩阵几处变化？为什么是这个数？）。再思考：第 1 章的 `WindowedDataset` 能否用这个技巧把逐样本切片优化成一次性视图？代价是什么？

---

## 本章小结与下一章预告

张量 = 一维 storage + (shape, stride, offset) 元数据。视图操作改元数据（O(1)），物化操作搬数据（O(n)）；广播靠 stride=0、转置靠交换 stride、`contiguous` 按逻辑序重排物理内存。LLM 内存账：推理 2 字节/参数起步，AdamW 混合精度训练 16 字节/参数——单卡装不下，是一切分布式训练技术的第一因。

**下一章（第 3 章）**：Autograd。张量还有一个我们刻意没讲的元数据字段：`grad_fn`。它把张量连成一张有向无环图——PyTorch 动态图机制的全部秘密。我们会手推反向传播、剖析计算图的构建与销毁，并回答本章遗留的问题："为什么 `total += loss` 不 detach 会内存泄漏"。
