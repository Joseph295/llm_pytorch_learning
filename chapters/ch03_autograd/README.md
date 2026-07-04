# 第 3 章 · Autograd 解剖：动态计算图与反向传播

> **本章目标**：拆开 PyTorch 的心脏。学完你应该能回答：
> 1. `loss.backward()` 一行代码背后发生了什么？图是什么时候建的、什么时候销毁的？
> 2. 为什么 `.grad` 默认**累加**而不是覆盖？这个设计换来了什么？
> 3. 训练时"激活值"为什么占那么多显存？和 autograd 什么关系？
> 4. 第 2 章遗留问题：`total += loss` 不 detach 为什么内存泄漏？

**前置**：第 2 章。 **硬件路径**：本地。 **预计用时**：5~6 小时（挑战题值得花一晚上）。

---

## 3.1 来龙去脉：求导的三条路，为什么赢家是"反向模式自动微分"

神经网络训练 = 沿梯度下降调参数。问题：几十亿参数的复合函数，导数怎么求？历史上三条路：

**路一：数值微分**。`f'(x) ≈ (f(x+ε) - f(x)) / ε`。每个参数要跑一次前向——70 亿参数跑 70 亿次前向，宣告死刑。但它简单可靠，至今仍是**验证**其它方法正确性的金标准（`torch.autograd.gradcheck` 就是它，本章实验用到）。

**路二：符号微分**。像 Mathematica 那样对表达式求导出公式。问题是表达式膨胀（expression swell）：复合函数的导数公式指数爆炸，且要求计算过程是封闭数学表达式——带分支循环的程序没法弄。

**路三：自动微分（AD）**。关键洞察：任何程序都是**基本运算的组合**，每个基本运算的导数是已知的，链式法则把它们串起来。不展开公式、不近似，按运算轨迹精确计算。AD 又分两个方向：

- **前向模式**：沿计算方向同步传播导数。一次遍历得到"输出对**一个输入**"的导数。成本 ∝ 输入数。
- **反向模式**：先正着算一遍并记录轨迹，再倒着传播。一次遍历得到"**一个输出**对所有输入"的导数。成本 ∝ 输出数。

深度学习的形态是：**上亿输入（参数）、一个输出（标量 loss）**。反向模式一次反向就拿到全部参数的梯度，前向模式要上亿次——胜负毫无悬念。反向传播（backpropagation）就是反向模式 AD 在神经网络上的应用。代价也在这里埋下：**"先正着算一遍并记录轨迹"意味着中间结果要留到反向用**——这就是激活值内存的来源（3.2-⑤）。

**动态图 vs 静态图**：记录轨迹有两种时机。TensorFlow 1.x 先声明完整图再执行（define-then-run，像提交 DAG 作业）；PyTorch 边执行边记录（define-by-run）——每次前向都现场重建图，Python 的 if/for 天然可用，调试时断点处处可停。研究界用脚投票选了后者。代价是图的优化空间小（看不到全局），这笔账第 11 章 `torch.compile` 会找回来。

---

## 3.2 核心原理

### ① 计算图长什么样：grad_fn 串起来的反向链

```python
x = torch.tensor([2.0], requires_grad=True)   # 叶子节点
y = x * 3                                     # y.grad_fn = <MulBackward0>
z = y ** 2                                    # z.grad_fn = <PowBackward0>
```

每个由"需要梯度的张量"参与运算产出的新张量，都带一个 `grad_fn`——该运算的**反向函数对象**。`grad_fn.next_functions` 指向上游运算的反向函数，一路链到叶子。所以"计算图"在 PyTorch 里的实体是**一张由 grad_fn 连成的反向有向图**，前向执行的副产品。

三类节点要分清：
- **叶子（leaf）**：用户直接创建的 `requires_grad=True` 张量（模型参数都是）。`x.is_leaf == True`，梯度最终落在 `x.grad`。
- **中间节点**：运算产出的张量。参与反向传播，但**默认不保留 `.grad`**（用完即弃，省内存）——想看要先 `t.retain_grad()`（易错点①）。
- **不追踪**：`requires_grad=False` 的张量（输入数据通常如此），是图的常数。

### ② backward()：一次拓扑序的反向遍历

`loss.backward()` 做的事：从 `loss.grad_fn` 出发，按**拓扑逆序**遍历反向图；每个节点用上游传来的梯度（grad_output）计算对各输入的梯度（链式法则的一步），传给下一层；到达叶子时**累加**进 `.grad`。

两个关键设计：

**梯度累加而非覆盖**。一个张量被多处使用时（比如残差连接里 x 既走主路又走捷径），多条反向路径的梯度要**求和**——这是多元链式法则的数学要求。PyTorch 把"求和"直接实现为"往 `.grad` 里累加"，顺便免费送了一个特性：多次 backward 自动累积梯度 → **梯度累积**训练技术（小显存模拟大 batch，第 6 章正式使用）的机制基础。代价：每步训练前必须 `optimizer.zero_grad()`，忘了就是易错点里的经典事故。

**图默认用完即毁**。backward 结束后，为反向保存的中间量（saved tensors）立即释放（这是显存的大头）。所以对同一张图二次 backward 会报错——除非 `backward(retain_graph=True)`（绝大多数"需要 retain_graph"的场合其实是代码写错了，见 3.7 案例 2）。

**非标量 backward**。`backward()` 无参数只对标量合法（loss 是标量所以日常无感）。对向量 y 反向要传"权重"：`y.backward(gradient=v)` 计算的是 `d(y·v)/dx`——本质是 vector-Jacobian product（VJP）。反向模式 AD 的原子操作就是 VJP，这个视角在读进阶资料时会反复出现。

### ③ 三种"关掉 autograd"的方式，语义各不同

| 工具 | 语义 | 典型场景 |
|---|---|---|
| `with torch.no_grad():` | 块内运算不建图 | 推理、手动更新参数 |
| `tensor.detach()` | 返回共享数据但**脱离图**的新张量 | 日志累加、把中间量当常数用 |
| `with torch.inference_mode():` | no_grad 加强版：产出的张量永远不能再进图（省掉版本计数等开销，更快） | 纯推理服务（第 17 章 mini-vLLM 用它） |

**`.detach()` vs `.data`**：老代码常见 `t.data`，效果类似 detach 但**绕过版本计数**（③ 的安全机制）——修改 `.data` 后 autograd 无法发现数据已过期，可能算出**静默错误的梯度**。铁律：新代码永远用 `detach()`，看到 `.data` 提高警惕（它没死是因为优化器内部等少数场景确实需要，那是"知道自己在干什么"的场合）。

### ④ in-place 操作与版本计数器

每个张量有个 `_version` 计数，原地操作（`add_`、`relu_`、切片赋值）让它 +1。反向函数如果依赖某个被保存的张量，backward 时会核对版本——对不上就抛 `RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation`。

为什么这么设计：`y = x.exp()` 的反向公式是 `grad * y`（复用前向结果），如果你事后 `y += 1`，保存的 y 已经不是当年的 y，梯度必错。版本计数器把"静默算错"变成"大声报错"——这是 PyTorch 设计品味的体现（对比 2.4-⑥ dtype 提升的静默，可以体会何时该吵何时该忍）。排查这个报错的完整流程见 3.7 案例 1。

### ⑤ 内存视角：激活值 = 为反向而扣押的中间结果

反向公式几乎都要用到前向的中间量（`exp` 要 y、`matmul` 要两个输入、`relu` 要掩码……）。autograd 把它们存在 `grad_fn` 的 `saved_tensors` 里，**从前向算出到反向用完，一直占着显存**。这就是"激活值内存"，它随 batch × 序列长 × 层数增长，训练大模型时常常**超过模型状态本身**（第 2 章 16 字节/参数之外的另一头大象）。

现在可以回答第 2 章的遗留问题了，并且要比坊间说法更精确（本章实验实测出的区分）：

```python
# 场景 A：训练循环（loss 已 backward）
loss.backward()
total += loss             # 坏习惯但非灾难：backward 已释放 saved_tensors，
                          # 残留的只是轻量图结构对象

# 场景 B：评估/指标循环（loss 从不 backward）—— 重灾区
loss = compute_val_loss(...)
total += loss             # ✗✗ 每步整张图连同全部激活值被 total 引住，永不释放
```

`total` 带着 grad_fn → 引用本步反向图 → 图引用所有 saved tensors。训练场景里 backward 是"泄压阀"；评估场景没有泄压阀，循环一千步就是一千份激活值滞留——**症状是评估阶段内存线性上涨，病根是一个没 detach 的累加**。正确姿势按优先级：评估整体套 `torch.no_grad()`（图根本不建，还更快）> 累加处 `detach()`。

省激活内存的正规军是**梯度检查点（gradient checkpointing）**：前向时不保存中间激活，反向时**重算**一遍换内存——用 33% 左右的额外计算换掉大头激活（第 11 章实测，第 14 章微调 7B 时是保命选项）。它的实现基础就是下面的自定义 Function。

### ⑥ 自定义 autograd.Function：亲手给一个运算定义导数

```python
class MyGELU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)          # 反向要用的自己存
        return 0.5 * x * (1 + torch.erf(x / 1.41421356))

    @staticmethod
    def backward(ctx, grad_out):          # 输入：上游梯度；输出：对每个 forward 输入的梯度
        (x,) = ctx.saved_tensors
        pdf = torch.exp(-0.5 * x * x) / 2.5066282746
        return grad_out * (0.5 * (1 + torch.erf(x / 1.41421356)) + x * pdf)
```

什么时候需要它：① 给不可导/非 torch 运算定义梯度（量化的直通估计器 STE，第 16 章 QAT 的核心）；② 融合多个运算省内存/提速（第 19 章手写算子的接口就是它）；③ 梯度检查点这类"篡改保存策略"的技术。写完必须用 `torch.autograd.gradcheck`（数值微分对拍）验证——本章实验演示全流程。

---

## 3.3 动手实验

```bash
uv run chapters/ch03_autograd/code/graph_anatomy.py      # 打印真实计算图 + 叶子/中间节点行为
uv run chapters/ch03_autograd/code/grad_accumulation.py  # 累加语义 + micro-batch 数学等价性验证
uv run chapters/ch03_autograd/code/custom_function.py    # 自定义 Function + gradcheck 全流程
uv run chapters/ch03_autograd/code/memory_leak_demo.py   # 亲眼看 total += loss 的内存泄漏曲线
```

第 4 个脚本用 MPS 内存计数把泄漏画成数字：同一个循环，带 detach 内存水平线，不带 detach 线性爬升——这个实验做过一次，终身免疫此坑。

---

## 3.4 易错点清单

**① 中间节点的 `.grad` 是 None**
```python
y = x * 3; z = y.sum(); z.backward()
print(y.grad)     # None！还伴随一条 UserWarning
```
→ **原因**：中间节点默认不保留梯度（省内存的刻意设计）。
→ **修正**：backward 前 `y.retain_grad()`；或用 hook：`y.register_hook(print)`。

**② 第二次 backward 报错**
```python
loss.backward(); loss.backward()   # ✗ Trying to backward through the graph a second time
```
→ **原因**：图在第一次 backward 后已销毁（saved tensors 释放）。
→ **修正**：真需要二次反向（如高阶导、GAN 的某些写法）用 `retain_graph=True`；但先自问是不是把"每步应重新前向"写成了"复用旧输出"（3.7 案例 2）。

**③ 原地操作打断反向**
```python
y = x.exp(); y += 1; y.sum().backward()   # ✗ version 不匹配
```
→ **修正**：热路径外优先非原地写法（`y = y + 1`）；确要原地，确认该张量不在任何反向公式的依赖里。ReLU 这类反向只依赖输出符号的运算有原地版（`relu_`）且安全——框架知道哪些能原地，你手写时要自己证明。

**④ 手动更新参数创建了新叶子**
```python
w = w - lr * w.grad     # ✗ 新张量 w 是中间节点！旧叶子失联，下步 .grad=None
```
→ **原因**：`=` 是重新绑定（第 1 章引用语义），新 w 由运算产出、不是叶子。
→ **修正**：`with torch.no_grad(): w -= lr * w.grad`（原地改叶子本体，且不建图）。第 6 章手写优化器时这是核心考点——`optimizer.step()` 内部正是这么做的。

**⑤ `total += loss` 内存泄漏**（3.2-⑤ 已剖析）
→ **口诀**：任何"跨迭代存活"的张量（累加器、日志列表、EMA、best_output）一律 `detach()`（必要时加 `.cpu()`）。

**⑥ 对 requires_grad 的张量做 `torch.tensor(x)` / `x.numpy()`**
```python
np_x = x.numpy()          # ✗ RuntimeError: Can't call numpy() on Tensor that requires grad
```
→ **原因**：NumPy 不认识计算图，这条路必须显式断开——torch 拒绝静默断图。
→ **修正**：`x.detach().numpy()`（明确宣告"我知道从这往后没有梯度"）。

---

## 3.5 开源项目的最佳实践

**① `derivatives.yaml`：PyTorch 所有算子的导数登记簿**
PyTorch 内置算子的反向公式不是散落在 C++ 里，而是集中登记在 [tools/autograd/derivatives.yaml](https://github.com/pytorch/pytorch/blob/main/tools/autograd/derivatives.yaml)，构建时代码生成。想知道任何算子的精确梯度公式（比如 `softmax` 的），直接查这个文件——比翻论文快且权威。**读源码技巧**：搜索 `- name: softmax` 一类的条目。

**② `torch.utils.checkpoint`：自定义 Function 的最重要应用**
[torch/utils/checkpoint.py](https://github.com/pytorch/pytorch/blob/main/torch/utils/checkpoint.py) 用本章 3.2-⑥ 的机制实现梯度检查点：forward 里 `no_grad` 跑真前向（不存激活），把**输入**存进 ctx；backward 里重新以 `enable_grad` 跑一遍前向补出激活，再正常反向。transformers 里每个模型的 `gradient_checkpointing_enable()` 都落到它。读懂这 100 行 = 同时吃透 no_grad/自定义 Function/激活内存三个概念。

**③ HF Trainer 的日志纪律**
transformers 的 Trainer 上报 loss 用的是 `loss.detach()` 后的标量聚合（跨步再 `.item()` 降频同步），三个易错点（泄漏、同步、累加语义）一次规避。写自己的训练循环时照抄这个纪律（第 6 章的循环模板就这么写）。

---

## 3.6 典型面试题

**Q1：反向模式和前向模式自动微分的区别？为什么深度学习用反向模式？什么场景前向模式反而合适？**

> **参考答案**：反向模式一次遍历算"单输出对全部输入"的梯度（成本 ∝ 输出数），前向模式一次算"全部输出对单输入"（成本 ∝ 输入数）。DL 是亿级参数 → 标量 loss，反向模式一次搞定。前向模式适合输入少输出多的场景（如对单个超参的敏感度分析）、以及不想保存激活的在线场景（`torch.func.jvp` 提供支持）。**加分点**：反向模式的空间代价是保存前向中间量（激活内存），前向模式几乎无额外内存——这是二者更本质的工程差异。

**Q2：为什么 PyTorch 的梯度是累加的？说出这个设计的一个数学原因和一个工程收益。**

> **参考答案**：数学原因——同一张量出现在多条计算路径（残差、共享 embedding 等权重共享）时，总梯度是各路径梯度之**和**，实现上"逐路径累加进 .grad"最自然。工程收益——梯度累积免费获得：把大 batch 拆成 K 个 micro-batch 依次 forward+backward 不清零，`.grad` 里自动是 K 步之和（数学上等价于大 batch，注意 loss 要除 K）。代价是必须显式 `zero_grad()`。**加分点**：`zero_grad(set_to_none=True)`（现默认）比置零更优：省一次内存写 + 让"忘了 backward 就 step"从静默变报错。

**Q3：训练显存中"激活值"指什么？为什么推理时没有这块开销？gradient checkpointing 怎么用计算换显存，代价多大？**

> **参考答案**：激活值 = 反向公式依赖的前向中间结果（autograd 存进 saved_tensors 的那些），量级 ∝ batch × 序列长 × 隐层宽 × 层数。推理不建图不保存，所以没有。Checkpointing：前向对被包裹段落不存激活只存输入，反向时重算该段前向补激活；额外代价约一次前向 ≈ 总计算 +33%（前向:反向 ≈ 1:2），换来激活内存从 O(层数) 降到 O(√层数) 或分段常数。**加分点**：说出"重算段落边界的选择"是权衡点，以及 FlashAttention 本质上也是"重算换显存"哲学在注意力内部的应用（第 11 章）。

**Q4：这段代码有什么 bug？`for x, y in loader: loss = model(x, y); loss.backward(); optimizer.step()`**

> **参考答案**：缺 `optimizer.zero_grad()`——梯度跨步累加，等效学习率越来越大，训练很快发散。位置放 step 之后或 backward 之前皆可。**加分点**：指出如果是故意做梯度累积，则 loss 要除以累积步数，且 step/zero_grad 每 K 步执行一次；再补一刀：真实训练还缺 `model.train()` 与梯度裁剪（第 6 章模板）。

---

## 3.7 疑难杂症排查

**案例 1：`RuntimeError: ... modified by an inplace operation`**

报错信息其实给了坐标：`[torch.FloatTensor [64, 128]], which is output 0 of ReluBackward0, is at version 2; expected version 1`。
排查流程：① 从 shape `[64,128]` 和 `ReluBackward0` 定位是哪个 relu 的输出；② 找它后面谁改了它——高发嫌疑：`+=`、`relu_`/`clamp_`、切片赋值 `x[mask] = 0`、优化器提前 step；③ 开 `torch.autograd.set_detect_anomaly(True)` 重跑，报错会带上**前向的调用栈**直接指认现场（很慢，只在排查时开）。
修复：把原地改非原地；或调整顺序让修改发生在 backward 之后。

**案例 2：`Trying to backward through the graph a second time`，但我只写了一次 backward**

真实病根常常不是"写了两次"，而是**图的一部分被跨迭代复用**了：循环外算好的某个中间量（如预计算的 embedding、缓存的 hidden）参与每步的 loss——第 1 步 backward 销毁了它的图，第 2 步再反向就撞墙。
排查：找 loss 的依赖里哪些张量不是本步新算的（`grad_fn` 链上溯）；把它们要么移进循环重算，要么 `detach()`（如果本就不需要它的梯度）。`retain_graph=True` 能压住报错，但多数时候是在给真 bug 盖被子，且激活内存不再释放。

**案例 3：`.grad` 一直是 None（三层排查树）**

① `x.is_leaf` 是 False？→ 中间节点，用 `retain_grad()` 或检查你是不是想看别的张量；
② `x.requires_grad` 是 False？→ 创建时忘了开，或上游被 `no_grad`/`detach` 断了（`x.grad_fn is None` 且非叶子即为断链证据）；
③ 前两者都对但仍 None？→ backward 根本没执行到它：loss 与 x 之间图不连通（常见于中途 `.numpy()` 转换、用了 Python float、或条件分支绕开了 x）。逐段打印 `t.grad_fn` 找断点。

---

## 3.8 练习题

### 基础 1：手推并验证两层网络的梯度
对 `z = (w2 * relu(w1 * x)).sum()`（标量 w1/w2/x 即可），手推 `dz/dw1`、`dz/dw2`，用 autograd 验证。再把 x 换成负数，观察 relu 掩码对梯度的影响。

### 基础 2：梯度累积等价性
用同一组数据验证：batch=8 一次算 vs batch=2 × 4 次累积（loss 除以 4），两种方式最终 `.grad` 相同（allclose）。解释为什么 loss 要除以累积步数。

### 进阶 1：计算图打印器
写函数 `print_graph(tensor)`，从 `tensor.grad_fn` 出发沿 `next_functions` 递归，缩进打印整张反向图（节点类名 + 是否 AccumulateGrad 叶子终点）。用它打印一个三层小网络的图，标出哪里体现了"残差连接 = 梯度双路"（给网络加一条 `x + f(x)` 观察 AddBackward 的两个上游）。

### 挑战 1：手写 micrograd（标量版 autograd 引擎）
不用 torch，写一个 `Value` 类：支持 `+`、`*`、`tanh`，每个运算记录 children 和局部导数，实现 `backward()`（拓扑排序 + 链式法则 + **梯度累加**）。用它训练一个 2-4-1 的小网络拟合 XOR，并与 PyTorch 对同一初始权重的梯度对拍（误差 < 1e-6）。这是 Karpathy micrograd 的自力更生版——写完它，autograd 对你不再有任何神秘感。

---

## 本章小结与下一章预告

Autograd = 前向时顺手记录反向图（grad_fn 链），backward 时拓扑逆序执行 VJP 并向叶子累加。理解了 saved_tensors，就理解了激活内存、`total += loss` 泄漏和 gradient checkpointing 的全部因果。版本计数器和"拒绝静默断图"体现了同一个设计哲学：宁可报错，不给错答案。

**下一章（第 4 章）**：nn.Module。参数（`nn.Parameter`）本质上就是"requires_grad=True 的叶子 + 自动注册"，而注册机制正是第 1 章 `__setattr__` 拦截的工业版。我们会拆 state_dict、hooks、模块树遍历——读懂一切开源模型代码的钥匙。
