# 第 4 章 · nn.Module 体系：读懂一切开源模型代码的钥匙

> **本章目标**：吃透 PyTorch 的模型组织机制。学完你应该能回答：
> 1. `nn.Parameter` 和普通张量差在哪？参数是怎么被"自动发现"的？
> 2. `state_dict` 里到底存了什么？为什么 DDP 存的权重单机载不进去？
> 3. `model.eval()` 和 `torch.no_grad()` 是一回事吗？（高频面试题）
> 4. hook 是什么？为什么 profiler、FSDP、特征提取全都寄生在它上面？

**前置**：第 1 章（`__setattr__` 拦截、MiniModule）、第 3 章（叶子张量）。 **硬件路径**：本地。 **预计用时**：4~5 小时。

---

## 4.1 来龙去脉：张量 + autograd 之后，还缺什么？

第 3 章结束时你已经能训练模型了——用裸张量。但裸张量方案在工程上立刻碰壁：

1. **参数收集**：优化器需要"全部参数"的列表。手工维护几百个张量的清单？一个 7B 模型有几百个权重矩阵，分散在几十层结构里。
2. **命名与寻址**：保存/加载/调试需要每个参数有稳定的名字（"第 3 层注意力的 Q 投影"）。
3. **批量操作**：整个模型搬 GPU、转 dtype、切换训练/推理模式、冻结某几层——都需要"对模型整体或子树做遍历操作"的能力。

`nn.Module` 就是为此而生的**组合模式（Composite Pattern）容器**：模块套模块构成一棵树，参数挂在树的节点上，一切批量操作都是树遍历。你在第 1 章手写的 `MiniModule`（`__setattr__` 拦截 + 递归收集）就是它的骨架——本章看工业版多出了什么。

---

## 4.2 核心原理

### ① nn.Parameter：会被自动登记的张量

`nn.Parameter` 是 `Tensor` 的子类，只多两件事：默认 `requires_grad=True`（第 3 章的叶子节点），以及**被 `Module.__setattr__` 识别并登记**：

```python
class MyLayer(nn.Module):
    def __init__(self):
        super().__init__()                            # 必须先调！它创建 _parameters 等登记簿
        self.w = nn.Parameter(torch.randn(4, 4))      # → 登记进 self._parameters
        self.sub = nn.Linear(4, 4)                    # → 登记进 self._modules
        self.mask = torch.ones(4)                     # → 只是普通属性，不登记！
```

`Module.__setattr__` 按类型分流：`Parameter` 进 `_parameters`，`Module` 进 `_modules`，其余当普通属性。这就是"参数自动发现"的全部机制——第 1 章 MiniModule 的工业版，多的是类型分流和防呆检查（比如 `super().__init__()` 没调会报友好错误）。

### ② 第三类登记对象：buffer——"随模型走但不学习"的状态

```python
self.register_buffer("freq_cis", precompute_rope(...))          # RoPE 的旋转矩阵缓存
self.register_buffer("causal_mask", mask, persistent=False)     # 不想存进 checkpoint 就 False
```

buffer 的待遇：随 `to(device)` 迁移、默认进 `state_dict`（`persistent=False` 除外）、**不进 `parameters()`**（优化器不碰它）。三类对象的判别口诀——**要学习的用 Parameter，不学习但属于模型状态的用 buffer，纯临时量用普通属性**。LLM 里 buffer 的常客：RoPE 频率表、因果掩码、BatchNorm 的 running stats（第 8 章都会用到）。

### ③ 模块树与命名：state_dict 的 key 从哪来

```python
model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
[name for name, _ in model.named_parameters()]
# ['0.weight', '0.bias', '2.weight', '2.bias']   ← 点分路径 = 树上的寻址
```

参数名 = 从根到叶的属性路径。`layers.3.attn.q_proj.weight` 这种名字你将在 LLM checkpoint 里天天见——它同时是**调试坐标**（第几层哪个投影）和**加载对齐的依据**。遍历 API 一族：`named_parameters()`（叶子参数）、`named_modules()`（所有节点含自身）、`children()`（直接子节点）、`apply(fn)`（对每个子模块调 fn，初始化的标准工具）。

**重要推论**：参数放进 Python 原生 `list`/`dict` 是**注册不到**的（`__setattr__` 只拦截直接属性赋值）。必须用 `nn.ModuleList`/`nn.ModuleDict`/`nn.ParameterList`——易错点①，transformer 堆 N 层必经之路。

### ④ state_dict：只有数值，没有结构

```python
sd = model.state_dict()      # OrderedDict: 名字 → 张量（参数 + persistent buffers）
model.load_state_dict(sd)    # 按名字对号入座
```

三个关键认知：
1. **它不含代码/结构**——加载前你得先用代码构造出同构的模型（对比 pickle 整个模型对象的旧做法：脆弱、不安全、版本绑死，已被抛弃）。safetensors（第 2 章）存的就是 state_dict。
2. **`load_state_dict(sd, strict=True)` 的语义**：缺 key、多 key 都报错。`strict=False` 返回 `(missing_keys, unexpected_keys)`——**必须检查这个返回值**，静默吞掉不匹配是"载入成功但效果全错"的头号来源（4.7 案例 1）。
3. **返回的张量是参数的引用**（第 1 章引用语义）：`sd` 不是快照！要快照需 `copy.deepcopy` 或逐项 `clone()`——保存"最优权重"时的经典坑。

`torch.save(sd, path)` / `torch.load(path, map_location="cpu", weights_only=True)`：`map_location` 让 GPU 存的权重能在无 GPU 机器加载（第 2 章 `.to(device)` 伏笔回收），生产惯例是**保存前搬 CPU、加载时显式指定设备**。

**安全注意**：`torch.load` 底层是 pickle，`weights_only=False` 时反序列化可以执行**任意代码**——加载来路不明的 `.pt`/`.bin` 等于运行陌生人的脚本。torch 2.6 起默认已改为 `weights_only=True`（只允许张量与基础容器），但读旧代码/旧版本环境时务必显式传参。这也是 safetensors（第 2 章）取代 pickle 格式的安全动机：纯字节 + 元数据，结构上不可能藏代码。下载社区模型优先选 `.safetensors` 文件。

### ⑤ train/eval 与 no_grad：正交的两根开关

高频面试题的完整答案，先记结论表：

| | `model.train()` / `model.eval()` | `torch.no_grad()` |
|---|---|---|
| 控制什么 | **模块行为**：Dropout 开/关、BatchNorm 用批统计/滑动统计 | **autograd**：是否建计算图 |
| 实现 | 递归设置 `module.training` 布尔位 | 线程本地的全局开关 |
| 影响显存/速度 | 几乎不 | 显著（不存激活） |
| 推理时 | 必须 `eval()`（否则 dropout 还在随机丢） | 必须（否则白建图） |

两者**正交**：eval 模式下不加 no_grad，照样建图耗显存；no_grad 下不 eval，dropout 照样开。正确的推理姿势永远是**两个一起**：`model.eval()` + `with torch.no_grad():`（或 `inference_mode`）。反向的坑：评估完回训练忘了 `model.train()`，dropout 全程没开，正则化静默失效。

### ⑥ hooks：模块树上的 AOP 切点

```python
def spy(module, args, output):
    print(f"{module.__class__.__name__}: {output.norm():.3f}")

handle = model.layers[3].register_forward_hook(spy)   # 不改模型代码，旁路观察
... 
handle.remove()                                       # 用完必须摘除！
```

四种常用 hook：`forward_pre_hook`（进 forward 前）、`forward_hook`（出 forward 后）、`full_backward_hook`（反向经过时）、参数级的 `register_post_accumulate_grad_hook`（梯度累加完成后，第 12 章 DDP 的挂载点之一）。这就是第 1 章"为什么必须 `model(x)` 不能 `model.forward(x)`"的原因——hooks 编排在 `__call__` 里。

工业级用途一览（后面章节逐个兑现）：特征提取与探针（本章实验）、NaN 哨兵（第 15 章排查 loss 爆炸）、profiler 插桩（第 11 章）、FSDP 在 pre-hook 里 all-gather 参数/post-hook 里释放（第 13 章）、量化观察器（第 16 章）。**hook 是 PyTorch 生态的暗线基础设施**。

### ⑦ `to()` 的语义陷阱：Module 就地，Tensor 返新

```python
model.to("mps")        # 就地修改（参数原对象换设备），返回 self 只是为了链式调用
tensor.to("mps")       # 返回新张量，原张量不动！
```

同名方法、相反语义——历史包袱，但必须记住。`tensor = tensor.to(device)` 的重新赋值不能少；模型则 `model.to(device)` 一句完事。优化器要在 `model.to()` **之后**创建（优化器持有参数引用，先创建再搬家会导致状态设备错位——第 6 章优化器状态的坑）。

---

## 4.3 动手实验

```bash
uv run chapters/ch04_nn_module/code/module_registry.py   # 注册机制/树遍历/state_dict 解剖
uv run chapters/ch04_nn_module/code/hooks_lab.py         # hook 抓激活分布 + NaN 哨兵实战
uv run chapters/ch04_nn_module/code/save_load.py         # 保存/加载/前缀修复/部分加载全流程
```

`hooks_lab.py` 里的 NaN 哨兵值得留意：给全模型每个子模块挂 forward hook，输出出现 NaN/Inf 时立刻报出**是哪一层**——这个 30 行的工具在第 15 章排查真实训练事故时会直接复用。

---

## 4.4 易错点清单

**① 参数藏在 list 里，优化器一无所知**
```python
self.layers = [nn.Linear(4, 4) for _ in range(8)]      # ✗ 普通 list 不触发注册
self.layers = nn.ModuleList(nn.Linear(4, 4) for _ in range(8))   # ✓
```
→ **现象**：不报错！`parameters()` 为空或缺一截，优化器只更新注册到的部分，loss 降一点就平——**静默型事故**。
→ **排查**：`sum(p.numel() for p in model.parameters())` 和理论参数量对账（本章实验演示）。

**② eval 忘切 / train 忘切回**
→ **现象**：忘 `eval()`——推理结果每次不同（dropout 在随机丢）；评估后忘 `train()`——之后的训练悄悄没了 dropout/BN 更新。
→ **修正**：训练循环模板里把 `model.train()` 写在**每个 epoch 训练段开头**（而不是循环外一次），评估函数开头 `model.eval()`，用完自动回来。

**③ `model.eval()` 当成 `no_grad` 用**
→ **现象**：推理显存爆炸——eval 只改模块行为，图照建。两个都要（4.2-⑤）。

**④ 保存"最优权重"存了个引用**
```python
best_sd = model.state_dict()          # ✗ 引用！模型继续训练，best 跟着变
best_sd = copy.deepcopy(model.state_dict())   # ✓
```
→ **现象**：早停后加载"最优"权重，效果却是最后一步的——因为存的是活引用。

**⑤ `super().__init__()` 忘调或晚调**
```python
def __init__(self):
    self.w = nn.Parameter(...)     # ✗ AttributeError: cannot assign parameters before
    super().__init__()             #    Module.__init__() call
```
→ **原因**：登记簿（`_parameters` 等）是 `Module.__init__` 创建的，没有它 `__setattr__` 无处登记。报错信息友好，但要理解为什么。

**⑥ hook 用完不摘，越挂越多**
→ **现象**：重复调用"注册 hook 的函数"后性能下降/日志重复/闭包引用导致显存不释放。
→ **修正**：保存 `handle` 并 `remove()`；批量场景用 `try/finally` 或把 hook 生命周期包进上下文管理器（第 1 章技能落地）。

---

## 4.5 开源项目的最佳实践

**① transformers 的 `from_pretrained`：state_dict 工程的集大成**
它做的远不止 `load_state_dict`：下载/缓存 → safetensors 分片按需读 → 键名映射（新旧版本兼容）→ `tie_weights`（embedding 与 lm_head 共享，本章挑战题）→ 设备分派（`device_map`）。读 [modeling_utils.py](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py) 里 `_load_pretrained_model` 对 missing/unexpected keys 的分级告警——这是 `strict=False` 的正确工程姿势：**丢什么可以容忍（如分类头）、丢什么必须报错（如主干），逐类决策而不是一刀切**。

**② `_init_weights` + `apply()`：初始化的标准模式**
几乎所有 HF 模型都有 `_init_weights(self, module)` 方法，配合 `self.apply(self._init_weights)` 树遍历执行。初始化用 `torch.no_grad()` 包裹（改参数值不该建图）。第 8 章我们的 Transformer 沿用此模式，第 9 章会看到初始化方差对训练稳定性的实际影响（GPT-2 对残差投影的 1/√N 缩放）。

**③ LLaMA 的结构惯例**
读 transformers 的 `LlamaModel`：`nn.ModuleList` 堆层、每层是独立 `LlamaDecoderLayer`、RoPE 频率表用 buffer、`lm_head` 与 `embed_tokens` 可选共享。开源 LLM 的结构代码高度同构——精读一个，通读所有（第 10 章逐行精读）。

---

## 4.6 典型面试题

**Q1：`model.eval()` 和 `torch.no_grad()` 的区别？推理时该用哪个？**

> **参考答案**：正交的两根开关——`eval()` 递归设置 `training=False`，改变 Dropout（关闭）和 BatchNorm（用滑动统计）等模块的**行为**；`no_grad()` 关闭 autograd 图构建，省显存提速但不改变任何模块行为。推理两个都要。**加分点**：`inference_mode()` 是 no_grad 的加强版（产出张量永久出图，免版本计数开销）；训练中的验证段用完要记得 `model.train()` 切回。

**Q2：`nn.Parameter`、buffer、普通张量属性，模型里三者怎么选？各自在 `state_dict`/`parameters()`/`to(device)` 中的待遇？**

> **参考答案**：要梯度更新的状态 → Parameter（三处全有份）；不更新但属于模型状态、需要持久化或随设备走的 → buffer（进 state_dict 与 to，不进 parameters）；纯派生缓存 → 普通属性（三处都不管，注意 to 不会帮你搬设备）。例子：权重矩阵/embedding 是 Parameter；BN 的 running_mean、RoPE 频率表是 buffer；临时的 attention mask 可以是普通属性但要自己管设备。**加分点**：`register_buffer(..., persistent=False)` 用于"要随设备走但不想进 checkpoint"的状态（可重算的缓存）。

**Q3：加载 checkpoint 报 `Missing key(s) ... Unexpected key(s): module.xxx`，怎么回事，怎么修？**

> **参考答案**：`module.` 前缀是 DDP 包装的痕迹——`DistributedDataParallel(model)` 把原模型挂在 `.module` 属性下，state_dict 的 key 全部多一层前缀。修法三选一：① 保存时就存 `model.module.state_dict()`（最佳实践）；② 加载时字符串处理剥前缀；③ 加载端也包一层 DDP 再载。**加分点**：`torch.compile` 会引入 `_orig_mod.` 前缀，同类问题；通用解法是写个 `strip_prefix` 工具函数，或用 `load_state_dict` 前先 `consume_prefix_in_state_dict_if_present`（torch 自带）。

**Q4：如何冻结一个模型的前 N 层只微调其余部分？冻结后还有什么要注意的？**

> **参考答案**：两步——对冻结参数 `p.requires_grad_(False)`；优化器只喂 `filter(lambda p: p.requires_grad, model.parameters())`。注意点：① 只设 requires_grad 不过滤优化器也能跑，但 AdamW 的 weight decay 实现差异可能仍碰参数，且优化器状态白占显存（每参数 8 字节！第 2 章的账）；② 冻结层的 BatchNorm/Dropout 行为不受 requires_grad 影响，需要的话对冻结子树单独 `.eval()`；③ 被冻结段的激活如果不需要（整个前缀冻结），可以 no_grad 跑前缀省激活显存——这正是第 14 章 LoRA 显存优势的一部分原理。

---

## 4.7 疑难杂症排查

**案例 1：权重"加载成功"，效果却像没训过**

高发病因：`load_state_dict(sd, strict=False)` 静默丢弃了大量不匹配的 key（改过类名/层名/前缀后尤其常见）。
排查：① 永远接住返回值 `missing, unexpected = model.load_state_dict(sd, strict=False)` 并打印数量；② 对比 `set(sd.keys())` 与 `set(model.state_dict().keys())` 的差集，看差在前缀还是结构；③ 抽一个确定加载了的权重算 norm，和 checkpoint 里的对账。
**方法论**：加载类问题永远先对 key，再对 shape，最后对数值。

**案例 2：同一模型两次构造，推理结果不同**

排查树：① 忘了 `eval()`（dropout 随机）→ 最常见；② 权重没加载（用了随机初始化，看 key 对账）；③ 真随机源：有模块在 forward 里用了 `torch.randn` 且没固定种子；④ 非确定算子（MPS/CUDA 上部分归约顺序不定，误差 1e-6 级——这是正常浮点噪声，不是 bug，判断标准见 allclose 的容差）。

**案例 3：加了 hook 之后显存缓慢上涨**

病根：hook 闭包里保存了输出张量（如 `activations.append(output)`）且没 detach——第 3 章场景 B 的变体：hook 抓的是**带图的**激活。修正：抓取即 `output.detach()`（需要统计就当场算完只存标量）；实验结束 `handle.remove()`。本章 `hooks_lab.py` 的统计器展示了规范写法。

---

## 4.8 练习题

### 基础 1：手写 MySequential
不用 `nn.Sequential`，实现一个接受任意个子模块、按序执行的容器（提示：`nn.ModuleList` 或在 `__init__` 里 `add_module`）。验证 `named_parameters()` 的命名是否与 `nn.Sequential` 一致。

### 基础 2：参数账本
写函数 `param_report(model)`：按 top-level 子模块分组统计参数量与占比，顺带检查"申报参数量"（构造时的理论值）与实际注册量是否一致。用它测一个故意用普通 list 藏层的模型，展示它如何暴露易错点①。

### 进阶 1：激活统计器（第 15 章的排查工具雏形）
用 forward hook 实现 `ActivationStats`：上下文管理器，进入时给每个叶子模块挂 hook 记录输出的 mean/std/max，退出时自动摘除并输出表格。要求：hook 内不保留张量（当场算标量），支持 `with ActivationStats(model) as stats:` 用法。

### 挑战 1：手写 tie_weights
实现一个 mini 语言模型骨架：`embedding (vocab→d)` + 线性层 `lm_head (d→vocab)`，让两者**共享同一份权重**（GPT-2/LLaMA 部分版本的做法，省 vocab×d 个参数）。要求：① 共享后 `parameters()` 不重复计数；② 训练一步后验证两处的梯度作用在同一张量上；③ state_dict 里如何体现共享？加载时会有什么坑？（第 9 章 miniGPT 直接使用你的结论。）

---

## 本章小结与下一章预告

nn.Module = 组合模式的模块树 + 三本登记簿（parameters/buffers/modules）+ 树遍历 API + hook 切点。state_dict 只是"名字→数值"的快照，结构永远由代码定义。train/eval 与 no_grad 正交。这些机制没有一个是魔法——全部是第 1 章语言特性 + 第 3 章叶子张量的组合。

**下一章（第 5 章）**：数据管线。模型准备好了，数据怎么高效喂进来？Dataset/DataLoader/Sampler 三件套、多进程加载的共享内存管道、以及大数据工程师会心一笑的那些主题：预取、背压、分片、shuffle 的代价。
