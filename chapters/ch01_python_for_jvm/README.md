# 第 1 章 · 写给 JVM 工程师的 Python 速成

> **本章目标**：不是教你 Python 语法（你能读能写），而是校准心智模型——Python 有一批特性长得像 Java 的某个东西，行为却完全不同（"假朋友"），而 PyTorch 的 API 设计大量依赖这些特性。学完本章你应该能回答：
> 1. `model(x)` 一个"对象加括号"为什么能执行前向传播？
> 2. `@torch.no_grad()` 和 `with torch.no_grad():` 是什么关系？
> 3. GIL 不是把 Python 锁成单线程了吗，PyTorch 怎么还能吃满 GPU？
> 4. `for batch in dataloader:` 背后是什么协议在工作？

**前置**：第 0 章。 **硬件路径**：本地。 **预计用时**：3~4 小时。

---

## 1.1 来龙去脉：为什么深度学习选择了 Python

这个问题值得认真回答，因为它决定了你怎么看待 PyTorch 的架构。

**Python 慢，但 PyTorch 不慢**，因为 PyTorch 是"Python 皮、C++ 骨"：你在 Python 里写 `a @ b`，实际执行的是 C++（ATen 库）里的矩阵乘 kernel，Python 只负责**调度**——告诉 C++ 层"接下来算这个"。粗略类比你熟悉的架构：Python 之于 PyTorch，如同 SQL 之于计算引擎——描述计算的语言不需要快，执行计算的引擎才需要快。第 19 章会拆开这个调用链（Python → dispatcher → ATen → kernel）。

由此推论出两个重要事实：

**① GIL 在这里不致命**。GIL（全局解释器锁）让一个进程里同一时刻只有一个线程执行 Python 字节码——相当于整个 JVM 只有一把大锁。但 PyTorch 的重活发生在 C++/GPU 层，那里**不持有 GIL**（C 扩展可以主动释放它）。GPU 在算矩阵乘时，Python 线程只是在等结果。GIL 真正伤害的是"纯 Python 的 CPU 密集代码"——这正是为什么 DataLoader 用**多进程**而不是多线程做数据预处理（第 5 章展开，1.6 节面试题先埋点）。

**② 动态性是选它的原因，不是缺陷**。深度学习研究需要"改两行马上跑"的迭代速度，Python 的动态类型 + 交互式环境（Jupyter）踩中了这个需求。PyTorch 早期对 TensorFlow 1.x 的胜利，本质是"动态图（define-by-run）"对"静态图（define-then-run）"的胜利——而动态图正是靠 Python 的动态性实现的（第 3 章 autograd 会看到这一点）。你从强类型世界带来的"编译期安全感"在这里换成了"运行时灵活性"，代价后面再谈（易错点清单就是账单）。

### "假朋友"总表

先给你一张地图，后面逐个拆：

| 长得像 Java 的… | 实际上… | PyTorch 里的落点 |
|---|---|---|
| 接口/多态 | 鸭子类型：不看类型看方法 | Dataset 只要求实现 `__getitem__`/`__len__` |
| 注解（Annotation） | 装饰器：直接改写函数 | `@torch.no_grad()` |
| try-with-resources | 上下文管理器 `with` | `with torch.no_grad():` |
| Iterator 接口 | 迭代器协议 + 生成器 `yield` | `for batch in dataloader:` |
| 运算符（不可重载） | 魔术方法重载一切运算符 | `a @ b` 就是 `a.__matmul__(b)` |
| 变量 = 值（基本类型） | 变量永远是引用 | 张量赋值不拷贝数据 |
| 类型声明 | type hints 只是注释，运行时不检查 | 源码里的 `def forward(x: Tensor) -> Tensor` 骗不了人也保护不了人 |

---

## 1.2 核心原理：七个特性，七个 PyTorch 锚点

### ① 一切皆引用：没有"基本类型"这回事

Java 里 `int a = b` 拷贝值，`Object a = b` 拷贝引用。Python 里**一切都是后者**——变量是贴在对象上的标签：

```python
a = torch.randn(3, 3)
b = a          # b 和 a 是同一个张量的两个名字，零拷贝
b[0, 0] = 999  # a[0, 0] 也变成 999
```

要独立副本必须显式 `b = a.clone()`。这个语义贯穿 PyTorch：函数传参传引用、`state_dict()` 返回的是参数的引用（第 4 章的坑）、视图操作共享底层存储（第 2 章的主线）。**从今天起，看到 `=` 默认想"别名"，看到需要副本想 `clone()`。**

### ② 鸭子类型：协议取代接口

Java 靠 `implements Iterable` 声明能力；Python 靠"你实现了对应方法，你就是"。PyTorch 的 `Dataset` 是最好的例子——它不要求你继承任何东西（继承 `torch.utils.data.Dataset` 只是惯例），只要求两个方法：

```python
class MyDataset:                    # 没继承任何类
    def __len__(self):  return 100
    def __getitem__(self, i): return torch.tensor([i], dtype=torch.float32)

len(MyDataset())      # 100 —— len() 调用 __len__
MyDataset()[42]       # tensor([42.]) —— [] 调用 __getitem__
# DataLoader 拿到它照常工作：它只按协议调用，不检查类型
```

这类 `__xxx__` 方法叫**魔术方法（dunder methods）**，是 Python 的"隐式接口"体系。你需要认识的高频成员：

| 魔术方法 | 触发语法 | PyTorch 里谁在用 |
|---|---|---|
| `__call__` | `obj()` | `nn.Module`：`model(x)` |
| `__getitem__` / `__len__` | `obj[i]` / `len(obj)` | `Dataset`、张量切片 |
| `__iter__` / `__next__` | `for x in obj` | `DataLoader` |
| `__matmul__` | `a @ b` | 矩阵乘 |
| `__enter__` / `__exit__` | `with obj:` | `torch.no_grad()` |
| `__repr__` | 打印对象 | 张量/模型的漂亮输出 |
| `__setattr__` | `self.x = ...` | `nn.Module` 拦截属性赋值来注册参数（第 4 章核心机制！） |

### ③ `__call__`：`model(x)` 的真相

Java 里"对象加括号"是语法错误；Python 里它调用 `__call__`。这是 PyTorch 最重要的一个约定：

```python
class Linear:
    def __init__(self, w):  self.w = w
    def __call__(self, x):  return x @ self.w

layer = Linear(torch.randn(4, 2))
y = layer(torch.randn(3, 4))     # 实际执行 layer.__call__(x)
```

真实的 `nn.Module.__call__` 不是直接调 `forward`，而是包了一层：`model(x)` → `Module.__call__` → 前置 hooks → `self.forward(x)` → 后置 hooks。**这就是为什么文档说"永远调用 `model(x)` 而不要直接调 `model.forward(x)`"**——直接调 forward 会跳过 hooks（第 4 章讲 hooks 是什么、第 11 章的 profiler 和第 13 章的 FSDP 都靠 hooks 工作）。

### ④ 装饰器：函数是一等公民的推论

Python 的函数是对象，能当参数传、当返回值返。装饰器就是"接收函数、返回增强版函数"的函数——本质是你熟悉的 **AOP/动态代理**，只是语法轻得多：

```python
def timer(fn):
    def wrapped(*args, **kwargs):          # *args/**kwargs = 透传任意参数
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        print(f"{fn.__name__}: {(time.perf_counter()-t0)*1000:.1f}ms")
        return out
    return wrapped

@timer                    # 语法糖，等价于 train_step = timer(train_step)
def train_step(): ...
```

`*args, **kwargs` 顺便说透：`*args` 收集多余的位置参数成 tuple，`**kwargs` 收集多余的关键字参数成 dict。**读 HuggingFace 源码你会看到海量的 `**kwargs` 透传**——参数从顶层 API 一路传到底层实现，中间层不关心内容。这是动态类型的典型风格（也是读源码时"参数到底传到哪去了"困惑的来源，1.7 节教你排查）。

### ⑤ 上下文管理器：`with` 是 try-finally 的协议化

```python
with torch.no_grad():        # 进入时调 __enter__：关闭梯度记录
    logits = model(x)        # 这个块里所有运算不建计算图（省内存、加速）
                             # 退出时调 __exit__：恢复原状态（异常也保证执行）
```

对照 Java：`with` ≈ try-with-resources，`__enter__/__exit__` ≈ `AutoCloseable`。精妙之处在于 `torch.no_grad` 同时实现了上下文管理器协议**和**装饰器协议，所以两种写法等价：

```python
with torch.no_grad():        # 用法一：包一个代码块
    y = model(x)

@torch.no_grad()             # 用法二：包整个函数（推理函数的标准写法）
def evaluate(model, x): return model(x)
```

同族的还有 `torch.autocast`（第 6 章混合精度）、`torch.profiler.profile`（第 11 章）——PyTorch 用上下文管理器表达一切"进入某种模式，离开时恢复"的语义。

### ⑥ 生成器：`yield` 是惰性流

```python
def read_batches(path):
    with open(path) as f:
        batch = []
        for line in f:                # 逐行读，不整文件进内存
            batch.append(line)
            if len(batch) == 32:
                yield batch           # 产出一批，暂停在这里等下次索取
                batch = []
```

调用 `read_batches(p)` 不执行任何代码，返回一个生成器对象；每次 `for` 索取时才执行到下一个 `yield`。这就是你熟悉的**惰性求值流**——语义上接近 Iterator 的惰性、Spark RDD 的 lazy transformation。PyTorch 落点：`DataLoader` 的迭代本质就是这套协议，流式数据集（`IterableDataset`，第 5 章讲预训练数据管线时的主角）直接要求你写生成器。

### ⑦ 广播（Broadcasting）：形状不同的张量如何运算

这是 NumPy 定下、torch 全盘继承的规则，**第 2 章起每一章都在用**，现在必须焊进直觉。两条规则：

1. **从右往左**逐维对齐比较；
2. 每一维上：相等 ✓ / 其中一个是 1（虚拟复制去匹配另一个）✓ / 其中一个不存在（当作 1）✓ / 否则报错。

```python
(3, 4) + (4,)      → (3, 4)      # (4,) 左侧补 1 → (1,4) → 第一维 1 扩成 3
(8, 1, 6) + (7, 6) → (8, 7, 6)   # 逐维: [8 vs 无→8] [1 vs 7→7] [6 vs 6→6]
(3, 4) + (3,)      → 报错!        # 从右对齐: 4 vs 3 不匹配
```

LLM 里的真实用例，感受一下它的表达力：

```python
scores = torch.randn(B, H, T, T)   # 注意力分数: batch, heads, seq, seq
mask = torch.tril(torch.ones(T, T))  # 因果掩码: (T, T)
scores = scores.masked_fill(mask == 0, float("-inf"))
# (T,T) 广播到 (B,H,T,T)：一份掩码服务所有 batch 和 head，零拷贝
```

关键认知：广播**不真的复制数据**（靠 stride=0 的视图实现，第 2 章揭底），所以既省内存又快。但它也是 bug 温床——形状意外对上了，静默算出错误结果（易错点⑥）。

---

## 1.3 动手实验

```bash
uv run chapters/ch01_python_for_jvm/code/java_false_friends.py   # 假朋友逐个演示
uv run chapters/ch01_python_for_jvm/code/mini_pytorch_protocols.py  # 用纯 Python 手写 PyTorch 三大协议
uv run chapters/ch01_python_for_jvm/code/broadcasting_drills.py  # 广播规则演练
```

第二个脚本值得精读：它用 60 行纯 Python（不用 torch 的任何机制）实现一个具备 `__call__`、参数注册、`no_grad` 上下文管理器的 **mini 框架骨架**——你会发现 PyTorch 的"外壳"没有任何魔法，全是本章的语言特性。第 4 章读真正的 `nn.Module` 源码时，你已经见过它的简化版了。

---

## 1.4 易错点清单

**① `is` vs `==`**
```python
a = torch.tensor([1, 2]); b = a.clone()
a == b   # tensor([True, True]) —— 逐元素比较（重载了 __eq__）
a is b   # False —— 身份比较（同一个对象吗）
```
→ **现象**：用 `==` 判断"是不是同一个张量"得到一个张量而不是布尔值，`if a == b:` 直接报错 `Boolean value of Tensor with more than one element is ambiguous`。
→ **原因**：Java 的 `==` 比引用、`equals` 比内容；Python 反过来，`is` 比身份、`==` 被张量重载成了逐元素运算。
→ **修正**：判断同一对象用 `is`；判断数值全等用 `torch.equal(a, b)`；判断近似相等（浮点！）用 `torch.allclose(a, b)`。

**② 可变默认参数**
```python
def append_log(x, logs=[]):    # ✗ 默认值在函数定义时创建一次，所有调用共享
    logs.append(x); return logs
append_log(1)  # [1]
append_log(2)  # [1, 2] ←—— 惊喜
```
→ **修正**：`def append_log(x, logs=None): logs = [] if logs is None else logs`。PyTorch 相关场景：自定义 Dataset/Module 的 `__init__` 默认参数里放 list/dict 时必踩。

**③ 变量遮蔽：把自己的文件命名为 `torch.py`**
→ **现象**：`import torch` 报 `AttributeError: module 'torch' has no attribute 'randn'`，或更诡异的错。
→ **原因**：Python 优先从当前目录解析 import，你的 `torch.py`/`dataset.py`/`types.py` 遮蔽了真库。Java 的包全名机制不存在这个问题。
→ **修正**：实验脚本永远不要用库名命名。排查口诀见 1.7 节案例 1。

**④ 闭包晚绑定**
```python
hooks = [lambda: print(i) for i in range(3)]
[h() for h in hooks]     # 打印 2 2 2，不是 0 1 2
```
→ **原因**：闭包捕获的是**变量**不是**值**（Java 的 lambda 强制 effectively final，恰好避开了这坑）。循环里注册 hook/回调（第 4 章 hook、第 13 章给每层挂通信钩子）时高危。
→ **修正**：默认参数固化 `lambda i=i: print(i)`，或用 `functools.partial`。

**⑤ 类型提示不是类型检查**
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    ...
forward(model, "hello")   # 运行时完全不拦截，进函数体才炸
```
→ **认知校准**：hints 服务于 IDE 和静态检查器（mypy/pyright），运行时是注释。好处是你读 PyTorch 源码时 hints 是可靠的文档；坏处是别指望它当护栏——传错类型的报错点可能在离调用处很远的地方（1.7 节排查方法论）。

**⑥ 广播静默给出"合法但错误"的结果**
```python
loss_per_token = torch.randn(32, 128)     # (batch, seq)
weights = torch.randn(32)                 # 想给每个样本加权
weighted = loss_per_token * weights       # ✗ 报错还算走运
weights_col = torch.randn(128)
weighted = loss_per_token * weights_col   # ✓✗ 不报错！但语义是"按 token 位置加权"
```
→ **现象**：不报错，loss 正常下降，模型效果莫名差——**最贵的一类 bug**。
→ **修正**：涉及降维/加权的运算，显式 reshape 表达意图：`weights[:, None]`（变 `(32,1)`，明确"按行广播"）。团队实践：关键张量运算写 shape 注释（`# (B, T) * (B, 1) -> (B, T)`），这是 LLM 开源代码的普遍风格。

---

## 1.5 开源项目的最佳实践

**① `nn.Module._call_impl`：`__call__` 协议的工业级实现**
读 [pytorch/torch/nn/modules/module.py](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/module.py) 中的 `_call_impl`（`Module.__call__` 的真身）。注意两点：快路径判断（没有任何 hook 时直接 `forward(*args, **kwargs)`，一次 dict 判空的开销都要省）；hooks 的执行顺序编排。60 行代码，是"协议 + 性能意识"的范本。

**② LLaMA 官方实现的 `@dataclass` 配置**
[meta-llama/llama/model.py](https://github.com/meta-llama/llama/blob/main/llama/model.py) 开头的 `ModelArgs`：用 `@dataclass` 装饰器自动生成 `__init__`/`__repr__`（类比 Lombok 的 `@Data`）。现代 LLM 代码的配置几乎都这么写，取代了字典满天飞的旧风格——你会在第 8 章我们自己的 Transformer 里采用同样模式。

**③ HuggingFace 的 kwargs 透传链**
`transformers` 的 `from_pretrained(..., **kwargs)` 把参数一路透传到 config/模型构造。学它的**分拆手法**：在每一层用 `kwargs.pop("xxx", default)` 取走自己关心的参数，剩下的继续往下传——这让 API 加参数不用改中间层签名。副作用是拼写错误的参数会被静默吞掉（`temperture=0.7` 不报错），所以新版加了严格校验。**教训**：设计 kwargs 链时，末端要对剩余未消费的 kwargs 报错或告警。

---

## 1.6 典型面试题

**Q1：GIL 是什么？为什么 PyTorch 的 DataLoader 用多进程而不是多线程？训练主循环受 GIL 影响吗？**

> **参考答案**：GIL 保证一个进程内同一时刻只有一个线程执行 Python 字节码。数据预处理（解码、增强、tokenize）是纯 Python/CPU 密集工作，多线程会被 GIL 串行化，所以 DataLoader 用 `num_workers` 个**子进程**并行（进程各有各的 GIL），通过共享内存把张量传回主进程（第 5 章细讲这条管道及其坑）。训练主循环基本不受 GIL 影响：矩阵运算在 C++/CUDA 层执行且主动释放 GIL。**加分点**：提到 Python 3.13+ 的 free-threaded 构建（PEP 703）正在试验去 GIL，以及 GPU 异步执行让 Python 的调度开销被计算流水掩盖（第 11 章）。

**Q2：解释 `model(x)` 到 `forward` 的完整调用链。为什么不建议直接调 `model.forward(x)`？**

> **参考答案**：`model(x)` → `Module.__call__`（实际是 `_call_impl`）→ 依序执行 forward pre-hooks → `forward(x)` → forward hooks → 返回。直接调 `forward` 跳过所有 hooks，而很多机制寄生在 hooks 上：profiler 插桩、量化观察器、FSDP 的参数聚散（第 13 章——FSDP 模型直接调 forward 会拿到空参数直接崩）。**加分点**：hooks 还分 module 级和全局级，`register_forward_hook` 返回的 handle 要保存以便 `remove()`。

**Q3：手写一个装饰器 `@retry(n=3)`，失败自动重试 n 次。（现场编码题）**

> **参考答案**要点：带参数的装饰器是**三层**函数（参数层→装饰层→包装层）：
> ```python
> def retry(n=3):
>     def deco(fn):
>         @functools.wraps(fn)            # 保住原函数的 __name__/__doc__
>         def wrapped(*args, **kwargs):
>             for i in range(n):
>                 try: return fn(*args, **kwargs)
>                 except Exception:
>                     if i == n - 1: raise
>         return wrapped
>     return deco
> ```
> **加分点**：`functools.wraps` 的作用（不加的话被装饰函数的元信息全变成 `wrapped`，堆栈难读）；对比 `@torch.no_grad()` 有括号、`@timer` 没括号——前者是"调用后返回装饰器"，后者本身就是装饰器。

**Q4：Python 的内存管理和 JVM GC 有什么本质区别？这对 PyTorch 显存管理意味着什么？**

> **参考答案**：CPython 主机制是**引用计数**（计数归零立即释放，确定性析构）+ 辅助的分代 GC 只为解决循环引用；JVM 是纯追踪式 GC（释放时机不确定）。对 PyTorch 的意义重大：张量的显存在引用归零那一刻**立即**归还给 PyTorch 的缓存分配器——所以 `del tensor` / 变量出作用域是有效的显存管理手段，`loss.backward()` 后及时 `del loss` 能实际降低峰值显存（第 11 章显存解剖、第 15 章 OOM 排查都建立在这个确定性之上）。**加分点**：循环引用的张量要等分代 GC 才释放，这是"明明 del 了显存没降"的一种病因。

---

## 1.7 疑难杂症排查

**案例 1：`AttributeError: module 'xxx' has no attribute 'yyy'`，但文档明明说有**

排查顺序：
1. `print(xxx.__file__)` ——看模块**实际从哪加载**。指向你项目目录？中招易错点③（文件名遮蔽）。
2. 指向 site-packages 但属性没有？查版本：`print(xxx.__version__)`——文档是新版的，你装的是旧版。
3. `__file__` 指向意外的 venv？回到第 0 章易错点④（用错解释器）。
**方法论**：`__file__` + `__version__` 两板斧先行，能解决此类问题的九成。

**案例 2：改了库的源码/自己的模块，重跑没生效**

场景：Jupyter 里 debug，改了 `my_model.py`，重新执行 cell 行为没变。
原因：Python 的 import 有进程级缓存（`sys.modules`），第二次 `import` 直接用缓存——JVM 热加载难，Python 默认也不热加载。
修正：Jupyter 里开自动重载（`%load_ext autoreload` + `%autoreload 2`）；脚本环境重启进程。**教训**：怀疑"改了没生效"时，在模块顶层加一行 print 验证加载，别浪费时间 debug 旧代码。

**案例 3：`TypeError: forward() got an unexpected keyword argument 'xxx'`（kwargs 链断裂）**

场景：给 HF 模型传自定义参数，在某个中间层爆炸。
排查：这类错误的报错点在链条**末端**，问题往往在**你传参的拼写**或**版本不匹配**（该版本还没有这个参数）。用 `inspect.signature(model.forward)` 直接打印目标函数真实签名对照，比翻文档快。
**方法论**：动态语言里"参数去哪了"的问题，用 `inspect` 模块现场求证，不猜。

**案例 4：进程无输出直接死亡，exit code 137**

场景：脚本没有任何报错，甚至没有任何输出，进程直接消失，shell 里 `echo $?` 显示 137。（本章配套代码第一版就真实踩过。）
解码：137 = 128 + 9，即被 `SIGKILL` 杀死，几乎总是**内核 OOM killer**——你的进程把内存吃爆了。Python 侧看不到任何 traceback，因为进程是被外力击毙的，连临终遗言都来不及说。
本例病根：`MiniDataset` 实现了 `__getitem__` 但忘了越界时抛 `IndexError`，而 Python 的旧式迭代协议**靠 IndexError 停止**，于是 `for x in ds` 无限索取，列表推导式无限累积直到内存耗尽。
排查思路：① exit 137/OOM 先怀疑无限循环或数据整份进内存；② 用 `head`（限制迭代次数）或加 print 定位失控点；③ macOS 看 `log show --predicate 'eventMessage CONTAINS "memory"'`，Linux 看 `dmesg | grep -i oom`。
**教训**：实现 `__getitem__` 协议时，越界抛 `IndexError` 不是可选项——它是协议的一部分（真 PyTorch 的 map-style Dataset 同理，配合 `__len__` 由 Sampler 控制范围，但直接 for 遍历时 IndexError 仍是唯一刹车）。

---

## 1.8 练习题

### 基础 1：手写 `@timer` 装饰器
实现一个能同时用于普通函数和方法的 `@timer`，打印毫秒耗时，要求用 `functools.wraps` 保留元信息。用它装饰一个做 1024×1024 矩阵乘的函数验证。

### 基础 2：广播 shape 心算
不运行代码，推断下列运算的结果 shape（或判断报错），写下推理过程，再用代码验证：
a) `(5, 1, 4) * (3, 4)`   b) `(2, 3) + (3, 2)`   c) `(8, 1) + (1, 8)`   d) `(6, 7) + (7,)`   e) `(4, 3, 2) + (3, 1)`

### 进阶 1：从零写一个符合 PyTorch 协议的 Dataset
不继承任何类，实现一个 `WindowedDataset`：接收一个一维张量和窗口大小 w，`[i]` 返回 `(前 w 个元素, 第 w+1 个元素)` 的元组（这就是第 9 章语言模型"输入前文预测下一个词"的数据形态）。要求支持 `len()`、下标访问、`for` 遍历，并写出它为什么能被 `torch.utils.data.DataLoader` 直接消费。

### 挑战 1：双协议的 `no_grad` 平替
用纯 Python 实现一个 `class my_mode:`，既能 `with my_mode():` 又能 `@my_mode()` 使用（模仿 `torch.no_grad` 的双协议设计），进入时把全局开关 `GRAD_ENABLED` 置 False、退出恢复**原值**（注意嵌套使用的正确性），异常时也要恢复。提示：装饰器协议可以借 `__call__` 实现，也可以研究 `contextlib.ContextDecorator` 后自己写一遍。

---

## 本章小结与下一章预告

Python 之于 PyTorch 不是"换个语法的 Java"，而是一套以**协议**（魔术方法）、**引用语义**、**运行时动态性**为根基的不同物种。你现在知道了 `model(x)`、`@no_grad`、`with`、`for batch in loader`、广播这五个日常语法背后的机制——第 2~5 章会反复兑现这些伏笔。

**下一章（第 2 章）**：张量的本质。我们会打开张量的"物理层"——storage、stride、view，回答"为什么 `transpose` 零拷贝而 `contiguous` 要搬数据"、"7B 模型到底要多少内存"这类问题。广播规则会在那里揭示它的实现秘密：stride=0。
