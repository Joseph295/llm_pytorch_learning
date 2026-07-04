"""第 1 章 · 用纯 Python 手写 PyTorch 的三大协议骨架

运行：uv run chapters/ch01_python_for_jvm/code/mini_pytorch_protocols.py

用 ~60 行不依赖 torch 机制的代码，复刻 PyTorch 外壳的三个核心设计：
  1. MiniModule   —— __call__ 调 forward + __setattr__ 自动注册参数（nn.Module 的骨架）
  2. MiniDataset  —— __len__/__getitem__ 协议（Dataset 的全部要求）
  3. mini_no_grad —— 上下文管理器 + 装饰器双协议（torch.no_grad 的骨架）

结论先行：PyTorch 的"外壳"没有魔法，全是第 1 章的语言特性。
第 4 章读真 nn.Module 源码时，你已经见过它的简化版。
"""


# ═══ 1. MiniModule：nn.Module 的 20 行骨架 ═══
class MiniModule:
    def __init__(self):
        # 用 object.__setattr__ 绕过下面自定义的拦截（先有登记簿才能登记）
        object.__setattr__(self, "_params", {})
        object.__setattr__(self, "_submodules", {})

    def __setattr__(self, name, value):
        """拦截所有 self.xxx = ... 赋值——这就是 nn.Module 自动发现参数的机制！

        真 nn.Module 拦截的是 nn.Parameter 和 Module 类型；我们用 float 模拟参数。
        """
        if isinstance(value, float):
            self._params[name] = value
        elif isinstance(value, MiniModule):
            self._submodules[name] = value
        object.__setattr__(self, name, value)

    def parameters(self):
        """递归收集自己和子模块的所有参数——state_dict/optimizer 的基础。"""
        yield from self._params.values()          # 生成器委托
        for sub in self._submodules.values():
            yield from sub.parameters()

    def __call__(self, *args, **kwargs):
        # 真 _call_impl 在这里编排 hooks；骨架只保留"约定调 forward"
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("子类必须实现 forward——和真 PyTorch 同样的约定")


class Scale(MiniModule):
    def __init__(self, k: float):
        super().__init__()
        self.k = k                     # float → 被 __setattr__ 拦截注册为"参数"

    def forward(self, x):
        return x * self.k


class Pipeline(MiniModule):
    def __init__(self):
        super().__init__()
        self.first = Scale(2.0)        # MiniModule → 注册为子模块
        self.second = Scale(3.0)

    def forward(self, x):
        return self.second(self.first(x))


net = Pipeline()
print(f"net(10) = {net(10)}                    ← 对象加括号：__call__ → forward")
print(f"自动发现的参数: {list(net.parameters())}   ← __setattr__ 拦截 + 递归收集")

# ═══ 2. MiniDataset：鸭子类型，不继承任何东西 ═══
class MiniDataset:
    """DataLoader 对 Dataset 的全部要求就是这两个方法（map-style）。"""

    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if i >= self.n:
            # 关键！旧式迭代协议靠 IndexError 停止。漏掉这个检查，
            # for 循环会无限索取 ds[0],ds[1],... 直到内存耗尽被 OOM 杀掉
            # （本脚本第一版就真的犯了这个错，exit code 137）
            raise IndexError(i)
        return i, i * i                # (输入, 标签)


ds = MiniDataset(5)
print(f"\nlen(ds) = {len(ds)}; ds[3] = {ds[3]}")
# 只实现了 __getitem__/__len__，for 循环也能工作——
# Python 的"旧式迭代协议"会用下标 0,1,2,... 依次调用 __getitem__
print(f"for 遍历: {[pair for pair in ds]}   ← 没写 __iter__ 也能迭代")

# ═══ 3. mini_no_grad：双协议（with + 装饰器）═══
GRAD_ENABLED = True


class mini_no_grad:
    """同一个类，两副面孔——torch.no_grad 的设计复刻。"""

    def __enter__(self):
        global GRAD_ENABLED
        self.prev = GRAD_ENABLED       # 保存原值：嵌套使用才正确
        GRAD_ENABLED = False

    def __exit__(self, exc_type, exc, tb):
        global GRAD_ENABLED
        GRAD_ENABLED = self.prev       # 异常也保证恢复（with 的语义）

    def __call__(self, fn):            # 被当装饰器用时：包住函数
        def wrapped(*args, **kwargs):
            with self.__class__():     # 复用自己的 with 逻辑
                return fn(*args, **kwargs)
        return wrapped


print(f"\n初始 GRAD_ENABLED = {GRAD_ENABLED}")
with mini_no_grad():
    print(f"with 块内       = {GRAD_ENABLED}")
print(f"with 块外       = {GRAD_ENABLED}   ← 自动恢复")


@mini_no_grad()                        # 有括号：先实例化，再把函数传给 __call__
def evaluate():
    return f"函数体内 GRAD_ENABLED = {GRAD_ENABLED}"


print(evaluate())
print(f"函数返回后      = {GRAD_ENABLED}")
