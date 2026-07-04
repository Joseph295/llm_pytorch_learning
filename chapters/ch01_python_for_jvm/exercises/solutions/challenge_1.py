"""挑战 1 参考答案：双协议的 no_grad 平替（with + 装饰器，支持嵌套与异常恢复）

运行：uv run chapters/ch01_python_for_jvm/exercises/solutions/challenge_1.py

考点拆解：
1. 双协议 = __enter__/__exit__（with）+ __call__ 返回包装函数（装饰器）
2. "恢复原值"而不是"置回 True"——嵌套时后者会提前打开开关，语义错误
3. 异常路径的恢复由 with 语义天然保证（__exit__ 必被调用）
4. 每次进入要开新实例/新状态槽，否则同一实例重入会互相覆盖 prev
"""

GRAD_ENABLED = True


class my_mode:
    def __enter__(self):
        global GRAD_ENABLED
        self.prev = GRAD_ENABLED          # 保存"进入时"的值——嵌套正确性的关键
        GRAD_ENABLED = False
        return self

    def __exit__(self, exc_type, exc, tb):
        global GRAD_ENABLED
        GRAD_ENABLED = self.prev          # 恢复原值；返回 None（不吞异常）

    def __call__(self, fn):
        import functools

        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            # 每次调用创建新实例：装饰的函数可能递归/并发调用，
            # 复用 self 会让 prev 被覆盖
            with self.__class__():
                return fn(*args, **kwargs)
        return wrapped


# ── 测试 1：基本 with ──
assert GRAD_ENABLED
with my_mode():
    assert not GRAD_ENABLED
assert GRAD_ENABLED
print("基本 with           ✓")

# ── 测试 2：嵌套——内层退出不应提前恢复成 True ──
with my_mode():
    with my_mode():
        assert not GRAD_ENABLED
    assert not GRAD_ENABLED, "内层退出后应恢复为'外层进入后'的 False，而不是 True！"
assert GRAD_ENABLED
print("嵌套恢复原值        ✓")

# ── 测试 3：异常时也恢复 ──
try:
    with my_mode():
        raise ValueError("boom")
except ValueError:
    pass
assert GRAD_ENABLED, "异常路径 __exit__ 也必须执行"
print("异常路径恢复        ✓")

# ── 测试 4：装饰器用法 ──
@my_mode()
def evaluate():
    assert not GRAD_ENABLED
    return "ok"

assert evaluate() == "ok" and GRAD_ENABLED
print("装饰器协议          ✓")

# ── 测试 5：被装饰函数递归调用（新实例策略的必要性）──
@my_mode()
def recurse(n):
    assert not GRAD_ENABLED
    return 0 if n == 0 else recurse(n - 1)

recurse(3)
assert GRAD_ENABLED
print("递归调用安全        ✓")

print("\n全部通过。对照阅读：torch.autograd.grad_mode 里的 no_grad 实现，")
print("真身用 ContextDecorator + C++ 层的线程本地开关（TLS），思想与此完全一致——")
print("线程本地是因为真实场景有多线程，全局变量会串台（我们的玩具版没处理这点）。")
