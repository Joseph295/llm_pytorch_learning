"""基础 1 参考答案：手写 @timer 装饰器

运行：uv run chapters/ch01_python_for_jvm/exercises/solutions/basic_1.py
"""

import functools
import time

import torch


def timer(fn):
    @functools.wraps(fn)               # 不加这行，matmul_demo.__name__ 会变成 'wrapped'
    def wrapped(*args, **kwargs):      # *args/**kwargs 透传 → 函数和方法通吃
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        print(f"[timer] {fn.__name__}: {(time.perf_counter() - t0) * 1000:.2f} ms")
        return out
    return wrapped


@timer
def matmul_demo(n: int = 1024):
    a = torch.randn(n, n)
    return (a @ a).sum().item()        # .item() 强制取回结果（CPU 上本就同步）


class Trainer:
    @timer                             # 方法也能用：self 走 *args 第一位
    def step(self):
        return matmul_demo(512)


matmul_demo()
Trainer().step()
assert matmul_demo.__name__ == "matmul_demo", "functools.wraps 保住了元信息"
print(f"元信息保留验证: matmul_demo.__name__ = {matmul_demo.__name__!r} ✓")
