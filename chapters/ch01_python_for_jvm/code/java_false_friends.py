"""第 1 章 · Java 假朋友逐个演示

运行：uv run chapters/ch01_python_for_jvm/code/java_false_friends.py

每个小节演示一个"长得像 Java、行为不一样"的特性（对应讲义 1.2/1.4 节）。
"""

import torch


def demo(title: str):
    print(f"\n{'─' * 56}\n▶ {title}\n{'─' * 56}")


# ═══ 1. 一切皆引用：赋值不拷贝 ═══
demo("赋值是别名，不是拷贝（Java 基本类型直觉在此失效）")
a = torch.zeros(2, 2)
b = a               # 别名
c = a.clone()       # 真副本
b[0, 0] = 999
print(f"改 b 后 a[0,0] = {a[0, 0].item()}   ← a 跟着变（同一对象）")
print(f"       c[0,0] = {c[0, 0].item()}   ← clone 的不受影响")

# ═══ 2. is vs ==：身份与内容 ═══
demo("is 比身份，== 被张量重载成逐元素比较")
x = torch.tensor([1, 2])
y = x.clone()
print(f"x == y      -> {x == y}   (是张量！不是布尔)")
print(f"x is y      -> {x is y}")
print(f"torch.equal -> {torch.equal(x, y)}   (数值全等判断用这个)")
try:
    if x == y:  # 多元素张量转 bool 是歧义的
        pass
except RuntimeError as e:
    print(f"if x == y:  -> RuntimeError: {str(e)[:48]}...")

# ═══ 3. 可变默认参数：定义时求值一次 ═══
demo("可变默认参数在函数定义时创建一次，所有调用共享")


def bad_append(v, logs=[]):  # noqa: B006 —— 故意演示反模式
    logs.append(v)
    return logs


print(f"第一次调用: {bad_append(1)}")
print(f"第二次调用: {bad_append(2)}   ← 上次的 [1] 还在！")


def good_append(v, logs=None):
    logs = [] if logs is None else logs
    logs.append(v)
    return logs


print(f"修正版两次: {good_append(1)} {good_append(2)}")

# ═══ 4. 闭包晚绑定 ═══
demo("闭包捕获变量而非值（Java lambda 的 effectively final 帮你挡过的坑）")
hooks_bad = [lambda: i for i in range(3)]
hooks_good = [lambda i=i: i for i in range(3)]  # 默认参数在定义时固化
print(f"晚绑定: {[h() for h in hooks_bad]}   ← 全是循环结束时的 2")
print(f"固化后: {[h() for h in hooks_good]}")

# ═══ 5. 类型提示不设防 ═══
demo("type hints 只是注释，运行时不检查")


def double(t: torch.Tensor) -> torch.Tensor:
    return t + t  # hints 说要 Tensor，但传啥都不拦


print(f'double("ab") = {double("ab")!r}   ← 没报错，字符串被"加倍"成拼接')
print('double([1,2]) =', double([1, 2]), '  ← 列表拼接，同样静默荒谬')

# ═══ 6. 广播的静默陷阱（易错点⑥，最贵的 bug）═══
demo("广播静默给出形状合法、语义错误的结果")
loss_per_token = torch.ones(4, 8)      # (batch=4, seq=8)
w_batch = torch.arange(4.0)            # 想按样本加权: (4,)
try:
    loss_per_token * w_batch           # (4,8)*(4,) 从右对齐 8 vs 4 -> 报错（走运）
except RuntimeError as e:
    print(f"(4,8)*(4,)  -> 报错（这算走运）: {str(e)[:50]}...")
w_wrong = torch.arange(8.0)            # (8,) 恰好匹配 seq 维
silent = loss_per_token * w_wrong      # 不报错！但语义成了"按 token 位置加权"
print(f"(4,8)*(8,)  -> 静默成功，shape={tuple(silent.shape)} ← 语义错了没人知道")
correct = loss_per_token * w_batch[:, None]   # (4,1) 显式表达"按行广播"
print(f"(4,8)*(4,1) -> shape={tuple(correct.shape)} ← [:, None] 显式表达意图")
