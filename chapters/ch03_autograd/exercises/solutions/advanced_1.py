"""进阶 1 参考答案：计算图打印器 + 残差双路观察

运行：uv run chapters/ch03_autograd/exercises/solutions/advanced_1.py
"""

import torch


def print_graph(t: torch.Tensor):
    """从张量出发递归打印反向图。AccumulateGrad = 叶子终点（参数）。

    坑位说明：seen 必须用 dict 持有 grad_fn 对象本身，不能只存 id()。
    grad_fn 的 Python 包装对象按需创建、不持引用即回收，id 会被后续对象
    复用，只存 id 会产生"假重复"误报（本答案初版就中了这一枪）。
    """
    seen = {}                              # id -> fn 对象（持有引用防 id 复用）

    def walk(fn, depth):
        if fn is None:
            return
        tag = ""
        if type(fn).__name__ == "AccumulateGrad":
            # AccumulateGrad 持有它服务的叶子张量
            tag = f"  ← 叶子 shape={tuple(fn.variable.shape)}"
        rep = "  " * depth + f"└─ {type(fn).__name__}{tag}"
        if id(fn) in seen and seen[id(fn)] is fn:
            print(rep + "  （↑重复节点：多条路径汇聚点）")
            return
        seen[id(fn)] = fn
        print(rep)
        for child, _ in getattr(fn, "next_functions", []):
            walk(child, depth + 1)

    walk(t.grad_fn, 0)


# 带残差连接的两层块：out = x + f(x)
lin1 = torch.nn.Linear(4, 4)
lin2 = torch.nn.Linear(4, 4)
x = torch.randn(2, 4, requires_grad=True)

hidden = torch.relu(lin1(x))
out = x + lin2(hidden)          # 残差：x 既是主路输入又走捷径
loss = out.sum()

print("带残差连接的反向图：")
print_graph(loss)

print("""
读图要点：
1. AddBackward0（残差加法）有两个上游分支——一条到 lin2 的主路，
   一条直通 x 的 AccumulateGrad：这就是"残差 = 梯度高速公路"的图证。
   反向时梯度沿两条路都走，捷径分支的梯度不经过任何权重矩阵、原样直达浅层——
   深网络能训得动（缓解梯度消失）的结构性原因（第 8 章 Transformer 残差流的理论地基）。
2. x 的 AccumulateGrad 节点在图里出现两次引用（我们标了"重复节点"）——
   两条路径的梯度在此相加，正是 3.2-② 的"多消费者求和"。
""")
