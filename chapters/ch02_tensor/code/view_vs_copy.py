"""第 2 章 · view 家族的边界：什么时候零拷贝，什么时候报错，什么时候静默拷贝

运行：uv run chapters/ch02_tensor/code/view_vs_copy.py
"""

import time

import torch


def same_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()


print("═══ 1. view 三种结局 ═══")
t = torch.randn(4, 6)

v = t.view(24)
print(f"连续张量 view(24)        : 成功，零拷贝 = {same_storage(t, v)}")

tt = t.transpose(0, 1)                      # (6,4) 非连续
try:
    tt.view(24)
except RuntimeError as e:
    print(f"转置后 view(24)          : ✗ {str(e)[:62]}...")

r = tt.reshape(24)                          # reshape 兜底
print(f"转置后 reshape(24)       : 成功，但零拷贝 = {same_storage(tt, r)} ← 静默付了拷贝的钱")

r2 = t.reshape(24)                          # 能视图时 reshape 也是视图
print(f"连续张量 reshape(24)     : 成功，零拷贝 = {same_storage(t, r2)} ← 能省则省")

print("\n═══ 2. expand vs repeat：一字之差，内存天壤 ═══")
kv = torch.randn(1, 1024, 128)              # 模拟 1 个 KV 头 (heads, seq, dim)
n_rep = 8                                   # GQA: 8 个 query 头共享它

e = kv.expand(n_rep, 1024, 128)             # 零拷贝
p = kv.repeat(n_rep, 1, 1)                  # 真复制
mb = lambda t_: t_.untyped_storage().nbytes() / 1024 / 1024
print(f"expand 后 storage: {mb(e):.1f} MB（与原张量共享），repeat 后: {mb(p):.1f} MB")
print(f"→ LLaMA GQA 的 repeat_kv 用 expand 的原因：KV cache 是推理显存大头（第 16 章）")

print("\n═══ 3. 非连续布局的性能税 ═══")
n = 2048
x = torch.randn(n, n)
xt = x.t()                                  # 非连续视图


def bench(fn, iters=20):
    fn()                                    # warmup
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) * 1000 / iters


t_contig = bench(lambda: x.sum(dim=1))       # 行求和，内存顺序访问
t_strided = bench(lambda: xt.sum(dim=1))     # 逻辑同样的行求和，物理上跨步跳读
xt_mat = xt.contiguous()
t_materialized = bench(lambda: xt_mat.sum(dim=1))
print(f"连续张量按行求和   : {t_contig:6.3f} ms")
print(f"转置视图按行求和   : {t_strided:6.3f} ms   ← 跨步访问的缓存代价")
print(f"物化后按行求和     : {t_materialized:6.3f} ms   ← 要反复用就先 contiguous 摊销")

print("\n═══ 4. dtype 静默提升的三档优先级（易错点⑥）═══")
h = torch.randn(8, 8, dtype=torch.float16)
print(f"fp16 × Python 标量 0.1          → {(h * 0.1).dtype}")
print(f"fp16 × fp32 零维 tensor(0.1)    → {(h * torch.tensor(0.1)).dtype}  ← 零维按标量对待，不提升")
print(f"fp16 × fp32 一维 tensor([0.1])  → {(h * torch.tensor([0.1])).dtype}  ← 带维度就提升，显存×2")
