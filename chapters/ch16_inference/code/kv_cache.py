"""第 16 章 · 给注意力加 KV Cache，验证正确性 + 实测加速

运行：uv run chapters/ch16_inference/code/kv_cache.py

KV cache 把自回归生成从 O(T²) 降到 O(T)：缓存历史 K/V，每步只算新 token。
本脚本实现带 cache 的注意力，验证与无 cache 结果一致（正确性），测加速比。
"""

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class CachedAttention(nn.Module):
    """因果自注意力，支持 KV cache。RoPE 省略（聚焦 cache 机制）。"""

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.H, self.D = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, cache=None, use_cache=False):
        """cache=None + use_cache=False：训练/无缓存前向（因果 mask）。
        use_cache=True：启用缓存。prefill 传 cache=None + use_cache=True（建新缓存）；
                        decode 传上一步的 cache（追加新 K/V）。
        """
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.H, self.D).transpose(1, 2)      # (B,H,T,D)
        k = k.view(B, T, self.H, self.D).transpose(1, 2)
        v = v.view(B, T, self.H, self.D).transpose(1, 2)

        is_prefill_or_train = cache is None
        if cache is not None:                                 # decode：追加到已有缓存
            past_k, past_v = cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_cache = (k, v) if use_cache else None
        # prefill / 训练：q 与 k 等长，需因果 mask；decode：单 query 对全部 key，无需 mask
        is_causal = is_prefill_or_train and T > 1
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out), new_cache


d, H, Tgen = 128, 4, 200
attn = CachedAttention(d, H).to(device)
prompt = torch.randn(1, 8, d, device=device)

# ═══ 1. 正确性：对同一输入序列，cache 与非 cache 应给出相同的最后位置输出 ═══
# 干净的正确性测试（避免生成反馈放大差异）：固定长度序列，比较最后位置的输出。
@torch.no_grad()
def last_output_no_cache(seq):
    out, _ = attn(seq)                                     # 全序列前向
    return out[:, -1:]                                     # 取最后位置


@torch.no_grad()
def last_output_with_cache(seq):
    _, cache = attn(seq[:, :-1], use_cache=True)           # 前 T-1 个做 prefill 建 cache
    out, _ = attn(seq[:, -1:], cache=cache, use_cache=True)  # 最后 1 个用 cache
    return out


test_seq = torch.randn(1, 32, d, device=device)
diff = (last_output_no_cache(test_seq) - last_output_with_cache(test_seq)).abs().max().item()
print(f"cache vs 非 cache 的最后位置输出差异: {diff:.2e}"
      f"（{'一致 ✓ 正确性通过' if diff < 1e-4 else '异常✗ cache 实现有 bug'}）")


# 用于加速测量的生成函数（贪心式，逐 token）
@torch.no_grad()
def generate_no_cache(x, steps):
    seq = x
    for _ in range(steps):
        out, _ = attn(seq)                                 # O(T²)：每步重算全序列
        seq = torch.cat([seq, out[:, -1:]], dim=1)
    return seq


@torch.no_grad()
def generate_with_cache(x, steps):
    out, cache = attn(x, use_cache=True)                   # prefill
    seq = torch.cat([x, out[:, -1:]], dim=1)
    for _ in range(steps - 1):
        out, cache = attn(seq[:, -1:], cache=cache, use_cache=True)   # O(T)：只算新 token
        seq = torch.cat([seq, out[:, -1:]], dim=1)
    return seq

# ═══ 2. 加速比：序列越长，KV cache 优势越大 ═══
def bench(fn, *args):
    fn(*args)
    if device.type == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    fn(*args)
    if device.type == "mps":
        torch.mps.synchronize()
    return (time.perf_counter() - t0) * 1000


print(f"\n{'生成长度':>8} | {'无cache(ms)':>12} | {'有cache(ms)':>12} | {'加速':>6}")
for steps in [50, 100, 200, 400]:
    t_no = bench(generate_no_cache, prompt, steps)
    t_yes = bench(generate_with_cache, prompt, steps)
    print(f"{steps:>8} | {t_no:>12.1f} | {t_yes:>12.1f} | {t_no / t_yes:>5.1f}×")

print("""
读数（16.2-①）：
- 无 cache：每步重算整个序列的 K/V，总计算 O(T²)——生成越长每步越慢
- 有 cache：每步只算 1 个新 token 的 K/V，总计算 O(T)——每步耗时恒定
- 序列越长，加速比越大（O(T²)/O(T) = O(T)）
- 代价是显存：缓存 2·L·n_kv_heads·T·d_head 字节（第 2 章账，第 10 章 GQA 减它）
真实实现还要处理 RoPE 位置索引（易错点①：新 token 位置=cache_len 不是 0）。
""")
