"""第 10 章 · MHA / MQA / GQA 三种注意力的 KV cache 账与质量对比

运行：uv run chapters/ch10_modern_llm/code/gqa_vs_mha.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


class VariableKVAttention(nn.Module):
    """一个类覆盖 MHA/MQA/GQA：n_kv_heads = n_heads→MHA, =1→MQA, 中间→GQA。"""

    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.H, self.KV = n_heads, n_kv_heads
        self.D = d_model // n_heads
        self.rep = n_heads // n_kv_heads
        self.wq = nn.Linear(d_model, n_heads * self.D, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.D, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.D, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.H, self.D).transpose(1, 2)
        k = self.wk(x).view(B, T, self.KV, self.D).transpose(1, 2)
        v = self.wv(x).view(B, T, self.KV, self.D).transpose(1, 2)
        if self.rep > 1:                                    # expand 零拷贝铺开（第 7 章）
            k = k[:, :, None].expand(B, self.KV, self.rep, T, self.D).reshape(B, self.H, T, self.D)
            v = v[:, :, None].expand(B, self.KV, self.rep, T, self.D).reshape(B, self.H, T, self.D)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(out.transpose(1, 2).contiguous().view(B, T, C))


def kv_cache_mb(n_kv_heads, L=32, T=4096, d_head=128, concurrency=32):
    return 2 * L * n_kv_heads * T * d_head * 2 * concurrency / 1024**2   # K+V, fp16


print("═══ KV cache 账（L=32, T=4096, d_head=128, 并发=32, fp16）═══")
print(f"{'方案':<8} | {'KV头数':>6} | {'KV cache':>10} | {'相对 MHA':>8}")
mha = kv_cache_mb(32)
for name, kv in [("MHA", 32), ("GQA-8", 8), ("GQA-4", 4), ("MQA", 1)]:
    c = kv_cache_mb(kv)
    print(f"{name:<8} | {kv:>6} | {c / 1024:>8.1f} GB | {c / mha:>7.0%}")
print("→ MQA 省到 1/32，GQA-8 省到 1/4——推理并发能力直接倍增（第 16 章续账）")

print("\n═══ 三种方案的参数量与前向一致性 ═══")
B, T, C, H = 4, 32, 256, 8
x = torch.randn(B, T, C)
for name, kv in [("MHA(kv=8)", 8), ("GQA(kv=2)", 2), ("MQA(kv=1)", 1)]:
    attn = VariableKVAttention(C, H, kv)
    n = sum(p.numel() for p in attn.parameters())
    out = attn(x)
    print(f"{name:<12} 参数量 {n:>6} | 输出 {tuple(out.shape)} | KV 投影省 {1 - kv / H:.0%} 参数")
print("→ query 投影不变，KV 投影随头数缩小：GQA 连参数都省一点（主要收益仍是推理 cache）")
