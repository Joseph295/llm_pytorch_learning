"""第 7 章 · 注意力从零构建：单头 → 因果 → 多头，与官方 SDPA 对拍

运行：uv run chapters/ch07_attention/code/attention_from_scratch.py

MultiHeadAttention 类是第 8 章 Transformer 的核心组件。
shape 注释是纪律，不是装饰（易错点④的双保险之一；另一个是最后的对拍）。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# ═══ 1. 单头注意力：公式的直译 ═══


def single_head_attention(q, k, v):
    """q,k,v: (T, d) → (T, d)。三行公式直译。"""
    scores = q @ k.T / math.sqrt(q.size(-1))     # (T, T)  相似度矩阵
    weights = scores.softmax(dim=-1)             # (T, T)  每行和为 1 的检索权重
    return weights @ v                           # (T, d)  加权取回 value


T, d = 5, 8
x = torch.randn(T, d)
out = single_head_attention(x, x, x)
print(f"单头（无投影版）: {tuple(x.shape)} → {tuple(out.shape)}")

# ═══ 2. 因果掩码 ═══


def causal_attention(q, k, v):
    T = q.size(0)
    scores = q @ k.T / math.sqrt(q.size(-1))
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
    scores = scores.masked_fill(~mask, float("-inf"))   # softmax 之前！（易错点②）
    return scores.softmax(dim=-1) @ v, scores.softmax(dim=-1)


out, w = causal_attention(x, x, x)
print(f"\n因果注意力权重（下三角，每行和=1）:\n{w.round(decimals=2)}")

# ═══ 3. 多头注意力：第 8 章的正式组件 ═══


class MultiHeadAttention(nn.Module):
    """因果多头自注意力。与 F.scaled_dot_product_attention 语义对齐。"""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.H = n_heads
        self.D = d_model // n_heads              # 每头维度（√d 缩放用它！易错点①）
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.H, self.D).transpose(1, 2)   # (B, H, T, D)
        k = self.wk(x).view(B, T, self.H, self.D).transpose(1, 2)   # (B, H, T, D)
        v = self.wv(x).view(B, T, self.H, self.D).transpose(1, 2)   # (B, H, T, D)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.D)        # (B, H, T, T)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        scores = scores.masked_fill(~mask, float("-inf"))           # (T,T) 广播到 (B,H,T,T)
        attn = scores.softmax(dim=-1)                               # (B, H, T, T)

        out = attn @ v                                              # (B, H, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, C)        # (B, T, C) ← 第 2 章实战：
        return self.wo(out)                                         # transpose 后必须 contiguous


B, T, C, H = 4, 10, 64, 8
mha = MultiHeadAttention(C, H)
x = torch.randn(B, T, C)
mine = mha(x)
print(f"\n多头: {tuple(x.shape)} → {tuple(mine.shape)}")

# ═══ 4. 与官方 SDPA 对拍（实现完成的验收标准）═══
q = mha.wq(x).view(B, T, H, C // H).transpose(1, 2)
k = mha.wk(x).view(B, T, H, C // H).transpose(1, 2)
v = mha.wv(x).view(B, T, H, C // H).transpose(1, 2)
ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)       # 官方统一入口
ref = mha.wo(ref.transpose(1, 2).contiguous().view(B, T, C))
print(f"与 F.scaled_dot_product_attention 对拍 allclose: "
      f"{torch.allclose(mine, ref, atol=1e-5)} ✓")

# ═══ 5. 因果泄漏测试（7.7 案例 1 的测试化）═══
x2 = x.clone()
x2[:, 5:, :] = torch.randn_like(x2[:, 5:, :])        # 篡改位置 5 之后的全部输入
leak = (mha(x)[:, :5] - mha(x2)[:, :5]).abs().max().item()
print(f"篡改未来后，前 5 个位置的输出变化: {leak:.2e}（应为 0——未来对过去不可见）✓")
