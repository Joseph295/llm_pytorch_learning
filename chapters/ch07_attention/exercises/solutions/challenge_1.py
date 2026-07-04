"""挑战 1 参考答案：GQA（分组查询注意力）——expand 零拷贝版

运行：uv run chapters/ch07_attention/exercises/solutions/challenge_1.py
"""

import math

import torch
import torch.nn as nn

torch.manual_seed(0)


class GQA(nn.Module):
    """H 个 query 头共享 G 组 KV 头。G=H 退化为 MHA，G=1 退化为 MQA。"""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.H, self.G = n_heads, n_kv_heads
        self.D = d_model // n_heads
        self.rep = self.H // self.G                       # 每组 KV 服务几个 Q 头
        self.wq = nn.Linear(d_model, self.H * self.D, bias=False)
        self.wk = nn.Linear(d_model, self.G * self.D, bias=False)   # KV 投影更瘦！
        self.wv = nn.Linear(d_model, self.G * self.D, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    @staticmethod
    def repeat_kv(x: torch.Tensor, rep: int) -> torch.Tensor:
        """(B, G, T, D) → (B, G*rep, T, D)，expand 零拷贝（LLaMA 官方同款手法）。"""
        B, G, T, D = x.shape
        if rep == 1:
            return x
        x = x[:, :, None, :, :].expand(B, G, rep, T, D)   # 新维度 stride=0，零拷贝
        return x.reshape(B, G * rep, T, D)                # reshape 此处才物化（喂给 matmul 前）

    def forward(self, x):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.H, self.D).transpose(1, 2)   # (B,H,T,D)
        k = self.wk(x).view(B, T, self.G, self.D).transpose(1, 2)   # (B,G,T,D)
        v = self.wv(x).view(B, T, self.G, self.D).transpose(1, 2)
        k, v = self.repeat_kv(k, self.rep), self.repeat_kv(v, self.rep)   # (B,H,T,D)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.D)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        scores = scores.masked_fill(~mask, float("-inf"))
        out = (scores.softmax(-1) @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(out)


# ═══ 与"显式复制 KV"的朴素实现对拍 ═══
B, T, C, H, G = 2, 12, 64, 8, 2
gqa = GQA(C, H, G)
x = torch.randn(B, T, C)
fast = gqa(x)

# 朴素版：repeat_interleave 真复制
q = gqa.wq(x).view(B, T, H, C // H).transpose(1, 2)
k = gqa.wk(x).view(B, T, G, C // H).transpose(1, 2).repeat_interleave(H // G, dim=1)
v = gqa.wv(x).view(B, T, G, C // H).transpose(1, 2).repeat_interleave(H // G, dim=1)
scores = q @ k.transpose(-2, -1) / math.sqrt(C // H)
mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
naive = (scores.masked_fill(~mask, float("-inf")).softmax(-1) @ v)
naive = gqa.wo(naive.transpose(1, 2).contiguous().view(B, T, C))
print(f"expand 版与 repeat_interleave 朴素版 allclose: {torch.allclose(fast, naive, atol=1e-6)} ✓")

# ═══ KV cache 账 ═══
H_, G_, T_, D_ = 32, 8, 4096, 128
mha_kv = 2 * H_ * T_ * D_ * 2          # K+V，fp16
gqa_kv = 2 * G_ * T_ * D_ * 2
print(f"""
KV cache 账（单层单请求，H={H_}, G={G_}, T={T_}, d_head={D_}, fp16）:
  MHA: {mha_kv / 1024**2:.0f} MB   GQA: {gqa_kv / 1024**2:.0f} MB   → 省 {1 - gqa_kv / mha_kv:.0%}
32 层 × 64 并发请求: MHA {mha_kv * 32 * 64 / 1024**3:.0f} GB vs GQA {gqa_kv * 32 * 64 / 1024**3:.0f} GB
→ 推理服务的并发能力直接翻 4 倍——这就是 LLaMA-2 70B 起全面转 GQA 的原因（第 16 章续账）
注意：GQA 省的是 KV 的'状态'（cache 与投影参数），注意力计算本身的 FLOPs 不省
（K 还是要逻辑上铺满 H 个头参与 QK 乘法）。
""")
