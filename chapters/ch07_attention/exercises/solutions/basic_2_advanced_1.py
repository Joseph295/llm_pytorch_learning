"""基础 2 + 进阶 1 参考答案：padding mask 支持 与 cross-attention

运行：uv run chapters/ch07_attention/exercises/solutions/basic_2_advanced_1.py
"""

import math

import torch
import torch.nn as nn

torch.manual_seed(0)


class MHA(nn.Module):
    """多头注意力：支持 self/cross 两用 + causal/padding 两种 mask。

    进阶 1 的答案就藏在签名里：forward(x_q, x_kv)——self-attention 只是
    x_q is x_kv 的特例。cross 模式下 Q 的长度 Tq 与 KV 的长度 Tk 可以不同。
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.H, self.D = n_heads, d_model // n_heads
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x_q, x_kv=None, causal=True, kv_padding_mask=None):
        x_kv = x_q if x_kv is None else x_kv
        B, Tq, C = x_q.shape
        Tk = x_kv.size(1)
        q = self.wq(x_q).view(B, Tq, self.H, self.D).transpose(1, 2)    # (B,H,Tq,D)
        k = self.wk(x_kv).view(B, Tk, self.H, self.D).transpose(1, 2)   # (B,H,Tk,D)
        v = self.wv(x_kv).view(B, Tk, self.H, self.D).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.D)            # (B,H,Tq,Tk)
        if causal:
            cm = torch.tril(torch.ones(Tq, Tk, dtype=torch.bool, device=x_q.device))
            scores = scores.masked_fill(~cm, float("-inf"))
        if kv_padding_mask is not None:                                 # (B,Tk) True=真实
            pm = kv_padding_mask[:, None, None, :]                      # 广播到 (B,1,1,Tk)
            scores = scores.masked_fill(~pm, float("-inf"))
        out = (scores.softmax(-1) @ v).transpose(1, 2).contiguous().view(B, Tq, C)
        return self.wo(out)


mha = MHA(32, 4)

# ═══ 基础 2 验证：padding 位置零影响 ═══
B, T = 2, 6
x_short = torch.randn(B, 4, 32)                       # "真实"序列长 4
x_padded = torch.cat([x_short, torch.randn(B, 2, 32)], dim=1)   # 补 2 个垃圾位
pad_mask = torch.tensor([[1, 1, 1, 1, 0, 0]] * B, dtype=torch.bool)

out_padded = mha(x_padded, kv_padding_mask=pad_mask)[:, :4]     # 只看真实位置
out_ref = mha(x_short)
print(f"基础2: padding 版前 4 位输出 == 无 padding 版: "
      f"{torch.allclose(out_padded, out_ref, atol=1e-6)} ✓")
print("→ mask 生效 = 垃圾位对真实位置的注意力权重为 0，输出与从未见过它们一致")

# ═══ 进阶 1 验证：cross-attention，长度不同 ═══
q_seq = torch.randn(B, 5, 32)                         # 解码侧 5 个位置
kv_seq = torch.randn(B, 9, 32)                        # 编码侧 9 个位置
out = mha(q_seq, kv_seq, causal=False)
print(f"\n进阶1: cross-attention (Tq=5, Tk=9) → {tuple(out.shape)} ✓")

print("""
cross-attention 需要因果掩码吗？——看 KV 序列的语义：
- KV 是"已完成的条件"（翻译的源句、图像特征）：不需要。条件整体可见，
  没有"未来"可言（经典 encoder-decoder 的 cross 层不加 causal）。
- KV 是"另一条也在生成的流"（如语音-文本同步生成）：需要按时间对齐的
  mask——本质还是"不许看尚未发生的东西"，只是"发生"的时钟在另一条序列上。
判断准则永远是信息因果，不是层的名字。
""")
