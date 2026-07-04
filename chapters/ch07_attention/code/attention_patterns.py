"""第 7 章 · 注意力学什么：玩具任务上可视化学到的模式

运行：uv run chapters/ch07_attention/code/attention_patterns.py

任务设计：序列里藏一个"指针"结构——每个位置的正确输出是它*两步之前*
位置的 token。模型必须学会"看固定偏移"的注意力模式才能解题。
训练后打印注意力矩阵，亲眼看到模式形成。
"""

import math

import torch
import torch.nn as nn

torch.manual_seed(1)

VOCAB, D, T = 16, 32, 8


class TinyAttnLM(nn.Module):
    """embedding + 单头因果注意力 + 输出头。刻意极简，让注意力独扛任务。"""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D)
        self.pos = nn.Embedding(T, D)                 # 可学位置编码（第 8 章讲 RoPE 前的临时方案）
        self.wq = nn.Linear(D, D, bias=False)
        self.wk = nn.Linear(D, D, bias=False)
        self.wv = nn.Linear(D, D, bias=False)
        self.head = nn.Linear(D, VOCAB)

    def forward(self, idx, return_attn=False):
        B, T_ = idx.shape
        x = self.embed(idx) + self.pos(torch.arange(T_, device=idx.device))
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        scores = q @ k.transpose(-2, -1) / math.sqrt(D)
        mask = torch.tril(torch.ones(T_, T_, dtype=torch.bool))
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = scores.softmax(-1)
        logits = self.head(attn @ v)
        return (logits, attn) if return_attn else logits


def make_batch(bsz=64):
    x = torch.randint(0, VOCAB, (bsz, T))
    y = torch.full_like(x, -100)                      # -100 = cross_entropy 的忽略标签
    y[:, 2:] = x[:, :-2]                              # 目标：两步前的 token
    return x, y


model = TinyAttnLM()
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
for step in range(400):
    x, y = make_batch()
    logits = model(x)
    loss = nn.functional.cross_entropy(logits.view(-1, VOCAB), y.view(-1), ignore_index=-100)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 100 == 0 or step == 399:
        print(f"step {step:>3}: loss={loss.item():.4f}")

x, y = make_batch(1)
logits, attn = model(x, return_attn=True)
acc = (logits.argmax(-1)[0, 2:] == y[0, 2:]).float().mean()
print(f"\n任务准确率: {acc:.0%}")
print("\n学到的注意力矩阵（行=query 位置，列=key 位置）：")
print("（若任务学会，位置 t 应把注意力集中在 t-2 —— 一条向左偏移 2 的对角线）")
for i, row in enumerate(attn[0].round(decimals=2).tolist()):
    marks = " ".join(f"{v:4.2f}" if v > 0 else " .  " for v in row)
    peak = max(range(len(row)), key=lambda j: row[j])
    print(f"  q{i}: {marks}   ← 最大注意力在 k{peak}" + ("  ✓" if peak == max(i - 2, 0) else ""))
print("\n→ 注意力权重不是玄学：任务需要什么检索模式，训练就把它刻进 QK 的几何里")
