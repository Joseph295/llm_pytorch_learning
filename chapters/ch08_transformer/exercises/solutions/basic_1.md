# 基础 1 参考答案：GPT-2 medium 参数量对账

配置：L=24, d=1024, V=50257（原版 GPT-2 medium，标称 355M）。

**手算（我们的公式 12Ld² + Vd）：**

```
主干:      12 × 24 × 1024²  = 301,989,888  ≈ 302.0M
embedding: 50257 × 1024     =  51,463,168  ≈  51.5M
合计（tie 后）              ≈ 353.5M
```

与标称 355M 的差（约 1.5M）来自：GPT-2 用 LayerNorm（每层 2 组 γ+β ≈ 4d/层）、
全部 Linear 带 bias、以及 learned position embedding（1024×1024 ≈ 1.0M）——
这些小项我们的现代化实现（RMSNorm 无 bias、RoPE 无参数）都省掉了。

**代码对账（用本章 gpt_model，同 L/d/V）：**

```python
from gpt_model import GPT, GPTConfig
cfg = GPTConfig(vocab_size=50257, block_size=1024, n_layer=24, n_head=16, n_embd=1024)
m = GPT(cfg)
print(f"{m.num_params()/1e6:.1f}M")   # ≈ 353.5M，与手算一致（差异仅 norm 的 γ）
```

**心算模板（面试现场就这三步）：**
1. `12 L d²`（记住 12 = 注意力 4 + FFN 8）
2. `+ V·d`（tie 后只算一份；不 tie 要 ×2）
3. SwiGLU 架构把 12 换成 ~14（FFN 三矩阵 3×(8/3)d² = 8d²…实际 LLaMA 系数略高），粗估仍用 12 起步
