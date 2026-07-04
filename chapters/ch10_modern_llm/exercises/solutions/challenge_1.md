# 挑战 1 参考答案：加载真实 LLaMA 权重对拍

目标：让你的 miniGPT 实现与 HF `transformers` 的 LlamaModel 完全一致，
能加载一个真实小模型（Qwen2.5-0.5B / TinyLlama / SmolLM-135M）的权重并前向对拍。
这是"读懂源码"的终极验收——数值对不上就说明你的实现和标准有差异。

## 为什么这是硬核验收

参数量对、shape 对都不够——**数值对拍才能暴露约定层面的错位**（10.7 案例 1 的四类坑：
QKV 排布、RoPE 配对方式、RMSNorm eps/精度、SwiGLU 门顺序）。能对上，说明你的
每一个约定都和标准实现一致。

## 步骤指引

**1. 拿一个小模型的权重**
```bash
uv run hf download Qwen/Qwen2.5-0.5B --local-dir /tmp/qwen05
# 或 HuggingFaceTB/SmolLM-135M（更小，M4 更快）
```

**2. 对齐配置**（读 `config.json`）：`hidden_size`/`num_hidden_layers`/
`num_attention_heads`/`num_key_value_heads`(GQA!)/`intermediate_size`/
`rms_norm_eps`/`rope_theta`/`vocab_size`/`tie_word_embeddings`。

**3. 逐层对拍的黄金方法**（不要指望一次全对）：
```python
from transformers import AutoModelForCausalLM
import torch

ref = AutoModelForCausalLM.from_pretrained("/tmp/qwen05", torch_dtype=torch.float32)
ref.eval()
ids = torch.tensor([[1, 2, 3, 4, 5]])

# 用 hook 抓官方实现每一层的输出（第 4 章 hook）
acts = {}
for name, m in ref.named_modules():
    m.register_forward_hook(lambda mod, i, o, n=name: acts.__setitem__(n, o))
with torch.no_grad():
    ref(ids)

# 你的实现喂相同权重、相同输入，逐层比对，找第一个 diverge 的层
# 第一个对不上的层 = 你的约定和官方不同的地方
```

## 四个最常见的 diverge 点与修法

| 症状（第一个发散的层） | 病根 | 修法 |
|---|---|---|
| 第一个 attention 就错 | QKV 权重排布约定：官方权重按 `(n_heads, head_dim, hidden)` 组织，拆头顺序要匹配 | 对齐 `view` 的维度顺序；必要时 permute 权重 |
| attention 数值接近但不等 | RoPE 的 half-rotation 配对：LLaMA 用"前后半分"(`rotate_half`)，某些实现用"交错" | 确认 `rotate_half` 与官方一致（我们的 gpt_model 已是 LLaMA 式） |
| RMSNorm 后有微小 diff | eps 值或是否上转 fp32 | 对齐 `rms_norm_eps`，内部 fp32 计算 |
| FFN 输出错 | SwiGLU 的 gate/up 顺序、或 GQA 的 KV 复制在 RoPE 前后 | gate 走 SiLU，KV 复制在 RoPE 之后（易错点①） |

## 验收标准

```python
# 全模型 logits 对拍
mine_logits = my_model(ids)
ref_logits = ref(ids).logits
print(torch.allclose(mine_logits, ref_logits, atol=1e-3))  # fp32 下应为 True
```

`atol=1e-3` 通过 = 你的实现与官方在数值上等价。这时你可以自信地说：
**"我不是会用 LLaMA，我能重新实现 LLaMA。"** 这正是本教程第二篇的终点。

## 提示

- 从最小的模型开始（SmolLM-135M），层少好排查。
- 先关掉所有 dropout、用 fp32、eval 模式（排除随机性与精度噪声）。
- Qwen2.5 有 QK-bias 和 QK-Norm 等细节（第 10.2-⑤），比纯 LLaMA 多几个部件——
  想少踩坑先选纯 LLaMA 架构的模型（TinyLlama）。
