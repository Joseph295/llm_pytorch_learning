# 第 9 章 · 练习题

题目详情见[讲义 9.8 节](../README.md#98-练习题把里程碑做深)。这些是"把里程碑做深"的开放题，
以指引与关键代码骨架为主（不是标准答案——你的语料、你的数字）。

| 题号 | 内容 | 难度 |
|---|---|---|
| 基础 1 | 换语料重训 | ★ |
| 基础 2 | resume 验证（含删 optimizer 对照） | ★ |
| 进阶 1 | 加梯度累积 | ★★ |
| 进阶 2 | scaling 迷你实验（三尺寸 val loss 曲线） | ★★ |
| 挑战 1 | KV cache 加速生成（第 16 章预习） | ★★★ |

## 关键提示

**基础 1**：只需替换 `projects/minigpt/data/raw.txt` 为任意 UTF-8 文本，删掉
`tokenizer.json`（触发重训 BPE），重跑 `prepare_data.py`。观察 `prepare_data`
末尾打印的"学到的多字节 token"如何反映新语料的高频词。

**进阶 1 骨架**（在 train.py 的循环里）：
```python
ACCUM = 4
for step in range(total_steps):
    for micro in range(ACCUM):
        x, y = get_batch(...)
        with torch.autocast(...):
            _, loss = model(x, y)
            loss = loss / ACCUM          # 第 3 章：必须除以 ACCUM
        loss.backward()                  # 不清零，梯度累加
    gnorm = clip_grad_norm_(...)         # 对累积后的总梯度裁剪一次
    opt.step(); opt.zero_grad(set_to_none=True)
```
对比点：相同"有效 batch"下，accum 版单步显存更低、wall-clock 更长——第 15 章
微调 7B 时这是在小卡上跑大 batch 的保命手段。

**挑战 1 思路**（KV cache，第 16 章正式展开）：给 `gpt_model.py` 的注意力
加一个可选的 `past_kv` 参数，缓存历史 K/V；`generate` 里每步只把**新 token**
喂进去，K/V append 到缓存，Q 只算 1 个位置。注意 RoPE 的位置索引要用
"缓存长度"而不是从 0 数。测量生成 500 token 的 wall-clock：朴素 O(T²) vs
cache O(T)，序列越长加速越明显。
