# 第 16 章 · 练习题

题目详情见[讲义 16.8 节](../README.md#168-练习题)。先自己做，再对照 `solutions/`。

| 题号 | 内容 | 难度 | 交付物 |
|---|---|---|---|
| 基础 1 | 给 miniGPT 加 KV cache | ★ | 见 code/kv_cache.py + 真实模型集成 |
| 基础 2 | prefill/decode 特征测量 | ★ | 见 code/prefill_decode.py |
| 进阶 1 | 手写 int8 量化 | ★★ | 见 code/quantization.py + per-tensor/channel 对比 |
| 挑战 1 | 投机解码模拟 | ★★★ | 见 code/speculative.py + 命中率-加速关系 |

基础/进阶实现基座已在 `code/` 提供。基础 1 的进阶版是把 KV cache 集成到
第 8 章的真实 GPT（含 RoPE 位置索引处理，易错点①）——这也是第 17 章
mini-vLLM 的前置。
