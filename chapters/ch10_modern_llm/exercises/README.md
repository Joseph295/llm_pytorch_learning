# 第 10 章 · 练习题

题目详情见[讲义 10.8 节](../README.md#108-练习题)。先自己做，再对照 `solutions/`。

| 题号 | 内容 | 难度 | 交付物 |
|---|---|---|---|
| 基础 1 | miniGPT 升级 GQA | ★ | 改模型 + val loss/cache 对比 |
| 基础 2 | SwiGLU 替换 FFN | ★ | 见 code/swiglu.py + 集成到 miniGPT |
| 进阶 1 | MoE 负载均衡可视化 | ★★ | 见 code/moe_toy.py 扩展 |
| 挑战 1 | 加载真实 LLaMA 权重对拍 | ★★★ | 见 solutions/challenge_1.md 指引 |

基础 2 / 进阶 1 的核心实现已在本章 `code/` 给出（swiglu.py / moe_toy.py），
练习是把它们集成回 miniGPT 并观察端到端效果。挑战 1 见下方指引文档。
