# 第 12 章 · 练习题

题目详情见[讲义 12.8 节](../README.md#128-练习题)。先自己做，再对照 `solutions/`。

| 题号 | 内容 | 难度 | 交付物 |
|---|---|---|---|
| 基础 1 | 手动 AllReduce 验证 | ★ | 见 code/allreduce_demo.py |
| 基础 2 | DDP vs 单卡等价性 | ★ | 见 code/ddp_minigpt.py + 对比 |
| 进阶 1 | 手写 Ring AllReduce | ★★ | 见 code/ring_allreduce.py + 通信量分析 |
| 挑战 1 | 云端 NCCL 实战 + scaling 分析 | ★★★ | 见 solutions/challenge_1.md 指引 |

基础/进阶的实现基座已在 `code/` 提供（gloo 后端，M4 免费跑）。挑战 1 是云端实战，
见指引文档。
