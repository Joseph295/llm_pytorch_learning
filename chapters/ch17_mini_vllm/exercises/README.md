# 第 17 章 · 练习题

题目详情见[讲义 17.8 节](../README.md#178-练习题)。实现基座在 `projects/mini-vllm/`。

| 题号 | 内容 | 难度 | 交付物 |
|---|---|---|---|
| 基础 1 | 块分配器 | ★ | 见 projects/mini-vllm/block_manager.py + test_paging.py |
| 基础 2 | 块表读写（跨块边界） | ★ | 见 test_paging.py 的 test_block_table_mapping |
| 进阶 1 | 完整调度循环 | ★★ | 见 scheduler.py + engine.py + demo.py |
| 挑战 1 | 前缀共享（copy-on-write） | ★★★ | 见 block_manager.py 的 share/ref_count |

`projects/mini-vllm/test_paging.py` 已覆盖基础 1/2 和挑战 1 的引用计数基础。
进阶 1 的完整调度循环在 `engine.py` + `demo.py`（多请求并发，观察动态进出）。

## 挑战 1 扩展方向（前缀共享的完整实现）

`block_manager.py` 的 `share()` + `ref_count` 是 copy-on-write 的基础。完整实现：
1. 多个请求有相同前缀（如同一 system prompt）时，prefill 阶段共享前缀的物理块
   （`share()` 增加引用计数，不分配新块）；
2. 当某请求要写入共享块（分叉点）时，先 copy-on-write（复制到新块，引用计数递减）；
3. 验证：共享后总块数减少；某请求的写入不影响其他共享请求。

这就是 vLLM 的 prefix caching（automatic prefix caching）和 SGLang 的 RadixAttention
的核心——多请求共享 KV cache 前缀，在多轮对话/相同 system prompt 场景省大量显存。
