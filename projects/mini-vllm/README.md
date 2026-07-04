# mini-vLLM — 里程碑三：手写推理引擎

配套教程：[第 17 章](../../chapters/ch17_mini_vllm/README.md)。第四篇（推理）的核心交卷。

## 运行

```bash
uv run projects/mini-vllm/test_paging.py   # 块管理单元测试（CPU，秒级）
uv run projects/mini-vllm/demo.py          # 多请求并发生成，观察调度
```

## 文件

| 文件 | 作用 | 对应 vLLM | 章节 |
|---|---|---|---|
| `block_manager.py` | 块分配器 + 块表（PagedAttention） | `core/block_manager.py` | 17.2-② |
| `scheduler.py` | continuous batching 调度器 | `core/scheduler.py` | 17.2-① |
| `engine.py` | 主循环（调度→前向→释放） | `engine/llm_engine.py` | 17.2-③ |
| `demo.py` | 多请求并发演示 | — | — |
| `test_paging.py` | 块管理单元测试 | — | — |

## 核心思想（你的系统背景是主场）

- **continuous batching** = token 级动态任务调度（请求随到随处理，完成即退出）
- **PagedAttention** = 操作系统虚拟内存搬到 KV cache（块表=页表，物理块=物理页）
- **前缀共享** = copy-on-write（引用计数管理共享块）

手写的简化版抓住了 vLLM 的核心结构。读懂它后，去读 vLLM 真实源码
（`core/scheduler.py` / `core/block_manager.py`）会毫无障碍——那才是本章的终极目标。

## 与真实 vLLM 的差距

- 本项目每请求独立前向（教学）；vLLM 用变长注意力 kernel 批量处理不 padding
- 本项目块管理器只做账本；vLLM 块表直接索引 CUDA attention kernel
- vLLM 还有：抢占换出、chunked prefill、prefix caching、量化/并行集成、CUDA 优化
