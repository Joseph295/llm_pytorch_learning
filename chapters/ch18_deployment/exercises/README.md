# 第 18 章 · 练习题

题目详情见[讲义 18.8 节](../README.md#188-练习题)。这些是部署实战题，以指引为主。

| 题号 | 内容 | 难度 | 交付物 |
|---|---|---|---|
| 基础 1 | M4 上跑量化 LLM | ★ | 见 code/mlx_run.py（装 mlx-lm）+ 测量报告 |
| 基础 2 | TTFT/TPOT/吞吐权衡 | ★ | 见 code/serving_metrics.py + 曲线分析 |
| 进阶 1 | 部署你微调的模型 | ★★ | 云端 vLLM 或本地 GGUF，行为一致性对比 |
| 挑战 1 | 端到端本地服务 | ★★★ | M4 上 MLX/Ollama + Web UI，端到端延迟测量 |

## 实战提示

**基础 1**：`uv pip install mlx-lm`，然后 `mlx_run.py` 会下载并运行量化模型。
测量加载时间、生成速度（token/s）、内存（活动监视器看统一内存占用）。
对比 4bit vs 8bit 的大小/速度/质量。

**进阶 1**：把第 15 章微调的 QLoRA adapter 合并进 base（`peft` 的 `merge_and_unload`），
- 云端：`vllm serve <merged-model>` 起服务，用 OpenAI 客户端测试
- 本地：转 GGUF（llama.cpp 的 `convert_hf_to_gguf.py` + `llama-quantize`），
  用 llama.cpp 或 Ollama 跑
关键验证：部署后的输出和微调时一致（对话模板/tokenizer 对齐，易错点⑥）。

**挑战 1**：`ollama create` + `ollama run`（封装 llama.cpp）或 `mlx_lm.server`（OpenAI 兼容），
配一个简单前端（甚至 curl OpenAI 接口）。测端到端延迟，讨论端侧 vs 云端的场景取舍。
