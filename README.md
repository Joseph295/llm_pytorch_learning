# PyTorch 从入门到专家（大模型方向）

一份为**大数据/分布式系统工程师**量身定制的 PyTorch 系统教程：用你已有的分布式心智模型做类比桥接，文档 + 可运行代码双轨，每章八段式（来龙去脉 / 核心原理 / 动手实验 / 易错点 / 开源最佳实践 / 面试题 / 疑难排查 / 练习题）。

完整设计见 [设计文档](docs/superpowers/specs/2026-07-03-pytorch-llm-tutorial-design.md)。
跨章疑难杂症索引见 [troubleshooting.md](docs/troubleshooting.md)。

## 快速开始

```bash
uv sync                                              # 精确复现环境（Python 3.12 + torch）
uv run chapters/ch00_environment/code/check_env.py   # 环境体检
```

## 路线图

### 第〇篇 · 起步
- [x] [第 0 章 · 环境与工具链](chapters/ch00_environment/README.md)
- [x] [第 1 章 · 写给 JVM 工程师的 Python 速成](chapters/ch01_python_for_jvm/README.md)

### 第一篇 · PyTorch 核心
- [x] [第 2 章 · 张量的本质](chapters/ch02_tensor/README.md)
- [x] [第 3 章 · Autograd 解剖](chapters/ch03_autograd/README.md)
- [x] [第 4 章 · nn.Module 体系](chapters/ch04_nn_module/README.md)
- [x] [第 5 章 · 数据管线](chapters/ch05_data_pipeline/README.md)
- [x] [第 6 章 · 训练循环完全解剖](chapters/ch06_training_loop/README.md)

### 第二篇 · Transformer 与 LLM 构建
- [x] [第 7 章 · 注意力机制从零推导](chapters/ch07_attention/README.md)
- [x] [第 8 章 · 完整 Transformer 实现](chapters/ch08_transformer/README.md)
- [x] [第 9 章 · 🏆 里程碑一：从零预训练 miniGPT](chapters/ch09_minigpt/README.md)
- [x] [第 10 章 · 现代 LLM 架构演进](chapters/ch10_modern_llm/README.md)

### 第三篇 · 训练工程
- [x] [第 11 章 · 单卡性能优化](chapters/ch11_performance/README.md)
- [x] [第 12 章 · 分布式训练原理](chapters/ch12_distributed/README.md)
- [x] [第 13 章 · 大模型并行策略](chapters/ch13_parallelism/README.md)
- [x] [第 14 章 · 微调工程](chapters/ch14_finetuning/README.md)
- [x] [第 15 章 · 🏆 里程碑二：云端微调 7B + 疑难杂症手册](chapters/ch15_finetune_7b/README.md)

### 第四篇 · 推理与部署
- [x] [第 16 章 · 推理优化原理](chapters/ch16_inference/README.md)
- [x] [第 17 章 · 🏆 里程碑三：mini-vLLM](chapters/ch17_mini_vllm/README.md)
- [x] [第 18 章 · 服务化与端侧部署](chapters/ch18_deployment/README.md)

### 第五篇 · 专家纵深
- [x] [第 19 章 · PyTorch 内部机制](chapters/ch19_pytorch_internals/README.md)
- [x] [第 20 章 · 面试全景与综合复盘](chapters/ch20_interview/README.md)

## 里程碑项目（作品集）

| 项目 | 内容 | 章节 |
|---|---|---|
| [minigpt](projects/minigpt/) | 从零 BPE + 数据管线 + 预训练 GPT | 第 9 章 |
| [finetune-7b](projects/finetune-7b/) | 云端 QLoRA 微调 7B（CUDA） | 第 15 章 |
| [mini-vllm](projects/mini-vllm/) | 手写推理引擎（continuous batching + PagedAttention） | 第 17 章 |

## 硬件路径

- 本地 Apple Silicon（MPS）可完成：第 0~11、16、18 章及里程碑一、三的逻辑层
- 需按需租云 GPU（讲义内标注预估费用）：第 12、13、15 章及里程碑二

## 学习约定

1. 每章按讲义顺序读 + 跑 `code/` + 做 `exercises/`（基础题必做），再进入下一章
2. 练习答案写在各章 `exercises/` 下并提交——本仓库就是你的学习档案
3. 环境出问题，第一步永远是 `uv run chapters/ch00_environment/code/check_env.py`
4. 遇到疑难杂症查 [troubleshooting.md](docs/troubleshooting.md)

## 设计特色

- **分布式类比桥接**：每章用你已精通的分布式系统概念（AllReduce↔shuffle、PagedAttention↔虚拟内存、checkpoint↔容错快照）建立心智模型
- **第一性原理**：每个技术都问"为解决什么问题、代价是什么、和已知的什么相通"，不背结论
- **真实踩坑**：配套代码全部实跑验证，过程中的真实 bug（如里程碑一的 loss spike 事故：β₂=0.95 对小模型是毒药）都写进疑难排查——比虚构案例有说服力
- **可验证专家**：三个里程碑项目 + 每章面试真题，学完即可通过大厂 LLM Infra / 算法岗的 PyTorch 环节
