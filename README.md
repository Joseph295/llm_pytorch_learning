# PyTorch 从入门到专家（大模型方向）

一份为**大数据/分布式系统工程师**量身定制的 PyTorch 系统教程：用你已有的分布式心智模型做类比桥接，文档 + 可运行代码双轨，每章八段式（来龙去脉 / 核心原理 / 动手实验 / 易错点 / 开源最佳实践 / 面试题 / 疑难排查 / 练习题）。

完整设计见 [设计文档](docs/superpowers/specs/2026-07-03-pytorch-llm-tutorial-design.md)。

## 快速开始

```bash
uv sync                                              # 精确复现环境（Python 3.12 + torch）
uv run chapters/ch00_environment/code/check_env.py   # 环境体检
```

## 路线图与进度

### 第〇篇 · 起步
- [x] [第 0 章 · 环境与工具链](chapters/ch00_environment/README.md)
- [x] [第 1 章 · 写给 JVM 工程师的 Python 速成](chapters/ch01_python_for_jvm/README.md)

### 第一篇 · PyTorch 核心
- [ ] 第 2 章 · 张量的本质：storage/stride/广播/内存布局
- [ ] 第 3 章 · Autograd 解剖
- [ ] 第 4 章 · nn.Module 体系
- [ ] 第 5 章 · 数据管线
- [ ] 第 6 章 · 训练循环完全解剖

### 第二篇 · Transformer 与 LLM 构建
- [ ] 第 7 章 · 注意力机制从零推导
- [ ] 第 8 章 · 完整 Transformer 实现
- [ ] 第 9 章 · 🏆 里程碑一：从零预训练 miniGPT
- [ ] 第 10 章 · 现代 LLM 架构演进（LLaMA/Qwen/MoE）

### 第三篇 · 训练工程
- [ ] 第 11 章 · 单卡性能优化
- [ ] 第 12 章 · 分布式训练原理（DDP 源码级）
- [ ] 第 13 章 · 大模型并行策略（ZeRO/FSDP/TP/PP）
- [ ] 第 14 章 · 微调工程（SFT/LoRA/DPO）
- [ ] 第 15 章 · 🏆 里程碑二：云端多卡微调 7B + 疑难杂症手册

### 第四篇 · 推理与部署
- [ ] 第 16 章 · 推理优化原理（KV Cache/量化/投机解码）
- [ ] 第 17 章 · 🏆 里程碑三：mini-vLLM
- [ ] 第 18 章 · 服务化与端侧（vLLM/llama.cpp/MLX）

### 第五篇 · 专家纵深
- [ ] 第 19 章 · PyTorch 内部机制（dispatcher/ATen/Triton）
- [ ] 第 20 章 · 面试全景与综合复盘

## 硬件路径

- 本地 Apple Silicon（MPS）可完成：第 0~11、16、18 章及里程碑一
- 需按需租云 GPU（讲义内标注预估费用）：第 12、13、15 章及里程碑二/三的性能验证

## 学习约定

1. 每章按讲义顺序读 + 跑 `code/` + 做 `exercises/`（基础题必做），再进入下一章
2. 练习答案写在各章 `exercises/` 下（如 `my_basic_1.md`）并提交——本仓库就是你的学习档案
3. 环境出问题，第一步永远是 `uv run chapters/ch00_environment/code/check_env.py`
