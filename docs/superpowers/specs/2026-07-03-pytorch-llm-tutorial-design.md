# PyTorch 大模型方向「从入门到专家」教程 — 设计文档

日期：2026-07-03
状态：已与学习者确认通过

## 1. 背景与目标

学习者画像：
- 多年大数据 / 分布式系统工程经验（主力语言 Java/Scala，Python 一般）
- 深度学习零实战基础
- 目标：转型大模型领域，**全栈覆盖，训练与推理并重**
- 硬件：本地 Apple M4 / 24GB 内存（MPS）；愿意按需租用云 GPU 做分布式与 CUDA 实战

成功标准：学完全部章节并完成三个里程碑项目后，具备以下可验证能力：
1. 不借助框架封装，从零实现并预训练一个小型 GPT（含 tokenizer、数据管线、训练循环）
2. 能在云端多卡环境独立完成 7B 级模型微调（含 LoRA/QLoRA、分布式配置、故障排查）
3. 能解释并动手实现推理核心优化（KV Cache、continuous batching、简化版 PagedAttention）
4. 能读懂 PyTorch / HuggingFace / vLLM / Megatron 关键源码，并通过大厂 LLM Infra / 算法岗面试的 PyTorch 相关环节

## 2. 设计原则

1. **心智模型映射**：利用学习者已有的分布式系统知识做类比桥接（DataLoader ≈ 数据管线背压；Ring AllReduce ≈ shuffle 通信模式；checkpoint ≈ 容错快照；FSDP 分片 ≈ 数据分区）。每章「来龙去脉」一节主动建立对照。
2. **按需拆黑盒**：概念按「第一次遇到问题 → 朴素解法 → 工业级解法」的因果链多次递进出现（例：KV Cache 在第 8、9、17 章三次深化），不一次性灌输结论。
3. **专家可验证**：每篇以里程碑自测项目收尾，通过里程碑才进入下一篇。

## 3. 教程形态

**文档 + 可运行代码双轨**：
- 每章一篇系统性 Markdown 讲义（八段式模板，见 §5）
- 配套可运行 Python 代码与分档练习题（含参考答案）
- 本地 M4 直接可跑；分布式/CUDA 章节提供「gloo 本地模拟（免费）+ 云 GPU 实战（标注预估费用）」双路径

语言：中文讲义，代码与术语保留英文原文。

## 4. 章节大纲（五篇二十章）

### 第〇篇 · 起步
- **第 0 章** 环境与工具链：uv、MPS、Jupyter、云 GPU 租用指南（含费用预估）
- **第 1 章** 写给 JVM 工程师的 Python 速成：鸭子类型与 `__call__`、装饰器、上下文管理器、生成器、NumPy 广播——只讲 PyTorch 高频用到且 Java 工程师易摔跤的部分

### 第一篇 · PyTorch 核心
- **第 2 章** 张量的本质：storage/stride/view、广播、内存布局与 contiguous
- **第 3 章** Autograd 解剖：动态计算图、反向传播、叶子节点、梯度累积、自定义 `autograd.Function`
- **第 4 章** nn.Module 体系：参数注册、`state_dict`、hook、模块组合
- **第 5 章** 数据管线：Dataset/DataLoader/Sampler、多进程加载的坑
- **第 6 章** 训练循环完全解剖：SGD→AdamW 演进、学习率调度、梯度裁剪、AMP 混合精度

### 第二篇 · Transformer 与 LLM 构建
- **第 7 章** 注意力机制从零推导：RNN 困境 → self-attention 必然性 → 手写多头注意力
- **第 8 章** 完整 Transformer 实现：Pre/Post-Norm、RMSNorm、RoPE、因果掩码
- **第 9 章** 🏆 里程碑一：从零预训练 miniGPT（自写 BPE、数据准备、完整训练，M4 本地完成）
- **第 10 章** 现代 LLM 架构演进：逐行读 LLaMA/Qwen 源码，GQA、SwiGLU、MoE

### 第三篇 · 训练工程
- **第 11 章** 单卡性能优化：Profiler、`torch.compile`、FlashAttention 原理、显存逐字节解剖
- **第 12 章** 分布式训练原理：通信原语（对照大数据 shuffle）、DDP 源码级剖析、gloo 模拟 + 云端 NCCL 实战
- **第 13 章** 大模型并行策略：ZeRO 三阶段、FSDP、TP/PP/EP、Megatron 与 DeepSpeed 设计取舍
- **第 14 章** 微调工程：SFT、LoRA/QLoRA 原理与手写实现、DPO、PEFT/TRL 最佳实践
- **第 15 章** 🏆 里程碑二：云端多卡微调 7B 模型 + 训练疑难杂症手册（loss spike、NaN、OOM、卡死）

### 第四篇 · 推理与部署
- **第 16 章** 推理优化原理：KV Cache 精讲、量化（GPTQ/AWQ 数学）、投机解码
- **第 17 章** 🏆 里程碑三：mini-vLLM——手写 continuous batching 与简化版 PagedAttention，再读 vLLM 真实源码
- **第 18 章** 服务化与端侧：vLLM/SGLang 生产部署、llama.cpp 与 MLX（M4 实战）

### 第五篇 · 专家纵深
- **第 19 章** PyTorch 内部机制：dispatcher、ATen、自定义 CUDA 算子、Triton 入门
- **第 20 章** 面试全景与综合复盘：按 LLM Infra / 算法岗真题体系梳理，每题回指对应章节

## 5. 每章八段式模板

1. **来龙去脉**：为解决什么问题而生，之前方案为什么不行（含分布式系统类比）
2. **核心原理**：深入浅出的推导
3. **动手实验**：`code/` 下可运行代码
4. **易错点清单**：每条给「错误代码 → 现象 → 原因 → 修正」四元组
5. **开源最佳实践**：指向 PyTorch/HuggingFace/vLLM/Megatron 具体源码文件的导读
6. **面试真题**：3~5 道 + 深度参考答案
7. **疑难排查**：对应本章知识的线上问题排查案例
8. **练习题**：基础/进阶/挑战三档，`exercises/solutions/` 提供带注释答案

## 6. 仓库结构

```
llm_pytorch_learning/
├── README.md                    # 总路线图 + 进度追踪
├── chapters/
│   └── chNN_<slug>/
│       ├── README.md            # 讲义（八段式）
│       ├── code/                # 可运行示例
│       └── exercises/           # 练习 + solutions/
├── projects/                    # 里程碑项目独立成工程
│   ├── minigpt/
│   ├── finetune-7b/
│   └── mini-vllm/
└── docs/
    └── troubleshooting.md       # 疑难杂症跨章总索引
```

## 7. 硬件路径约定

- 第 0~11、16、18 章：本地 M4（MPS/CPU）全程可完成
- 第 12、13、15 章：gloo 后端多进程本地模拟通信语义（免费）+ 云 GPU 多卡实战（每次实验预估 ¥20~100，讲义中逐一标注）
- 第 17 章：逻辑层本地可跑，性能验证上云

## 8. 写作顺序与验收

- 按章节顺序写作（第 0 章起），每章完成的定义：讲义八段齐全、`code/` 全部在 M4 上实际运行通过、练习题含参考答案
- 学习者完成一章并反馈后再进入下一章，允许根据反馈调整后续章节深度

## 9. 明确不在范围内（YAGNI）

- 不教 Python 语法基础（只讲 PyTorch 相关的高频特性）
- 不覆盖 CV / 语音 / 多模态（专注 LLM；多模态可作为学完后的扩展方向）
- 不覆盖 Prompt 工程与 Agent 应用层开发
- 不自建数学教材（线代/概率按需在章内补直觉，不独立成章）
