# 第 18 章 · 服务化与端侧部署：vLLM、llama.cpp、MLX

> **本章目标**：从"引擎原理"到"生产/本地部署"。学完你应该能回答：
> 1. 生产环境怎么用 vLLM/SGLang 部署一个模型服务？关键调优参数有哪些？
> 2. 服务化的性能指标（TTFT/TPOT/吞吐）怎么权衡？
> 3. 为什么 llama.cpp/MLX 能让 7B 在你的 M4 上流畅跑？GGUF 量化是什么？
> 4. 端侧和云端部署的取舍是什么？

**前置**：第 16 章（推理优化）、第 17 章（引擎原理）。 **硬件路径**：MLX/llama.cpp 在你的 M4 实战；vLLM 部署上云。 **预计用时**：4~5 小时。
**收尾视角**：第 17 章你手写了引擎，本章用生产引擎；训好、优化好的模型，最终要"服务出去"或"装进设备"。

---

## 18.1 来龙去脉：模型训好了，然后呢

一个训好、微调好的模型，有两条落地路径：

1. **云端服务化**：部署成 API 服务，高并发对外提供推理（ChatGPT、你公司的 LLM 服务）。关注吞吐、延迟、成本、可用性——这是你的分布式系统背景的主场。
2. **端侧部署**：把模型装进用户设备（手机、笔记本、你的 M4）本地运行。关注模型大小、内存占用、隐私、离线可用——量化推理的极致工程。

两条路径的技术栈不同，但都建立在第 16/17 章的原理上。本章带你把手写的 mini-vLLM 升级到生产引擎，并在你的 M4 上真正跑起来 7B 模型。

---

## 18.2 核心原理

### ① 服务化的性能指标：三个数字的权衡

LLM 服务的性能不是单一数字，是一组相互制约的指标：

- **TTFT（Time To First Token）**：从请求到第一个 token 的延迟。由 prefill 决定（处理 prompt 的时间）。影响"感觉快不快"。
- **TPOT（Time Per Output Token）**：生成每个后续 token 的时间。由 decode 决定。影响"生成流畅度"。
- **吞吐（Throughput）**：每秒处理的总 token 数（所有请求）。由 batching 效率决定。影响"成本"。

**核心权衡**：大 batch 提升吞吐（成本低）但增加单请求延迟（TTFT/TPOT 变长）；小 batch 反之。生产要根据场景取舍：面向用户的对话优先低延迟（小 batch + 优先调度），离线批处理优先吞吐（大 batch）。这就是你熟悉的"延迟 vs 吞吐"权衡，在 LLM 服务里的具体形态。

### ② vLLM 生产部署：一行起服务

```bash
# 启动一个 OpenAI 兼容的 API 服务
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --tensor-parallel-size 2 \        # 张量并行（第 13 章，大模型跨卡）
    --max-model-len 8192 \            # 最大上下文
    --gpu-memory-utilization 0.9 \    # 显存利用率上限（留余量给激活）
    --quantization awq                # 量化（第 16 章）
```

它把第 17 章你手写的一切（PagedAttention、continuous batching）+ 生产特性（OpenAI API、张量并行、量化、prefix caching）打包成一个命令。关键调优参数：
- `--max-num-seqs`：最大并发请求数（batch 上限）——吞吐/延迟权衡的旋钮；
- `--gpu-memory-utilization`：给 KV cache 留多少显存——越高并发越多但风险越大；
- `--enable-prefix-caching`：开启前缀共享（第 17 章挑战题）——多轮对话/相同 system prompt 场景大幅提效。

### ③ GGUF 量化与 llama.cpp：端侧推理的极致

llama.cpp 用 **GGUF** 格式（量化权重 + 元数据的单文件）让大模型在 CPU/消费级设备跑。它的量化命名（`Q4_K_M` 等）编码了量化策略：
- `Q4` = 4-bit，`Q8` = 8-bit（位数越低越小越快、精度越低）；
- `K` = k-quant（分块量化，比传统更精确）；
- `M`/`S`/`L` = 中/小/大（同位数下的精度档位，重要层用更高精度）。

**常用选择**：`Q4_K_M` 是大小/质量的甜点（7B 约 4.4GB，质量损失小）；追求质量用 `Q5_K_M`/`Q6_K`；极限压缩用 `Q3`/`Q2`（质量明显下降）。这些是第 16 章量化原理的工程化封装。

### ④ MLX：Apple Silicon 的原生框架

**MLX** 是 Apple 为自家芯片设计的深度学习框架，充分利用统一内存（第 0 章）和 Metal。相比 PyTorch MPS：MLX 是为 Apple Silicon 从头设计的（惰性求值、统一内存零拷贝），推理更快、内存更省。`mlx-lm` 让你几行代码在 M4 上跑量化 LLM：

```python
from mlx_lm import load, generate
model, tokenizer = load("mlx-community/Qwen2.5-7B-Instruct-4bit")
print(generate(model, tokenizer, prompt="解释注意力机制", max_tokens=200))
```

本章实验用 MLX 在你的 M4 上真正跑一个量化 7B——**你的笔记本变成本地 LLM 推理机**。这是端侧部署最贴近你硬件的实战。

### ⑤ 部署架构的其他拼图

- **服务编排**：Kubernetes + 自动扩缩容（你的老本行）、负载均衡、健康检查。
- **模型路由**：多模型/多版本流量管理、A/B 测试。
- **监控**：QPS、延迟分位数（p50/p99）、GPU 利用率、KV cache 占用、错误率——你熟悉的 SRE 指标体系。
- **成本优化**：竞价实例、请求批处理、缓存（相同请求/前缀）、模型级联（小模型兜底大模型）。

这些和你部署大数据/微服务系统的经验高度相通——LLM 服务本质是一种特殊的高成本、有状态（KV cache）计算服务。

---

## 18.3 动手实验

```bash
# 端侧实战（你的 M4）
uv run chapters/ch18_deployment/code/mlx_run.py         # 用 MLX 跑量化 LLM（需装 mlx-lm）
uv run chapters/ch18_deployment/code/serving_metrics.py  # TTFT/TPOT/吞吐 权衡模拟

# llama.cpp（本地编译或装预编译）
# ./llama-cli -m qwen2.5-7b-instruct-q4_k_m.gguf -p "你好"
```

`serving_metrics.py` 模拟不同 batch 策略下的 TTFT/TPOT/吞吐，让你直观看到"延迟 vs 吞吐"的权衡曲线（不需要真实 GPU）。`mlx_run.py` 是端侧实战——在你的 M4 上跑真实量化模型（首次运行会下载模型）。

---

## 18.4 易错点清单

**① gpu-memory-utilization 设太高导致 OOM**
→ 设 0.95+ 看似能多服务请求，但激活/临时缓冲的峰值会超预留，运行中 OOM。留余量（0.85-0.9）。

**② max-model-len 设太大浪费显存**
→ KV cache 按 max-model-len 预留（PagedAttention 缓解但仍有上限）。设成远超实际需求的值浪费显存、限制并发。按实际场景设。

**③ 量化档位选错**
→ Q2/Q3 追求极限压缩但质量明显下降（尤其代码/推理任务）；生产一般 Q4_K_M 起步，质量敏感用 Q5/Q6。别为省一点内存牺牲太多质量——先评估再选。

**④ 端侧内存预估不足**
→ M4 24GB 统一内存要同时装模型 + KV cache + 系统 + 其他 app。7B Q4（4.4GB）+ 长上下文 KV cache 可能超预算。预估总内存（第 2/16 章的账）。

**⑤ 服务化不做请求超时/限流**
→ 恶意/异常的超长 prompt 或无限生成会占死资源。生产必须设 max_tokens、请求超时、限流——你部署服务的常识，LLM 服务同样适用。

**⑥ 忽略 tokenizer/模板一致性**
→ 部署时的对话模板必须和训练时一致（第 14/15 章），否则模型行为异常。vLLM 的 chat 接口会用模型的 chat_template，自定义时要对齐。

---

## 18.5 开源项目的最佳实践

**① vLLM 的生产部署模式**
`vllm serve` + OpenAI 兼容 API 是事实标准。看它的 `--help` 全部参数理解调优空间；生产用 Docker + K8s 部署，配合 Prometheus 监控（KV cache 利用率、吞吐、延迟分位数）。多副本 + 负载均衡做高可用。

**② llama.cpp / Ollama 的端侧生态**
[Ollama](https://github.com/ollama/ollama) 封装 llama.cpp，`ollama run qwen2.5` 一行本地跑模型——端侧部署的"Docker"。看它的 Modelfile（模型 + 参数 + 模板的封装）。适合本地开发、隐私敏感、离线场景。

**③ MLX 的 Apple 生态**
[mlx-lm](https://github.com/ml-explore/mlx-examples) 在 Apple Silicon 上是最快的本地推理方案（比 PyTorch MPS 快、比 llama.cpp Metal 后端更原生）。`mlx-community` 有大量预量化模型。你的 M4 用它跑 LLM 是最优解。

**④ 推理服务的成本优化实践**
模型级联（小模型 handle 简单请求、复杂的升级大模型）、语义缓存（相同/相似请求命中缓存）、prefix caching（相同 system prompt 共享）、竞价实例——这些和你优化大数据成本的思路一致：分层、缓存、弹性。

---

## 18.6 典型面试题

**Q1：LLM 服务的关键性能指标有哪些？如何权衡？**

> **参考答案**：TTFT（首 token 延迟，prefill 决定，影响响应感）、TPOT（每 token 延迟，decode 决定，影响流畅度）、吞吐（每秒总 token，batching 决定，影响成本）。核心权衡：大 batch 高吞吐低成本但高延迟，小 batch 反之。场景取舍：对话优先低延迟（小 batch、优先调度、投机解码），离线批处理优先吞吐（大 batch）。**加分点**：prefill/decode 分离部署（P/D disaggregation）分别优化；SLO 驱动的调度；延迟分位数（p99）比均值更重要；continuous batching 让吞吐和延迟的权衡更优。

**Q2：如何在生产环境部署一个 7B 模型服务？关键调优参数？**

> **参考答案**：用 vLLM/SGLang 起 OpenAI 兼容服务；配置张量并行（大模型跨卡）、量化（AWQ/GPTQ 省显存）、max-num-seqs（并发上限，吞吐/延迟旋钮）、gpu-memory-utilization（KV cache 显存，留激活余量）、enable-prefix-caching（相同前缀共享）；Docker + K8s + 自动扩缩容 + Prometheus 监控；请求超时/限流/max_tokens 防滥用。**加分点**：多副本高可用、模型级联降成本、KV cache 利用率监控、chat template 一致性。

**Q3：端侧部署（如手机/笔记本）跑 LLM 的关键技术？和云端有何不同？**

> **参考答案**：关键是量化（GGUF Q4_K_M 等把 7B 压到 ~4GB）+ 高效推理框架（llama.cpp/MLX 针对消费级硬件/Apple Silicon 优化）。与云端不同：端侧优先模型大小/内存/隐私/离线，单请求（无需高并发 batching），用 CPU/集成 GPU/统一内存；云端优先吞吐/并发/成本，用数据中心 GPU。**加分点**：Apple 统一内存的优势（第 0 章，无 PCIe 传输）；端侧量化更激进（内存受限）；MLX 为 Apple Silicon 原生设计比 PyTorch MPS 快；隐私是端侧的核心卖点。

**Q4：GGUF 的 Q4_K_M 是什么意思？如何选量化档位？**

> **参考答案**：Q4=4-bit，K=k-quant（分块量化更精确），M=中等精度档（重要层用更高精度）。选择：Q4_K_M 是大小/质量甜点（7B ~4.4GB，损失小，生产常用）；质量敏感用 Q5_K_M/Q6_K；极限压缩用 Q3/Q2（质量明显降，尤其代码/推理）。原则：先评估目标任务的质量，再选能接受的最小档位。**加分点**：k-quant 对不同层差异化位数；量化对不同任务敏感度不同（生成 vs 推理）；importance matrix（imatrix）量化用校准数据进一步提质。

---

## 18.7 疑难杂症排查

**案例 1：vLLM 服务吞吐上不去 / 延迟高**

① max-num-seqs 太小限制了并发 batch——调大（受显存限制）；② gpu-memory-utilization 太低，KV cache 空间不足，能并发的请求少——调高（留余量）；③ 没开 prefix caching，相同 system prompt 重复 prefill——开启；④ prefill 和 decode 相互干扰——考虑 chunked prefill 或 P/D 分离。方法论：先看 KV cache 利用率和 batch 大小，再调对应参数。

**案例 2：端侧跑模型很慢 / 内存爆**

① 量化档位太高（模型太大）——降到 Q4_K_M；② 上下文太长，KV cache 吃内存——减 context 或用 KV 量化；③ 用了 PyTorch MPS 而非 MLX/llama.cpp（针对性优化差）——换原生框架；④ 后台其他 app 占内存——统一内存要和系统/app 共享（易错点④）。

**案例 3：部署后模型输出和微调时不一致**

几乎都是 chat template / tokenizer 不一致（易错点⑥）——确认部署用的模板和训练一致；特殊 token（BOS/EOS/system 标记）对齐；采样参数（temperature/top_p）设置合理。第 15 章微调的模型部署时尤其注意。

---

## 18.8 练习题

### 基础 1：M4 上跑量化 LLM
用 `mlx_run.py`（装 `mlx-lm`）在你的 M4 上跑一个量化 7B（如 Qwen2.5-7B-4bit）。测量：加载时间、TTFT、生成速度（token/s）、内存占用。对比不同量化档位（4bit vs 8bit）。

### 基础 2：TTFT/TPOT/吞吐权衡
用 `serving_metrics.py`，模拟不同 batch size 下的三个指标，画出权衡曲线。回答：对话场景和离线批处理场景各该选什么 batch 策略？

### 进阶 1：部署你微调的模型
把第 15 章微调的模型（QLoRA adapter 合并后）用 vLLM 部署（云端）或转成 GGUF 用 llama.cpp（本地）。对比部署前后的行为一致性（易错点⑥）。

### 挑战 1：端到端本地服务
在你的 M4 上搭一个完整的本地 LLM 服务：MLX/Ollama 起模型 + 一个简单的 Web UI（或用 OpenAI 兼容接口）。测量端到端延迟，讨论：什么场景端侧部署比云端更优（隐私/离线/成本/延迟）？

---

## 本章小结与第四篇回顾

部署分两条路：云端服务化（vLLM/SGLang + 生产特性，关注 TTFT/TPOT/吞吐权衡，是你的分布式主场）和端侧部署（GGUF/llama.cpp/MLX，量化推理的极致，让 7B 跑进你的 M4）。两者都建立在第 16/17 章的推理原理上。

**第四篇完结**：从推理优化原理（第 16 章）到手写引擎（第 17 章）到生产/端侧部署（第 18 章），你掌握了把训好的模型高效服务出去的全栈。**你现在能优化推理、能造引擎、能部署上线。**

**下一章（第 19 章，第五篇开篇）**：PyTorch 内部机制。到这里你会用 PyTorch 的一切，但它内部怎么工作？dispatcher 如何把 `a + b` 路由到具体 kernel？ATen 是什么？怎么写自定义 CUDA/Triton 算子？第五篇深入 PyTorch 的引擎室，让你从"专家用户"变成"能改 PyTorch 的人"。
