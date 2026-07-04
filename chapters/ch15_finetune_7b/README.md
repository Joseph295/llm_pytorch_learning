# 第 15 章 · 🏆 里程碑二：云端 QLoRA 微调 7B + 训练疑难杂症手册

> **这是第二次交卷，也是训练部分的总决战**。你将租一张云 GPU，用 QLoRA 把一个 7B base 模型微调成能听指令的 instruct 模型；同时，本章是全教程一路踩坑的**集大成排查手册**——loss spike、NaN、OOM、多卡卡死的完整排查树。
>
> 学完你将拥有：一次真实的云端 7B 微调经历、一个属于你的指令模型、以及一套系统的训练故障排查方法论。

**前置**：第 6/11/12/13/14 章全部。 **硬件路径**：云 GPU（预估 ¥30~50）。 **项目位置**：`projects/finetune-7b/`。

---

## 15.1 来龙去脉：为什么微调是大多数人的实战起点

预训练（第 9-13 章）是大厂的游戏——万卡、百万美元。但**微调是每个人都能做、且工业界需求最大的**：用开源 base 模型（Qwen2.5、Llama-3、DeepSeek）+ 你自己的数据，几十块钱、几小时，就能得到一个懂你领域的模型。

第 14 章讲了微调的方法（SFT/LoRA/QLoRA/DPO），本章把它落到真实的 7B 模型和真实的云环境——**从"知道怎么做"到"真的做出来"**。这中间的鸿沟全是工程细节：环境配置、显存管理、数据格式、故障排查。里程碑一（第 9 章）教会你训练的骨架，里程碑二教会你在真实规模下把骨架跑起来、跑稳。

---

## 15.2 项目：用 QLoRA 微调 Qwen2.5-7B

**目标**：把 Qwen2.5-7B（base）微调成能遵循中文指令的模型，用一个小的指令数据集（如 Alpaca 中文版的子集，几千条）。

**技术栈**（第 14 章的落地）：
- **QLoRA**：4-bit 量化主干（NF4）+ LoRA 旁路，7B 微调塞进单张 24GB 卡
- **transformers + peft + trl + bitsandbytes**：工业标准工具链
- **SFTTrainer**：自动处理模板、掩码、packing

### 15.2.1 云环境准备（第 0 章纪律的实战）

```bash
# 1. 租实例：AutoDL / RunPod，选 24GB+ 卡（4090/A10）+ 官方 PyTorch 镜像
# 2. 装依赖（镜像已有 torch，只补这些）
uv pip install transformers peft trl bitsandbytes datasets accelerate
# 3. 下载模型（用国内镜像加速，第 0 章）
export HF_ENDPOINT=https://hf-mirror.com
hf download Qwen/Qwen2.5-7B --local-dir ./qwen7b
```

### 15.2.2 QLoRA 微调脚本核心

完整脚本见 `projects/finetune-7b/finetune_qlora.py`，核心配置：

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# 4-bit 量化配置（14.2-③ 的全部）
bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained("./qwen7b", quantization_config=bnb, device_map="auto")

# LoRA 配置（14.2-②）
lora = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","v_proj","k_proj","o_proj"],
                  lora_dropout=0.05, task_type="CAUSAL_LM")

# SFTTrainer 自动处理掩码/packing（14.5-②）
trainer = SFTTrainer(model=model, peft_config=lora, args=SFTConfig(
    per_device_train_batch_size=4, gradient_accumulation_steps=4,
    gradient_checkpointing=True,          # 省激活（第 11 章）
    bf16=True, learning_rate=2e-4,        # LoRA 用较大 lr（第 14 章）
    max_steps=500, optim="paged_adamw_8bit",  # 8-bit 优化器（第 11 章菜单）
    logging_steps=10, save_steps=100,
), train_dataset=dataset)
trainer.train()
```

### 15.2.3 验收

```bash
# 微调前：base 模型对指令只会续写
# 微调后：加载 LoRA adapter，模型能遵循指令回答
uv run projects/finetune-7b/chat.py --adapter ./output/checkpoint-500
```

对比同一条指令，微调前后的行为差异——这就是 SFT 把"续写机"变成"指令跟随者"的直观证据。

---

## 15.3 动手：显存不够时的完整菜单

7B QLoRA 微调最常见的问题是显存。按"从便宜到贵"依次打开（这张菜单是第 11/14 章知识的汇总，也是 OOM 排查的行动清单）：

| 手段 | 省什么 | 代价 | 章节 |
|---|---|---|---|
| 4-bit 量化主干（QLoRA） | 主干 14GB→3.5GB | 反量化开销 | 14.2-③ |
| LoRA（冻结主干） | 梯度+优化器状态 | 表达力受 r 限 | 14.2-② |
| gradient checkpointing | 激活值 | +33% 算力 | 11.2-⑤ |
| 减 batch + 梯度累积 | 激活值 | 更慢 | 6.2 |
| 8-bit 优化器 | 优化器状态 8→2 字节 | 轻微精度 | 11.5-② |
| 减 max_length | 激活 O(T²) | 截断长样本 | 7.2-⑤ |
| CPU offload | 把状态卸载到内存 | 大幅变慢 | 13.5-① |

**排查 OOM 的方法论**：先 `nvidia-smi` 看占用峰值出现在哪个阶段（加载/前向/反向/optimizer step），再对照第 11 章"显存分四类"定位是哪类爆了，然后从菜单选对应手段。

---

## 15.4 训练疑难杂症手册（全教程踩坑集大成）

这一节是本教程一路真实踩过的坑的系统整理——一张"症状 → 病因 → 排查 → 修复"的总表。**建议打印贴在工位上**。

### ① loss 不降

见第 6.7 案例 1 的五层排查树。核心：**过拟合单 batch 黄金测试**（几十步内应→0，不行就是代码 bug）→ lr 扫描 → 查数据错位/泄漏 → 查冻结参数。

### ② loss spike / 训练不稳定（本教程里程碑一的真实事故）

**真实案例复盘**：教程作者训练里程碑一的 miniGPT 时，loss 反复在 3 和 11 之间震荡，最终生成乱码。完整排查经过：

1. **怀疑 lr**：1e-3 → 6e-4 → 3e-4，逐级降低——**仍然震荡**（排除单纯 lr 过高）；
2. **控制实验（改一个变量）**：bf16 vs fp32 各跑 150、600 步——**两者都稳定**（排除混合精度）；
3. **隔离配置**：param groups、warmup 长度、cosine 调度、autocast wrapper、non_blocking——**逐个测试全部稳定**；
4. **验证数据**：检查 token id 越界（无）、batch/init 校验和（与稳定版逐位相同）；
5. **定性结论**：同一份代码、同一批数据、同一初始化，某些运行收敛到 loss 3.3、某些发散到 11——**这是 MPS 浮点非确定性在边缘稳定模型上的表现**：梯度范数偏高（5~13，健康应 <1），模型处于失稳边界，run-to-run 的微小数值差异被放大。

**修复（多管齐下，通用于任何 loss spike）**：
- **更紧的梯度裁剪**（clip 1.0 → 0.5）：直接限制单步伤害；
- **更保守的 lr + 更长 warmup**：远离失稳边界；
- **更平滑的 Adam β₂**（0.95 → 0.999）：二阶矩适应慢，不放大偶发大梯度；
- **按验证集存最优 checkpoint**（而非存最后一步）：即便偶发尖峰，也保证最终拿到的是好模型——**这是最关键的工程保险**。

**教训**：① loss spike 的根因未必是单一的，边缘稳定 + 数值非确定性会制造"幽灵 bug"；② 最可靠的防御不是消除所有尖峰，而是"最优 checkpoint 保存"这类让偶发故障不影响最终产物的工程设计；③ 大规模训练同理——loss spike 常见，工业做法是监控 + 自动回滚到 spike 前的 checkpoint（第 6 章梯度范数监控 + 本节的 best-checkpoint）。

### ③ loss 变 NaN

见第 6.7 案例 3 + 第 8.7 案例 2。排查序：fp16→bf16（一半问题消失）→ NaN 哨兵（第 4 章）二分定位第一枚 NaN 的层 → 查 log(0)/除0/超大 logits/全-inf 注意力行（第 7 章易错点③）→ 查数据。开 `torch.autograd.set_detect_anomaly(True)` 定位反向 NaN 源。

### ④ OOM（显存不足）

见 15.3 菜单 + 第 11.7 案例 3。方法论：`nvidia-smi` 看峰值阶段 → 第 11 章四类显存定位 → 菜单选手段。注意"第一次 optimizer step 才 OOM"= 优化器状态（第 6 章）；"缓慢上涨"= 泄漏（第 3 章 total+=loss / 第 4 章 hook 存张量）。

### ⑤ 多卡训练卡住（hang）

见第 12.7 案例 1。核心：**找"谁没到齐"**（集合通信 rank 不齐），不是找"谁错了"。排查：各 rank 打印到达点 → 数据分片不等长（DistributedSampler drop_last）→ rank-dependent 控制流少调了 collective → `TORCH_DISTRIBUTED_DEBUG=DETAIL`。

### ⑥ 微调后模型复读指令 / 效果差

见第 14.7 案例 1。查 SFT loss 掩码（指令部分是否 -100）、对话模板训练/推理一致性、LoRA 加的层和 r。

### ⑦ 环境/加载问题

见第 0.7 + 第 4.7。`check_env.py` 三件套（版本/设备/dtype）→ 权重加载先对 key 再对 shape 再对数值 → DDP/compile 前缀（第 4 章 Q3）。

---

## 15.5 开源项目的最佳实践

**① axolotl / LLaMA-Factory：配置驱动的微调框架**
生产微调很少手写脚本，而用 axolotl（YAML 配置一切：模型/数据/QLoRA/超参）或 LLaMA-Factory（带 WebUI）。它们把本章的 QLoRA 配置、数据处理、多卡、断点续训全部封装。理解底层（第 14 章手写 LoRA）后用这些框架是自然选择。

**② 训练监控：wandb / trackio + 梯度范数**
生产训练必接实验跟踪（loss/lr/gnorm/显存 曲线）。第 6 章讲的"三线监控"（loss/gnorm/lr）在这里是自动化的——梯度范数曲线突然抬头是 spike 的前兆，配合本章 ② 的自动回滚。

**③ 数据质量 > 数据数量**
SFT 效果的天花板是数据质量。LIMA 论文（1000 条高质量样本胜过 5 万条平庸样本）、以及各种数据清洗/去重/质量打分流程——微调工程有一半是数据工程（你的老本行！）。

---

## 15.6 典型面试题

**Q1：手把手描述用 QLoRA 微调一个 7B 模型的完整流程和关键配置。**

> **参考答案**：① 环境（24GB 卡 + transformers/peft/trl/bitsandbytes）；② 4-bit 加载（BitsAndBytesConfig: nf4 + double_quant + bf16 compute）；③ LoRA 配置（r=16, target q/k/v/o, alpha=32）；④ 数据（指令-回答对，SFTTrainer 自动掩码 packing）；⑤ 训练配置（小 batch + 梯度累积、gradient_checkpointing、bf16、paged_adamw_8bit、lr 2e-4）；⑥ 保存 adapter，推理时加载合并。关键权衡：r 大小、target 模块选择、显存菜单。**加分点**：为什么 lr 比全参大（只训少量参数）、QLoRA 反量化开销、数据质量决定上限。

**Q2：训练 loss 突然 spike，如何排查和防护？**

> **参考答案**：排查——查是否毒 batch（长尾/异常样本触发大梯度）、lr 是否过高、数值溢出（fp16→bf16）、梯度范数曲线（spike 前兆）。防护——梯度裁剪（限单步伤害）、warmup、监控 gnorm、**按 val 存最优 checkpoint + 自动回滚到 spike 前**（最关键：让偶发尖峰不影响最终产物）。**加分点**：区分"真 bug 导致的系统性发散"和"边缘稳定+数值噪声的偶发尖峰"，后者靠工程保险而非消除；大规模训练的 spike 监控与自动回滚实践。

**Q3：微调后模型不听指令、复读，可能的原因？**

> **参考答案**：① SFT loss 掩码错（对指令算了 loss，模型学着生成指令）——最常见；② 对话模板训练/推理不一致（特殊 token 对不上）；③ LoRA 加的层不对或 r 太小欠拟合；④ 学习率不当（太小没学到、太大破坏能力）；⑤ 数据质量差或格式错。排查：打印一个 batch 的 labels 确认掩码、对比训练/推理模板。**加分点**：base vs instruct 模型的区别、few-shot 能力来自预训练。

**Q4：全参微调、LoRA、QLoRA 的显存和效果如何权衡？7B 各需多少显存？**

> **参考答案**：全参微调 112GB（16 字节/参数）、效果最好但最贵；LoRA 冻结主干省梯度+优化器状态，7B 约需 14GB（主干 fp16）+ 少量旁路，效果接近全参；QLoRA 再 4-bit 量化主干到 3.5GB，单张 24GB 卡可跑，效果略低于 LoRA（量化损失）但差距小。选择：显存够用 LoRA，不够用 QLoRA，追求极致效果且资源充足才全参。**加分点**：具体数字来自第 2 章的账；QLoRA 的量化只在只读前向不损梯度质量；效果差距在多数任务上可忽略。

---

## 15.7 疑难杂症排查

本章 15.4 已是全教程的排查总手册。这里补充**云端微调特有**的问题：

**案例 1：bitsandbytes 导入失败 / 4-bit 加载报错**
→ bitsandbytes 强依赖 CUDA 版本匹配。查 `python -m bitsandbytes` 的诊断输出；确认 CUDA 版本与 bnb 编译版本一致（第 0 章的版本矩阵）；用官方镜像避免自己编译。

**案例 2：device_map="auto" 把模型切到多卡但推理慢/错**
→ `device_map="auto"` 自动分层到多卡（模型并行的自动版），层间传输有开销。单卡放得下就别用 auto（指定 `device_map={"":0}`）；多卡推理理解它的分层逻辑。

**案例 3：微调 loss 正常但保存的 adapter 加载后无效**
→ ① adapter 保存/加载路径不对（peft 的 save_pretrained/from_pretrained）；② base 模型版本不一致（adapter 是相对某个 base 的增量）；③ 合并时精度不对（第 14 章易错点⑥）。验证：加载 adapter 后对比有无 adapter 的输出差异。

---

## 15.8 练习题

### 基础 1：完成 7B QLoRA 微调
按 15.2 跑通 Qwen2.5-7B 的 QLoRA 微调（或换更小的 Qwen2.5-3B 省钱）。对比微调前后同一指令的回答。记录：显存占用、训练时间、花费。

### 基础 2：显存菜单实验
在微调脚本上，逐个开关 15.3 菜单的手段（gradient_checkpointing、batch size、8-bit 优化器），记录每个对显存和速度的影响。找到"刚好能在你的卡上跑"的最小配置。

### 进阶 1：数据质量对比
用两个数据集微调同一模型：一个 5000 条平庸数据，一个 500 条精选数据。对比效果，验证"质量 > 数量"（LIMA 的家庭作坊版）。

### 挑战 1：完整对齐流程
在 SFT 之后加一步 DPO（第 14 章）：构造偏好数据（对 SFT 模型的输出人工/规则打分），用 TRL 的 DPOTrainer 做偏好对齐。对比 SFT-only 和 SFT+DPO 的输出风格差异。这是完整的"预训练→SFT→对齐"三阶段的最后一环。

---

## 本章小结与第三篇回顾

你在云端用 QLoRA 真正微调了一个 7B 模型——从 base 到 instruct，几十块钱几小时。更重要的是，你拥有了一套系统的训练故障排查方法论（15.4 手册），它汇总了本教程一路踩过的每一个坑。里程碑一的真实 loss spike 事故告诉你：**最可靠的工程不是消除所有故障，而是让偶发故障不影响最终产物**。

**第三篇完结**：从单卡性能（第 11 章）到分布式（第 12 章）、并行策略（第 13 章）、微调（第 14 章）、云端 7B 实战（第 15 章），你已经掌握了 LLM 训练工程的全栈。**你现在能训练、能微调、能排障。**

**下一章（第 16 章，第四篇开篇）**：推理优化。训练是一次性的，推理是每天亿万次的——成本的大头在推理侧。KV Cache 为什么是自回归生成的命根子？量化怎么把模型压小 4 倍还能用？投机解码如何让大模型"预测未来"加速？第四篇转向"如何把训好的模型高效地服务出去"。
