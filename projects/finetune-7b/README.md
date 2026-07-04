# 里程碑二 — 云端 QLoRA 微调 7B

配套教程：[第 15 章](../../chapters/ch15_finetune_7b/README.md)。第三篇（训练工程）的总交卷。

> ⚠️ **本项目需要 NVIDIA GPU**（bitsandbytes 4-bit 量化依赖 CUDA），在 M4/CPU 上无法运行。
> 这是设计如此——里程碑二就是云端实战。预估费用 ¥30~50（单张 24GB 卡，几小时）。

## 云端步骤

```bash
# 1. 租实例：AutoDL/RunPod，24GB+ 卡（4090/A10），官方 PyTorch 镜像（第 0 章纪律）
# 2. 装依赖
uv pip install transformers peft trl bitsandbytes datasets accelerate
# 3. 微调
export HF_ENDPOINT=https://hf-mirror.com          # 国内加速
python finetune_qlora.py                          # ~500 步，约 1-2 小时
# 4. 验收：对比微调前后
python chat.py --adapter ./output/final --prompt "用一句话解释注意力机制"
python chat.py --base_only  --prompt "用一句话解释注意力机制"
# 5. 实验完立即关机！（第 0 章纪律：按小时计费）
```

## 文件

| 文件 | 作用 | 复用章节 |
|---|---|---|
| `finetune_qlora.py` | QLoRA 微调主脚本 | 第 14 章 QLoRA/LoRA、第 11 章显存菜单 |
| `chat.py` | 加载 adapter 对比微调前后 | 第 14 章 adapter 加载 |

## 配置要点（第 14/15 章）

- **4-bit NF4 量化** + 双重量化 + bf16 计算：7B 主干 14GB→3.5GB
- **LoRA** r=16，加在 q/k/v/o 投影
- **显存菜单全开**：gradient_checkpointing + 8-bit 优化器 + 梯度累积 + packing
- **lr=2e-4**（LoRA 用较大 lr）、cosine 调度、梯度裁剪

## 省钱选项

- 换更小模型：`Qwen2.5-3B` 或 `1.5B`（改 finetune_qlora.py 的 MODEL）
- 减 max_steps / max_seq_length
- 选竞价实例（spot）

> `output/` 已 git 忽略（adapter 权重）。
