# miniGPT — 里程碑一：从零预训练一个 GPT

配套教程：[第 9 章](../../chapters/ch09_minigpt/README.md)。前八章知识的第一次总组装。

## 快速开始

```bash
uv run projects/minigpt/prepare_data.py          # ① 数据（首次 ~5-7 分钟）
uv run projects/minigpt/train.py --smoke         # ② 冒烟（~20 秒，验证管线）
uv run projects/minigpt/train.py                 # ③ 训练（M4 ~15 分钟，3000 步）
uv run projects/minigpt/generate.py --prompt 话说 # ④ 续写
```

## 文件

| 文件 | 作用 | 复用的章节知识 |
|---|---|---|
| `tokenizer.py` | 从零字节级 BPE | 第 5 章数据、字节/Unicode |
| `prepare_data.py` | 语料→BPE→tokenize→memmap | 第 5 章离线管线/packing |
| `train.py` | 六步训练循环+AMP+调度+ckpt | 第 3/6 章训练，第 0/2 章设备与显存 |
| `generate.py` | 加载 checkpoint 续写 | 第 4 章 state_dict，第 8 章 generate |
| 模型 | `chapters/ch08_transformer/code/gpt_model.py` | 第 7/8 章注意力与 Transformer |

## 配置

~12M 参数：`L=6, d=384, H=6, block_size=256, vocab=4096`。语料《红楼梦》（Gutenberg 公版）。
改配置直接编辑 `train.py` 里的 `GPTConfig`。改 `vocab` 需重跑 `prepare_data`；改 `block_size` 不需要。

## 实测基线（M4 / 24GB，已验证）

- 数据准备：BPE 训练 ~6 分钟（一次性缓存）+ 分块 encode ~14 秒
- 训练：3000 步约 15 分钟，loss 从 8.41（≈ln 4096）降至 train 3.8 / 最优 val 6.34
- 稳定配置：**β₂=0.999**（0.95 会致 loss spike，见第 9 章排查）+ best-checkpoint
- 生成：真实输出含 黛玉/襲人/探春/怡紅院 等人名地名 + 对话结构，文白腔调清晰

> `data/` 已 git 忽略（含权重与语料）。克隆后重跑 `prepare_data.py` 重建。
