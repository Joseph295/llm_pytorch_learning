"""第 14 章 · SFT 的 loss 掩码：只训练回答部分

运行：uv run chapters/ch14_finetuning/code/sft_masking.py

演示为什么指令部分要 label=-100：不掩码则模型学着复读指令，
掩码后才学"生成回答"。这是 SFT 与预训练的唯一实质区别。
"""

import torch
import torch.nn.functional as F

IGNORE = -100

# 模拟一条 SFT 样本的 token id（简化：用整数代表 token）
# 结构: [<user>] 指令token... [<assistant>] 回答token... [<end>]
instruction = [1, 20, 21, 22, 23]          # <user> + 指令
response = [2, 40, 41, 42, 3]              # <assistant> + 回答 + <end>
tokens = instruction + response

print("═══ 掩码构造 ═══")
input_ids = torch.tensor(tokens[:-1])      # 输入 = 去掉最后一个
targets = torch.tensor(tokens[1:])         # 目标 = 右移一位（第 5 章）

# 关键：把"指令部分"对应的 target 位置设为 IGNORE
# 指令占 len(instruction) 个 token，其预测目标（前 len(instruction)-1 个 target 位）要屏蔽
targets_masked = targets.clone()
n_instr = len(instruction)
targets_masked[: n_instr - 1] = IGNORE     # 指令内部的 next-token 预测不算 loss

print(f"input:          {input_ids.tolist()}")
print(f"target(全算):    {targets.tolist()}")
print(f"target(掩码后):  {targets_masked.tolist()}   ← 前 {n_instr - 1} 位 -100，只训回答")

# ═══ loss 对比 ═══
torch.manual_seed(0)
vocab = 50
logits = torch.randn(len(input_ids), vocab)

loss_all = F.cross_entropy(logits, targets, ignore_index=IGNORE)
loss_masked = F.cross_entropy(logits, targets_masked, ignore_index=IGNORE)
n_all = (targets != IGNORE).sum().item()
n_masked = (targets_masked != IGNORE).sum().item()
print(f"\n全序列算 loss: 参与 {n_all} 个位置 → 模型学习生成'指令+回答'（会复读指令！）")
print(f"掩码后算 loss: 参与 {n_masked} 个位置 → 模型只学习生成'回答'（正确）")

print("""
为什么这样做（14.2-①）：
- SFT 目标是让模型学会'给定指令，生成回答'，不是'生成指令'
- 指令部分作为条件（context）参与前向注意力，但不作为学习目标
- 忘了掩码 = 易错点①：模型 SFT 后开始复读或续写指令而非回答
- ignore_index=-100 是 cross_entropy 的约定（第 7 章），被忽略的位置不贡献 loss/梯度
真实实现：对话模板拼接后，用 tokenizer 的 offset 定位指令/回答边界，批量设 -100。
TRL 的 SFTTrainer 的 DataCollatorForCompletionOnlyLM 自动做这件事（14.5-②）。
""")
