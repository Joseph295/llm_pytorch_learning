"""第 8 章 · 置换等变性：注意力为什么"分不清词序"，位置编码如何修复

运行：uv run chapters/ch08_transformer/code/permutation_test.py
"""

import math

import torch
import torch.nn as nn

torch.manual_seed(0)


def attention(x):
    """无位置信息的自注意力（无投影简化版，结论对带投影同样成立）。"""
    scores = x @ x.transpose(-2, -1) / math.sqrt(x.size(-1))
    return scores.softmax(-1) @ x


T, d = 6, 16
x = torch.randn(T, d)
perm = torch.randperm(T)

out_then_perm = attention(x)[perm]          # 先算后打乱
perm_then_out = attention(x[perm])          # 先打乱后算

print("═══ 1. 置换等变性证明 ═══")
print(f"打乱输入再算 == 算完再打乱: {torch.allclose(out_then_perm, perm_then_out, atol=1e-6)}")
print("→ '猫追狗'和'狗追猫'对裸注意力是同一个集合——顺序信息不存在于机制中")

print("\n═══ 2. 加入位置编码后，等变性被打破 ═══")
pos = nn.Embedding(T, d)
x_pos = x + pos(torch.arange(T))
a = attention(x_pos)[perm]
b = attention(x_pos[perm])                  # 注意：打乱的是"已加位置"的向量
# 更严格的对照：打乱 token 但位置照旧（真实场景语序变了位置编码不变）
x_permuted_tokens = x[perm] + pos(torch.arange(T))
c = attention(x_permuted_tokens)
print(f"语序改变后输出是否只是原输出的重排: {torch.allclose(a, c, atol=1e-6)}")
print("→ False = 位置编码让'不同语序'产生真正不同的表示（不再只是重排）")

print("\n═══ 3. 从困惑度角度看意义 ═══")
print("""无位置信息的语言模型只能学到词袋统计——'的 我 猫 爱'与'我爱猫的'等价，
下一词预测的上限极低。第 9 章训练时若忘了位置编码（或 RoPE 没生效），
症状就是 loss 卡在明显偏高的位置不动（8.7 案例 3 的一种病因）。""")
