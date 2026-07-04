"""基础 2 参考答案：sinusoidal 位置编码

运行：uv run chapters/ch08_transformer/exercises/solutions/basic_2.py
"""

import math

import torch


def sinusoidal_pe(max_len: int, d: int) -> torch.Tensor:
    """原始 Transformer 的固定位置编码：(max_len, d)。"""
    pe = torch.zeros(max_len, d)
    pos = torch.arange(max_len).float().unsqueeze(1)              # (T, 1)
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)                            # 偶数维 sin
    pe[:, 1::2] = torch.cos(pos * div)                            # 奇数维 cos
    return pe


T, d = 8, 32
pe = sinusoidal_pe(T, d)


def attention(x):
    return (x @ x.transpose(-2, -1) / math.sqrt(x.size(-1))).softmax(-1) @ x


torch.manual_seed(0)
x = torch.randn(T, d)
perm = torch.randperm(T)
x_pos = x + pe
same = torch.allclose(attention(x_pos)[perm], attention(x[perm] + pe), atol=1e-6)
print(f"加 sinusoidal 后，换语序仍只是输出重排吗: {same}（False = 等变性已破坏 ✓）")

print("""
vs learned 的优点：任意长度有定义（公式算出来的，没有查表越界问题），
  且零参数；相邻位置的编码天然相近（平滑性是免费的归纳偏置）。
vs RoPE 的缺点：加在 embedding 上只注入一次'绝对'位置，注意力分数中的
  相对性只是间接近似；RoPE 直接作用于每层的 QK 匹配、点积严格只依赖
  相对位移（第 8 章 rope_explained.py 的验证），外推与插值的可操作性也更强。
冷知识：sinusoidal 与 RoPE 同源——都是用多尺度旋转角编码位置，
  区别在'加在向量上'还是'旋转向量'。RoPE 可以看作 sinusoidal 的正确打开方式。
""")
