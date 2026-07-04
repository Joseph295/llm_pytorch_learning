"""第 5 章 · LLM 预训练数据管线：packing（第 9 章 miniGPT 直接复用）

运行：uv run chapters/ch05_data_pipeline/code/llm_packing.py

演示 padding 与 packing 两种方案的算力浪费对比，并实现完整 packing 管线：
  变长文档 → tokenize → 拼接（插 <eos>）→ 定长切块 → 语言模型样本 (x, y)
"""

import torch
from torch.utils.data import DataLoader, Dataset

EOS = 0                                  # 文档分隔符 token id
SEQ_LEN = 16                             # 演示用小块；第 9 章用 256+


def fake_tokenize(doc: str) -> list[int]:
    """演示用 tokenizer：字符转 id（第 9 章换成真 BPE）。"""
    return [ord(c) % 96 + 1 for c in doc]


docs = [
    "the quick brown fox",
    "jumps over",
    "a very long document that keeps going and going and going",
    "short",
    "medium length text here",
]


# ═══ 方案对比：padding 的浪费 ═══
tokenized = [fake_tokenize(d) for d in docs]
max_len = max(len(t) for t in tokenized)
total_real = sum(len(t) for t in tokenized)
padded_cells = len(tokenized) * max_len
print("═══ padding vs packing ═══")
print(f"5 篇文档 token 数: {[len(t) for t in tokenized]}")
print(f"padding 到 max_len={max_len}: 有效 token 占比 {total_real / padded_cells:.0%}"
      f"（{padded_cells - total_real} 个 pad 白算）")


# ═══ packing 管线 ═══
def pack(token_lists: list[list[int]], seq_len: int) -> torch.Tensor:
    """拼接全部文档（文档间插 EOS），切成 (n_blocks, seq_len+1) 的定长块。

    +1 是语言模型的错位需要：x = block[:-1], y = block[1:]（第 1 章
    WindowedDataset 的思想，窗口=整块）。尾部不足一块的 token 丢弃
    （大语料下损失可忽略；也可留到下个 epoch 开头，工业实现两种都有）。
    """
    stream = []
    for toks in token_lists:
        stream.extend(toks)
        stream.append(EOS)
    n_blocks = (len(stream) - 1) // seq_len
    usable = stream[: n_blocks * seq_len + 1]
    blocks = torch.tensor(usable[:-1]).view(n_blocks, seq_len)
    targets = torch.tensor(usable[1:]).view(n_blocks, seq_len)
    return blocks, targets


class PackedLMDataset(Dataset):
    def __init__(self, token_lists, seq_len):
        self.x, self.y = pack(token_lists, seq_len)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


ds = PackedLMDataset(tokenized, SEQ_LEN)
print(f"\npacking 后: {len(ds)} 个定长块，每块 {SEQ_LEN} token，零 padding，有效占比 100%")
x0, y0 = ds[0]
print(f"块 0 输入: {x0.tolist()}")
print(f"块 0 目标: {y0.tolist()}   ← 恰好右移一位：位置 t 的目标是 t+1 的 token")

loader = DataLoader(ds, batch_size=2, shuffle=True)
xb, yb = next(iter(loader))
print(f"\nDataLoader 出品 batch: x{tuple(xb.shape)} y{tuple(yb.shape)} —— 定长稠密，GPU 最爱")

print("""
工程注记（第 9 章按此实施）：
1. 真实管线是离线的：tokenize + pack 一次性完成，存 np.memmap 二进制；
   训练时按索引读块——数据加载快到不需要 worker（5.5-③ 铁律）。
2. 块可能跨文档：EOS 告知边界。GPT-2/3 连注意力都不隔离照样训练；
   讲究的实现在块内做文档级 attention mask（代价是 mask 构造复杂度）。
3. shuffle 发生在"块"级别（切好后打乱块序），文档级顺序在离线阶段打乱。
""")
