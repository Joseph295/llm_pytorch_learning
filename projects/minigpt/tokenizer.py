"""miniGPT · 从零实现字节级 BPE tokenizer

三个核心操作（与 GPT-2/minbpe 同一算法家族）：
  train  : 统计最高频相邻字节对，迭代合并 vocab_size-256 次
  encode : 按训练时的合并次序（rank 越小优先级越高）贪心合并
  decode : 查表拼接字节，utf-8 解码

字节级的意义：任何 Unicode 文本都先变成 0~255 的字节流——词表天然覆盖
一切语言，不存在 <UNK>。对中文的有趣现象：常用汉字（3 字节）会在训练
早期被 BPE "重新发明"成单 token，然后才开始组词。
"""

import json
from collections import Counter


class BPETokenizer:
    def __init__(self):
        self.merges: dict[tuple[int, int], int] = {}   # (a, b) -> 新 token id（也是 rank）
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    # ─────────────────────── train ───────────────────────
    def train(self, text: str, vocab_size: int, verbose: bool = True):
        assert vocab_size > 256
        ids = list(text.encode("utf-8"))
        if verbose:
            print(f"训练语料: {len(text):,} 字符 → {len(ids):,} 字节")

        for step in range(vocab_size - 256):
            stats = Counter(zip(ids, ids[1:]))          # C 速度的相邻对计数
            if not stats:
                break
            pair = stats.most_common(1)[0][0]
            new_id = 256 + step
            ids = self._merge(ids, pair, new_id)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose and (step + 1) % 200 == 0:
                piece = self.vocab[new_id].decode("utf-8", errors="replace")
                print(f"  merge {step + 1:>4}/{vocab_size - 256}: {pair} -> {new_id} "
                      f"({piece!r})  序列压缩到 {len(ids):,}")

    @staticmethod
    def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        out, i = [], 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    # ─────────────────────── encode / decode ───────────────────────
    def _encode_chunk(self, ids: list[int]) -> list[int]:
        while len(ids) >= 2:
            # 找当前序列里 rank 最小（最早学到）的可合并对——贪心次序必须与训练一致
            stats = set(zip(ids, ids[1:]))
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = self._merge(ids, pair, self.merges[pair])
        return ids

    def encode(self, text: str, chunk_on: str = "\n") -> list[int]:
        """按分隔符分块后逐块编码。

        为什么分块（第 5 章 GPT-2 的经验）：朴素 encode 是 O(n·merges)，
        对整篇文档一次编码会退化到几十亿次操作而卡死。GPT-2 用正则按词切分
        正是为了把 n 限制在单词长度。这里用换行切分（中文没有天然词边界），
        代价是 merge 不跨行——对语言模型质量无实质影响。
        """
        sep_ids = self._encode_chunk(list(chunk_on.encode("utf-8"))) if chunk_on else []
        out = []
        for i, part in enumerate(text.split(chunk_on)):
            if i > 0:
                out.extend(sep_ids)               # 把分隔符本身也编码进去（无损）
            out.extend(self._encode_chunk(list(part.encode("utf-8"))))
        return out

    def decode(self, ids: list[int]) -> str:
        data = b"".join(self.vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace")   # 字节级可能切开多字节字符，replace 兜底

    # ─────────────────────── save / load ───────────────────────
    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({f"{a},{b}": v for (a, b), v in self.merges.items()}, f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        tok = cls()
        with open(path) as f:
            raw = json.load(f)
        for key, v in raw.items():
            a, b = map(int, key.split(","))
            tok.merges[(a, b)] = v
        for (a, b), v in sorted(tok.merges.items(), key=lambda kv: kv[1]):
            tok.vocab[v] = tok.vocab[a] + tok.vocab[b]
        return tok

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)


if __name__ == "__main__":
    tok = BPETokenizer()
    sample = "人有悲欢离合，月有阴晴圆缺，此事古难全。" * 50 + "hello world " * 50
    tok.train(sample, vocab_size=300, verbose=False)
    ids = tok.encode("悲欢离合 hello")
    print(f"往返测试: {tok.decode(ids)!r}")
    assert tok.decode(tok.encode(sample[:100])) == sample[:100], "无损往返！"
    ratio = len(sample.encode()) / len(tok.encode(sample))
    print(f"44 个 merge 后的压缩比: {ratio:.2f} 字节/token")
    print("最先学到的 merge:", [tok.vocab[256 + i].decode('utf-8', 'replace') for i in range(6)])
