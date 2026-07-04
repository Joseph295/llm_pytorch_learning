"""miniGPT · 数据准备：下载语料 → 训练 BPE → tokenize → packing → memmap 二进制

运行：uv run projects/minigpt/prepare_data.py

落实第 5 章的铁律：离线 tokenize + packing，存成定长可 mmap 的二进制。
训练时（train.py）"数据加载"退化为按索引读块——快到不需要 worker。

语料：《红楼梦》（Project Gutenberg 公版）。中文古典文本信息密度高、
风格统一，是小模型学"续写"的好素材。
"""

import os
import urllib.request

import numpy as np

from tokenizer import BPETokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
URL = "https://www.gutenberg.org/files/24264/24264-0.txt"
VOCAB_SIZE = 4096


def clean(raw: str) -> str:
    """去掉 Gutenberg 的英文页眉页脚，保留正文。"""
    start = raw.find("第一回")
    end = raw.find("End of the Project Gutenberg")
    if end == -1:
        end = len(raw)
    body = raw[start:end] if start != -1 else raw
    # 去掉 CRLF 和 BOM，压缩多余空白
    return body.replace("\r\n", "\n").replace("﻿", "").strip()


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    raw_path = os.path.join(DATA_DIR, "raw.txt")

    if not os.path.exists(raw_path):
        print(f"下载语料: {URL}")
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read().decode("utf-8")
        with open(raw_path, "w") as f:
            f.write(raw)
    else:
        raw = open(raw_path).read()

    text = clean(raw)
    print(f"清洗后正文: {len(text):,} 字符")

    # 训练 BPE（4096 词表，M4 上约 5~7 分钟）。已训练过则直接复用。
    tok_path = os.path.join(DATA_DIR, "tokenizer.json")
    if os.path.exists(tok_path):
        print("复用已训练的 tokenizer.json")
        tok = BPETokenizer.load(tok_path)
    else:
        tok = BPETokenizer()
        print(f"训练 BPE（vocab={VOCAB_SIZE}）...")
        tok.train(text, VOCAB_SIZE, verbose=True)
        tok.save(tok_path)

    # tokenize 全文 + packing（这里数据小，直接整体 tokenize）
    print("tokenize 全文...")
    ids = tok.encode(text)
    ratio = len(text.encode()) / len(ids)
    print(f"→ {len(ids):,} tokens，压缩比 {ratio:.2f} 字节/token")

    # 90/10 划分 train/val，存 uint16 memmap（vocab<65536 所以 uint16 够）
    ids = np.array(ids, dtype=np.uint16)
    n = int(len(ids) * 0.9)
    ids[:n].tofile(os.path.join(DATA_DIR, "train.bin"))
    ids[n:].tofile(os.path.join(DATA_DIR, "val.bin"))
    print(f"train.bin: {n:,} tokens | val.bin: {len(ids) - n:,} tokens")
    print(f"词表大小实际: {tok.vocab_size}（train.py 的 GPTConfig.vocab_size 用它）")

    # 抽样展示 tokenizer 学到了什么
    print("\n几个学到的多字节 token:")
    samples = [tok.vocab[i].decode("utf-8", "replace") for i in [300, 800, 1500, 3000, 4000]]
    print("  ", [s for s in samples if s.strip()])


if __name__ == "__main__":
    main()
