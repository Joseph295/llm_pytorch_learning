"""miniGPT · 用训练好的 checkpoint 续写文本

运行：uv run projects/minigpt/generate.py --prompt "话说" --tokens 300
"""

import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "chapters", "ch08_transformer", "code"))
from gpt_model import GPT, GPTConfig  # noqa: E402
from tokenizer import BPETokenizer  # noqa: E402

DATA_DIR = os.path.join(HERE, "data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="话说")
    ap.add_argument("--tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    tok = BPETokenizer.load(os.path.join(DATA_DIR, "tokenizer.json"))

    ck = torch.load(os.path.join(DATA_DIR, "ckpt.pt"), map_location=device, weights_only=True)
    cfg = GPTConfig(**ck["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ck["model"])
    print(f"加载 checkpoint（step {ck['step']}，{model.num_params() / 1e6:.1f}M 参数）")

    ids = torch.tensor([tok.encode(args.prompt)], device=device)
    out = model.generate(ids, args.tokens, temperature=args.temperature, top_k=args.top_k)
    print(f"\n{'─' * 50}\n{tok.decode(out[0].tolist())}\n{'─' * 50}")


if __name__ == "__main__":
    main()
