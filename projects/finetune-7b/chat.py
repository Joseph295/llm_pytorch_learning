"""里程碑二 · 加载微调后的 LoRA adapter，对比微调前后的指令跟随能力

⚠️ 云端 CUDA 运行（同 finetune_qlora.py）。

运行：
  python chat.py --adapter ./output/final --prompt "用一句话解释什么是注意力机制"
  python chat.py --base_only --prompt "..."     # 对比：base 模型只会续写
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL = "Qwen/Qwen2.5-7B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="./output/final")
    ap.add_argument("--base_only", action="store_true", help="不加载 adapter，看 base 行为")
    ap.add_argument("--prompt", default="用一句话解释什么是注意力机制")
    args = ap.parse_args()

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map={"": 0})
    tok = AutoTokenizer.from_pretrained(MODEL)

    if not args.base_only:
        model = PeftModel.from_pretrained(model, args.adapter)    # 挂上 LoRA 旁路
        print(f"已加载 adapter: {args.adapter}")
    else:
        print("base 模型（无 adapter）")

    msgs = [{"role": "user", "content": args.prompt}]
    inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to("cuda")
    out = model.generate(inputs, max_new_tokens=256, temperature=0.7, do_sample=True)
    print("\n" + "─" * 50)
    print(tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True))
    print("─" * 50)
    print("\n对比 --base_only：base 模型对指令往往续写更多指令，而非回答——")
    print("这就是 SFT 把'续写机'变成'指令跟随者'的直观证据（第 14/15 章）。")


if __name__ == "__main__":
    main()
