"""里程碑二 · QLoRA 微调 7B（云端 CUDA 运行）

⚠️ 本脚本需要 NVIDIA GPU（bitsandbytes 的 4-bit 量化依赖 CUDA），
   在 M4/CPU 上无法运行。这是设计如此——里程碑二就是云端实战（第 15 章）。

云端运行（AutoDL/RunPod 等，24GB+ 卡）：
  uv pip install transformers peft trl bitsandbytes datasets accelerate
  export HF_ENDPOINT=https://hf-mirror.com     # 国内加速
  python finetune_qlora.py

把 Qwen2.5-7B（base）微调成能遵循中文指令的模型。
落实第 14 章：QLoRA(4-bit NF4 + LoRA) + SFT(掩码/packing) + 第 11 章显存菜单。
"""

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

MODEL = "Qwen/Qwen2.5-7B"           # 省钱可换 Qwen2.5-3B / 1.5B
OUTPUT = "./output"


def main():
    # ── 4-bit 量化配置（14.2-③ 的全部：NF4 + 双重量化 + bf16 计算）──
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, quantization_config=bnb, device_map={"": 0},  # 单卡放得下就别用 "auto"（15.7）
        attn_implementation="flash_attention_2",             # 第 11 章 FlashAttention
    )
    model.config.use_cache = False                           # 训练时关 KV cache（省显存）
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── LoRA 配置（14.2-②）：加在注意力四个投影 ──
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    # ── 数据：中文指令数据集（示例用 alpaca 中文，实战换你的领域数据）──
    ds = load_dataset("llamafactory/alpaca_zh_demo", split="train")

    def format_example(ex):
        # 用 Qwen 的 chat 模板拼接（SFTTrainer 会自动对指令部分掩码，14.5-②）
        msgs = [{"role": "user", "content": ex["instruction"] + "\n" + ex.get("input", "")},
                {"role": "assistant", "content": ex["output"]}]
        return {"text": tokenizer.apply_chat_template(msgs, tokenize=False)}

    ds = ds.map(format_example)

    # ── 训练配置：第 11/14 章的显存菜单全开 ──
    args = SFTConfig(
        output_dir=OUTPUT,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,          # 有效 batch 16（第 6 章）
        gradient_checkpointing=True,            # 省激活（第 11 章）
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        learning_rate=2e-4,                     # LoRA 用较大 lr（第 14 章）
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_steps=500,
        optim="paged_adamw_8bit",               # 8-bit 优化器（第 11 章菜单）
        max_grad_norm=1.0,                      # 梯度裁剪（第 6 章）
        logging_steps=10,
        save_steps=100,
        max_seq_length=1024,
        packing=True,                           # packing 提效（第 5 章）
        dataset_text_field="text",
    )

    trainer = SFTTrainer(model=model, peft_config=lora, args=args, train_dataset=ds)
    trainer.train()
    trainer.save_model(f"{OUTPUT}/final")       # 只存 LoRA adapter（几十 MB）
    print(f"微调完成，adapter 保存到 {OUTPUT}/final")
    print("验收: uv run chat.py --adapter ./output/final")


if __name__ == "__main__":
    main()
