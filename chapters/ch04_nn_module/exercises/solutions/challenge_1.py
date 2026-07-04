"""挑战 1 参考答案：手写 tie_weights（embedding 与 lm_head 权重共享）

运行：uv run chapters/ch04_nn_module/exercises/solutions/challenge_1.py

背景：语言模型的输入 embedding (vocab→d) 和输出投影 lm_head (d→vocab)
形状互为转置，共享一份权重可省 vocab×d 个参数（GPT-2 的 50257×768 ≈ 3900万，
占其总参数的 30%+）。第 9 章 miniGPT 直接采用本题结论。
"""

import torch
import torch.nn as nn

VOCAB, D = 100, 32


class TiedLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D)              # weight: (vocab, d)
        self.body = nn.Linear(D, D)
        self.lm_head = nn.Linear(D, VOCAB, bias=False)   # weight: (vocab, d) —— 形状恰好相同！
        self.lm_head.weight = self.embed.weight          # ← tie：让两个属性指向同一个 Parameter

    def forward(self, idx):
        return self.lm_head(torch.relu(self.body(self.embed(idx))))


model = TiedLM()

# ── ① parameters() 不重复计数 ──
n_params = sum(p.numel() for p in model.parameters())
expect = VOCAB * D + (D * D + D)     # 共享的算一份 + body
print(f"参数量 {n_params} == 理论 {expect}: {n_params == expect}")
print("→ named_parameters 按对象身份去重，共享的 Parameter 只出现一次"
      f"（名字取先注册的：{[n for n, _ in model.named_parameters()][0]}）")

# ── ② 梯度作用在同一张量上 ──
loss = model(torch.randint(0, VOCAB, (4, 7))).sum()
loss.backward()
print(f"embed.weight.grad is lm_head.weight.grad: {model.embed.weight.grad is model.lm_head.weight.grad}")
print("→ 两条反向路径（输入侧 + 输出侧）的梯度自动累加进同一个 .grad（第 3 章多消费者求和）")

# ── ③ state_dict 的体现与坑 ──
sd = model.state_dict()
print(f"\nstate_dict keys: {list(sd)}")
print(f"两个 key 指向同一 storage: "
      f"{sd['embed.weight'].data_ptr() == sd['lm_head.weight'].data_ptr()}")
print("""
坑位分析：
1. state_dict 里共享权重出现两次（各自的路径名各一份）——保存文件时体积翻倍。
   safetensors 拒绝保存共享张量（会报错），HF 的做法是保存前解开、只存一份，
   加载后再 tie 回去（transformers 的 tie_weights() 在 from_pretrained 里自动补做）。
2. 最大的坑：先 load_state_dict 再 tie 没问题；但"先 tie、再 load 一个
   没 tie 的 checkpoint"会把共享解开吗？——不会解开引用，但两个 key 先后
   copy_ 到同一个张量，最后写入的那个生效，前一个被覆盖。若 ckpt 里两份
   数值不同（比如后来单独微调过 lm_head），会静默丢失一份。
3. 因此工程惯例：tie 操作放在"构造之后、加载之后"各调一次（幂等），
   并在加载后验证 data_ptr 一致。
""")

# 验证坑 2：构造一个两份权重不同的 state_dict
sd2 = {k: v.clone() for k, v in sd.items()}
sd2["lm_head.weight"] += 1.0                     # 假装 lm_head 被单独动过
model2 = TiedLM()
model2.load_state_dict(sd2)
same = torch.equal(model2.embed.weight, model2.lm_head.weight)
matches_lm = torch.allclose(model2.embed.weight, sd2["lm_head.weight"])
print(f"加载'两份不同'的 ckpt 后仍共享: {same}；最终数值等于后写入的 lm_head: {matches_lm}")
print("→ embed 的那份被静默覆盖——这就是为什么加载后要做一致性断言")
