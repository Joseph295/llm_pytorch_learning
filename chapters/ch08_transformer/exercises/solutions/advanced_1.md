# 进阶 1 参考答案：dropout 的三处正确安放

对 `gpt_model.py` 的修改点（模型代码已预留 `cfg.dropout`，注意力/FFN 处已就位）：

```python
# ① embedding 之后（已有）：self.drop = nn.Dropout(cfg.dropout)
x = self.drop(self.embed(idx))

# ② 注意力权重上（已通过 SDPA 的 dropout_p 参数实现）：
F.scaled_dot_product_attention(q, k, v, is_causal=True,
                               dropout_p=self.dropout if self.training else 0.0)
#    注意 self.training 判断——SDPA 不像 nn.Dropout 会自动感知 eval 模式！

# ③ 每个子层写回残差流之前（MLP 已有 self.drop(self.proj(...))；
#    注意力同理可在 wo 之后加）：
x = x + self.dropout_layer(self.attn(self.norm1(x), cos, sin))
```

**错误位置对照**（易错点⑤）：`x = self.drop(x + sublayer(...))` 把 dropout 压在
残差主干道上——随机把"信息高速公路"整段断路，破坏 Pre-Norm 的恒等通路，
伤害远大于正则收益。口诀：**dropout 管支路，不碰主干道**。

**验证清单：**

1. train/eval 行为差异（第 4 章）：
```python
model.train(); a, b = model(x)[0], model(x)[0]     # 两次不同（随机丢）
model.eval();  c, d = model(x)[0], model(x)[0]     # 两次相同
```

2. 过拟合单 batch（dropout=0.1 时）：黄金测试仍应通过，只是需要更多步
（约 1.5~2 倍）。如果完全过拟合不了，检查是不是把 dropout 放上了主干道，
或 p 设得离谱（>0.5）。

**工程注记**：现代 LLM 预训练普遍 dropout=0（数据海量，欠拟合才是主要矛盾，
正则反而拖慢）；SFT/小数据微调时 0.05~0.1 常见。所以 gpt_model.py 的默认
值是 0——第 9 章预训练用默认，第 14 章微调再讨论开不开。
