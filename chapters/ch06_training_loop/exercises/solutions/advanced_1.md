# 进阶 1 参考答案：buggy_loop.py 的 5 个 bug

| # | Bug | 位置 | 预期症状 |
|---|---|---|---|
| 1 | **没有 `optimizer.zero_grad()`** | 训练循环内缺失 | 梯度跨步累加，等效 lr 持续放大——前期 loss 降得"格外快"，随后震荡/发散。最迷惑的是小任务上可能侥幸收敛，换个任务就炸 |
| 2 | **裁剪在 `optimizer.step()` 之后** | `clip_grad_norm_` 位置 | 等于没裁：更新已用未裁的梯度完成。毒 batch 出现时单步事故不设防（平时看不出差别——这类"平时无症状"的 bug 最危险） |
| 3 | **优化器在 `model.to(device)` 之前创建** | 开头三行的顺序 | AdamW 的 state 惰性创建所以本例侥幸无恙，但这是踩线写法：若优化器实现会预建 state（或 load 了 checkpoint），CUDA 上直接设备不匹配报错。顺序纪律：建模 → to(device) → 建优化器 |
| 4 | **`evaluate()` 结束没切回 `model.train()`** | evaluate 函数 | 第一次 eval（step 0）之后，模型永远停在 eval 模式——Dropout 全程失效。训练 loss 假性变好（没有 dropout 噪声），泛化受损且无报错。对照第 4 章易错点② |
| 5 | **`scheduler.step()` 每个训练 step 调，但 T_max=100 是按"总步数=100"设的吗？** | 调度配置 | 本例恰好 100 步所以周期对了——真正的 bug 是**耦合脆弱**：改训练步数忘改 T_max，lr 会提前衰完或衰不完。修复：T_max 与总步数用同一个变量；或换 LambdaLR 显式写 warmup+cosine（本例还完全没有 warmup，训练大模型时是第 6 个隐患） |

## 修复版核心循环

```python
model = build().to(device)                       # ③ 先搬设备
optimizer = torch.optim.AdamW(...)               #    再建优化器
TOTAL = 100
scheduler = LambdaLR(optimizer, make_warmup_cosine(warmup=10, total=TOTAL))  # ⑤ 解耦+warmup

for step in range(TOTAL):
    model.train()                                # ④ 保证每步都在 train 模式
    x, y = get_batch()
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # ② backward 后 step 前
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)        # ① 清零
```

## 复盘要点

- 5 个 bug **没有一个会报错**——训练代码的 bug 谱系里，"报错的是慈悲，静默的才要命"。
- 对应的防御手段（按性价比排序）：过拟合单 batch 测试（案例 1 黄金测试）、
  梯度范数/loss/lr 三线监控、固定训练模板不自由发挥、code review 时按六步次序清单核对。
