# 第 6 章 · 训练循环完全解剖：从 SGD 到 AdamW + AMP

> **本章目标**：拼出工业级训练循环。学完你应该能回答：
> 1. Adam 和 AdamW 差在哪？为什么 LLM 清一色 AdamW？
> 2. 为什么要 warmup？为什么 LLM 训练必开梯度裁剪？
> 3. bf16 和 fp16 差在哪一位？为什么混合精度还需要 fp32 主参数？
> 4. 一个正确的训练 step，六件事的顺序是什么、错序会怎样？

**前置**：第 3 章（累加语义）、第 4 章（参数管理）、第 5 章（数据）。 **硬件路径**：本地（bf16 AMP 在 MPS 实测）。 **预计用时**：5~6 小时。

---

## 6.1 来龙去脉：更新规则的演化史，每一步都在修一个具体的病

梯度下降的原始形态 `w -= lr * grad` 有一串临床症状，优化器的演化就是逐个治病的历史：

**病 1：mini-batch 梯度噪声大，走路摇摇晃晃** → **Momentum**（1980s）：维护梯度的指数滑动平均当作"速度"，噪声相互抵消、一致方向获得惯性。`v = β·v + g; w -= lr·v`。

**病 2：不同参数需要的步长天差地别**。embedding 里低频词的梯度稀疏微弱，LayerNorm 的 scale 梯度稠密强烈——统一 lr 顾此失彼 → **RMSProp/Adagrad**：用梯度平方的滑动平均给每个参数**自适应缩放**步长：`w -= lr · g / (√E[g²] + ε)`——梯度常年大的参数走小步，稀疏微弱的走大步。

**病 3：把 1 和 2 合起来** → **Adam**（2015）：一阶矩 m（momentum）+ 二阶矩 v（RMSProp）+ 偏差修正（m、v 从 0 初始化，早期严重低估，要除以 `1-β^t` 校正）。代价就是第 2 章算过的账：**每参数 8 字节的优化器状态**。

**病 4：Adam 的 L2 正则失效** → **AdamW**（2017）：传统做法把 L2 惩罚加进梯度（`g += λw`），但 Adam 会把这个惩罚项也除以 `√v` 自适应缩放——梯度大的参数正则被稀释，正则强度和梯度大小耦合，语义完全变形。AdamW 把 weight decay 从梯度里**解耦**出来，直接作用于权重：`w -= lr·λ·w`（decay）然后再做 Adam 更新。**这就是"decoupled weight decay"的全部含义**——一个实现细节的修正，效果显著到成为 LLM 训练的默认（LLaMA/GPT/Qwen 无一例外）。

**学习率**是这套机器上最重要的旋钮，但"一个恒定的 lr"在大模型上不工作，于是有了调度（6.2-③）。这一整套演化的经验教训值得体会：**优化器的进步大多不是新数学，而是对"哪里和直觉不符"的工程修正**。

---

## 6.2 核心原理

### ① 优化器解剖：state、param_groups、step

```python
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
opt.state          # dict: 参数 → {'step': t, 'exp_avg': m, 'exp_avg_sq': v}
opt.param_groups   # list[dict]: 参数分组，每组可有独立的 lr/weight_decay
```

三个立即有用的推论：
1. **state 是惰性创建的**——第一次 `step()` 才为每个参数分配 m/v。显存曲线在第一步跳涨 8 字节/参数就是它（OOM 常发生在第一步 step 而不是前向，排查时的重要线索）。
2. **param_groups 是分组治理的机制**：LLM 惯例是二维矩阵参数（weight）用 weight_decay，**bias 和 norm 参数不 decay**（它们不是"复杂度"的来源，decay 只会伤害表达；nanoGPT 的 `configure_optimizers` 是标准实现，见 6.5）。分层学习率（微调时底层小 lr）也走这个机制。
3. **`step()` 内部就是第 3 章易错点④的官方解法**：`with torch.no_grad()` 里原地更新参数本体，不建图、不换对象。

### ② 学习率调度：warmup + cosine 是 LLM 的标配形状

```
lr ▲    ／￣￣＼
   │   ／       ＼＿
   │  ／            ＼＿＿
   │ ／                   ＼＿＿＿＿
   └─────────────────────────────────→ step
     warmup         cosine decay
```

**为什么必须 warmup**（从 0 线性升到峰值，LLM 典型 500~2000 步）：训练初期两个不稳定因素叠加——Adam 的二阶矩 v 只见过几个 batch，估计噪声极大，自适应缩放会放出危险的大步子；同时深层网络初始化点附近的 loss 面粗糙。小步热身让 v 的统计变得可靠、让网络先走进平缓盆地。跳过 warmup 训大模型，前几百步 loss spike 甚至直接 NaN 的概率显著上升。

**为什么 cosine decay**：训练后期需要小步精修；cosine 的形状（先缓降后陡降再缓收尾）在实践中稳定好用，成为 GPT-3 以来的惯例。终点值通常不是 0 而是峰值的 10%。**注意 `scheduler.step()` 在 LLM 训练里是每个 optimizer step 调一次**（不是每 epoch——预训练根本没有 epoch 概念，只有 token 预算）。

### ③ 梯度裁剪：loss spike 的第一道保险丝

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

计算**所有参数梯度拼起来的全局 L2 范数**，若超过 max_norm 就整体等比缩小。语义：保方向、限步长。为什么 LLM 必开（max_norm=1.0 几乎是行业常数）：长尾 batch（罕见 token 组合、超长文档）会偶发地产生异常大梯度，一步大更新把参数踢出稳定区 → loss spike → 可能雪崩到 NaN。裁剪把这种"单步事故"的伤害限制住。**顺序关键**：backward 之后、step 之前（易错点③）。它同时是免费的监控探针：`clip_grad_norm_` 返回裁剪前的范数，**记录它**——梯度范数曲线是训练健康度的心电图（第 15 章排查的主要依据之一）。

### ④ 混合精度（AMP）：一场关于 5 个指数位的权衡

先看三种浮点格式的位布局：

```
fp32: 1 符号 | 8 指数 | 23 尾数   → 范围 ~1e38，精度高
fp16: 1 符号 | 5 指数 | 10 尾数   → 范围只到 65504！容易上溢/下溢
bf16: 1 符号 | 8 指数 |  7 尾数   → 范围同 fp32，精度粗
```

混合精度的动机：half 精度算得快（新 GPU 的 Tensor Core 对 half 有数倍吞吐）、激活省一半显存。问题是**范围**：fp16 的 65504 上限，注意力 logits、loss、梯度都可能越过——上溢成 inf；小梯度又下溢成 0。两条治理路线：

- **fp16 + GradScaler**（老路线）：loss 乘一个大系数（如 2¹⁶）再 backward，把小梯度抬进 fp16 可表达区，step 前再除回去；遇到 inf/nan 就跳过这步并调小系数。复杂、有额外失败模式。
- **bf16**（现代路线）：指数位和 fp32 一样多 → 范围问题消失，**不需要 GradScaler**。尾数只有 7 位精度粗，但深度学习对精度粗糙的容忍度远高于对范围溢出的容忍度。**Ampere（A100）之后的 GPU 和 Apple Silicon 都支持 bf16，LLM 训练已全面转向 bf16**——你的 M4 与云端 A100 用同一配置（第 0 章练习验证过 bf16 ✓）。

`torch.autocast` 做的事：**按算子分派精度**——matmul 等计算密集算子进 half，softmax/norm/loss 等数值敏感算子留 fp32。这就是"混合"的含义，不是全网 half。

**为什么还要 fp32 主参数**（第 2 章账本的最后一块拼图）：参数更新量 `lr·梯度` 相对参数本身往往小于 half 的分辨率——`1.0 + 0.0001` 在 bf16 里等于 1.0，**更新被整个吞掉**（尤其训练后期 lr 小的时候）。解法：参数的权威副本保持 fp32，更新累积在 fp32 上，前向时才转 half。手动 bf16 训练小模型可以不用主参数（本章实验会做对照），工业训练必用。

### ⑤ 把一切拼起来：训练 step 的六步次序

```python
for step, (x, y) in enumerate(loader):
    x, y = x.to(device), y.to(device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        loss = model(x, y) / accum_steps                    # ① 前向（AMP 区内）
    loss.backward()                                         # ② 反向（图在 AMP 区外销毁）
    if (step + 1) % accum_steps == 0:
        norm = clip_grad_norm_(model.parameters(), 1.0)     # ③ 裁剪（并记录 norm！）
        optimizer.step()                                    # ④ 更新
        scheduler.step()                                    # ⑤ 调度（每 optimizer step）
        optimizer.zero_grad(set_to_none=True)               # ⑥ 清梯度
```

顺序错乱的代价：裁剪放 step 后 = 没裁；zero_grad 放 backward 与 step 之间 = 白训；scheduler 忘调 = lr 恒定在 warmup 起点附近。这个模板（加上 checkpoint 保存与日志纪律）就是第 9 章 miniGPT 的训练循环——本章实验先在小模型上跑通全部零件。

---

## 6.3 动手实验

```bash
uv run chapters/ch06_training_loop/code/optimizer_anatomy.py   # 手写 SGD/AdamW 与官方对拍 + state 显存实测
uv run chapters/ch06_training_loop/code/lr_and_clip.py         # warmup+cosine 曲线 + 裁剪救 spike 演示
uv run chapters/ch06_training_loop/code/amp_training.py        # bf16 AMP 实测 + 完整六步模板训练
```

第三个脚本会在 MPS 上实测 bf16 autocast 的吞吐收益，并演示"没有 fp32 主参数时小更新被吞"的现象——④ 的两个论断都用数字验证。

---

## 6.4 易错点清单

**① weight decay 无差别应用到所有参数**
→ **现象**：不报错，但 norm 层的 scale 被往 0 拉、bias 被压制，收敛质量默默变差。
→ **修正**：param_groups 分组——`dim >= 2` 的参数（矩阵/embedding）decay，其余（bias、norm 的 weight）不 decay。nanoGPT 写法见 6.5-①。

**② zero_grad 的位置/遗漏**（第 3 章 Q4 的复习）
→ 梯度跨步累加 → 等效 lr 越来越大 → 发散。模板固定顺序，别自由发挥。梯度累积时只在累积边界清。

**③ 裁剪与 step 的顺序**
```python
optimizer.step()
clip_grad_norm_(model.parameters(), 1.0)    # ✗ 更新已经发生，裁了个寂寞
```
→ 裁剪必须在 backward 之后、step 之前。做梯度累积时：在**最后一个 micro-batch 的 backward 之后**裁一次（对累积后的总梯度裁）。

**④ resume 训练只存了模型权重**
→ **现象**：断点续训后 loss 先跳升一截再慢慢回落。
→ **原因**：AdamW 的 m/v 状态和 scheduler 的步数没存——优化器失忆，前几百步的自适应统计重新冷启动（等于又要 warmup 一次却没 warmup）。
→ **修正**：checkpoint 四件套：`model.state_dict()` + `optimizer.state_dict()` + `scheduler.state_dict()` + step/epoch 计数（+ 采样器/数据位置，严格复现还要 RNG 状态）。

**⑤ fp16 思维用在 bf16 上，或反之**
→ bf16 下套 GradScaler：无害但多余，还可能因 scaler 的 inf 检测逻辑引入无谓的跳步。fp16 下不用 scaler：小梯度大面积下溢，训练极慢或不收敛。**MPS 注**：MPS 上 fp16 的 GradScaler 支持不完整，又一个用 bf16 的理由。

**⑥ 优化器创建早于 model.to(device)**
→ **现象**（CUDA 上）：state 在 CPU 上创建，step 时设备不匹配报错或静默慢（跨设备）。
→ **修正**：顺序永远是 建模型 → `to(device)` → 建优化器。（加载 checkpoint 的 optimizer state 时，torch 会跟随参数设备，但自定义优化器未必——又一个买保险的理由。）

---

## 6.5 开源项目的最佳实践

**① nanoGPT 的 `configure_optimizers`：参数分组的教科书**
[karpathy/nanoGPT](https://github.com/karpathy/nanoGPT/blob/master/model.py) 里 30 行：按 `p.dim() >= 2` 分 decay/no_decay 两组；打印每组参数量（对账习惯！）；CUDA 上启用 `fused=True` 的 AdamW（把逐参数的更新循环融合成单 kernel，第 11 章会理解为什么快）。第 9 章我们照此实现。

**② LLaMA 论文里的真实超参**（把抽象讨论落到数字上）
7B 配置：AdamW(β₁=0.9, β₂=0.95, wd=0.1)、峰值 lr 3e-4、warmup 2000 步、cosine 衰减到峰值 10%、clip 1.0、bf16。**β₂=0.95 而不是默认 0.999** 是 LLM 圈的集体选择：二阶矩对最近梯度更敏感，配合大 batch 更稳（面试加分细节）。

**③ HF Trainer / accelerate 的 AMP 封装**
`TrainingArguments(bf16=True)` 一个开关背后就是本章 ④ 的整套机制（autocast 区域管理 + 必要时的 scaler + fp32 主参数由 optimizer 侧处理）。读 [accelerate 的 `prepare()`](https://github.com/huggingface/accelerate) 如何把 model/optimizer/loader 一起包装——第 12 章它还会替你处理分布式，同一个 API 面。

---

## 6.6 典型面试题

**Q1：Adam 和 AdamW 的区别？为什么 LLM 训练默认 AdamW？**

> **参考答案**：区别只在 weight decay 的施加位置——Adam 把 λw 加进梯度（随后被 1/√v 自适应缩放，正则强度与梯度幅度耦合、被稀释），AdamW 解耦成独立的 `w -= lr·λ·w` 步骤（正则语义纯净、强度可控）。实证上 AdamW 的泛化一致更好，且超参迁移性强，故成为 LLM 默认。**加分点**：报出 LLaMA 的 (0.9, 0.95, wd=0.1) 配置和 β₂ 调低的原因；提到 decay 不作用于 bias/norm 的分组惯例。

**Q2：为什么训练大模型要 warmup？warmup 步数怎么定？**

> **参考答案**：两个不稳定源——Adam 的二阶矩早期由极少样本估计、自适应步长不可靠地偏大；初始化点附近 loss 面粗糙，大步更新易把网络推进坏区域（激活/注意力分布崩坏）。warmup 用小步让统计和网络同时"进入状态"。步数经验：总步数的 0.1%~1%（LLM 常见 500~2000 步），batch 越大、模型越大越需要。**加分点**：warmup 不足的症状是前期 loss spike/NaN；相关技术如 embedding lr 单独调低、μP 参数化在超大模型上部分替代 warmup 的作用。

**Q3：bf16 和 fp16 各自的位布局与失败模式？为什么混合精度还需要 fp32 master weights？**

> **参考答案**：fp16 = 5 指数 + 10 尾数，范围仅 ±65504，失败模式是上/下溢（需 GradScaler 动态缩放 loss 规避）；bf16 = 8 指数 + 7 尾数，范围同 fp32，失败模式是精度粗（一般可容忍），无需 scaler。master weights：更新量 lr·g 常小于 half 的相对分辨率（如 bf16 只有 ~2-3 位十进制有效数字），`w += tiny` 会被舍入吞掉，训练后期整体停滞；fp32 权威副本上累积更新可避免。**加分点**：这正是第 2 章 16 字节/参数中那 4 字节的由来；推理时无更新，故纯 half 无碍。

**Q4：`clip_grad_norm_` 的语义？max_norm 怎么选？它和 loss spike 什么关系？**

> **参考答案**：对全部参数梯度的**全局** L2 范数设上限，超出按比例整体缩小——保方向、限模长（对比 clip_value 逐元素截断会改变方向）。max_norm=1.0 是 LLM 事实标准，本质是"每步参数位移不超过 lr 量级"的约束。与 spike：长尾 batch 的异常大梯度是 spike 的常见触发器，裁剪限制单步伤害；同时裁剪前范数是重要监控信号——持续逼近/超过阈值说明训练接近不稳定区，该查数据或调 lr 了。**加分点**：裁剪在梯度累积时机上的正确位置；DDP 下要在梯度同步后裁（第 12 章）。

---

## 6.7 疑难杂症排查

**案例 1：loss 完全不降——五层排查树**

① **能过拟合单 batch 吗？**（黄金测试：抓一个 batch 反复训练，正常实现应几十步内 loss→0）不能 → 代码 bug（模型/损失/优化器连接问题：参数没进优化器、loss 写错、梯度断链）；
② 能 → lr 扫描：3e-3/3e-4/3e-5 三档各跑几百步，全都不动 → 查数据（标签错位、输入全同）；
③ 有一档动 → lr 问题，从那档细调；
④ 前期动后期平 → 调度器问题（lr 衰没了）或容量不足；
⑤ 检查 `sum(p.requires_grad for p in model.parameters())`——是不是大半参数被冻着（易错点遗产）。

**案例 2：loss 有规律地周期性跳动**

看跳动周期对得上什么：对上 epoch 边界 → 数据侧（shuffle 关了/dataset 有序 + 周期重置）；对上 checkpoint/eval 间隔 → eval 后忘 `model.train()`（第 4 章易错点②）或 eval 污染了 RNG 状态；对上梯度累积周期 → loss 除以 accum_steps 忘了或裁剪位置错。**方法论**：周期性异常先找同周期的事件源。

**案例 3：AMP 一开 loss 变 NaN**

排查序：① 用的 fp16？先换 bf16（一半的此类问题直接消失）；② bf16 还 NaN → 病根多半不在精度而在数值稳定性边界被放大：找 log(0)/除 0/超大 logits（第 4 章 NaN 哨兵上场，二分定位第一枚 NaN 在哪层）；③ autocast 区域是否包了不该包的（loss 计算建议留在 fp32——cross_entropy 内部有 log_softmax）；④ 检查有没有手写 `.half()` 强转参数（应该让 autocast 管理，参数保持 fp32）。

---

## 6.8 练习题

### 基础 1：手写 AdamW
按 6.1 的公式实现 `my_adamw_step(params, state, lr, betas, eps, wd)`（含偏差修正与解耦 decay），与 `torch.optim.AdamW` 对同一模型跑 10 步对拍（参数 allclose）。

### 基础 2：warmup + cosine 调度器
实现 `get_lr(step, warmup, total, peak, floor)` 纯函数版，画出（打印数据点即可）2000 步的曲线，标出三个阶段的边界。对拍 `torch.optim.lr_scheduler.LambdaLR` 封装版。

### 进阶 1：找 bug 专项
`exercises/buggy_loop.py` 里有一个埋了 5 个 bug 的训练循环（本章与前章的易错点混编）。找齐、修复、并说明每个 bug 的预期症状。答案里有埋雷清单。

### 挑战 1：优化器对比实验
用合成的两团高斯分类任务（几百样本的小 MLP），对比四种配置的收敛曲线：SGD / SGD+momentum / AdamW 恒定 lr / AdamW+warmup+cosine。每种跑 3 个种子取均值。给出你的结论：什么时候 Adam 系明显赢？（提示：把特征尺度故意做成不均匀的——第 6.1 节"病 2"的重现。）

---

## 本章小结与下一章预告

训练循环 = 六步固定次序（前向/反向/裁剪/更新/调度/清零）。AdamW 是"每参数自适应 + 解耦正则"的当前答案（8 字节/参数的代价）；warmup+cosine 是 lr 的标准形状；bf16 让混合精度回归简单；fp32 主参数保住微小更新。至此，**第一篇完结**——模型、数据、梯度、更新四大件你已全部拆解过。

**下一章（第 7 章，第二篇开篇）**：注意力机制。从"RNN 为什么不行"出发，一步步推导出 self-attention 的必然形状，然后手写多头注意力——第 1 章的因果掩码广播、第 2 章的 view/transpose 折腾、本章的数值稳定性直觉，全部在 60 行核心代码里会师。
