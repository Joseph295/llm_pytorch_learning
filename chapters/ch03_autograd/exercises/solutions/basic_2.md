# 基础 2 参考答案：梯度累积等价性

完整可运行版本即本章 `code/grad_accumulation.py`（该脚本就是按这道题的要求写的），
验证结果：batch=8 整批 vs batch=2×4 次累积（每次 loss/4），`.grad` allclose ✓。

**为什么 loss 要除以累积步数 K：**

mean 型 loss 下，整批梯度 = ∇(1/N)Σℓᵢ。K 个 micro-batch 各自的 loss 是
(K/N)Σ_micro ℓᵢ 的均值，直接累加后梯度等于整批的 K 倍。除以 K 才对齐。

**不除以 K 的后果**：等效学习率放大 K 倍。K=8、16 时训练大概率发散；
更隐蔽的是 K 较小时不发散但超参失真——你调出来的学习率换个 K 就不对了。

**边界情况（第 9 章会遇到真实版）**：语言模型按 token 平均的 loss，若各
micro-batch 的有效 token 数不同（padding 不均），简单除以 K 不再严格等价，
应按 token 数加权：`loss * n_tokens_micro / n_tokens_total`。
