"""第 16 章 · 投机解码模拟：用小模型草稿 + 大模型验证，不损分布

运行：uv run chapters/ch16_inference/code/speculative.py

投机解码：草稿模型生成 K 个候选，目标模型一次前向并行验证，拒绝采样保证
输出分布与纯目标模型一致。本脚本用两个"分布"模拟，验证分布一致 + 加速。
"""

import torch

torch.manual_seed(0)

VOCAB = 32


# 用固定的"模型"= 给定上下文的下一 token 概率分布（真实中来自 LM 前向）
def target_dist(context_sum):
    """目标模型（大、准）的下一 token 分布。"""
    logits = torch.randn(VOCAB, generator=torch.Generator().manual_seed(context_sum % 10000))
    return logits.softmax(0)


def draft_dist(context_sum):
    """草稿模型（小、快、略偏）：目标分布 + 噪声。"""
    logits = torch.randn(VOCAB, generator=torch.Generator().manual_seed(context_sum % 10000))
    logits += 0.3 * torch.randn(VOCAB, generator=torch.Generator().manual_seed((context_sum + 1) % 10000))
    return logits.softmax(0)


def speculative_step(context, K, rng):
    """一步投机：草稿生成 K 个，目标验证（拒绝采样）。返回接受的 token 列表。"""
    ctx = list(context)
    # 1. 草稿模型自回归生成 K 个候选
    drafts, draft_probs = [], []
    for _ in range(K):
        p = draft_dist(sum(ctx))
        tok = torch.multinomial(p, 1, generator=rng).item()
        drafts.append(tok); draft_probs.append(p[tok].item())
        ctx.append(tok)
    # 2. 目标模型一次前向并行验证（真实中是一次前向算 K+1 个位置）
    ctx = list(context)
    accepted = []
    for i, tok in enumerate(drafts):
        p_target = target_dist(sum(ctx))
        p_t, p_d = p_target[tok].item(), draft_probs[i]
        # 拒绝采样：以 min(1, p_target/p_draft) 概率接受（保证分布 = 目标分布）
        if torch.rand(1, generator=rng).item() < min(1.0, p_t / max(p_d, 1e-9)):
            accepted.append(tok); ctx.append(tok)
        else:
            # 拒绝：从修正分布 (p_target - p_draft)+ 重采样一个，停止
            resid = torch.clamp(target_dist(sum(ctx)) - draft_dist(sum(ctx)), min=0)
            resid = resid / resid.sum()
            accepted.append(torch.multinomial(resid, 1, generator=rng).item())
            break
    return accepted


# ═══ 1. 分布一致性验证：投机解码 vs 纯目标模型 ═══
def pure_target_gen(context, n, rng):
    ctx = list(context)
    out = []
    for _ in range(n):
        p = target_dist(sum(ctx))
        tok = torch.multinomial(p, 1, generator=rng).item()
        out.append(tok); ctx.append(tok)
    return out


# 统计第一个生成 token 的分布，对比两种方法
ctx0 = [5, 12]
N = 20000
rng = torch.Generator().manual_seed(1)
pure_first = torch.zeros(VOCAB)
spec_first = torch.zeros(VOCAB)
for _ in range(N):
    pure_first[pure_target_gen(ctx0, 1, rng)[0]] += 1
    spec_first[speculative_step(ctx0, 4, rng)[0]] += 1
pure_first /= N; spec_first /= N
tv_dist = 0.5 * (pure_first - spec_first).abs().sum().item()    # 总变差距离
print(f"投机解码 vs 纯目标模型 第一 token 分布的总变差距离: {tv_dist:.4f}")
print(f"（接近 0 = 分布一致，投机解码不损质量的数学保证 ✓）")

# ═══ 2. 加速：接受率决定加速比 ═══
rng = torch.Generator().manual_seed(2)
total_accepted, total_calls = 0, 0
for _ in range(2000):
    acc = speculative_step([5, 12], K=4, rng=rng)
    total_accepted += len(acc)
    total_calls += 1                                     # 每次投机 = 1 次目标模型前向
avg_accept = total_accepted / total_calls
print(f"\n平均每次目标模型前向产出 {avg_accept:.2f} 个 token（K=4）")
print(f"→ 加速比 ≈ {avg_accept:.1f}×（纯目标模型每次前向只产 1 个 token）")
print("""
读数（16.2-④）：
- 投机解码用草稿模型的便宜生成 + 目标模型的一次并行验证换取加速
- 拒绝采样规则保证输出分布 = 纯目标模型分布（不损质量，数学保证，非近似）
- 加速比 = 每次目标前向的平均接受 token 数，取决于草稿命中率
- 草稿模型越接近目标（同系列小模型），命中率越高，加速越大（通常 2-3×）
- 利用了 decode 的 memory-bound 特性：目标模型一次前向验证 K 个 token
  几乎和验证 1 个一样便宜（空闲算力，16.2-②）
- 变体：Medusa（自投机多头）、EAGLE（特征级投机）省掉独立草稿模型
""")
