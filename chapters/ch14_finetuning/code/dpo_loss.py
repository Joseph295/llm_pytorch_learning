"""第 14 章 · DPO 损失实现 + 偏好概率提升验证

运行：uv run chapters/ch14_finetuning/code/dpo_loss.py

DPO 跳过奖励模型和强化学习，用一个分类损失直接在偏好对上训练。
本脚本实现 DPO 损失，并在玩具设置上验证它确实提高了偏好回答的相对概率。
"""

import torch
import torch.nn.functional as F

torch.manual_seed(0)


def dpo_loss(policy_chosen_logp, policy_rejected_logp,
             ref_chosen_logp, ref_rejected_logp, beta=0.1):
    """DPO 损失（14.2-④ 的公式）。

    输入是四个"序列对数概率"（log π(y|x) = 序列中各 token 的 log prob 之和）。
    直觉：提高 chosen 相对 rejected 的对数概率比（相对冻结的参考模型）。
    """
    # 策略模型的对数概率比 vs 参考模型的对数概率比
    policy_ratio = policy_chosen_logp - policy_rejected_logp
    ref_ratio = ref_chosen_logp - ref_rejected_logp
    logits = beta * (policy_ratio - ref_ratio)
    loss = -F.logsigmoid(logits).mean()
    # 隐式奖励（DPO 定义的奖励 = β·log π/π_ref）
    chosen_reward = beta * (policy_chosen_logp - ref_chosen_logp).detach()
    rejected_reward = beta * (policy_rejected_logp - ref_rejected_logp).detach()
    return loss, chosen_reward.mean(), rejected_reward.mean()


# ═══ 玩具设置：一个可训练的"打分"向量代表策略，冻结副本代表参考 ═══
# 简化：用标量 log-prob 模拟。真实场景 log-prob 来自模型对回答 token 的 log_softmax 求和。
class ToyPolicy(torch.nn.Module):
    """极简策略：对 chosen/rejected 各输出一个 log-prob。"""

    def __init__(self):
        super().__init__()
        # 8 个偏好对，每对有 chosen/rejected 两个"回答特征"
        self.chosen_logp = torch.nn.Parameter(torch.zeros(8))
        self.rejected_logp = torch.nn.Parameter(torch.zeros(8))


policy = ToyPolicy()
# 参考模型 = 策略的初始冻结副本（易错点⑤：参考必须冻结）
ref_chosen = policy.chosen_logp.detach().clone()
ref_rejected = policy.rejected_logp.detach().clone()

opt = torch.optim.AdamW(policy.parameters(), lr=0.1)

print("═══ DPO 训练：提高 chosen 相对 rejected 的概率 ═══")
print(f"{'step':>5} | {'loss':>7} | {'chosen奖励':>10} | {'rejected奖励':>12} | {'margin':>7}")
for step in range(60):
    loss, cr, rr = dpo_loss(policy.chosen_logp, policy.rejected_logp,
                            ref_chosen, ref_rejected, beta=0.1)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 15 == 0 or step == 59:
        print(f"{step:>5} | {loss.item():>7.4f} | {cr.item():>10.4f} | {rr.item():>12.4f} | {(cr - rr).item():>7.4f}")

print("""
读数：
- chosen 奖励上升、rejected 奖励下降 → margin（差距）增大 = 偏好被学到
- 损失下降 = σ(β·margin) 趋近 1 = 模型越来越偏好 chosen
- β 控制偏离参考的强度：太大→过度偏离丢能力，太小→学得慢（挑战题扫描）

vs RLHF-PPO（14.2-④）：
- DPO 只需 2 个模型（策略 + 冻结参考），标准梯度下降，稳定
- PPO 需 4 个模型 + 强化学习，复杂不稳定
- 真实 DPO 的 log-prob 来自模型对回答序列每个 token 的 log_softmax 求和
  （对指令部分不算，类似 SFT 掩码），此处用标量简化演示核心机制
""")
