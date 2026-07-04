"""第 18 章 · 服务化性能指标模拟：TTFT / TPOT / 吞吐 的权衡

运行：uv run chapters/ch18_deployment/code/serving_metrics.py

模拟不同 batch size 下的三个指标，展示"延迟 vs 吞吐"权衡曲线（无需真实 GPU）。
"""

# 简化模型（基于第 16 章 memory-bound 特性）：
# - prefill 时间 ∝ prompt 长度（compute-bound）
# - decode 每步时间：读一次权重(固定) + batch 内并行（batching 摊薄）
WEIGHT_READ_MS = 20.0        # 读一遍模型权重的时间（decode 的主要成本，memory-bound）
PER_TOKEN_COMPUTE_MS = 0.5   # 单 token 的计算（batch 内累加）
PREFILL_MS_PER_TOKEN = 0.3   # prefill 每 token


def simulate(batch_size, prompt_len=200, gen_len=100):
    """返回 (TTFT_ms, TPOT_ms, 吞吐_token/s)。"""
    # TTFT：prefill 整个 prompt（batch 内各请求 prefill，简化为串行估计）
    ttft = prompt_len * PREFILL_MS_PER_TOKEN
    # TPOT：每个 decode step 读一次权重（固定）+ batch 内每请求的计算
    # 关键：权重读取被 batch 内所有请求摊薄（16.2-②）
    tpot_per_step = WEIGHT_READ_MS + batch_size * PER_TOKEN_COMPUTE_MS
    tpot = tpot_per_step / batch_size          # 每请求每 token 的等效时间... 不对
    # 修正：TPOT 是单请求视角的每 token 延迟 = 整个 step 时间（所有请求同步生成一个 token）
    tpot = tpot_per_step
    # 吞吐：每步产出 batch_size 个 token
    throughput = batch_size / (tpot_per_step / 1000)   # token/s
    return ttft, tpot, throughput


print("═══ batch size 对 TTFT / TPOT / 吞吐的影响 ═══\n")
print(f"{'batch':>6} | {'TTFT(ms)':>9} | {'TPOT(ms)':>9} | {'吞吐(tok/s)':>12} | {'单请求延迟':>10}")
print("-" * 60)
for B in [1, 2, 4, 8, 16, 32, 64]:
    ttft, tpot, thr = simulate(B)
    single_req_latency = ttft + 100 * tpot        # 首 token + 100 个后续 token
    print(f"{B:>6} | {ttft:>9.1f} | {tpot:>9.1f} | {thr:>12.0f} | {single_req_latency:>9.0f}ms")

print("""
读数（18.2-①）：
- batch 越大，吞吐越高（权重读取被摊薄，memory-bound 的免费午餐，16.2-②）
- 但 batch 越大，每步时间越长 → TPOT 增大 → 单请求延迟增大
- 这就是"延迟 vs 吞吐"的核心权衡：
    * 对话场景（用户等着看回复）：小 batch，优先低延迟（TTFT/TPOT 小）
    * 离线批处理（如批量总结文档）：大 batch，优先高吞吐（成本低）
- continuous batching（第 17 章）让这个权衡更优：动态调度使 GPU 始终满载，
  在给定延迟约束下最大化吞吐
生产调优：max-num-seqs（batch 上限）就是这条曲线上的工作点选择（18.2-②）。
""")

# ═══ 延迟约束下的最优 batch ═══
print("═══ 给定 TPOT ≤ 50ms 约束下的最大吞吐 batch ═══")
best_b, best_thr = 1, 0
for B in [1, 2, 4, 8, 16, 32, 64, 128]:
    _, tpot, thr = simulate(B)
    if tpot <= 50 and thr > best_thr:
        best_b, best_thr = B, thr
print(f"满足 TPOT≤50ms 的最大吞吐：batch={best_b}，吞吐={best_thr:.0f} token/s")
print("→ SLO 驱动的调度：在延迟约束内最大化吞吐，是生产服务的核心优化目标")
