"""挑战 1 参考答案：shuffle buffer 的随机性定量分析

运行：uv run chapters/ch05_data_pipeline/exercises/solutions/challenge_1.py
"""

import random
import statistics


def shuffle_buffer(stream, buffer_size: int, seed: int = 0):
    """流式打乱：buffer 填满后，每来一条随机换出一条。O(B) 内存打乱无限流。"""
    rng = random.Random(seed)
    buf = []
    for item in stream:
        if len(buf) < buffer_size:
            buf.append(item)
        else:
            j = rng.randrange(buffer_size)
            yield buf[j]                    # 随机弹出一条
            buf[j] = item                   # 新条目补位
    rng.shuffle(buf)                        # 流结束，清空 buffer
    yield from buf


N = 10_000
print(f"输入：有序流 [0..{N - 1}]，观察输出位置相对输入位置的漂移\n")
print(f"{'buffer':>7} | {'平均|漂移|':>10} | {'漂移P95':>8} | {'最大漂移':>8} | 备注")
print("-" * 62)
for B in [10, 100, 1000, 10_000]:
    out = list(shuffle_buffer(iter(range(N)), B, seed=42))
    assert sorted(out) == list(range(N)), "必须是一个排列（不重不漏）"
    drifts = [abs(pos - val) for pos, val in enumerate(out)]
    mean_d = statistics.mean(drifts)
    p95 = sorted(drifts)[int(0.95 * N)]
    note = "≈全局打乱" if B >= N else f"只在 ~{B} 邻域内打乱"
    print(f"{B:>7} | {mean_d:>10.1f} | {p95:>8} | {max(drifts):>8} | {note}")

print(f"""
结论：
1. 漂移量级 ≈ buffer 大小：B=100 时元素基本只在原位置 ±百级范围内挪动——
   若数据按类别/来源排序，B 太小等于"没打乱"，模型仍按块学习（易错点⑤的近亲）。
2. 定量选法：B 应 ≳ 数据中"同质块"的长度（比如单个 shard 的样本数），
   工业默认 1e4~1e6 条。配合"离线全局 shuffle 一次"双保险（5.2-⑤）。
3. 与水库抽样的联系/区别：同样是"固定内存处理无限流 + 均匀随机"，
   水库抽样解决的是"等概率抽 k 条"（丢弃其余），shuffle buffer 解决的是
   "全量输出但顺序随机"（不丢数据）。核心技巧同源：新元素与 buffer 内
   元素等概率换位。区别在输出语义——抽样是子集，shuffle 是排列，
   且 shuffle buffer 的随机性受 buffer 大小硬限制（漂移界），
   水库抽样的均匀性则是精确的。
""")
