"""第 11 章 · 用 torch.profiler 剖析 miniGPT 一步训练

运行：uv run chapters/ch11_performance/code/profile_minigpt.py

profiler 是 GPU 的火焰图：找出时间花在哪几个算子，判断是否有融合机会。
方法论与读 Spark UI 找慢 stage 一致——先看占比，再钻热点。
"""

import os
import sys

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "ch08_transformer", "code"))
from gpt_model import GPT, GPTConfig  # noqa: E402

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

cfg = GPTConfig(vocab_size=4096, block_size=256, n_layer=6, n_head=6, n_embd=384)
model = GPT(cfg).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
x = torch.randint(0, cfg.vocab_size, (16, cfg.block_size), device=device)
y = torch.randint(0, cfg.vocab_size, (16, cfg.block_size), device=device)


def step():
    _, loss = model(x, y)
    opt.zero_grad(); loss.backward(); opt.step()


# warmup（易错点②：不 warmup 数据全错）
for _ in range(5):
    step()
if device.type == "mps":
    torch.mps.synchronize()

# profiler 支持 CPU 与 CUDA activity；MPS 的 GPU 侧统计有限，
# 这里用 CPU activity 看算子调用与自身耗时（趋势与热点一致）
acts = [ProfilerActivity.CPU]
if device.type == "cuda":
    acts.append(ProfilerActivity.CUDA)

with profile(activities=acts, record_shapes=True) as prof:
    for _ in range(10):
        step()
    if device.type == "mps":
        torch.mps.synchronize()

print("═══ 算子耗时排名（Top 12）═══")
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=12))

print("""
怎么读这张表（方法论）：
1. 看排名靠前的算子——时间花在哪（帕累托，优化前 20% 热点）
2. 分类：matmul/addmm/attention 是 compute-bound（大矩阵，Tensor Core 主场）；
   add/mul/rms_norm/gelu/copy 是 memory-bound 胶水（融合机会）
3. 大量微小 kernel（每个耗时小但数量多）= torch.compile 的用武之地
4. 若某个 .item()/同步操作耗时异常 = 同步气泡（第 11.2-②）

导出 Chrome trace 看时间线（可视化气泡）：
    prof.export_chrome_trace("trace.json")   # chrome://tracing 或 perfetto.dev 打开
""")
