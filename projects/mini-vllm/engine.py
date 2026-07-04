"""mini-vLLM · 推理引擎主循环：组装 scheduler + block manager + 模型

把 continuous batching（token 级动态调度）+ PagedAttention（块管理）+ KV cache
串成一个引擎。用第 8 章的 GPT 做模型，服务多个并发请求。

注：为聚焦"调度 + 分页"的系统逻辑，每请求用简单的逐请求 KV cache 做实际注意力，
    块管理器负责显存块的分配/释放账本（真实 vLLM 用块表直接索引 KV kernel）。
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                "chapters", "ch08_transformer", "code"))
from block_manager import BlockManager  # noqa: E402
from scheduler import Request, Scheduler  # noqa: E402


class MiniVLLM:
    """极简推理引擎：continuous batching 调度 + 分页 KV 块管理。"""

    def __init__(self, model, eos_token: int, num_kv_blocks=64, block_size=16, max_running=8):
        self.model = model.eval()
        self.eos = eos_token
        self.manager = BlockManager(num_kv_blocks, block_size)
        self.scheduler = Scheduler(self.manager, max_running)
        self.device = next(model.parameters()).device
        self.step_count = 0

    def add_request(self, req_id, prompt_tokens, max_new_tokens=50):
        self.scheduler.add_request(Request(req_id, list(prompt_tokens), max_new_tokens))

    @torch.no_grad()
    def _forward_step(self, requests):
        """对所有活跃请求各生成一个 token。

        简化：每请求独立跑一次前向（真实引擎会 batch + 变长 kernel）。
        重点展示调度逻辑，不是 kernel 优化。
        """
        for req in requests:
            idx = torch.tensor([req.all_tokens[-self.model.cfg.block_size:]], device=self.device)
            logits, _ = self.model(idx)
            next_tok = logits[0, -1].argmax().item()      # 贪心（演示用）
            req.output_tokens.append(next_tok)
            # 为新 token 记账一个 KV 槽（分页：满块则分配新块）
            req.block_table.append_token()
            # 结束条件：EOS 或达到 max_new_tokens
            if next_tok == self.eos or len(req.output_tokens) >= req.max_new_tokens:
                req.finished = True

    def run(self, verbose=True):
        """主循环（17.2-③）：调度 → 前向 → 释放，直到所有请求完成。"""
        log = []
        while self.scheduler.has_work:
            running = self.scheduler.schedule()           # 1. 新请求入队（若有空闲块）
            if not running:
                break
            self._forward_step(running)                   # 2-4. 各生成一个 token + 记账块
            self.scheduler.free_finished()                # 5. 完成的退出、释放块
            self.step_count += 1
            if verbose and self.step_count % 10 == 0:
                log.append(f"  step {self.step_count:3d}: running={len(self.scheduler.running)} "
                           f"waiting={len(self.scheduler.waiting)} "
                           f"finished={len(self.scheduler.finished)} "
                           f"空闲KV块={self.manager.num_free}/{self.manager.num_blocks}")
        if verbose:
            print("\n".join(log))
        return self.scheduler.finished
