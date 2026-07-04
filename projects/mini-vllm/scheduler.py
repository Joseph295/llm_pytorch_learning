"""mini-vLLM · continuous batching 调度器

把调度粒度从"整个 batch"降到"单个 token 步"：每步所有活跃请求各生成一个
token，完成的立即退出、新请求立即填入。这就是动态任务调度搬到推理引擎。
"""

from dataclasses import dataclass, field

from block_manager import BlockManager, BlockTable


@dataclass
class Request:
    req_id: int
    prompt_tokens: list[int]
    max_new_tokens: int
    output_tokens: list[int] = field(default_factory=list)
    block_table: BlockTable = None
    finished: bool = False

    @property
    def all_tokens(self):
        return self.prompt_tokens + self.output_tokens

    @property
    def cur_pos(self):
        """当前生成位置（不同请求各不相同——RoPE 位置要各算各的，易错点③）。"""
        return len(self.all_tokens)


class Scheduler:
    """维护 running / waiting / finished 三队列，每步推进（17.2-①）。"""

    def __init__(self, manager: BlockManager, max_running: int = 8):
        self.manager = manager
        self.max_running = max_running
        self.waiting: list[Request] = []
        self.running: list[Request] = []
        self.finished: list[Request] = []

    def add_request(self, req: Request):
        self.waiting.append(req)

    def schedule(self) -> list[Request]:
        """把等待队列里的新请求加入 running（若有空位和足够 KV 块）。"""
        while self.waiting and len(self.running) < self.max_running:
            req = self.waiting[0]
            need = self.manager.blocks_needed(len(req.prompt_tokens) + 1)
            if self.manager.num_free < need:
                break                                   # 显存不足，等待（真实 vLLM 会抢占）
            req.block_table = BlockTable(self.manager)
            for _ in req.prompt_tokens:                 # prefill：为 prompt 分配块
                req.block_table.append_token()
            self.running.append(self.waiting.pop(0))
        return self.running

    def free_finished(self):
        """把完成的请求移出 running、释放 KV 块（易错点②）。"""
        still_running = []
        for req in self.running:
            if req.finished:
                req.block_table.free()                  # 释放块回空闲池
                self.finished.append(req)
            else:
                still_running.append(req)
        self.running = still_running

    @property
    def has_work(self):
        return bool(self.waiting or self.running)
