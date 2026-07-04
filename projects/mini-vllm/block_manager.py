"""mini-vLLM · PagedAttention 的块分配器 + 块表

把 KV cache 切成固定大小的块按需分配（不要求连续），每个请求用块表
（=页表）记录用了哪些物理块。这就是操作系统虚拟内存搬到 KV cache 管理。
"""


class BlockManager:
    """固定大小块的分配器 + 引用计数（支持 copy-on-write 前缀共享）。"""

    def __init__(self, num_blocks: int, block_size: int = 16):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.free_blocks = list(range(num_blocks))      # 空闲块池
        self.ref_count = {i: 0 for i in range(num_blocks)}   # 每块引用计数

    def allocate(self, n: int) -> list[int]:
        """分配 n 个块，返回块 id 列表。不够则抛异常（真实 vLLM 会触发抢占）。"""
        if len(self.free_blocks) < n:
            raise MemoryError(f"KV 块不足：需要 {n}，剩余 {len(self.free_blocks)}")
        blocks = [self.free_blocks.pop() for _ in range(n)]
        for b in blocks:
            self.ref_count[b] = 1
        return blocks

    def free(self, blocks: list[int]):
        """释放块（引用计数归零才真正回到空闲池，支持共享块，易错点②）。"""
        for b in blocks:
            self.ref_count[b] -= 1
            if self.ref_count[b] == 0:
                self.free_blocks.append(b)

    def share(self, blocks: list[int]):
        """共享块（前缀共享用）：增加引用计数，不分配新块（copy-on-write 的基础）。"""
        for b in blocks:
            self.ref_count[b] += 1

    def blocks_needed(self, n_tokens: int) -> int:
        """n 个 token 需要几个块（向上取整）。"""
        return (n_tokens + self.block_size - 1) // self.block_size

    @property
    def num_free(self) -> int:
        return len(self.free_blocks)


class BlockTable:
    """一个请求的块表：逻辑 token 位置 → 物理块 + offset（=页表）。"""

    def __init__(self, manager: BlockManager):
        self.manager = manager
        self.blocks: list[int] = []                     # 本请求占用的物理块（有序）
        self.n_tokens = 0

    def append_token(self):
        """新增一个 token，必要时分配新块。返回它的 (物理块, offset)。"""
        if self.n_tokens % self.manager.block_size == 0:   # 当前块满了/还没块
            self.blocks.extend(self.manager.allocate(1))
        block_idx = self.n_tokens // self.manager.block_size
        offset = self.n_tokens % self.manager.block_size
        self.n_tokens += 1
        return self.blocks[block_idx], offset

    def locate(self, pos: int) -> tuple[int, int]:
        """逻辑位置 pos → (物理块, offset)。块表映射的核心（易错点①）。"""
        if pos >= self.n_tokens:
            raise IndexError(f"位置 {pos} 超出已分配 {self.n_tokens}")
        return self.blocks[pos // self.manager.block_size], pos % self.manager.block_size

    def free(self):
        """请求结束，释放所有块（易错点②）。"""
        self.manager.free(self.blocks)
        self.blocks = []
        self.n_tokens = 0
