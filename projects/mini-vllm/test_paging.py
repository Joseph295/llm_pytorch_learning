"""mini-vLLM · 块分配器与块表单元测试（基础练习 1/2）

运行：uv run projects/mini-vllm/test_paging.py
"""

from block_manager import BlockManager, BlockTable


def test_allocate_free():
    m = BlockManager(num_blocks=4, block_size=16)
    assert m.num_free == 4
    b = m.allocate(2)
    assert m.num_free == 2 and len(b) == 2
    m.free(b)
    assert m.num_free == 4                              # 释放后回到空闲池
    print("✓ 分配/释放：4 → 分配2 → 剩2 → 释放 → 回到4")


def test_exhaust():
    m = BlockManager(num_blocks=2, block_size=16)
    m.allocate(2)
    try:
        m.allocate(1)
        assert False, "应抛 MemoryError"
    except MemoryError:
        print("✓ 耗尽：块用完时正确抛 MemoryError（真实 vLLM 触发抢占）")


def test_block_table_mapping():
    """块表映射：逻辑位置 → 物理块 + offset，覆盖跨块边界（易错点①）。"""
    m = BlockManager(num_blocks=4, block_size=16)
    t = BlockTable(m)
    for _ in range(20):                                # 20 个 token 跨 2 个块
        t.append_token()
    # 位置 0~15 在第 0 块，位置 16~19 在第 1 块
    blk0, off0 = t.locate(0)
    blk15, off15 = t.locate(15)                        # 第 0 块最后一个
    blk16, off16 = t.locate(16)                        # 第 1 块第一个（跨块！）
    assert blk0 == blk15 and off0 == 0 and off15 == 15
    assert blk16 != blk0 and off16 == 0
    print(f"✓ 块表映射：pos0→(块{blk0},off0) pos15→(块{blk15},off15) "
          f"pos16→(块{blk16},off0) — 跨块边界正确")


def test_ref_count_sharing():
    """前缀共享：共享块引用计数 >1，全部释放才回收（挑战题基础）。"""
    m = BlockManager(num_blocks=4, block_size=16)
    b = m.allocate(1)
    m.share(b)                                          # 第二个请求共享这个块
    assert m.ref_count[b[0]] == 2
    m.free(b)                                           # 一个请求释放
    assert m.num_free == 3, "引用计数还有 1，块不应回收"
    m.free(b)                                           # 另一个也释放
    assert m.num_free == 4, "引用归零，块回收"
    print("✓ 引用计数共享：共享块引用=2，全部释放才回收（copy-on-write 基础）")


if __name__ == "__main__":
    print("═══ PagedAttention 块管理单元测试 ═══")
    test_allocate_free()
    test_exhaust()
    test_block_table_mapping()
    test_ref_count_sharing()
    print("\n全部通过 ✓ 这就是 vLLM block_manager 的核心逻辑（简化版）")
