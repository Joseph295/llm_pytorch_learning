# 基础 2 参考答案：算术强度手算

算术强度 = 计算量(FLOP) / 访存量(byte)。与硬件的 FLOPS/带宽比（ridge point）
比较：高于则 compute-bound，低于则 memory-bound。

**a) `(4096,4096) @ (4096,4096)` 矩阵乘**
- FLOP = 2·N³ = 2·4096³ ≈ 1.37×10¹¹
- 访存 = 读 A + 读 B + 写 C = 3·4096²·4 ≈ 2.0×10⁸ byte（fp32）
- **算术强度 ≈ 683 FLOP/byte** → 远高于任何硬件的 ridge point（A100 ≈ 156）→ **compute-bound**
- 优化方向：Tensor Core、低精度（bf16 翻倍吞吐）、更大矩阵摊薄

**b) `(1M,)` 张量的 `x + 1`**
- FLOP = 10⁶（每元素一次加）
- 访存 = 读 x + 写结果 = 2·10⁶·4 = 8×10⁶ byte
- **算术强度 = 0.125 FLOP/byte** → 远低于 ridge point → **memory-bound**
- 优化方向：融合（和相邻操作合并，一次读写）、就地操作（省一次写）

**c) `(4096,4096)` 的 RMSNorm**
- FLOP：平方(N²) + 求和(N²) + rsqrt(N行) + 乘(N²) ≈ 3·N² ≈ 5×10⁷
- 访存：读 x(N²·4) + 读 γ + 写结果(N²·4) ≈ 1.3×10⁸ byte
- **算术强度 ≈ 0.4 FLOP/byte** → **memory-bound**
- 这解释了第 8 章的实测意外：手写 RMSNorm（多个分立 kernel，多次读写 x）
  比融合 kernel 慢——RMSNorm 卡在访存，kernel 数量决定一切

**验证机器的 ridge point**（roofline 的拐点）：
```
ridge point = 峰值FLOPS / 峰值带宽
A100: 312e12 / 2e12 ≈ 156 FLOP/byte
M4（约）: 实测峰值FLOPS / 实测带宽（用 code/roofline.py 测）
```
算术强度 > ridge point → compute-bound，反之 memory-bound。

**总结这张表**：矩阵乘是 LLM 里唯一大宗的 compute-bound 操作；norm/激活/
逐元素全是 memory-bound。这就是为什么 torch.compile 和 FlashAttention
主攻"融合 memory-bound 操作"——它们是优化的主战场。
