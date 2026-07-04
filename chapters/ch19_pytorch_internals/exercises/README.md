# 第 19 章 · 练习题

题目详情见[讲义 19.8 节](../README.md#198-练习题)。

| 题号 | 内容 | 难度 | 交付物 |
|---|---|---|---|
| 基础 1 | 追踪分发路径 | ★ | 见 code/dispatch_trace.py + 分析 |
| 基础 2 | 读源码找 kernel | ★ | 见 code/read_source.py + 追踪路径文档 |
| 进阶 1 | torch.library 自定义算子 | ★★ | 见 code/custom_op.py + fake 实现 |
| 挑战 1 | 手写 Triton kernel（需 CUDA，上云） | ★★★ | Triton fused kernel + 对拍加速 |

基础/进阶实现基座在 `code/`。挑战 1 需要 CUDA GPU（Triton 只支持 NVIDIA），
上云实战。

## 挑战 1 指引（Triton fused kernel）

```python
import triton, triton.language as tl, torch

@triton.jit
def fused_gelu_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < n                                  # 边界 mask（易错点③）
    x = tl.load(x_ptr + off, mask=mask)
    # GELU: 0.5*x*(1+erf(x/sqrt(2)))，用 tanh 近似
    out = 0.5 * x * (1 + tl.math.tanh(0.7978845608 * (x + 0.044715 * x*x*x)))
    tl.store(out_ptr + off, out, mask=mask)

def fused_gelu(x):
    out = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    fused_gelu_kernel[grid](x, out, n, BLOCK=1024)
    return out

# 对拍 + 测速
x = torch.randn(1<<20, device='cuda')
assert torch.allclose(fused_gelu(x), torch.nn.functional.gelu(x, approximate='tanh'), atol=1e-4)
```

对比：手写 Triton kernel vs PyTorch eager vs torch.compile 生成的 kernel。
这就是 FlashAttention（第 11 章）这类工作的底层——理解它 = 摸到性能优化的天花板。
