"""第 19 章 · 用 torch.library 注册自定义算子

运行：uv run chapters/ch19_pytorch_internals/code/custom_op.py

现代自定义算子的正确姿势：torch.library.custom_op + register_fake，
让算子能被 autograd 和 torch.compile 识别（19.2-③，易错点④）。
"""

import torch

# ═══ 注册一个融合算子：swish_bias(x, b) = x * sigmoid(x) + b ═══
# 用 @torch.library.custom_op 注册（PyTorch 2.4+ 的现代 API）
try:
    @torch.library.custom_op("mylib::swish_bias", mutates_args=())
    def swish_bias(x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x) + b

    # fake（meta）实现：告诉 torch.compile 输出的形状/dtype，无需真实计算（易错点④）
    @swish_bias.register_fake
    def _(x, b):
        return torch.empty_like(x)

    # 反向（让它支持 autograd）
    def backward(ctx, grad):
        x, = ctx.saved_tensors
        sig = torch.sigmoid(x)
        dx = grad * (sig + x * sig * (1 - sig))          # d/dx[x·σ(x)]
        return dx, grad                                   # 对 x 和 b 的梯度

    def setup_context(ctx, inputs, output):
        x, b = inputs
        ctx.save_for_backward(x)

    swish_bias.register_autograd(backward, setup_context=setup_context)

    HAS_CUSTOM_OP = True
except (AttributeError, TypeError) as e:
    HAS_CUSTOM_OP = False
    print(f"此 torch 版本的 custom_op API 不同（{e}）；概念见讲义 19.2-③")


if HAS_CUSTOM_OP:
    print("═══ 1. 自定义算子正确性 ═══")
    x = torch.randn(4, requires_grad=True)
    b = torch.randn(4, requires_grad=True)
    out = torch.ops.mylib.swish_bias(x, b)
    ref = x * torch.sigmoid(x) + b
    print(f"自定义算子输出 == 参考实现: {torch.allclose(out, ref)}")

    print("\n═══ 2. autograd 正确性（gradcheck，第 3 章）═══")
    xd = torch.randn(4, dtype=torch.float64, requires_grad=True)
    bd = torch.randn(4, dtype=torch.float64, requires_grad=True)
    ok = torch.autograd.gradcheck(lambda x, b: torch.ops.mylib.swish_bias(x, b), (xd, bd))
    print(f"gradcheck: {ok}（自定义算子的反向数值正确）")

    print("\n═══ 3. 能被 torch.compile 处理（无 graph break，易错点④）═══")
    def model(x, b):
        return torch.ops.mylib.swish_bias(x, b).sum()

    try:
        compiled = torch.compile(model)
        r = compiled(torch.randn(4), torch.randn(4))
        print(f"torch.compile 成功处理自定义算子（fake 实现让它理解形状）✓")
    except Exception as e:
        print(f"compile: {str(e)[:60]}（MPS 上 compile 支持有限，CUDA 上完整）")

    print("""
要点（19.2-③）：
- torch.library.custom_op 注册的算子进入 dispatcher，享受和内置算子一样的待遇
- register_fake：提供形状/dtype 推断，让 torch.compile 能处理（否则 graph break）
- register_autograd：提供反向，让它支持梯度
- 这比老式 cpp_extension 更好：一等公民，compile/autograd/分布式都协同
- 真正的性能算子会在这里放 Triton/CUDA kernel（挑战题）
""")
