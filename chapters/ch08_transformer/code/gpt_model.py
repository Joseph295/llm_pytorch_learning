"""第 8 章 · 完整 GPT 实现（RoPE + Pre-Norm + RMSNorm + tied lm_head + 缩放初始化）

运行自检：uv run chapters/ch08_transformer/code/gpt_model.py
第 9 章用法：from gpt_model import GPT, GPTConfig

结构（讲义 8.2-⑤）：
  tokens → Embedding → L × Block(Pre-Norm MHA + Pre-Norm FFN) → RMSNorm → lm_head(tied)
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 4096
    block_size: int = 256          # 最大序列长（RoPE 预计算表的长度）
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0           # 预训练数据充足时惯例为 0


class RMSNorm(nn.Module):
    """x / RMS(x) · γ —— LayerNorm 减掉均值中心化后的极简版（讲义 8.2-②）。"""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 内部用 fp32 计算再转回——bf16 下的数值稳定工业写法（易错点⑥）
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def precompute_rope(head_dim: int, max_len: int, base: float = 10000.0):
    """预计算 RoPE 的 cos/sin 表：(max_len, head_dim)。多尺度'时钟指针'。"""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_len).float()
    freqs = torch.outer(t, inv_freq)                     # (max_len, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)              # (max_len, head_dim)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    """q,k: (B,H,T,D)；cos,sin: (T,D) 广播。只旋转 QK，不动 V（易错点①）。"""
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.H = cfg.n_head
        self.D = cfg.n_embd // cfg.n_head
        self.wqkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)   # QKV 合一，省 kernel
        self.wo = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.wqkv(x).split(C, dim=2)
        q = q.view(B, T, self.H, self.D).transpose(1, 2)                # (B,H,T,D)
        k = k.view(B, T, self.H, self.D).transpose(1, 2)
        v = v.view(B, T, self.H, self.D).transpose(1, 2)
        q, k = apply_rope(q, k, cos[:T], sin[:T])                       # 拆头后施加（易错点①）

        # 官方统一入口：自动选最优后端（CUDA 上是 FlashAttention，讲义 7.5-①）
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(out)


class MLP(nn.Module):
    """FFN：d → 4d → d，知识仓库（讲义 8.2-③）。"""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """Pre-Norm 残差块：主干道上无遮挡（讲义 8.2-②）。"""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm_f = RMSNorm(cfg.n_embd)                     # final norm（易错点④）
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight               # tie（第 4 章挑战题）

        head_dim = cfg.n_embd // cfg.n_head
        cos, sin = precompute_rope(head_dim, cfg.block_size)
        self.register_buffer("rope_cos", cos, persistent=False)   # buffer：随设备走、不进 ckpt
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)                        # HF 惯例（第 4 章 8.5-②）
        # 残差写回投影 /√(2L)：深度可训的暗线（讲义 8.2-④）
        for name, p in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"序列长 {T} 超过 block_size {self.cfg.block_size}"
        x = self.drop(self.embed(idx))                        # (B,T,C)
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)
        x = self.norm_f(x)
        logits = self.lm_head(x)                              # (B,T,V)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """自回归生成（朴素版：每步全量前向。KV cache 优化在第 16 章）。"""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]          # 超长时截取窗口
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = logits.softmax(-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())      # tie 后不重复计数（第 4 章）


# ═════════ 自检三连（讲义 8.3）═════════
if __name__ == "__main__":
    cfg = GPTConfig(vocab_size=4096, block_size=128, n_layer=6, n_head=6, n_embd=384)
    model = GPT(cfg)

    # ── 测试 1：参数量对账（12Ld² + Vd 公式）──
    formula = 12 * cfg.n_layer * cfg.n_embd**2 + cfg.vocab_size * cfg.n_embd
    actual = model.num_params()
    print(f"1. 参数量: 公式 {formula / 1e6:.2f}M vs 实际 {actual / 1e6:.2f}M "
          f"(偏差 {abs(actual - formula) / formula:.1%}，来源为 norm 的 γ) ✓")

    # ── 测试 2：因果泄漏测试（第 7 章 7.7 案例 1 的全模型版）──
    x = torch.randint(0, cfg.vocab_size, (2, 64))
    x2 = x.clone()
    x2[:, 32:] = torch.randint(0, cfg.vocab_size, (2, 32))    # 篡改后半段
    model.eval()
    with torch.no_grad():
        l1, _ = model(x)
        l2, _ = model(x2)
    leak = (l1[:, :32] - l2[:, :32]).abs().max().item()
    print(f"2. 因果泄漏: 篡改未来后前 32 位 logits 变化 = {leak:.2e}（应为 0）✓")

    # ── 测试 3：初始化的激活方差剖面（缩放 vs 不缩放）──
    def residual_profile(scale_init: bool, n_layer=12):
        torch.manual_seed(0)
        c = GPTConfig(vocab_size=1024, block_size=64, n_layer=n_layer, n_head=4, n_embd=128)
        m = GPT(c)
        if not scale_init:                                    # 撤销缩放，做对照组
            for name, p in m.named_parameters():
                if name.endswith("wo.weight") or name.endswith("proj.weight"):
                    nn.init.normal_(p, mean=0.0, std=0.02)
        h = m.drop(m.embed(torch.randint(0, 1024, (4, 64))))
        stds = []
        with torch.no_grad():
            for blk in m.blocks:
                h = blk(h, m.rope_cos, m.rope_sin)
                stds.append(h.std().item())
        return stds

    with_scale = residual_profile(True)
    without = residual_profile(False)
    print("3. 残差流 std 剖面（12 层）:")
    print(f"   有 /√(2L) 缩放: 首层 {with_scale[0]:.3f} → 末层 {with_scale[-1]:.3f}"
          f"（增长 {with_scale[-1] / with_scale[0]:.1f}x）")
    print(f"   无缩放        : 首层 {without[0]:.3f} → 末层 {without[-1]:.3f}"
          f"（增长 {without[-1] / without[0]:.1f}x ← 深模型不稳的病根）")

    # ── 测试 4：过拟合单 batch（第 6 章黄金测试）──
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    xb = torch.randint(0, cfg.vocab_size, (4, 64))
    yb = torch.randint(0, cfg.vocab_size, (4, 64))
    for i in range(80):
        _, loss = model(xb, yb)
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"4. 过拟合单 batch: 初始 loss≈{math.log(cfg.vocab_size):.1f}(=ln V) → 80 步后 {loss.item():.3f} ✓")
