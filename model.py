"""
GPT-style decoder-only Transformer model.
Components: TokenEmbedding, PositionalEncoding, MultiHeadMaskedSelfAttention,
            FeedForward, LayerNorm, residual connections, stacked TransformerBlocks,
            and a final linear language-model head.
"""

from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GPTConfig


# ─────────────────────────────────────────────────────────────────────────────
# Sub-modules
# ─────────────────────────────────────────────────────────────────────────────

class MultiHeadMaskedAttention(nn.Module):
    """Scaled dot-product multi-head attention with causal mask."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head

        # Fused QKV projection
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.out_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        # Causal mask (lower-triangular)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.context_length, cfg.context_length))
            .view(1, 1, cfg.context_length, cfg.context_length),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        # QKV split
        q, k, v = self.qkv(x).split(C, dim=2)
        # Reshape to (B, n_head, T, head_dim)
        def reshape(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q, k, v = reshape(q), reshape(k), reshape(v)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale                  # (B, H, T, T)
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v                                             # (B, H, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)      # (B, T, C)
        return self.resid_drop(self.out_proj(out))


class FeedForward(nn.Module):
    """Position-wise feed-forward network with GELU activation."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-LN transformer block: LN → Attention → residual → LN → FFN → residual."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = MultiHeadMaskedAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.ffn = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

class MiniGPT(nn.Module):
    """Minimal GPT-style decoder-only language model."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.n_embd)
        self.emb_drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # Weight tying: token embedding ↔ lm_head
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    # ── Initialisation ───────────────────────────────────────────────────

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        assert T <= self.cfg.context_length, (
            f"Sequence length {T} exceeds context_length {self.cfg.context_length}"
        )
        device = idx.device
        positions = torch.arange(T, device=device).unsqueeze(0)   # (1, T)

        x = self.emb_drop(self.token_emb(idx) + self.pos_emb(positions))
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                                   # (B, T, vocab)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss

    # ── Utilities ────────────────────────────────────────────────────────

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
