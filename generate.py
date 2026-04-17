"""Autoregressive text generation with temperature and top-k sampling."""

from __future__ import annotations
import argparse

import torch
import torch.nn.functional as F

from config import GPTConfig
from model import MiniGPT
from tokenizer import CharTokenizer, BPETokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Sampling
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(
    model: MiniGPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    """
    Autoregressively generate `max_new_tokens` tokens.

    Args:
        model:          trained MiniGPT (eval mode).
        idx:            (1, T) seed token tensor.
        max_new_tokens: number of tokens to generate.
        temperature:    >1 → more random, <1 → more deterministic.
        top_k:          if set, restricts sampling to top-k logits.

    Returns:
        (1, T + max_new_tokens) token tensor.
    """
    model.eval()
    ctx_len = model.cfg.context_length

    for _ in range(max_new_tokens):
        # Crop to context window
        idx_cond = idx[:, -ctx_len:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]                  # (1, vocab_size)

        # Temperature scaling
        logits = logits / max(temperature, 1e-8)

        # Top-k filtering
        if top_k is not None and top_k > 0:
            k = min(top_k, logits.size(-1))
            topk_vals, _ = torch.topk(logits, k)
            threshold = topk_vals[:, -1].unsqueeze(-1)
            logits = logits.masked_fill(logits < threshold, float("-inf"))

        probs = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_tok], dim=1)

    return idx


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="MiniGPT text generation")
    parser.add_argument("--checkpoint", default="checkpoints/model.pt")
    parser.add_argument("--tokenizer", default="checkpoints/tokenizer.json")
    parser.add_argument("--tokenizer_type", default="char", choices=["char", "bpe"])
    parser.add_argument("--prompt", default="\n")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    if args.tokenizer_type == "bpe":
        tokenizer = BPETokenizer.load(args.tokenizer)
    else:
        tokenizer = CharTokenizer.load(args.tokenizer)

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg: GPTConfig = ckpt["cfg"]
    model = MiniGPT(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])

    # Encode prompt
    ids = tokenizer.encode(args.prompt) or [0]
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    # Generate
    out = generate(
        model,
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
