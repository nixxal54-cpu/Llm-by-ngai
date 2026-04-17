"""Training loop for MiniGPT."""

from __future__ import annotations
import os
import time
from pathlib import Path

import torch

from config import GPTConfig
from dataset import TextDataset
from model import MiniGPT
from tokenizer import CharTokenizer, BPETokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_loss(
    model: MiniGPT,
    dataset: TextDataset,
    cfg: GPTConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    out: dict[str, float] = {}
    for split in ("train", "val"):
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            x, y = dataset.get_batch(split, cfg.batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def build_tokenizer(cfg: GPTConfig, text: str):
    if cfg.tokenizer_type == "bpe":
        tok = BPETokenizer(vocab_size=512).fit(text)
    else:
        tok = CharTokenizer().fit(text)
    return tok


# ─────────────────────────────────────────────────────────────────────────────
# Main training entry point
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: GPTConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device: {device}")

    # ── Tokenizer ────────────────────────────────────────────────────────
    text = Path(cfg.data_path).read_text(encoding="utf-8")
    tokenizer = build_tokenizer(cfg, text)
    cfg.vocab_size = tokenizer.vocab_size
    print(f"[train] vocab_size: {cfg.vocab_size}")

    Path(cfg.tokenizer_path).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(cfg.tokenizer_path)

    # ── Dataset ──────────────────────────────────────────────────────────
    dataset = TextDataset(cfg.data_path, tokenizer, cfg.context_length)

    # ── Model ────────────────────────────────────────────────────────────
    model = MiniGPT(cfg).to(device)
    print(f"[train] parameters: {model.num_parameters():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=0.1,
        betas=(0.9, 0.95),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.max_iters
    )

    # ── Loop ─────────────────────────────────────────────────────────────
    model.train()
    t0 = time.time()

    for step in range(1, cfg.max_iters + 1):
        # Evaluation checkpoint
        if step % cfg.eval_interval == 0 or step == 1:
            losses = estimate_loss(model, dataset, cfg, device)
            elapsed = time.time() - t0
            print(
                f"step {step:>5d}/{cfg.max_iters} | "
                f"train loss: {losses['train']:.4f} | "
                f"val loss: {losses['val']:.4f} | "
                f"lr: {scheduler.get_last_lr()[0]:.2e} | "
                f"time: {elapsed:.1f}s"
            )
            # Save checkpoint
            torch.save(
                {
                    "step": step,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "cfg": cfg,
                    "val_loss": losses["val"],
                },
                cfg.checkpoint_path,
            )

        x, y = dataset.get_batch("train", cfg.batch_size, device)
        _, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

    print(f"[train] done. checkpoint saved to {cfg.checkpoint_path}")


if __name__ == "__main__":
    cfg = GPTConfig()
    train(cfg)
