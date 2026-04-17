"""Hyperparameter configuration for MiniGPT."""

from dataclasses import dataclass, field


@dataclass
class GPTConfig:
    # ── Vocabulary ──────────────────────────────────────────────────────
    vocab_size: int = 256          # set after tokenizer.fit()

    # ── Architecture ────────────────────────────────────────────────────
    context_length: int = 256      # max sequence length (block size)
    n_embd: int = 256              # embedding dimension
    n_head: int = 4                # number of attention heads
    n_layer: int = 4               # number of transformer blocks
    dropout: float = 0.1

    # ── Training ────────────────────────────────────────────────────────
    batch_size: int = 64
    learning_rate: float = 3e-4
    max_iters: int = 5000
    eval_interval: int = 500
    eval_iters: int = 200
    grad_clip: float = 1.0

    # ── Generation ──────────────────────────────────────────────────────
    temperature: float = 0.8
    top_k: int = 40

    # ── Paths ────────────────────────────────────────────────────────────
    data_path: str = "data/corpus.txt"
    tokenizer_path: str = "checkpoints/tokenizer.json"
    checkpoint_path: str = "checkpoints/model.pt"
    tokenizer_type: str = "char"   # "char" | "bpe"

    def __post_init__(self) -> None:
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
