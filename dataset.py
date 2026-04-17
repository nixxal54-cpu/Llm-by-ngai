"""Dataset utilities: load text corpus and generate training batches."""

from __future__ import annotations
from pathlib import Path

import torch


class TextDataset:
    """Holds encoded token ids and yields random context/target pairs."""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        context_length: int,
        train_split: float = 0.9,
    ) -> None:
        text = Path(data_path).read_text(encoding="utf-8")
        ids = tokenizer.encode(text)
        data = torch.tensor(ids, dtype=torch.long)

        n = int(len(data) * train_split)
        self.train_data = data[:n]
        self.val_data = data[n:]
        self.context_length = context_length

        print(
            f"[dataset] tokens: {len(data):,} | "
            f"train: {len(self.train_data):,} | "
            f"val: {len(self.val_data):,}"
        )

    def get_batch(
        self,
        split: str,
        batch_size: int,
        device: torch.device | str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.train_data if split == "train" else self.val_data
        max_start = len(data) - self.context_length - 1
        if max_start < 1:
            raise ValueError("Corpus too small for the given context_length.")
        ix = torch.randint(max_start, (batch_size,))
        x = torch.stack([data[i : i + self.context_length] for i in ix])
        y = torch.stack([data[i + 1 : i + self.context_length + 1] for i in ix])
        return x.to(device), y.to(device)
