"""Character-level tokenizer with optional BPE-style merges."""

from __future__ import annotations
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional


class CharTokenizer:
    """Simple character-level tokenizer."""

    def __init__(self) -> None:
        self.char2idx: dict[str, int] = {}
        self.idx2char: dict[int, str] = {}
        self.vocab_size: int = 0

    def fit(self, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        self.char2idx = {ch: i for i, ch in enumerate(chars)}
        self.idx2char = {i: ch for ch, i in self.char2idx.items()}
        self.vocab_size = len(chars)
        return self

    def encode(self, text: str) -> list[int]:
        return [self.char2idx[ch] for ch in text if ch in self.char2idx]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.idx2char.get(i, "") for i in ids)

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps({"char2idx": self.char2idx}))

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        data = json.loads(Path(path).read_text())
        tok = cls()
        tok.char2idx = data["char2idx"]
        tok.idx2char = {int(i): ch for ch, i in tok.char2idx.items()}
        tok.vocab_size = len(tok.char2idx)
        return tok


class BPETokenizer:
    """Minimal byte-pair encoding tokenizer trained from scratch."""

    def __init__(self, vocab_size: int = 512) -> None:
        self.target_vocab_size = vocab_size
        self.merges: list[tuple[str, str]] = []
        self.vocab: dict[str, int] = {}
        self.inv_vocab: dict[int, str] = {}

    # ------------------------------------------------------------------ #
    #  Training                                                            #
    # ------------------------------------------------------------------ #

    def _get_pairs(self, vocab: dict[tuple[str, ...], int]) -> dict[tuple[str, str], int]:
        pairs: dict[tuple[str, str], int] = defaultdict(int)
        for word, freq in vocab.items():
            for a, b in zip(word, word[1:]):
                pairs[(a, b)] += freq
        return pairs

    def _merge_vocab(
        self,
        pair: tuple[str, str],
        vocab: dict[tuple[str, ...], int],
    ) -> dict[tuple[str, ...], int]:
        new_vocab: dict[tuple[str, ...], int] = {}
        bigram = " ".join(pair)
        replacement = "".join(pair)
        for word, freq in vocab.items():
            new_word: list[str] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                    new_word.append(replacement)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_vocab[tuple(new_word)] = freq
        return new_vocab

    def fit(self, text: str) -> "BPETokenizer":
        # Start with character-level tokens; mark word boundaries with Ġ
        word_freq: dict[tuple[str, ...], int] = defaultdict(int)
        for word in re.findall(r"\S+|\s+", text):
            token_word = tuple(list(word) + ["</w>"])
            word_freq[token_word] += 1

        base_chars: set[str] = set()
        for word in word_freq:
            base_chars.update(word)

        vocab = dict(word_freq)
        num_merges = self.target_vocab_size - len(base_chars)

        for _ in range(max(0, num_merges)):
            pairs = self._get_pairs(vocab)
            if not pairs:
                break
            best = max(pairs, key=lambda p: pairs[p])
            vocab = self._merge_vocab(best, vocab)
            self.merges.append(best)

        # Build final vocabulary
        all_tokens: set[str] = set()
        for word in vocab:
            all_tokens.update(word)
        all_tokens.update(base_chars)
        all_tokens = sorted(all_tokens)

        self.vocab = {tok: i for i, tok in enumerate(all_tokens)}
        self.inv_vocab = {i: tok for tok, i in self.vocab.items()}
        return self

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # ------------------------------------------------------------------ #
    #  Encode / Decode                                                     #
    # ------------------------------------------------------------------ #

    def _tokenize_word(self, word: str) -> list[str]:
        chars = list(word) + ["</w>"]
        for a, b in self.merges:
            i = 0
            while i < len(chars) - 1:
                if chars[i] == a and chars[i + 1] == b:
                    chars = chars[:i] + [a + b] + chars[i + 2 :]
                else:
                    i += 1
        return chars

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for word in re.findall(r"\S+|\s+", text):
            for tok in self._tokenize_word(word):
                if tok in self.vocab:
                    ids.append(self.vocab[tok])
        return ids

    def decode(self, ids: list[int]) -> str:
        tokens = [self.inv_vocab.get(i, "") for i in ids]
        return "".join(tokens).replace("</w>", " ").rstrip()

    def save(self, path: str) -> None:
        Path(path).write_text(
            json.dumps({"vocab": self.vocab, "merges": self.merges})
        )

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        data = json.loads(Path(path).read_text())
        tok = cls()
        tok.vocab = data["vocab"]
        tok.merges = [tuple(m) for m in data["merges"]]
        tok.inv_vocab = {int(i): ch for ch, i in tok.vocab.items()}
        return tok
