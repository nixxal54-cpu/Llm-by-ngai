# MiniGPT — GPT-style LM from Scratch

## Folder Structure

```
minigpt/
├── config.py          # Hyperparameters dataclass
├── tokenizer.py       # CharTokenizer + BPETokenizer
├── model.py           # MiniGPT transformer architecture
├── dataset.py         # TextDataset + batch loader
├── train.py           # Training loop
├── generate.py        # Autoregressive text generation
├── requirements.txt
└── data/
    └── download.py    # Fetch Tiny Shakespeare corpus
```

## Run Commands

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download corpus
python data/download.py

# 3. Train (char-level, ~6M params, ~20 min on CPU)
python train.py

# 4. Generate text
python generate.py \
    --prompt "ROMEO:" \
    --max_new_tokens 300 \
    --temperature 0.8 \
    --top_k 40

# ── Optional: BPE tokenizer ──────────────────────────────────────────
# Edit config.py → tokenizer_type = "bpe"
# Then re-run train.py

# ── Custom corpus ────────────────────────────────────────────────────
# Place any .txt file at data/corpus.txt and run train.py
```

## Config Knobs (config.py)

| Key              | Default | Description                        |
|------------------|---------|------------------------------------|
| `n_embd`         | 256     | Embedding dimension                |
| `n_head`         | 4       | Attention heads                    |
| `n_layer`        | 4       | Transformer blocks                 |
| `context_length` | 256     | Max sequence length                |
| `max_iters`      | 5000    | Training steps                     |
| `tokenizer_type` | `char`  | `"char"` or `"bpe"`               |

## Parameter Count

With defaults: **~6.3M parameters** (well under 10M cap).
