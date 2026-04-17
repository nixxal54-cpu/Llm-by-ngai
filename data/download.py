"""Download the Tiny Shakespeare dataset for training."""

import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
OUT = Path("data/corpus.txt")


def download() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        print(f"[data] already exists: {OUT} ({OUT.stat().st_size:,} bytes)")
        return
    print(f"[data] downloading {URL} ...")
    urllib.request.urlretrieve(URL, OUT)
    print(f"[data] saved to {OUT} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    download()
