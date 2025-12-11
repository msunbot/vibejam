# vibejam/data.py
import torch
from typing import Tuple
from .config import DataConfig

class CharDataset:
    """
    Minimal character-level dataset around a single text.
    Responsible for:
    - building vocab
    - encoding/decoding
    - providing train/val tensors
    """

    def __init__(self, text: str, cfg: DataConfig):
        self.cfg = cfg

        # build vocabulary
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

        # encode entire corpus
        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

        n = int(cfg.train_frac * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def encode(self, s: str):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    def get_batch(self, split: str, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.train_data if split == "train" else self.val_data
        block_size = self.cfg.block_size

        ix = torch.randint(0, len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])

        return x, y