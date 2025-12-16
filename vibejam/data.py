# vibejam/data.py
import torch
from typing import Tuple
from .config import DataConfig
# from vibejam.tokenizer_bpe import BPETokenizer 
from vibejam.tokenizer_char import CharTokenizer

def load_tokenizer(cfg):
    if cfg.tokenizer_type == "char":
        return CharTokenizer.load(cfg.vocab_path)
    elif cfg.tokenizer_type == "bpe":
        from vibejam.tokenizer_bpe import BPETokenizer
        return BPETokenizer.load(cfg.tokenizer_path)
    else:
        raise ValueError(cfg.tokenizer_type)


def build_char_tokenizer_from_text(text: str) -> CharTokenizer:
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return CharTokenizer(stoi=stoi, itos=itos)

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

        # Safety: require data longer than block_size
        if len(data) <= block_size + 1:
            raise ValueError(
                f"Data too short for split='{split}': "
                f"{len(data)} tokens with block_size={block_size}."
            )
        ix = torch.randint(0, len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])

        return x, y
    
class CharDatasetWithVocab:
    """
    Char dataset that uses a fixed vocab mapping (stoi/itos) instead of rebuilding it.
    Unknown characters are dropped (or you can map them to a fallback).
    """

    def __init__(self, text: str, cfg: DataConfig, stoi: dict, itos: dict):
        import torch
        self.cfg = cfg
        self.stoi = stoi
        self.itos = itos
        self.vocab_size = len(stoi)

        ids = []
        for c in text:
            if c in self.stoi:
                ids.append(self.stoi[c])

        data = torch.tensor(ids, dtype=torch.long)
        n = int(cfg.train_frac * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def encode(self, s: str):
        # match training behavior: drop unknown chars
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids if i in self.itos)

    def get_batch(self, split: str, batch_size: int):
        import torch
        data = self.train_data if split == "train" else self.val_data
        block_size = self.cfg.block_size

        if len(data) <= block_size + 1:
            raise ValueError(
                f"Data too short for split='{split}': {len(data)} tokens "
                f"with block_size={block_size}."
            )

        ix = torch.randint(0, len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        return x, y
    
class TokenDataset:
    """
    Token-level dataset driven by an external tokenizer (char or BPE).
    tokenizer must implement: encode(str)->List[int], decode(List[int])->str, vocab_size
    """
    def __init__(self, text: str, cfg: DataConfig, tokenizer):
        import torch
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size

        ids = tokenizer.encode(text)
        data = torch.tensor(ids, dtype=torch.long)

        n = int(cfg.train_frac * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def encode(self, s: str):
        return self.tokenizer.encode(s)

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    def get_batch(self, split: str, batch_size: int):
        import torch
        data = self.train_data if split == "train" else self.val_data
        block_size = self.cfg.block_size

        if len(data) <= block_size + 1:
            raise ValueError(f"Data too short: {len(data)} tokens with block_size={block_size}")

        ix = torch.randint(0, len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        return x, y