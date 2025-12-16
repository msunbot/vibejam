# vibejam/config.py
from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int     # must be set once dataset is known 
    block_size: int = 64
    n_embd: int = 64
    n_layer: int = 2
    n_head: int = 4
    dropout: float = 0.1

@dataclass
class TrainConfig:
    batch_size: int = 16
    learning_rate: float = 3e-4
    max_iters: int = 2000
    eval_interval: int = 200
    eval_iters: int = 50
    device: str = "cuda"  # fallback to cpu at runtime if needed
    ckpt_path: str | None = None # where to save model; None = don't save

@dataclass
class DataConfig:
    block_size: int = 64
    train_frac: float = 0.9

    # Tokenization controls
    tokenizer_type: str = "char"  # "char" or "bpe"
    tokenizer_path: str = ""      # used if tokenizer_type == "bpe"
    vocab_path: str = "checkpoints/vibejam_vocab.json"  # used if tokenizer_type == "char"