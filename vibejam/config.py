# vibejam/config.py
from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int = 64
    n_embd: int = 64
    n_layer: int = 2
    n_head: int = 4
    dropout: float = 0.1

@dataclass
class TrainConfig:
    batch_size: int = 16
    learning_rate: float = 3e-4
    max_iters: int = 500
    eval_interval: int = 100
    device: str = "cuda"  # will downgrade to "cpu" if no GPU

@dataclass
class DataConfig:
    block_size: int = 64
    train_frac: float = 0.9