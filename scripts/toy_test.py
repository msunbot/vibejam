# scripts/toy_test.py
import torch

from vibejam.config import ModelConfig, DataConfig
from vibejam.data import CharDataset
from vibejam.model import GPTModel

raw_text = """
vibejam is a tiny personal-style GPT.
this is just a toy corpus, but it will learn some patterns :)
"""

data_cfg = DataConfig(block_size=64)
dataset = CharDataset(raw_text, data_cfg)

model_cfg = ModelConfig(
    vocab_size=dataset.vocab_size,
    block_size=data_cfg.block_size,
    n_embd=64,
    n_layer=2,
    n_head=4,
    dropout=0.1,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = GPTModel(model_cfg).to(device)

x, y = dataset.get_batch("train", batch_size=4)
x, y = x.to(device), y.to(device)

logits, loss = model(x, y)
print("logits shape:", logits.shape)  # expect (4, block_size, vocab_size)
print("loss:", loss.item())