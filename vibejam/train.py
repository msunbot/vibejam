# vibejam/train.py

import time
from dataclasses import asdict
from typing import Tuple

import torch

from .config import ModelConfig, TrainConfig, DataConfig
from .data import CharDataset
from .model import GPTModel

# -----------------------------
# Utility: choose device
# -----------------------------
def get_device(train_cfg: TrainConfig) -> str:
    """Pick the actual device string to use."""
    if train_cfg.device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return train_cfg.device

# -----------------------------
# Dataset creation
# -----------------------------
def load_dataset_from_file(path: str, data_cfg: DataConfig) -> CharDataset:
    """Read a text file and wrap it in a CharDataset."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    dataset = CharDataset(text, data_cfg)
    return dataset

# -----------------------------
# get_batch & estimate_loss
# -----------------------------
def get_batch(dataset: CharDataset,
              split: str, 
              batch_size: int,
              device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a random batch of (x,y) from the dataset.

    x: input tokens, shape (B, T)
    y: target tokens, shape (B, T) = next char for each position in x
    """
    x, y = dataset.get_batch(split, batch_size)
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model: GPTModel,
                  dataset: CharDataset,
                  train_cfg: TrainConfig) -> dict: 
    """
    Compute average train/val loss over a few mini-batches
    Model is put in eval() mode temporarily
    """
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = []

        # Check if this split is long enough; if not, skip
        data = dataset.train_data if split == "train" else dataset.val_data
        if len(data) <= dataset.cfg.block_size + 1:
            out[split] = float("nan")
            continue

        for _ in range(train_cfg.eval_iters):
            xb, yb = get_batch(
                dataset,
                split=split,
                batch_size=train_cfg.batch_size,
                device=next(model.parameters()).device,
            )
            _, loss = model(xb, yb)
            losses.append(loss.item())

        out[split] = sum(losses) / len(losses)
        
    model.train()
    return out

# -----------------------------
# main training loop
# -----------------------------
def train_lm(text_path: str,
             model_cfg: ModelConfig,
             train_cfg: TrainConfig,
             data_cfg: DataConfig, 
             resume_path: str | None = None, 
             grad_clip: float | None = 1.0) -> Tuple[GPTModel, CharDataset]:
    """
    High level training function: 
    - loads text from file
    - builds dataset
    - builds model
    - runs training loop
    - optionally saves checkpoint
    """
    device = get_device(train_cfg)
    print(f"[vibejam] Using device: {device}")

    # 1) Dataset
    dataset = load_dataset_from_file(text_path, data_cfg)
    print(f"[vibejam] Loaded dataset from {text_path}")
    print(f"    vocab_size = {dataset.vocab_size}")
    print(f"    train tokens = {len(dataset.train_data)}, val tokens = {len(dataset.val_data)}")

    # 2) Model config must know vocab_size & block_size
    model_cfg = ModelConfig(
        vocab_size = dataset.vocab_size,
        block_size = data_cfg.block_size,
        n_embd=model_cfg.n_embd,
        n_layer=model_cfg.n_layer,
        n_head=model_cfg.n_head,
        dropout=model_cfg.dropout,
    )

    # 3) Model & optimizer
    model = GPTModel(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.learning_rate)

    print("[vibejam] Model config:", asdict(model_cfg))
    print("[vibejam] Train config:", asdict(train_cfg))

    start_iters = 0 
    if resume_path is not None: 
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_iter = ckpt.get("iter", 0)
        print(f"[vibejam] Resumed from {resume_path} at iter {start_iter}")

    # 4) Training loop
    start_time = time.time()
    for it in range(train_cfg.max_iters): 
        # sample a batch
        xb, yb = get_batch(dataset, "train", train_cfg.batch_size, device)

        # forward pass: logits & loss
        logits, loss = model(xb, yb)

        # backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # logging
        if it % train_cfg.eval_interval == 0 or it == train_cfg.max_iters - 1: 
            losses = estimate_loss(model,dataset, train_cfg)
            elapsed = time.time() - start_time
            print(
                f"iter {it:5d} | "
                f"train {losses['train']:.3f} | "
                f"val {losses['val']:.3f} | "
                f"time {elapsed:6.1f}s"
            )
    # 5) Optional checkpoint
    if train_cfg.ckpt_path is not None:
        ckpt = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iter": it,
            "model_cfg": asdict(model_cfg),
            "train_cfg": asdict(train_cfg),
            "data_cfg": asdict(data_cfg),
        }
        torch.save(ckpt, train_cfg.ckpt_path)
        print(f"[vibejam] Saved checkpoint to {train_cfg.ckpt_path}")
    
    return model, dataset