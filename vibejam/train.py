# vibejam/train.py

import time
from dataclasses import asdict
from typing import Tuple, Any
from pathlib import Path

import torch

from .config import ModelConfig, TrainConfig, DataConfig
from .data import CharDataset, TokenDataset
from .model import GPTModel

from vibejam.tokenizer_bpe import BPETokenizer
from vibejam.tokenizer_char import CharTokenizer


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
def load_dataset_from_file(path: str, data_cfg: DataConfig):
    text = Path(path).read_text(encoding="utf-8")

    if data_cfg.tokenizer_type == "bpe":
        if not data_cfg.tokenizer_path:
            raise ValueError("tokenizer_type=bpe requires data_cfg.tokenizer_path")
        tok = BPETokenizer.load(data_cfg.tokenizer_path)
        dataset = TokenDataset(text, data_cfg, tok)
    else:
        dataset = CharDataset(text, data_cfg)

    return dataset


# -----------------------------
# get_batch & estimate_loss
# -----------------------------
def get_batch(
    dataset: Any,
    split: str,
    batch_size: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a random batch of (x,y) from the dataset.

    x: (B, T) token ids
    y: (B, T) next-token ids
    """
    x, y = dataset.get_batch(split, batch_size)
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(
    model: GPTModel,
    dataset: Any,
    train_cfg: TrainConfig,
) -> dict:
    """
    Compute average train/val loss over a few mini-batches.
    """
    out = {}
    model.eval()

    for split in ["train", "val"]:
        losses = []

        # Check if split is long enough; if not, skip
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
def train_lm(
    text_path: str,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    data_cfg: DataConfig,
    resume_path: str | None = None,
    grad_clip: float | None = 1.0,
):
    """
    High level training function:
    - loads text from file
    - builds dataset
    - builds model
    - runs training loop
    - saves checkpoints at eval intervals if ckpt_path is provided
    """
    device = get_device(train_cfg)
    print(f"[vibejam] Using device: {device}")

    # 1) Dataset
    dataset = load_dataset_from_file(text_path, data_cfg)
    print(f"[vibejam] Loaded dataset from {text_path}")
    print(f"    vocab_size = {dataset.vocab_size}")
    print(f"    train tokens = {len(dataset.train_data)}, val tokens = {len(dataset.val_data)}")

    # 2) Tokenizer metadata (for reproducibility)
    tokenizer_meta = {"tokenizer_type": data_cfg.tokenizer_type}
    if data_cfg.tokenizer_type == "char":
        tok = CharTokenizer(stoi=dataset.stoi, itos=dataset.itos)
        tok.save(data_cfg.vocab_path)
        tokenizer_meta["vocab_path"] = data_cfg.vocab_path
        tokenizer_meta["stoi"] = dataset.stoi
        tokenizer_meta["itos"] = dataset.itos
        print(f"[vibejam] saved char vocab to {data_cfg.vocab_path}")
    else:
        tokenizer_meta["tokenizer_path"] = data_cfg.tokenizer_path
        print(f"[vibejam] using BPE tokenizer from {data_cfg.tokenizer_path}")

    # 3) Model config must know vocab_size & block_size
    model_cfg = ModelConfig(
        vocab_size=dataset.vocab_size,
        block_size=data_cfg.block_size,
        n_embd=model_cfg.n_embd,
        n_layer=model_cfg.n_layer,
        n_head=model_cfg.n_head,
        dropout=model_cfg.dropout,
    )

    # 4) Model & optimizer
    model = GPTModel(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.learning_rate)

    print("[vibejam] Model config:", asdict(model_cfg))
    print("[vibejam] Train config:", asdict(train_cfg))

    # 5) Resume (optional)
    start_iter = 0
    if resume_path is not None:
        resume_ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(resume_ckpt["model_state_dict"])
        if "optimizer_state_dict" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        start_iter = int(resume_ckpt.get("iter", 0)) + 1
        print(f"[vibejam] Resumed from {resume_path} at iter {start_iter}")

    # Ensure ckpt dir exists
    if train_cfg.ckpt_path:
        Path(train_cfg.ckpt_path).parent.mkdir(parents=True, exist_ok=True)

    # 6) Training loop
    start_time = time.time()
    for it in range(start_iter, train_cfg.max_iters):
        xb, yb = get_batch(dataset, "train", train_cfg.batch_size, device)
        _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # Eval/log/save at interval
        if it % train_cfg.eval_interval == 0 or it == train_cfg.max_iters - 1:
            losses = estimate_loss(model, dataset, train_cfg)
            elapsed = time.time() - start_time
            print(
                f"iter {it:5d} | "
                f"train {losses['train']:.3f} | "
                f"val {losses['val']:.3f} | "
                f"time {elapsed:6.1f}s"
            )

            if train_cfg.ckpt_path:
                ckpt = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "iter": it,
                    "model_cfg": asdict(model_cfg),
                    "train_cfg": asdict(train_cfg),
                    "data_cfg": asdict(data_cfg),
                    "tokenizer": tokenizer_meta,
                }
                torch.save(ckpt, train_cfg.ckpt_path)
                print(f"[vibejam] Saved checkpoint: {train_cfg.ckpt_path}")

    return model, dataset