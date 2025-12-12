import argparse
from pathlib import Path
import torch

from vibejam.config import ModelConfig, TrainConfig, DataConfig
from vibejam.data import CharDataset, CharDatasetWithVocab
from vibejam.model import GPTModel
from vibejam.train import get_device, estimate_loss, get_batch

# Build base vocab from original corpus (guarantees vocab_size=645)
base_text = Path("data/personal_corpus.txt").read_text(encoding="utf-8")
base_vocab_ds = CharDataset(base_text, data_cfg)
stoi, itos = base_vocab_ds.stoi, base_vocab_ds.itos

# Build rewrite-pairs dataset using fixed vocab
pairs_text = Path(args.pairs_path).read_text(encoding="utf-8")
dataset = CharDatasetWithVocab(pairs_text, data_cfg, stoi=stoi, itos=itos)
print("[vibejam] rewrite-pairs vocab_size (forced):", dataset.vocab_size)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs-path", type=str, required=True, help="data/rewrite_pairs.txt")
    p.add_argument("--base-ckpt", type=str, required=True, help="pretrained LM ckpt")
    p.add_argument("--out-ckpt", type=str, required=True, help="fine-tuned ckpt path")

    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-iters", type=int, default=3000)
    p.add_argument("--eval-interval", type=int, default=300)
    p.add_argument("--eval-iters", type=int, default=50)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()

def main():
    args = parse_args()

    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    print("[vibejam] Using device:", device)

    # Load base checkpoint
    base = torch.load(args.base_ckpt, map_location="cpu")
    base_model_cfg = ModelConfig(**base["model_cfg"])

    # Make sure block_size matches what we want for finetune
    # (In v0 char-level land, simplest is to keep the same block_size as base.)
    data_cfg = DataConfig(block_size=args.block_size, train_frac=0.95)

    text = Path(args.pairs_path).read_text(encoding="utf-8")
    dataset = CharDataset(text, data_cfg)
    print("[vibejam] rewrite-pairs vocab_size:", dataset.vocab_size)

    # Build model using base architecture but with vocab_size from this dataset
    # IMPORTANT: For char-level to work, vocab must match base. Easiest: use same tokenizer vocab.
    # Since we rebuild vocab from rewrite_pairs, it MUST contain the same character set as base training.
    # If it doesn't, you should generate pairs from the same corpus to keep char set consistent.
    model_cfg = ModelConfig(
        vocab_size=dataset.vocab_size, # now 645
        block_size=args.block_size,
        n_embd=base_model_cfg.n_embd,
        n_layer=base_model_cfg.n_layer,
        n_head=base_model_cfg.n_head,
        dropout=base_model_cfg.dropout,
    )

    model = GPTModel(model_cfg).to(device)

    # Load weights from base ckpt (works if vocab_size is identical; otherwise head shape mismatch)
    missing, unexpected = model.load_state_dict(base["model_state_dict"], strict=True)
    if missing or unexpected:
        print("[vibejam] load_state_dict notes:")
        print("  missing:", missing)
        print("  unexpected:", unexpected)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_iters=args.max_iters,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        device=device,
        ckpt_path=args.out_ckpt,
    )

    out_path = Path(args.out_ckpt)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for it in range(train_cfg.max_iters):
        xb, yb = get_batch(dataset, "train", train_cfg.batch_size, device)
        _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()

        if it % train_cfg.eval_interval == 0 or it == train_cfg.max_iters - 1:
            losses = estimate_loss(model, dataset, train_cfg)
            print(f"iter {it:5d} | train {losses['train']:.3f} | val {losses['val']:.3f}")

            ckpt = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "iter": it,
                "model_cfg": model_cfg.__dict__,
                "train_cfg": train_cfg.__dict__,
                "data_cfg": data_cfg.__dict__,
            }
            torch.save(ckpt, args.out_ckpt)
            print(f"[vibejam] Saved finetune ckpt: {args.out_ckpt}")

if __name__ == "__main__":
    main()