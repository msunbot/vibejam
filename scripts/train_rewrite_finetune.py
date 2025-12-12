import argparse
from pathlib import Path
import torch

from vibejam.config import ModelConfig, TrainConfig, DataConfig
from vibejam.data import CharDataset, CharDatasetWithVocab
from vibejam.model import GPTModel
from vibejam.train import estimate_loss, get_batch


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

    # IMPORTANT: we need the original corpus to rebuild the same char vocab as the base LM
    p.add_argument("--corpus-path", type=str, default="data/personal_corpus.txt")

    return p.parse_args()


def main():
    args = parse_args()

    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    print("[vibejam] Using device:", device)

    # Load base checkpoint (architecture + weights)
    base = torch.load(args.base_ckpt, map_location="cpu")
    base_model_cfg = ModelConfig(**base["model_cfg"])

    # Data config for finetune dataset windows
    data_cfg = DataConfig(block_size=args.block_size, train_frac=0.95)

    # 1) Build the *base vocab* from the original corpus (ensures vocab_size matches base checkpoint)
    base_text = Path(args.corpus_path).read_text(encoding="utf-8")
    base_vocab_ds = CharDataset(base_text, data_cfg)
    stoi, itos = base_vocab_ds.stoi, base_vocab_ds.itos
    print("[vibejam] Base vocab_size:", len(stoi))

    # 2) Build rewrite-pairs dataset using the *same vocab*
    pairs_text = Path(args.pairs_path).read_text(encoding="utf-8")
    dataset = CharDatasetWithVocab(pairs_text, data_cfg, stoi=stoi, itos=itos)
    print("[vibejam] rewrite-pairs vocab_size (forced):", dataset.vocab_size)

    # 3) Build model with base architecture, but vocab_size MUST match base (645)
    model_cfg = ModelConfig(
        vocab_size=dataset.vocab_size,
        block_size=args.block_size,
        n_embd=base_model_cfg.n_embd,
        n_layer=base_model_cfg.n_layer,
        n_head=base_model_cfg.n_head,
        dropout=base_model_cfg.dropout,
    )
    model = GPTModel(model_cfg).to(device)

    # Now vocab_size matches, strict load works
    model.load_state_dict(base["model_state_dict"], strict=True)

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

    # Training loop
    for it in range(train_cfg.max_iters):
        xb, yb = get_batch(dataset, "train", train_cfg.batch_size, device)
        _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()

        # Log + save periodically
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