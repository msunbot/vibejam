import argparse
from pathlib import Path
import torch

from vibejam.config import ModelConfig, TrainConfig, DataConfig
from vibejam.data import CharDataset, CharDatasetWithVocab, TokenDataset
from vibejam.model import GPTModel
from vibejam.train import estimate_loss, get_batch

from vibejam.tokenizer_bpe import BPETokenizer
from vibejam.tokenizer_char import CharTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs-path", type=str, required=True, help="data/rewrite_pairs.txt")
    p.add_argument("--base-ckpt", type=str, required=True, help="pretrained LM ckpt")
    p.add_argument("--out-ckpt", type=str, required=True, help="fine-tuned ckpt path")

    # If not provided, we will default to base checkpoint block_size for compatibility.
    p.add_argument("--block-size", type=int, default=None)

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-iters", type=int, default=3000)
    p.add_argument("--eval-interval", type=int, default=300)
    p.add_argument("--eval-iters", type=int, default=50)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--device", type=str, default="cuda")

    # Tokenizer selection
    p.add_argument("--tokenizer-type", type=str, default="char", choices=["char", "bpe"])
    p.add_argument("--tokenizer-path", type=str, default="", help="Required for --tokenizer-type bpe")
    p.add_argument("--vocab-path", type=str, default="checkpoints/vibejam_vocab.json", help="Used for char runs")

    # For char runs: used to rebuild the same vocab if vocab_path is missing
    p.add_argument("--corpus-path", type=str, default="data/personal_corpus.txt")

    return p.parse_args()


def main():
    args = parse_args()

    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    print("[vibejam] Using device:", device)

    # Load base checkpoint (architecture + weights)
    base = torch.load(args.base_ckpt, map_location="cpu")
    base_model_cfg = ModelConfig(**base["model_cfg"])

    # Default block_size to base checkpoint block_size (must match for strict load)
    block_size = args.block_size if args.block_size is not None else base_model_cfg.block_size
    if block_size != base_model_cfg.block_size:
        raise ValueError(
            f"block_size mismatch: base_ckpt block_size={base_model_cfg.block_size} but you set {block_size}. "
            f"Use --block-size {base_model_cfg.block_size} or omit --block-size."
        )

    # Data config for finetune dataset windows
    data_cfg = DataConfig(block_size=block_size, train_frac=0.95)

    # 1) Build dataset (char or bpe)
    pairs_text = Path(args.pairs_path).read_text(encoding="utf-8")

    tokenizer_meta = {"tokenizer_type": args.tokenizer_type}

    if args.tokenizer_type == "bpe":
        if not args.tokenizer_path:
            raise ValueError("--tokenizer-path is required when --tokenizer-type bpe")

        tok = BPETokenizer.load(args.tokenizer_path)
        tokenizer_meta["tokenizer_path"] = args.tokenizer_path
        tokenizer_meta["vocab_size"] = tok.vocab_size

        dataset = TokenDataset(pairs_text, data_cfg, tok)
        print("[vibejam] BPE finetune dataset vocab_size:", dataset.vocab_size)

    else:
        # Char mode: prefer loading persisted vocab mapping for exact alignment
        if Path(args.vocab_path).exists():
            tok = CharTokenizer.load(args.vocab_path)
            tokenizer_meta["vocab_path"] = args.vocab_path
            stoi, itos = tok.stoi, tok.itos
            print("[vibejam] Loaded char vocab from:", args.vocab_path)
        else:
            # Fallback: rebuild vocab from corpus (older behavior)
            base_text = Path(args.corpus_path).read_text(encoding="utf-8")
            base_vocab_ds = CharDataset(base_text, data_cfg)
            stoi, itos = base_vocab_ds.stoi, base_vocab_ds.itos
            print("[vibejam] Rebuilt char vocab from corpus; vocab_size:", len(stoi))

        dataset = CharDatasetWithVocab(pairs_text, data_cfg, stoi=stoi, itos=itos)
        print("[vibejam] Char finetune dataset vocab_size (forced):", dataset.vocab_size)

    # 2) Build model with base architecture; vocab_size MUST match base checkpoint
    if dataset.vocab_size != base_model_cfg.vocab_size:
        raise ValueError(
            f"vocab_size mismatch: base_ckpt vocab_size={base_model_cfg.vocab_size} "
            f"but dataset vocab_size={dataset.vocab_size}. "
            f"Check tokenizer artifacts and dataset construction."
        )

    model_cfg = ModelConfig(
        vocab_size=dataset.vocab_size,
        block_size=block_size,
        n_embd=base_model_cfg.n_embd,
        n_layer=base_model_cfg.n_layer,
        n_head=base_model_cfg.n_head,
        dropout=base_model_cfg.dropout,
    )
    model = GPTModel(model_cfg).to(device)

    # Strict load should work (same vocab_size, same block_size, same architecture)
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
                "tokenizer": tokenizer_meta,
            }
            torch.save(ckpt, args.out_ckpt)
            print(f"[vibejam] Saved finetune ckpt: {args.out_ckpt}")


if __name__ == "__main__":
    main()