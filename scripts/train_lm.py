# scripts/train_lm.py

import argparse
from vibejam.config import ModelConfig, TrainConfig, DataConfig
from vibejam.train import train_lm


def parse_args():
    p = argparse.ArgumentParser(description="Train vibejam LM on a text corpus.")

    p.add_argument("--text-path", type=str, required=True)
    p.add_argument("--ckpt-path", type=str, default=None)
    p.add_argument("--resume-path", type=str, default=None)

    # Data
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--train-frac", type=float, default=0.95)

    # Model
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)

    # Train
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-iters", type=int, default=10000)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--eval-iters", type=int, default=50)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--grad-clip", type=float, default=1.0)

    return p.parse_args()


def main():
    args = parse_args()

    data_cfg = DataConfig(block_size=args.block_size, train_frac=args.train_frac)

    # vocab_size will be set inside train_lm after dataset loads
    model_cfg = ModelConfig(
        vocab_size=0,
        block_size=data_cfg.block_size,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        dropout=args.dropout,
    )

    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_iters=args.max_iters,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        device=args.device,
        ckpt_path=args.ckpt_path,
    )

    # we’ll thread grad_clip into train_lm next
    train_lm(
        text_path=args.text_path,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        data_cfg=data_cfg,
        resume_path=args.resume_path,
        grad_clip=args.grad_clip,
    )


if __name__ == "__main__":
    main()