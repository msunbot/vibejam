# scripts/train_lm.py

import argparse

from vibejam.config import ModelConfig, TrainConfig, DataConfig
from vibejam.train import train_lm


def parse_args():
    parser = argparse.ArgumentParser(description="Train a tiny vibejam LM on a text file.")
    parser.add_argument(
        "--text-path",
        type=str,
        required=True,
        help="Path to the input .txt corpus",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=None,
        help="Optional path to save model checkpoint (.pt)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # You can tweak these defaults as needed
    data_cfg = DataConfig(block_size=64, train_frac=0.9)
    model_cfg = ModelConfig(vocab_size=0, block_size=data_cfg.block_size)  # vocab_size set inside train_lm
    train_cfg = TrainConfig(
        batch_size=32,
        learning_rate=3e-4,
        max_iters=2000,
        eval_interval=200,
        eval_iters=50,
        device="cuda",
        ckpt_path=args.ckpt_path,
    )

    train_lm(
        text_path=args.text_path,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        data_cfg=data_cfg,
    )


if __name__ == "__main__":
    main()