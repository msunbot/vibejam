# scripts/sample_lm.py

import argparse
import torch

from vibejam.config import ModelConfig, DataConfig
from vibejam.train import load_dataset_from_file
from vibejam.model import GPTModel
from vibejam.sample import generate_text


def parse_args():
    parser = argparse.ArgumentParser(description="Sample from a trained vibejam LM.")
    parser.add_argument("--text-path", type=str, required=True,
                        help="Path to the same corpus used for training")
    parser.add_argument("--ckpt-path", type=str, required=True,
                        help="Path to the saved checkpoint (.pt)")
    parser.add_argument("--prompt", type=str, default="",
                        help="Prompt text to condition on")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    # 1) Load checkpoint
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    model_cfg_dict = ckpt["model_cfg"]
    data_cfg_dict = ckpt["data_cfg"]

    model_cfg = ModelConfig(**model_cfg_dict)
    data_cfg = DataConfig(**data_cfg_dict)

    # 2) Rebuild dataset to get vocab/encode/decode
    dataset = load_dataset_from_file(args.text_path, data_cfg)

    # 3) Rebuild model and load weights
    model = GPTModel(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 4) Generate text
    text = generate_text(
        model,
        dataset,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print("=== SAMPLE ===")
    print(text)


if __name__ == "__main__":
    main()