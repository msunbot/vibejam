# scripts/sample_lm.py

import argparse
import torch

from vibejam.config import ModelConfig, DataConfig
from vibejam.train import load_dataset_from_file
from vibejam.build import build_model
from vibejam.sample import generate_text


def parse_args():
    parser = argparse.ArgumentParser(description="Sample from a trained vibejam LM.")
    parser.add_argument("--text-path", type=str, required=True,
                        help="Path to the same corpus used for training")
    parser.add_argument("--ckpt-path", type=str, required=True,
                        help="Path to the saved checkpoint (.pt)")
    parser.add_argument("--arch", type=str, default=None,
                        help="Override architecture (default: checkpoint arch or 'gpt')")
    parser.add_argument("--prompt", type=str, default="",
                        help="Prompt text to condition on")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Seed for repeatable sampling")
    return parser.parse_args()


def main():
    args = parse_args()

    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    model_cfg = ModelConfig(**ckpt["model_cfg"])
    data_cfg = DataConfig(**ckpt["data_cfg"])
    arch = args.arch if args.arch is not None else ckpt.get("arch", "gpt")

    dataset = load_dataset_from_file(args.text_path, data_cfg)

    model = build_model(arch, model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    text = generate_text(
        model,
        dataset,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
    )
    print("=== SAMPLE ===")
    print(text)


if __name__ == "__main__":
    main()