# scripts/compare_arch.py

import argparse
import torch

from vibejam.config import ModelConfig, DataConfig
from vibejam.train import load_dataset_from_file
from vibejam.build import build_model
from vibejam.sample import generate_text


def parse_args():
    p = argparse.ArgumentParser(description="Compare sampling across checkpoints.")
    p.add_argument("--text-path", type=str, required=True)
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--ckpt", action="append", required=True, help="Repeat for multiple checkpoints")

    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def main():
    args = parse_args()

    first = torch.load(args.ckpt[0], map_location="cpu")
    data_cfg = DataConfig(**first["data_cfg"])
    dataset = load_dataset_from_file(args.text_path, data_cfg)

    print("\n" + "=" * 100)
    print(f"PROMPT: {args.prompt!r}")
    print(f"temp={args.temperature} top_k={args.top_k} max_new={args.max_new_tokens} seed={args.seed}")
    print("=" * 100)

    for path in args.ckpt:
        ckpt = torch.load(path, map_location="cpu")
        model_cfg = ModelConfig(**ckpt["model_cfg"])
        arch = ckpt.get("arch", "gpt")

        model = build_model(arch, model_cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        out = generate_text(
            model=model,
            dataset=dataset,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed,
        )

        print("\n" + "-" * 100)
        print(f"CKPT: {path}")
        print(f"ARCH: {arch}")
        print("-" * 100)
        print(out)


if __name__ == "__main__":
    main()