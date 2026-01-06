# scripts/eval_rewrite.py

import argparse
import json
from pathlib import Path
import torch

from vibejam.config import ModelConfig, DataConfig
from vibejam.build import build_model
from vibejam.rewrite import rewrite_text, RewriteConfig
from vibejam.prompts import build_rewrite_prompt


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate rewrite behavior on a fixed jsonl set.")
    p.add_argument("--text-path", type=str, required=True, help="Training corpus path (for tokenizer/dataset rebuild)")
    p.add_argument("--ckpt-path", type=str, required=True, help="Checkpoint .pt")
    p.add_argument("--eval-path", type=str, default="data/eval_rewrite.jsonl", help="Fixed eval jsonl")
    p.add_argument("--out-path", type=str, default="outputs/rewrite_eval_out.jsonl")
    p.add_argument("--arch", type=str, default=None, help="Override arch (default: checkpoint arch or 'gpt')")

    # Sampling defaults for eval
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", type=str, default=None, help="cpu/cuda (default auto)")
    return p.parse_args()


def _load_eval_items(path: str):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def main():
    args = parse_args()

    # Load ckpt
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    model_cfg = ModelConfig(**ckpt["model_cfg"])
    data_cfg = DataConfig(**ckpt["data_cfg"])
    arch = args.arch if args.arch is not None else ckpt.get("arch", "gpt")

    # Rebuild dataset (we rely on train.load_dataset_from_file)
    from vibejam.train import load_dataset_from_file
    dataset = load_dataset_from_file(args.text_path, data_cfg)

    # Build model
    model = build_model(arch, model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    cfg = RewriteConfig(
        temperature=args.temperature,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    items = _load_eval_items(args.eval_path)
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)

    with open(args.out_path, "w", encoding="utf-8") as out_f:
        for ex in items:
            draft = ex["draft"]
            prompt = build_rewrite_prompt(draft)
            rewrite = rewrite_text(model=model, dataset=dataset, draft=draft, prompt=prompt, cfg=cfg)

            record = {
                "id": ex.get("id", None),
                "draft": draft,
                "rewrite": rewrite,
                "arch": arch,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "max_new_tokens": args.max_new_tokens,
                "seed": args.seed,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            print("\n" + "=" * 80)
            print(f"ID: {record['id']} | arch={arch} | temp={args.temperature} | top_k={args.top_k} | seed={args.seed}")
            print("- DRAFT -")
            print(draft)
            print("- REWRITE -")
            print(rewrite)

    print(f"\n[vibejam] wrote: {args.out_path}")


if __name__ == "__main__":
    main()