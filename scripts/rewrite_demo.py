# scripts/rewrite_demo.py

import argparse
import torch

from vibejam.config import ModelConfig, DataConfig
from vibejam.build import build_model
from vibejam.rewrite import rewrite_text

def parse_args():
    parser = argparse.ArgumentParser(description="Rewrite text in your vibejam style.")
    parser.add_argument(
        "--text-path",
        type=str,
        required=True,
        help="Path to the corpus used for training (same as in train_lm)",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint (.pt)",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default=None,
        help="Override architecture (default: checkpoint arch or 'gpt')",
    )
    parser.add_argument(
        "--draft",
        type=str,
        required=True,
        help="Draft text to rewrite",
    )
    parser.add_argument(
        "--vocab-path",
        type=str,
        default="checkpoints/vibejam_vocab.json",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=160,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
    )
    parser.add_argument("--tokenizer-type", type=str, default="char", choices=["char", "bpe"])
    parser.add_argument("--tokenizer-path", type=str, default="")

    return parser.parse_args()


def main():
    args = parse_args()
    from vibejam.prompts import build_rewrite_prompt

    # 0) Load the corpus text (needed to rebuild dataset tensors)
    with open(args.text_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 1) Load checkpoint
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    model_cfg_dict = ckpt["model_cfg"]
    data_cfg_dict = ckpt["data_cfg"]

    model_cfg = ModelConfig(**model_cfg_dict)
    data_cfg = DataConfig(**data_cfg_dict)

    arch = args.arch if args.arch is not None else ckpt.get("arch", "gpt")

    # 2) Rebuild dataset with SAME mapping as training
    from vibejam.tokenizer_bpe import BPETokenizer
    from vibejam.tokenizer_char import CharTokenizer
    from vibejam.data import TokenDataset, CharDatasetWithVocab

    if args.tokenizer_type == "bpe":
        if not args.tokenizer_path:
            raise ValueError("--tokenizer-path is required when --tokenizer-type bpe")
        tok = BPETokenizer.load(args.tokenizer_path)
        dataset = TokenDataset(text=text, cfg=data_cfg, tokenizer=tok)
    else:
        tok = CharTokenizer.load(args.vocab_path)
        dataset = CharDatasetWithVocab(text=text, cfg=data_cfg, stoi=tok.stoi, itos=tok.itos)

    # 3) Rebuild model and load weights
    model = build_model(arch, model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 4) Build prompt
    prompt = build_rewrite_prompt(args.draft)

    # 5) Run rewrite
    rewritten = rewrite_text(
        model=model,
        dataset=dataset,
        draft=args.draft,
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    print("=== DRAFT ===")
    print(args.draft)
    print("\n=== REWRITE ===")
    print(rewritten)


if __name__ == "__main__":
    main()