import argparse
import torch
from pathlib import Path

from vibejam.config import ModelConfig, DataConfig
from vibejam.train import load_dataset_from_file
from vibejam.model import GPTModel
from vibejam.rewrite import rewrite_text

TEST_DRAFTS = [
    "Today I went for a walk and felt pretty good about the progress on vibejam.",
    "I want a simple plan for next week: two deep learning lectures, one blog post, and one small coding sprint.",
    "Please rewrite this message to be concise, warm, and direct: Thanks for your time — I’ll follow up next week.",
    "I’m feeling behind, but I also know consistency matters. Rewrite this into a confident, pragmatic tone.",
    "Explain in plain English why block_size matters for a small Transformer.",
]

def load_model(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_cfg = ModelConfig(**ckpt["model_cfg"])
    model = GPTModel(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--text-path", type=str, required=True, help="personal_corpus.txt (for vocab)")
    p.add_argument("--base-ckpt", type=str, required=True)
    p.add_argument("--ft-ckpt", type=str, required=True)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--max-new-tokens", type=int, default=250)
    return p.parse_args()

def main():
    args = parse_args()

    # Use the same dataset/vocab
    # NOTE: this rebuilds vocab from text-path; in char-level world this must match training char set.
    data_cfg = DataConfig(block_size=128, train_frac=0.95)
    dataset = load_dataset_from_file(args.text_path, data_cfg)

    base = load_model(args.base_ckpt)
    ft = load_model(args.ft_ckpt)

    print("=== EVAL: BASE vs FINETUNED ===\n")

    for i, d in enumerate(TEST_DRAFTS, start=1):
        print(f"[{i}] DRAFT:\n{d}\n")

        out_base = rewrite_text(
            model=base,
            dataset=dataset,
            draft=d,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        print("BASE REWRITE:\n", out_base, "\n")

        out_ft = rewrite_text(
            model=ft,
            dataset=dataset,
            draft=d,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        print("FT REWRITE:\n", out_ft)
        print("\n" + "-"*80 + "\n")

if __name__ == "__main__":
    main()