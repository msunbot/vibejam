# scripts/style_engine_demo.py
import argparse
from vibejam.engine import VibejamEngine

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-path", required=True)
    p.add_argument("--corpus-path", required=True)
    p.add_argument("--mode", choices=["rewrite", "generate"], default="rewrite")
    p.add_argument("--text", required=True, help="Draft (rewrite) or prompt (generate)")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--max-new-tokens", type=int, default=250)
    return p.parse_args()

def main():
    args = parse_args()
    engine = VibejamEngine(
        ckpt_path=args.ckpt_path,
        corpus_path=args.corpus_path,
        block_size=128,
    )

    if args.mode == "rewrite":
        out = engine.rewrite(args.text, temperature=args.temperature, top_k=args.top_k, max_new_tokens=args.max_new_tokens)
    else:
        out = engine.generate(args.text, temperature=args.temperature, top_k=args.top_k, max_new_tokens=args.max_new_tokens)

    print(out)

if __name__ == "__main__":
    main()