# scripts/train_tokenizer_bpe.py
import argparse
from pathlib import Path
from vibejam.tokenizer_bpe import train_bpe

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-merges", type=int, default=8000)
    ap.add_argument("--min-pair-freq", type=int, default=2)
    args = ap.parse_args()

    text = Path(args.corpus_path).read_text(encoding="utf-8")
    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    tok = train_bpe(chunks, num_merges=args.num_merges, min_pair_freq=args.min_pair_freq)
    tok.save(args.out)
    print("saved", args.out, "vocab_size", tok.vocab_size, "merges", len(tok.merges))

if __name__ == "__main__":
    main()