# scripts/prepare_data.py

import argparse
from pathlib import Path

DOC_SEP = "\n<|doc_end|>\n"

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare a vibejam text corpus from a folder.")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing .txt / .md files",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        required=True,
        help="Output .txt file for concatenated corpus",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    out_path = Path(args.out_path)

    texts = []
    for ext in ("*.txt", "*.md"):
        for fp in input_dir.glob(ext):
            with open(fp, "r", encoding="utf-8") as f:
                txt = f.read().strip()
                if not txt:
                    continue
                # attach a filename marker if you want
                texts.append(txt)

    if not texts:
        raise ValueError(f"No .txt or .md files found in {input_dir}")

    corpus = DOC_SEP.join(texts)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(corpus)

    print(f"[vibejam] Wrote concatenated corpus to {out_path}")
    print(f"  Documents: {len(texts)}")
    print(f"  Corpus length (characters): {len(corpus)}")


if __name__ == "__main__":
    main()