import argparse, json
from pathlib import Path

DOC_SEP = "<|doc_end|>"

def split_into_chunks(text: str, min_chars: int, max_chars: int):
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if not cur:
            cur = p
        elif len(cur) + 2 + len(p) <= max_chars:
            cur = cur + "\n\n" + p
        else:
            if len(cur) >= min_chars:
                chunks.append(cur)
            cur = p
    if cur and len(cur) >= min_chars:
        chunks.append(cur)
    return chunks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--max-chars", type=int, default=800)
    args = ap.parse_args()

    corpus = Path(args.corpus_path).read_text(encoding="utf-8")
    chunks = split_into_chunks(corpus.replace(DOC_SEP, "\n\n"), args.min_chars, args.max_chars)

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        for i, ch in enumerate(chunks, start=1):
            _id = f"chunk_{i:06d}"
            # draft = slightly “worse” version; for now just truncate the chunk
            draft = ch[: max(80, len(ch)//2)]
            f.write(json.dumps({"id": _id, "draft": draft}, ensure_ascii=False) + "\n")

    print(f"Wrote {len(chunks)} drafts to {outp}")

if __name__ == "__main__":
    main()