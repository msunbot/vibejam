import argparse
import json
from pathlib import Path

DOC_SEP = "<|doc_end|>"

TEMPLATE = """<|sample|>
You are vibejam, a personal style engine.
Task: Rewrite the draft in my usual style. Keep meaning. Keep it concise.

Draft:
{draft}

Rewrite:
{rewrite}
<|end|>
"""

def split_into_chunks(text: str, min_chars: int, max_chars: int):
    """
    Split text into roughly paragraph-sized chunks.
    Very simple heuristic:
    - split by blank lines
    - pack paragraphs together until max_chars
    """
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    cur = ""

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

def load_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-path", type=str, required=True, help="e.g. data/personal_corpus.txt")
    ap.add_argument("--drafts-jsonl", type=str, required=True, help="JSONL with {id, draft}")
    ap.add_argument("--out-path", type=str, required=True, help="e.g. data/rewrite_pairs.txt")
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--max-chars", type=int, default=800)
    args = ap.parse_args()

    corpus_path = Path(args.corpus_path)
    drafts_path = Path(args.drafts_jsonl)
    out_path = Path(args.out_path)

    corpus = corpus_path.read_text(encoding="utf-8")

    # Split corpus into candidate "rewrite" chunks
    chunks = split_into_chunks(corpus.replace(DOC_SEP, "\n\n"), args.min_chars, args.max_chars)

    # We need IDs to match drafts to rewrites. We’ll assign ids chunk_000001, chunk_000002, ...
    id_to_rewrite = {f"chunk_{i:06d}": ch for i, ch in enumerate(chunks, start=1)}

    drafts = load_jsonl(drafts_path)
    id_to_draft = {row["id"]: row["draft"] for row in drafts if "id" in row and "draft" in row}

    # Intersect IDs
    ids = sorted(set(id_to_draft.keys()) & set(id_to_rewrite.keys()))
    if not ids:
        raise ValueError(
            "No matching ids between drafts and rewrite chunks.\n"
            "Expected ids like chunk_000001 etc. Ensure your JSONL uses the same ids."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for _id in ids:
            draft = id_to_draft[_id].strip()
            rewrite = id_to_rewrite[_id].strip()

            if not draft or not rewrite:
                continue

            f.write(TEMPLATE.format(draft=draft, rewrite=rewrite))
            f.write("\n")
            n_written += 1

    print(f"[vibejam] Wrote {n_written} rewrite-pairs to {out_path}")
    print("Example id range:", ids[0], "…", ids[-1])

if __name__ == "__main__":
    main()