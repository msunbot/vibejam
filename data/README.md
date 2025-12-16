# data/

This directory contains **local-only training data** and is intentionally not tracked in git.

Typical contents (ignored by `.gitignore`):
- `personal_corpus.txt`
- `rewrite_pairs.txt`
- `rewrite_drafts*.jsonl`
- `raw/` (source documents)

Only small, non-sensitive demo files (e.g. `toy.txt`) should be committed.