# vibejam/tokenizer_bpe.py
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

SPECIALS = ["<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>", "<|end|>"]

def bytes_to_syms(bs: bytes) -> List[str]:
    return [f"\\x{b:02x}" for b in bs]

def syms_to_bytes(syms: List[str]) -> bytes:
    out = bytearray()
    for s in syms:
        if not s.startswith("\\x") or len(s) != 4:
            raise ValueError(f"Bad byte sym: {s}")
        out.append(int(s[2:], 16))
    return bytes(out)

def get_stats(words: List[List[str]]) -> Dict[Tuple[str, str], int]:
    stats: Dict[Tuple[str, str], int] = {}
    for w in words:
        for i in range(len(w) - 1):
            pair = (w[i], w[i+1])
            stats[pair] = stats.get(pair, 0) + 1
    return stats

def merge_pair(words: List[List[str]], pair: Tuple[str, str], merged: str) -> List[List[str]]:
    a, b = pair
    out = []
    for w in words:
        i = 0
        nw = []
        while i < len(w):
            if i < len(w) - 1 and w[i] == a and w[i+1] == b:
                nw.append(merged)
                i += 2
            else:
                nw.append(w[i])
                i += 1
        out.append(nw)
    return out

@dataclass
class BPETokenizer:
    vocab: Dict[str, int]
    merges: List[Tuple[str, str]]
    id_to_sym: List[str]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str, add_bos: bool=False, add_eos: bool=False) -> List[int]:
        syms = bytes_to_syms(text.encode("utf-8"))
        for a, b in self.merges:
            merged = a + b
            syms = self._apply_one_merge(syms, (a, b), merged)

        ids = []
        if add_bos:
            ids.append(self.vocab["<|bos|>"])
        unk = self.vocab["<|unk|>"]
        for s in syms:
            ids.append(self.vocab.get(s, unk))
        if add_eos:
            ids.append(self.vocab["<|eos|>"])
        return ids

    def decode(self, ids: List[int], skip_specials: bool=True) -> str:
        syms = []
        for i in ids:
            s = self.id_to_sym[i]
            if skip_specials and s in SPECIALS:
                continue
            syms.append(s)

        flat = []
        for s in syms:
            if s.startswith("\\x") and len(s) == 4:
                flat.append(s)
            else:
                # merged symbol is concatenation of byte syms
                flat.extend([s[j:j+4] for j in range(0, len(s), 4)])

        return syms_to_bytes(flat).decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        obj = {"specials": SPECIALS, "merges": self.merges, "vocab": self.vocab}
        p.write_text(json.dumps(obj), encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> "BPETokenizer":
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        vocab = {k: int(v) for k, v in obj["vocab"].items()}
        merges = [tuple(x) for x in obj["merges"]]
        id_to_sym = [""] * len(vocab)
        for sym, idx in vocab.items():
            id_to_sym[idx] = sym
        return BPETokenizer(vocab=vocab, merges=merges, id_to_sym=id_to_sym)

    def _apply_one_merge(self, syms: List[str], pair: Tuple[str, str], merged: str) -> List[str]:
        a, b = pair
        out = []
        i = 0
        while i < len(syms):
            if i < len(syms) - 1 and syms[i] == a and syms[i+1] == b:
                out.append(merged)
                i += 2
            else:
                out.append(syms[i])
                i += 1
        return out

def train_bpe(texts: Iterable[str], num_merges: int=8000, min_pair_freq: int=2) -> BPETokenizer:
    words = [bytes_to_syms(t.encode("utf-8")) for t in texts]
    merges: List[Tuple[str, str]] = []

    for _ in range(num_merges):
        stats = get_stats(words)
        if not stats:
            break
        (a, b), freq = max(stats.items(), key=lambda kv: kv[1])
        if freq < min_pair_freq:
            break
        merges.append((a, b))
        words = merge_pair(words, (a, b), a + b)

    vocab_syms = set()
    for w in words:
        vocab_syms.update(w)

    all_syms = SPECIALS + sorted(vocab_syms)
    vocab = {s: i for i, s in enumerate(all_syms)}
    return BPETokenizer(vocab=vocab, merges=merges, id_to_sym=all_syms)