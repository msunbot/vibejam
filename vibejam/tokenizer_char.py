# vibejam/tokenizer_char.py
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

@dataclass
class CharTokenizer:
    stoi: Dict[str, int]
    itos: Dict[int, str]

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, s: str) -> List[int]:
        # strict: unknown chars are dropped (matches your CharDatasetWithVocab behavior)
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos[i] for i in ids if i in self.itos)

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        obj = {"stoi": self.stoi, "itos": {str(k): v for k, v in self.itos.items()}}
        p.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def load(path: str) -> "CharTokenizer":
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        stoi = {k: int(v) for k, v in obj["stoi"].items()}
        itos = {int(k): v for k, v in obj["itos"].items()}
        return CharTokenizer(stoi=stoi, itos=itos)