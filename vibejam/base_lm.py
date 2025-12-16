# vibejam/base_lm.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Optional, Dict, Any, List
import torch

class BaseLM(Protocol):
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        idx: (B, T) int64 token ids
        targets: (B, T) or None
        returns dict with:
          - logits: (B, T, V)
          - loss: scalar (optional if targets is None)
        """
        ...

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float, top_k: int) -> torch.Tensor:
        """
        idx: (B, T)
        returns: (B, T + max_new_tokens)
        """
        ...

    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "BaseLM": ...