# vibejam/lm_interface.py
from __future__ import annotations

from typing import Protocol, Optional, Tuple
import torch


class BaseLM(Protocol):
    """
    Minimal architecture-agnostic LM interface used by the vibejam harness.

    Mental model:
    - forward() is the *only* required compute path (train + eval + sampling).
    - get_block_size() tells the harness how much context the model can see.
    - configure_optimizers() lets each architecture define parameter grouping if needed.

    This is intentionally tiny so RWKV-lite and MoE-FFN can conform without ceremony.
    """

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        idx: (B, T) int64 token ids
        targets: (B, T) int64 token ids or None
        returns:
          logits: (B, T, V)
          loss: scalar tensor or None
        """
        ...

    def get_block_size(self) -> int:
        """Max context length the model supports."""
        ...

    def configure_optimizers(self, train_cfg) -> torch.optim.Optimizer:
        """
        Return an optimizer for this model.
        train_cfg is expected to have at least: learning_rate
        """
        ...