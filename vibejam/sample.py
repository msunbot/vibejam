# vibejam/sample.py

import torch
import torch.nn.functional as F
from typing import Any, Optional

from .lm_interface import BaseLM

def _apply_top_k(logits: torch.Tensor, top_k: Optional[int]) -> torch.Tensor:
    if top_k is None:
        return logits
    k = int(top_k)
    if k <= 0: 
        return logits
    v, _ = torch.topk(logits, k)
    min_v = v[:, -1].unsqueeze(-1)
    return torch.where(logits < min_v, torch.full_like(logits, -float("inf")), logits)

def generate_text(
    model: BaseLM,
    dataset: Any,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_k: int | None = None,
    stop_str: str | None = None,
    max_consecutive_newlines: int = 6,
    seed: int | None = None, 
) -> str:
    """
    Generate text from the model, conditioned on a text prompt.

    - prompt is turned into token IDs via dataset.encode
    - we feed those into model-forward sampling with temperature & top_k
    """
    device = next(model.parameters()).device
    model.eval()

    if seed is not None: 
        # Make eval runs comparable (still not perfectly deterministic across all CUDA ops,
        # but good enough for small-scale comparisons.)
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    # Encode prompt to token IDs
    if prompt:
        context_ids = dataset.encode(prompt)
    else:
        context_ids = [0]  # start from some token if empty

    idx = torch.tensor([context_ids], dtype=torch.long, device=device)

    block_size = int(model.get_block_size())
    consecutive_newlines = 0 

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)

        # last time step
        logits = logits[:, -1, :]   # (B=1, vocab_size)

        # temperature (avoid divide-by-zero)
        temp = max(float(temperature), 1e-6)
        logits = logits / temp

        # top-k
        logits = _apply_top_k(logits, top_k)

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
        idx = torch.cat([idx, next_token], dim=1)

        # ---- runaway guard: consecutive newlines ----
        if max_consecutive_newlines is not None and max_consecutive_newlines > 0:
            token_id = int(next_token.item())
            ch = None
            # CharDataset exposes itos; TokenDataset usually doesn't.
            if hasattr(dataset, "itos") and isinstance(dataset.itos, (list, tuple)):
                if 0 <= token_id < len(dataset.itos):
                    ch = dataset.itos[token_id]

            if ch == "\n":
                consecutive_newlines += 1
                if consecutive_newlines >= int(max_consecutive_newlines):
                    break
            else:
                consecutive_newlines = 0
        # -------------------------------------------

    out_ids = idx[0].tolist()
    text = dataset.decode(out_ids)

    if stop_str:
        j = text.find(stop_str)
        if j != -1:
            text = text[: j + len(stop_str)]

    return text