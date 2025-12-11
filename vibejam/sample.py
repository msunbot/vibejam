# vibejam/sample.py

import torch
import torch.nn.functional as F

from .model import GPTModel
from .data import CharDataset


def generate_text(
    model: GPTModel,
    dataset: CharDataset,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> str:
    """
    Generate text from the model, conditioned on a text prompt.

    - prompt is turned into token IDs via dataset.encode
    - we feed those into model.generate-like logic, but with temperature & top_k
    """

    device = next(model.parameters()).device
    model.eval()

    # Encode prompt to token IDs
    if prompt:
        context_ids = dataset.encode(prompt)
    else:
        context_ids = [0]  # start from some token if empty

    idx = torch.tensor([context_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.cfg.block_size :]
        logits, _ = model(idx_cond)

        # take logits at last time step
        logits = logits[:, -1, :]   # (B=1, vocab_size)

        # optionally apply top_k filtering
        if top_k is not None: 
            v, _ = torch.topk(logits, top_k)
            min_v = v[:, -1].unsqueeze(-1)
            logits = torch.where(logits < min_v, torch.full_like(logits, -float("inf")), logits)

        # apply temperature
        logits = logits / temperature

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
        idx = torch.cat([idx, next_token], dim=1)

    out_ids = idx[0].tolist()
    return dataset.decode(out_ids)
