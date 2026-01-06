# vibejam/rewrite.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Any

from vibejam.prompts import REWRITE_STOP, build_rewrite_prompt
from vibejam.sample import generate_text


@dataclass
class RewriteConfig:
    # Defaults optimized for "rewrite" behavior (less chaotic than free sampling)
    temperature: float = 0.6
    top_k: int = 50
    max_new_tokens: int = 256

    # Stop control
    stop_str: str = REWRITE_STOP

    # Sampling hygiene
    max_consecutive_newlines: int = 6
    seed: int | None = 123  # make eval comparable by default


def _extract_rewrite(full_text: str) -> str:
    """
    Extract the "Rewrite:" span from the model output.

    We assume prompts contain a "Rewrite:" tag and an explicit stop token REWRITE_STOP.
    """
    key = "Rewrite:\n"
    i = full_text.rfind(key)
    if i != -1:
        full_text = full_text[i + len(key):]

    j = full_text.find(REWRITE_STOP)
    if j != -1:
        full_text = full_text[:j]

    return full_text.strip()


def rewrite_text(
    model: Any,
    dataset: Any,
    draft: str,
    prompt: str | None = None,
    cfg: Optional[RewriteConfig] = None,
    # Back-compat overrides:
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
) -> str:
    """
    Rewrite a draft into vibejam style.

    Priority order:
      explicit overrides > cfg > defaults
    """
    if prompt is None:
        prompt = build_rewrite_prompt(draft)

    if cfg is None:
        cfg = RewriteConfig()

    mnt = int(max_new_tokens) if max_new_tokens is not None else int(cfg.max_new_tokens)
    temp = float(temperature) if temperature is not None else float(cfg.temperature)
    tk = int(top_k) if top_k is not None else int(cfg.top_k)

    full = generate_text(
        model=model,
        dataset=dataset,
        prompt=prompt,
        max_new_tokens=mnt,
        temperature=temp,
        top_k=tk,
        stop_str=cfg.stop_str,
        max_consecutive_newlines=cfg.max_consecutive_newlines,
        seed=cfg.seed,
    )
    return _extract_rewrite(full)