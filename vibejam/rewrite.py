# vibejam/rewrite.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from vibejam.prompts import REWRITE_STOP, build_rewrite_prompt
from vibejam.sample import generate_text

@dataclass
class RewriteConfig:
    temperature: float = 0.7
    top_k: int = 50
    max_new_tokens: int = 256
    # Safety: if stop not found, we still extract up to max_new_tokens.
    stop_str: str = REWRITE_STOP

def extract_rewrite_span(full_text: str, fmt: RewriteFormat) -> str:
    """
    full_text contains: Draft... Rewrite... <|end|> ...
    We return text after fmt.rewrite_tag up to fmt.stop.
    """
    key = fmt.rewrite_tag
    idx = full_text.rfind(key)
    if idx == -1:
        # Fallback: return tail (better than nothing)
        return full_text.strip()

    after = full_text[idx + len(key):].lstrip("\n").rstrip()

    # Stop token
    stop_i = after.find(fmt.stop)
    if stop_i != -1:
        after = after[:stop_i]

    return after.strip()

def _extract_rewrite(full_text: str) -> str:
    # 1) Find the last occurrence of "Rewrite:\n"
    key = "Rewrite:\n"
    i = full_text.rfind(key)
    if i != -1: 
        full_text = full_text[i + len(key):]
    
    # 2) Truncate at stop token
    j = full_text.find(REWRITE_STOP)
    if j != -1: 
        full_text = full_text[:j]
    
    return full_text.strip()

def rewrite_text(
    model, 
    dataset, 
    draft: str,
    prompt: str | None = None, 
    max_new_tokens: int = 250, 
    temperature: float = 0.7, 
    top_k: int | None = 50,
) -> str:
    if prompt is None: 
        prompt = build_rewrite_prompt(draft)
    
    full = generate_text(
        model=model,
        dataset=dataset,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        stop_str=REWRITE_STOP,
    )
    return _extract_rewrite(full)