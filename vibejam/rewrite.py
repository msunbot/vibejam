# vibejam/rewrite.py

from .model import GPTModel
from .data import CharDataset
from .sample import generate_text

# Simple prompt template for rewrite mode
REWRITE_PROMPT = "<|sample|>\nDraft:\n{draft}\nRewrite:\n"

def build_rewrite_prompt(draft: str) -> str:
    """Format the rewrite instruction prompt around the draft."""
    draft = draft.strip()
    return REWRITE_PROMPT.format(draft=draft)


def rewrite_text(
    model: GPTModel,
    dataset: CharDataset,
    draft: str,
    max_new_tokens: int = 600,
    temperature: float = 0.7,
    top_k: int | None = 40,
) -> str:
    """
    High-level rewrite API:

    - wraps the user's draft inside a rewrite prompt
    - calls the LM to generate continuation
    - extracts the part after 'Rewrite:\\n' as the rewritten text
    """
    prompt = build_rewrite_prompt(draft)
    full = generate_text(
        model=model,
        dataset=dataset,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    marker = "Rewrite:\n"
    end_marker = "<|end|>"

    # If the model re-generated the template, keep the LAST rewrite section
    last_idx = full.rfind(marker)
    if last_idx == -1:
        return full.strip()

    rewritten = full[last_idx + len(marker):]

    # Stop at end marker if present
    end_idx = rewritten.find(end_marker)
    if end_idx != -1:
        rewritten = rewritten[:end_idx]

    # Light cleanup: remove any accidental template tokens
    rewritten = rewritten.replace("<|sample|>", "").strip()

    return rewritten.strip()