# vibejam/rewrite.py

from .model import GPTModel
from .data import CharDataset
from .sample import generate_text

# Simple prompt template for rewrite mode
REWRITE_PROMPT = (
    "Below is some draft text. Rewrite it in my usual style, keeping the same meaning "
    "but using my tone and cadence.\n\n"
    "Draft:\n"
    "{draft}\n\n"
    "Rewrite:\n"
)

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

    # Extract only the rewritten portion (after "Rewrite:\n")
    marker = "Rewrite:\n"
    idx = full.find(marker)
    if idx == -1:
        # Fallback: if marker not found, just return everything
        return full

    rewritten = full[idx + len(marker):]
    return rewritten.strip()