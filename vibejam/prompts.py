# vibejam/prompts.py

from dataclasses import dataclass

# Keep this *identifcal* across: dataset building, finetune examples, inference.
# Rule: model sees Draft -> Rewrite, and we only return the Rewrite span.
REWRITE_STOP = "<|end|>"

def build_rewrite_prompt(draft: str) -> str:
    return f"Draft:\n{draft.strip()}\n\nRewrite:\n"

def wrap_rewrite_example(draft: str, rewrite: str) -> str:
    return build_rewrite_prompt(draft) + rewrite.strip() + "\n" + REWRITE_STOP + "\n"


