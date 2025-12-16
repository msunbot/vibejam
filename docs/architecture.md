## Architecture Overview

vibejam consists of four explicit layers:

1. Tokenizer
   - CharTokenizer (legacy)
   - BPETokenizer (byte-level, persisted artifact)

2. Dataset
   - TokenDataset: tokenizer-agnostic sliding window dataset
   - Produces (x, y) pairs for causal LM training

3. Model
   - GPT-style Transformer implemented from scratch
   - No external ML frameworks

4. Training Loop
   - Explicit control over checkpointing, evaluation, and resumption

Key invariant:
**token IDs must be stable across training and inference**.