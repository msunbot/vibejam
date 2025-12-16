# vibejam  
*A tiny personal-style language model, built from scratch.*

vibejam is a small, educational language model that learns from your own writing and acts as a **“last-mile style engine”** on top of larger LLMs.

It is intentionally minimal and transparent, designed to deepen intuition about the *entire* language model stack — from tokenization to training to inference — rather than to maximize output quality.

---

## What vibejam is

At its core, vibejam is:

- a decoder-only Transformer trained from scratch in PyTorch  
- a controllable pipeline for learning personal writing style  
- a systems-first ML project that treats tokenization, datasets, and checkpoints as first-class artifacts  

It supports two operating modes:

- **LM mode** → generate text in your general “voice”  
- **Rewrite mode (v0)** → take a draft and rewrite it in your style (prototype)

---

## Why vibejam exists

### As a product seed

vibejam is a prototype for a lightweight “style engine” that could sit behind large LLMs:

- A large LLM (GPT-4, Claude, etc.) handles reasoning and semantics  
- vibejam adapts that content to your tone, cadence, and structure  

The hypothesis is that *style* can be modeled locally and cheaply, without retraining a full-scale model.

### As a learning artifact

vibejam is also a deliberately transparent implementation of a language model stack, meant to build intuition about:

- tokenization (char-level vs BPE)  
- token and positional embeddings  
- causal self-attention  
- residual blocks and LayerNorm  
- next-token prediction training loops  
- ML engineering patterns: configs, CLIs, checkpoints, resumption  

---

## Features (Layer 1 + 1.5)

- Decoder-only Transformer (nanoGPT-style)
- Explicit tokenization layer:
  - CharTokenizer (legacy / educational)
  - BPETokenizer (byte-level, persisted artifact)
- Tokenizer-agnostic dataset abstraction (TokenDataset)
- Configurable training pipeline (train_lm)
- Periodic checkpointing with full config + tokenizer metadata
- Sampling API with temperature and top-k
- Rewrite mode v0 with instruction-style prompt and stop token
- CLI-first workflow for training, sampling, and rewriting

---

## Project structure

    vibejam/
      README.md
      PROGRESS.md
      docs/
        architecture.md
        lessons-learned.md

      vibejam/
        config.py        (ModelConfig, TrainConfig, DataConfig)
        data.py          (CharDataset, TokenDataset)
        tokenizer_char.py
        tokenizer_bpe.py
        model.py         (GPTModel)
        train.py         (training loop, checkpointing)
        sample.py        (sampling utilities)
        rewrite.py       (rewrite logic + prompt)

      scripts/
        train_tokenizer_bpe.py
        prepare_rewrite_pairs.py
        train_lm.py
        train_rewrite_finetune.py
        sample_lm.py
        rewrite_demo.py

---

## Installation

From the repo root:

    pip install -e .

Requires Python 3.9+ and PyTorch.

---

## Quickstart

### 1. Build a personal text corpus

Put your .txt / .md files under data/raw/, then run:

    python -m scripts.prepare_data \
      --input-dir data/raw \
      --out-path data/personal_corpus.txt

This produces a single corpus file with document separators.

---

### 2. Train a BPE tokenizer

    python scripts.train_tokenizer_bpe.py \
      --corpus-path data/personal_corpus.txt \
      --out checkpoints/vibejam_tokenizer_bpe.json

This creates a persisted tokenizer artifact that is reused across training and inference.

---

### 3. Train a base language model (BPE)

    python scripts.train_lm.py \
      --text-path data/personal_corpus.txt \
      --ckpt-path checkpoints/vibejam_bpe_personal.pt \
      --block-size 128 \
      --tokenizer-type bpe \
      --tokenizer-path checkpoints/vibejam_tokenizer_bpe.json

This trains a small Transformer from scratch using BPE tokenization.

---

### 4. Generate text

    python -m scripts.sample_lm \
      --text-path data/personal_corpus.txt \
      --ckpt-path checkpoints/vibejam_bpe_personal.pt \
      --tokenizer-type bpe \
      --tokenizer-path checkpoints/vibejam_tokenizer_bpe.json \
      --prompt "Today I feel"

Sampling supports temperature and top-k for controllable generation.

---

### 5. Rewrite text in your voice (v0)

    python -m scripts.rewrite_demo \
      --text-path data/personal_corpus.txt \
      --ckpt-path checkpoints/vibejam_bpe_personal.pt \
      --tokenizer-type bpe \
      --tokenizer-path checkpoints/vibejam_tokenizer_bpe.json \
      --draft "Today I went for a walk and felt good about the progress on my project."

Rewrite mode wraps the draft into a structured instruction prompt and returns the model’s continuation after the Rewrite section.

Outputs are prototype-grade; the goal is validating the end-to-end pipeline rather than producing polished rewrites.

---

## Design decisions & tradeoffs

### Tokenization

- Char-level tokenization was used initially for transparency and learning.
- BPE was introduced in Layer 1.5 to improve semantic modeling and efficiency.
- Tokenizers are treated as persisted artifacts, not implicit preprocessing steps.

### Model scale

- The default model is intentionally small and CPU-friendly.
- This makes failures understandable and iteration fast.
- The project optimizes for learning, not benchmark performance.

### Objective

- Pure next-token prediction throughout.
- Rewrite behavior emerges from prompt formatting and (optionally) fine-tuning.

---

## Limitations (v0.1)

- Rewrite mode is not yet fine-tuned on Draft→Rewrite pairs using BPE.
- Context length is short compared to large LLMs.
- Output quality is uneven at this scale.

These are deliberate tradeoffs for an early, learning-focused release.

---

## Roadmap

### Layer 1.5+
- BPE rewrite fine-tuning on Draft→Rewrite datasets
- Larger context windows and longer training runs (GPU)

### Layer 2
- Introduce a BaseLM interface and model factory
- Clean integration with external LLMs as a style adapter

### Layer 3 (Labs)
- Alternative architectures (Mamba, gated models, etc.)
- Experiments in continual and nested learning on personal data

---

## Progress

See PROGRESS.md for a detailed, day-by-day account of development, debugging, and lessons learned.

---

## License

MIT (or update as desired).