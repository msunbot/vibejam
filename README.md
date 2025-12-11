# vibejam  
*A tiny personal-style GPT, built from scratch on a nanoGPT-style Transformer.*

vibejam is a small, educational language model that learns from your own writing and acts as a **“last-mile style engine”** on top of larger LLMs.

Layer 1 focuses on:

- training a small decoder-only Transformer on your personal text, and  
- providing:  
  - **LM mode** → generate text in your general “voice”  
  - **rewrite mode v0** → take a draft and rewrite it in your style (prototype)

It is intentionally minimal: just PyTorch, a char-level tokenizer, and clean scripts.

---

## Why vibejam exists

**As a product seed**

- A lightweight “style engine” that can sit behind GPT-4 / Claude:
  - Big LLM handles reasoning and semantics.
  - vibejam adapts that content to your tone and cadence.

**As a learning artifact**

- A fully transparent Transformer implementation to deepen intuition about:
  - token/position embeddings  
  - causal self-attention  
  - residual blocks + LayerNorm  
  - next-token prediction training loops  
  - ML engineering patterns: configs, CLIs, checkpoints, sampling

---

## Features (Layer 1 v0)

- nanoGPT-style Transformer LM (decoder-only, causal self-attention)
- Char-level tokenizer for simplicity and transparency
- Configurable training pipeline (`train_lm`)
- Checkpoint saving with model + configs
- Sampling API (`generate_text`) with temperature and top-k
- Rewrite mode v0 (`rewrite_text`) using an instruction-style prompt
- Folder-to-corpus ingestion (`prepare_data`)

---

## Project structure

    vibejam/
      README.md
      PROGRESS.md
      pyproject.toml
      requirements.txt
      .gitignore

      vibejam/
        __init__.py
        config.py       (ModelConfig, TrainConfig, DataConfig)
        data.py         (CharDataset + batch sampling)
        model.py        (GPTModel: Transformer LM)
        train.py        (train_lm, estimate_loss)
        sample.py       (generate_text)
        rewrite.py      (rewrite_text + prompt template)

      scripts/
        prepare_data.py     (raw folder → single corpus file)
        train_lm.py         (CLI wrapper for train_lm)
        sample_lm.py        (CLI for sampling)
        rewrite_demo.py     (CLI for rewrite mode)

      notebooks/
        01_toy_transformer.ipynb
        02_train_personal_lm.ipynb
        03_rewrite_in_my_voice.ipynb

---

## Installation

From the repo root:

    pip install -e .

Requires Python 3.9+ and PyTorch.

---

## Quickstart

### 1. Build a personal text corpus

1. Put your `.txt` / `.md` files under `data/raw/`.  
2. Run:

       python -m scripts.prepare_data \
         --input-dir data/raw \
         --out-path data/personal_corpus.txt

This writes a single `data/personal_corpus.txt` with `<|doc_end|>` separators between documents.

---

### 2. Train a small LM on your corpus

Train on CPU or GPU (GPU is auto-detected if available):

    python -m scripts.train_lm \
      --text-path data/personal_corpus.txt \
      --ckpt-path checkpoints/vibejam_personal.pt

Defaults (configured via `TrainConfig` / `DataConfig`):

- char-level tokenizer  
- small Transformer (2 layers, 64-dim embeddings, 4 heads)  
- configurable `block_size` (context length)  
- periodic train/val loss logging  
- checkpoint with model weights + configs saved to `--ckpt-path`

To train on Colab GPU:

- Enable GPU runtime.  
- Clone the repo in the notebook.  
- Run the same `train_lm` command.

---

### 3. Generate text from your model

    python -m scripts.sample_lm \
      --text-path data/personal_corpus.txt \
      --ckpt-path checkpoints/vibejam_personal.pt \
      --prompt "Today I feel"

Options:

- `--max-new-tokens` (length of continuation)  
- `--temperature`:
  - < 1.0 → more deterministic
  - > 1.0 → more random/creative  
- `--top-k` to restrict sampling to the top-k tokens

---

### 4. Rewrite text in your voice (v0)

    python -m scripts.rewrite_demo \
      --text-path data/personal_corpus.txt \
      --ckpt-path checkpoints/vibejam_personal.pt \
      --draft "Today I went for a walk and felt good about the progress on my project."

What happens:

- `rewrite_demo.py`:
  - reloads model + configs from checkpoint
  - rebuilds `CharDataset` to get vocab/encode/decode
  - calls `rewrite_text(...)` with your draft

- `rewrite_text(...)`:
  - wraps the draft into an instruction prompt:

        Below is some draft text. Rewrite it in my usual style, keeping the same meaning but using my tone and cadence.

        Draft:
        <draft>

        Rewrite:

  - generates a continuation  
  - returns the text after `"Rewrite:\n"` as the rewritten output

Note: with a small char-level model and no explicit Draft→Rewrite fine-tuning, outputs are prototype-grade and often noisy. The goal in v0 is to validate the architecture and end-to-end flow.

---

## Design decisions & tradeoffs

**Char-level tokenizer (for v0)**

- Pros:
  - simple and easy to reason about  
  - no external tokenizer dependencies  
  - direct mapping from IDs to characters  

- Cons:
  - large vocab (hundreds of characters)  
  - longer sequences needed to represent semantics  
  - outputs tend to “semi-English babble” at this scale

**Small Transformer**

- 2 layers, 64-dim embeddings, 4 heads by default.  
- Easy to train on CPU / Colab and to inspect.  
- Intentionally underpowered for production; ideal for learning and iteration.

**Pure next-token prediction**

- Both LM mode and rewrite mode v0 share the same objective.  
- No special loss term for rewrite; behavior is driven entirely by how we format the text.

**CLI-first workflow**

- Training, sampling, and rewriting are simple CLI commands.  
- Notebooks are used for experiments, not as the primary API surface.

---

## Limitations (current v0)

- Char-level modeling; no BPE/tokenizer model yet.  
- Short context compared to “real” LLMs; rewrite only sees local context.  
- Rewrite mode is not fine-tuned on actual Draft→Rewrite pairs.  
- Model is small; style is captured locally rather than globally.

This is deliberate: Layer 1 is about building and understanding a complete stack, not maximizing output quality.

---

## Roadmap

**Layer 1.5 — Stronger style modeling**

- Retrain on GPU with:
  - larger `block_size` (64–128)  
  - more layers / larger embeddings  
  - more training steps  
- Introduce a BPE tokenizer backend for better semantics and efficiency.  
- Build a (draft, rewrite) dataset and add `train_rewrite_finetune.py` for instruction-style fine-tuning.

**Layer 2 — Model interface & big LLM integration**

- Introduce `BaseLM` and `build_model(arch=...)` to support multiple architectures.  
- Wrap vibejam into a `StyleEngine`:
  - big LLM (GPT-4, Claude, etc.) generates content  
  - vibejam adapts it to your voice  

**Layer 3 — Labs**

- Use vibejam’s training harness to experiment with:
  - Mamba  
  - gated attention  
  - xLSTM  
  - nested / continual learning ideas on personal style data  

---

## Progress

For a detailed, day-by-day breakdown of the work (including debugging, design choices, and learnings), see:

- `PROGRESS.md`

---

## License

MIT (or update to your preferred license).