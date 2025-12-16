# vibejam Progress  
*Layer 1 — Personal Style GPT (nanoGPT-based)*  
*Started: Dec 11, 2025*

---

## Week 1: Layer 1 Foundations (Days 1–6)

### Day 1 (Dec 11) — Repo & Package Skeleton ✅  
**Phase 0 — Setup & Tiny Transformer**

- [x] Created `vibejam/` repo with clear Python package structure:
  - `vibejam/` (package), `scripts/`, `notebooks/`
- [x] Added `pyproject.toml` and `requirements.txt`; enabled editable install via `pip install -e .`
- [x] Verified package imports (`from vibejam.model import GPTModel`) using module mode (`python -m scripts.toy_test`)
- [x] Set up `.gitignore` for virtualenv, raw data, checkpoints, OS artifacts  
**Result:** Clean, importable package and repo ready for iterative ML work.

**Key Learnings:**
- Python only treats directories as packages when they have `__init__.py`.
- Running scripts via `python -m scripts.name` is the clean way to get package imports working.
- Editable install (`pip install -e .`) makes local development + Colab reuse straightforward.

---

### Day 2 (Dec 11) — Minimal GPTModel (Karpathy-style) ✅  
**Phase 0 → Phase 1 — Model Core**

- [x] Implemented core Transformer components in `model.py` (typed by hand):
  - `Head` (causal self-attention with tril mask)
  - `MultiHeadAttention` (parallel heads + projection)
  - `FeedForward` (2-layer MLP with residual)
  - `Block` (pre-LN + residual around attention + MLP)
  - `GPTModel` (token embeddings + positional embeddings + stacked blocks + LM head)
- [x] Implemented `forward(idx, targets=None)` returning `(logits, loss)` with shape checks
- [x] Implemented `generate(idx, max_new_tokens)` for autoregressive sampling  
**Result:** A working nanoGPT-style decoder-only Transformer LM, fully understood at tensor-shape level.

**Key Learnings:**
- Attention shapes: `x: (B, T, C)`, `q,k,v: (B, T, head_size)`, attention weights: `(B, T, T)`.
- Causal masking is just a lower-triangular matrix applied before softmax.
- Flattening logits and targets to `(B*T, vocab_size)` / `(B*T,)` is the standard CE pattern.

---

### Day 3 (Dec 11) — Data Pipeline, Toy Test & Loss Sanity ✅  
**Phase 1 — Data & Local Sanity Checks**

- [x] Implemented `CharDataset` in `data.py`:
  - Builds char-level vocab (`stoi`, `itos`)
  - Encodes full corpus into a 1D tensor
  - Splits into `train_data` / `val_data`
  - Provides `get_batch(split, batch_size)` → `(x, y)` windows
- [x] Added `ModelConfig`, `TrainConfig`, `DataConfig` via `dataclasses` in `config.py`
- [x] Created `scripts/toy_test.py` to run a forward pass and print logits shape + loss
- [x] Debugged `ModuleNotFoundError: No module named 'vibejam'` by:
  - Ensuring correct folder nesting (single `vibejam/` package under repo root)
  - Using `python -m scripts.toy_test` from repo root  
**Result:** End-to-end forward pass working locally; verified model + data integration.

**Key Learnings:**
- Off-by-one issues in context windows matter: `len(data) > block_size + 1` is a hard requirement.
- Keeping block size small initially (e.g. 16) makes debugging easier.
- Char-level vocab on real text can be surprisingly large (hundreds of symbols).

---

### Day 4 (Dec 11) — Training Loop & CLI Training Script ✅  
**Phase 1 → Phase 2 — Training Infrastructure**

- [x] Implemented `train_lm(...)` in `vibejam/train.py`:
  - Loads text file → `CharDataset`
  - Creates `GPTModel` + AdamW optimizer
  - Core loop: `get_batch → model(xb, yb) → loss.backward() → optimizer.step()`
  - `estimate_loss()` for periodic train/val loss logging
  - Optional checkpoint saving (model + configs) to `.pt`
- [x] Implemented `scripts/train_lm.py` CLI:
  - `--text-path`, `--ckpt-path`
  - Builds configs and calls `train_lm(...)`
- [x] Debugged `random_ expects 'from' < 'to'` errors:
  - Root cause: validation split shorter than `block_size` → `torch.randint(0, len(data)-block_size, ...)` invalid
  - Added explicit safety checks and `val=nan` fallback when val is too short
- [x] Confirmed training on `data/toy.txt` quickly reached near-zero train loss (memorization as expected)  
**Result:** Usable LM trainer that runs both on tiny toy corpora and larger personal corpora.

**Key Learnings:**
- Very small corpora + relatively big models → quick overfitting and tiny loss are expected (model memorizes).
- Validation must have enough tokens for the chosen `block_size`; otherwise, centralized safety checks are essential.
- Logging both train and val losses gives quick intuition about under/overfitting even in character-level settings.

---

### Day 5 (Dec 11) — Personal Corpus Training & Sampling ✅  
**Phase 2 → Phase 3 — Voice Model v0**

- [x] Implemented `scripts/prepare_data.py`:
  - Reads all `.txt` / `.md` from `data/raw/`
  - Concatenates them with `<|doc_end|>` separators into `data/personal_corpus.txt`
- [x] Trained vibejam on `personal_corpus.txt`:
  - `vocab_size ≈ 645`, `train tokens ≈ 317k`, `val tokens ≈ 35k`
  - `block_size=16` and `block_size=64` experiments
  - Final val loss ≈ 2.32–2.45 (perplexity ~10–11), train and val closely matched
- [x] Implemented `generate_text(...)` in `sample.py`:
  - Supports `temperature` scaling and optional `top_k` filtering
  - Generates text conditioned on an initial prompt
- [x] Observed:
  - With tiny toy corpus → near-perfect memorization (loss → 0)
  - With personal corpus → stable, nontrivial modeling of character-level structure  
**Result:** First working “voice model” trained on real personal text, with sampling hooks in place.

**Key Learnings:**
- Higher `block_size` does not automatically reduce loss; it makes the modeling problem richer and sometimes harder with the same capacity.
- Temperature:
  - `<1` → sharper, more deterministic outputs
  - `>1` → more random and creative, but often noisier
- Char-level + large vocab leads to noisy, semi-English outputs; good enough for plumbing, but BPE will be important later.

---

### Day 6 (Dec 11) — Rewrite Mode v0, Style Engine & Repo Polish ✅  
**Phase 4 → Phase 5 → Phase 6 — Rewrite & Abstraction Hooks**

- [x] Implemented rewrite prompt + `rewrite_text(...)` in `rewrite.py`:
  - Prompt pattern: `"Below is some draft text. Rewrite it in my usual style..."` with `Draft:` and `Rewrite:` sections
  - Uses `generate_text` under the hood and extracts the portion after `"Rewrite:\n"`
- [x] Added `scripts/rewrite_demo.py`:
  - Loads checkpoint + corpus
  - Takes `--draft` from CLI
  - Prints original draft and vibejam’s rewritten output
- [x] Verified end-to-end pipeline:
  - Train LM on personal corpus
  - Save checkpoint
  - Load checkpoint + dataset
  - Run `rewrite_demo` on a sample draft
- [x] Observed v0 rewrite behavior:
  - Model produces character-level, semi-English babble with hints of rhythm but no true semantic fidelity
  - Limitations traced to:
    - char-level modeling
    - small model + short block_size
    - no instruction-tuned (Draft → Rewrite) training yet
- [x] Added `pyproject.toml` and cleaned `.gitignore` to exclude raw data and checkpoints from Git history  
**Result:** vibejam v0 Layer 1 complete:
- Small personal-style GPT model
- End-to-end training, sampling, and rewrite demo on real personal text
- Repo ready to push as a clean, educational project.

**Key Learnings:**
- Instruction patterns like `Draft:` / `Rewrite:` don’t “work” automatically; the LM must actually see such formats during training or fine-tuning.
- For meaningful rewrite, the next steps are:
  - larger context (`block_size`), more capacity, and more steps (ideally on GPU)
  - subword (BPE) tokenization
  - explicit (draft, rewrite) fine-tuning data
- Layer 2 integration (big LLM upstream + vibejam downstream) naturally builds on the current `rewrite_text` API.

---

## Technical Decisions

1. **Char-level for Layer 1 v0**  
   - Chosen for maximal transparency and fewer moving parts.
   - Future milestone: swap tokenizer to BPE (nanoGPT → nanochat-style tokenizer).

2. **Simple, nanoGPT-like architecture**  
   - 2-layer Transformer, pre-LN blocks, residual + MLP.
   - Easy to extend (more layers/heads) once GPU training is stable.

3. **Trainer as a reusable abstraction**  
   - `train_lm(...)` in `vibejam/train.py` separates:
     - dataset loading
     - model config
     - training loop (+ logging + checkpointing)
   - Prepares for future multi-architecture Layer 2 (`BaseLM` + `build_model()`).

4. **CLI-first UX**  
   - `train_lm.py`, `prepare_data.py`, `sample_lm.py`, `rewrite_demo.py` provide simple, composable entrypoints.
   - Keeps notebooks focused on experiments, not core plumbing.

---

## Next (Layer 1.5 and Layer 2 Start)

**Short-Term (Layer 1.5 — Better Style Modeling)**  
- [ ] Retrain vibejam on GPU (Colab):
      - Increase `block_size` (64–128)
      - Increase `n_embd`, `n_layer` for more capacity
      - Train for more steps (e.g., 10k–30k iters)
- [ ] Introduce BPE tokenizer (nanoGPT → nanochat-style upgrade) to reduce sequence length and improve semantics
- [ ] Build a `(draft, rewrite)` fine-tuning dataset:
      - Use real edits (LLM draft → Michelle edit)
      - Or synthetic pairs (noise/corruption → original)
- [ ] Add `scripts/train_rewrite_finetune.py` to fine-tune vibejam on the rewrite format

**Medium-Term (Layer 2 — Model Interface & Big LLM Integration)**  
- [ ] Introduce `BaseLM` interface + `build_model(arch="transformer")` factory
- [ ] Wrap vibejam into a `StyleEngine` class:
      - `generate(instruction)` calls a big LLM to produce a draft
      - `rewrite(...)` passes the draft to vibejam for style adaptation
- [ ] Add a simple notebook / script showing:
      - GPT-4/Claude draft → vibejam rewrite → side-by-side comparison

**Long-Term (Layer 3 — Labs & Post-Transformer Experiments)**  
- [ ] Create `labs/` area for:
      - Mamba, gated attention, xLSTM experiments
      - Nested learning / continual learning setups on personal style data
- [ ] Use the existing trainer/data pipeline as the common evaluation harness
      (same corpus, same loss metrics, different architectures).

---
## Week 2: Layer 1.5 — System Correctness & BPE Migration (Days 7–10)
---

### Day 7 (Dec 16) — System Correctness & Failure Debugging ✅  
**Phase 3 → Phase 4 — Making the Pipeline Correct**

- [x] Diagnosed and fixed multiple *structural* failures unrelated to model quality:
  - Tokenizer/vocab mismatch between training and inference
  - Prompt/template leakage between rewrite training data and inference prompts
  - Block size mismatch causing positional embedding incompatibility
  - Invalid JSON inputs due to smart quotes masquerading as valid JSON
  - Config wiring issues where assumptions leaked across layers
- [x] Enforced a single canonical rewrite format:
  - `Draft:` → `Rewrite:` → `<|end|>`
  - Ensured this format was used consistently in:
    - rewrite-pair generation
    - fine-tuning data
    - inference prompt
    - extraction logic
- [x] Refactored rewrite pipeline so that:
  - stop tokens are respected
  - only the intended rewrite span is returned
- [x] Verified that remaining rewrite quality issues were due to char-level modeling, not bugs  

**Result:**  
A fully correct end-to-end system where failures were explainable and reproducible.  
No remaining “mystery bugs” in the rewrite path.

**Key Learnings:**
- Most early ML failures are *systems bugs*, not learning bugs.
- Prompt format is training data; mismatch guarantees garbage output.
- Token IDs are part of the model contract, not a preprocessing detail.

---

### Day 8 (Dec 16) — Tokenization as a First-Class Artifact (BPE Prep) ✅  
**Phase 4 — Tokenizer Abstraction**

- [x] Cleanly separated responsibilities:
  - Tokenizer → Dataset → Model → Training loop
- [x] Introduced `TokenDataset`, making datasets tokenizer-agnostic
- [x] Implemented a byte-level BPE tokenizer from scratch:
  - Learned merges from corpus
  - Guaranteed coverage via byte fallback
  - Persisted tokenizer artifact (`tokenizer_bpe.json`)
- [x] Updated data pipeline to support:
  - Char-level tokenization (legacy)
  - BPE tokenization (Layer 1.5)
- [x] Added sanity checks to verify:
  - BPE vocab size
  - encode/decode roundtrips
  - compatibility with existing training/sampling code  

**Result:**  
Tokenization is now explicit, persisted, and swappable without touching model or trainer code.

**Key Learnings:**
- Tokenizers must be treated as versioned artifacts.
- Clean abstractions dramatically reduce downstream fragility.
- BPE migration should only happen after system correctness is established.

---

### Day 9 (Dec 16) — BPE Base LM Training (from Scratch) ✅  
**Phase 4 → Phase 5 — New Base Model**

- [x] Trained a new base language model from scratch using BPE:
  - `vocab_size ≈ 7.5k`
  - `block_size = 128`
  - Reduced model size for CPU feasibility (96-dim, 3 layers, 3 heads)
- [x] Debugged and fixed multiple training-loop bugs:
  - Incorrect checkpoint save scope (`ckpt` uninitialized)
  - Duplicate checkpoint writers
  - Char-specific metadata leaking into BPE runs
- [x] Refactored checkpoint format:
  - Char runs embed `stoi/itos`
  - BPE runs store a pointer to `tokenizer_bpe.json`
- [x] Successfully ran full BPE training (1500 iters on CPU):
  - Train loss decreased from ~9.07 → ~6.10
  - Validation loss tracked reasonably (~7.65)
  - Periodic checkpoints saved cleanly  

**Result:**  
A fully working BPE-based base LM checkpoint (`vibejam_bpe_personal.pt`) produced from scratch.

**Key Learnings:**
- Block size is architectural; mismatches invalidate checkpoints.
- Checkpoint logic must have exactly one authority.
- BPE dramatically reduces “spelling burden,” even in small models.

---

### Day 10 (Dec 16) — BPE Inference, Release Prep & Reflection ✅  
**Phase 5 — Stabilization & Release**

- [x] Updated `rewrite_demo.py` to support:
  - `--tokenizer-type bpe`
  - loading `BPETokenizer` + `TokenDataset` for inference
- [x] Successfully ran rewrite inference using the BPE base model:
  - Outputs still prototype-grade but structurally more word-like than char-level
- [x] Completed Layer 1.5 goals:
  - Correct, reproducible pipeline
  - BPE-based base training
  - Working inference path
- [x] Reflected on the development process:
  - Identified that the majority of effort went into system correctness
  - Confirmed BPE was the right next step *after* fixing fundamentals
- [x] Prepared repo for GitHub release:
  - Updated README to reflect actual architecture and capabilities
  - Documented lessons learned and architectural decisions
  - Positioned BPE rewrite fine-tuning as the next post-release milestone  

**Result:**  
vibejam v0.1 is ready for release as a clean, honest, learning-focused systems ML project.

**Key Learnings:**
- Shipping a correct system beats chasing output quality too early.
- BPE migration is straightforward *only* when abstractions are clean.
- A small, well-understood model is a better foundation than a large, opaque one.

---

## Current Status

- **Layer 1:** Complete  
- **Layer 1.5 (BPE):** Complete (base training + inference)  
- **Layer 2:** Not started (intentionally)  
- **Layer 3:** Not started  

---

## Next (Layer 1.5 and Layer 2 Start)

**Short-Term (Layer 1.5 — Better Style Modeling / Make Rewrite Real)**

- [x] Introduce BPE tokenizer (byte-level) + persist tokenizer artifact
- [x] Make dataset tokenizer-agnostic (`TokenDataset`) and train a BPE base LM checkpoint
- [ ] Improve rewrite inference hygiene (base model):
      - lower temperature / top-k defaults for rewrite
      - add guardrails against newline/whitespace loops
      - add a small fixed “rewrite eval set” (10 drafts) for consistent comparison
- [ ] BPE rewrite fine-tuning (this is the big quality jump):
      - regenerate clean `rewrite_pairs.txt` in canonical format:
            Draft:\n...\n\nRewrite:\n...\n<|end|>\n
      - update `train_rewrite_finetune.py` to use BPETokenizer + TokenDataset
      - fine-tune from `checkpoints/vibejam_bpe_personal.pt`
      - add early stopping / save-best-val checkpoint
- [ ] Optional: selective fine-tuning ladder (only after full FT works):
      - freeze lower blocks / tune head
      - lightweight adapters (LoRA-style, minimal)

**Medium-Term (Layer 2 — Model Interface & Big LLM Integration)**

- [ ] Introduce a minimal `BaseLM` interface:
      - forward(idx, targets) -> logits/loss
      - generate(idx, ...) -> idx
      - save/load
- [ ] Add `build_model(arch, cfg)` factory:
      - transformer baseline (existing GPTModel)
      - labs models registered without touching stable code
- [ ] Wrap vibejam as a `StyleEngine`:
      - big LLM generates content (semantics)
      - vibejam rewrites into personal style (tone/cadence)
- [ ] Add a side-by-side comparison script:
      - draft vs vibejam rewrite vs (optional) human edit

**Long-Term (Layer 3 — Labs)**

- [ ] Create `labs/` experiments with shared harness:
      - RWKV-lite baseline (fast first non-attention plug-in)
      - Mamba-style SSM block (later; higher complexity)
- [ ] Add slice-based evaluation for style tasks:
      - short drafts vs long drafts
      - bullet-heavy vs narrative
      - “semantic fidelity” stress tests

---
