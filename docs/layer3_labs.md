# Layer 3 Labs: RWKV-lite + MoE-FFN  
_Post-Transformer Intuition via Minimal, Comparable Experiments_

Layer 3 in vibejam is a set of **small, readable labs** designed to build intuition about post-Transformer architectures while reusing the **exact same training, sampling, and evaluation harness**.

This is intentionally not a benchmark suite or a reproduction of cutting-edge papers.  
The goal is understanding, not leaderboard performance.

---

## What “Layer 3” Means

Layer 3 experiments follow these constraints:

- Same dataset
- Same training loop
- Same sampling defaults
- Same eval scripts

Only the **model architecture changes**.

This allows direct comparison of architectural ideas rather than confounding differences in tooling or data.

---

## Quickstart

Train three models:

- GPT baseline
- RWKV-lite (recurrence, no attention)
- MoE-FFN (conditional compute, attention unchanged)

Then compare them using the same prompt, seed, and decoding settings.

---

## Lab 1: RWKV-lite  
_Recurrence + State, No Attention_

### What It Is

RWKV-lite is a **recurrent language model** that replaces attention with a **time-mixing mechanism** and an explicit recurrent state.

Instead of attending over all previous tokens, the model carries forward a compressed memory state that is updated at each timestep.

This lab is a simplified, educational version inspired by RWKV-style models.

---

### Why It Matters

RWKV-lite exposes the core tradeoff between:

- **Attention**: direct access to history, strong long-range structure
- **Recurrence**: cheap stateful mixing, but information is compressed

This lab makes that tradeoff visible and measurable.

---

### Key Diagnostic: Memory Half-Life

RWKV-lite logs an estimated **memory half-life (in tokens)** derived from the learned decay parameter.

This gives a concrete answer to:
“How long does information persist in the recurrent state?”

A critical lesson from this lab:

- Poor decay initialization makes the model effectively memoryless (half-life ≈ 1 token)
- Reasonable initialization (e.g. ≈ 16 tokens) allows recurrence to actually function
- During training, half-life often settles into a shorter but usable range (e.g. 8–10 tokens)

---

### Observations

- RWKV-lite trains smoothly under the same harness as GPT
- It learns corpus-local structure and “note-like” patterns
- Outputs are choppier and less globally coherent than GPT at the same scale
- This is expected: recurrence compresses history rather than retrieving it precisely

---

## Lab 2: MoE-FFN  
_Conditional Compute via Expert Routing_

### What It Is

MoE-FFN replaces the standard Transformer feed-forward network with a **Mixture-of-Experts**:

- Multiple small FFN “experts”
- A router assigns each token to one (or more) experts
- Only selected experts are evaluated (conditional compute)

Attention is left unchanged.

---

### Why It Matters

MoE introduces a **scaling motif**:

- Increase model capacity (more parameters)
- Keep per-token compute roughly constant
- Pay the price in routing complexity and training stability

This lab explores those tradeoffs at small scale.

---

### Routing Diagnostics

During training and sampling, MoE logs routing statistics:

- **usage**: approximate token counts per expert
- **top1_frac**: fraction of tokens routed to the most-used expert  
  - ≈ 0.25 for 4 experts = balanced  
  - high values (e.g. > 0.7) = collapse
- **entropy**: router distribution entropy  
  - higher = more spread / uncertainty
- **lb**: load-balancing weight (0.0 means disabled)

These diagnostics make routing behavior observable rather than implicit.

---

### Load Balancing (Aux Loss)

An optional auxiliary loss encourages expert usage to stay closer to uniform.

Observations at this scale:

- Routing does not automatically collapse
- Load balancing often has little effect early
- Its main value is preventing long-term drift toward collapse

This reinforces a key lesson: MoE is not “free performance” — it introduces new failure modes that must be measured.

---

### Configuration Sweeps

The following configurations were explored:

- 4 experts, top-k = 1
- 4 experts, top-k = 2
- 8 experts, top-k = 1 (stress test)

Key takeaways:

- top-k = 1 is simpler, faster, and easier to interpret
- top-k = 2 smooths routing but adds mixing and compute
- Increasing expert count at small scale did not meaningfully improve quality

---

## GPT Baseline (Context)

The GPT baseline remains the most globally coherent model at this scale.

This is expected: attention provides direct access to past tokens, which recurrence and MoE approximate or compress.

The value of these labs is not beating GPT, but understanding **why** it behaves differently.

---

## Cross-Lab Comparison Summary

- GPT: strongest long-range coherence
- RWKV-lite: efficient recurrence, sensitive to decay, good local structure
- MoE-FFN: higher capacity, but requires routing discipline and longer training to pay off

All three run under the same scripts, which makes these differences attributable to architecture rather than tooling.

---

## Key Takeaways

1. Recurrence only works if memory time constants are well-initialized
2. MoE is about capacity scaling, not instant quality gains
3. Routing collapse is real and must be measured, not assumed
4. Stable evaluation (fixed prompts, seeds, decoding) is mandatory
5. Minimal labs teach more than complex reproductions

---

## What’s Next

- Document MoE behavior at larger scale or with BPE tokenization
- Explore additional post-Transformer motifs (e.g. SSMs) using the same harness
- Keep Layer 3 focused on insight, not performance chasing