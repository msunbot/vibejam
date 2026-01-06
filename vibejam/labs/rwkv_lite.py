# labs/rwkv_lite.py
"""
RWKV-lite: a minimal recurrent language model (no attention) designed for learning.

Design goals:
- Minimal moving parts
- Explicit "state over time" in code you can read
- Conforms to vibejam BaseLM interface (forward/get_block_size/configure_optimizers)
- Not a faithful reproduction of RWKV-7; intentionally simplified.

Day 4-6 goals:
- Make recurrence "real": a normalized decaying key-value memory (a/b form).
- Keep code readable (explicit scan over time).
- Expose state so you can reason about it.

This is not RWKV-7. It's a didactic RWKV-ish skeleton.

Core idea:
- TimeMix: mixes current token with a recurrent summary of the past.
- ChannelMix: per-token MLP (like Transformer FFN).
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from vibejam.config import ModelConfig


class TimeMixKV(nn.Module):
    """
    RWKV-ish time mixing with a decaying key-value memory.

    For each time step t, we compute:
      k_t, v_t, r_t ("receptance"/gate), w_t (decay), u_t (bonus)

    We keep two recurrent states per batch:
      a_t: (B, C) numerator accumulator
      b_t: (B, C) denominator accumulator

    Update:
      a <- exp(-w) * a + exp(k) * v
      b <- exp(-w) * b + exp(k)

    Readout:
      mem = a / (b + eps)
      y_t = r * mem

    Intuition:
    - exp(k) acts like an importance weight for the current token content
    - exp(-w) acts like a learned time decay (forgetting)
    - a/b normalization keeps magnitudes stable (critical for training)
    - r gate decides how much memory to expose to residual stream
    """

    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.n_embd = n_embd
        self.ln = nn.LayerNorm(n_embd)

        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=True)

        # decay and bonus are learned per-channel parameters (simpler than per-token)
        # We keep them as parameters instead of projecting from x for simplicity/stability.

        # Initialize decay to a *reasonable memory* so the model actually uses recurrence early.
        # We parameterize:
        #   decay_per_step d = exp(-softplus(w))  in (0, 1]
        # Half-life h satisfies: d^h = 0.5  =>  d = exp(log(0.5)/h)
        #
        # Choose an initial half-life (tokens). 16 is a good starting point for block_size=128.
        init_half_life = 16.0
        init_decay = math.exp(math.log(0.5) / init_half_life)  # d in (0,1)

        # We want softplus(w) ~= -log(d)
        target = -math.log(init_decay)

        # softplus^{-1}(y) = log(exp(y) - 1)
        w0 = math.log(math.exp(target) - 1.0)

        self.time_decay = nn.Parameter(torch.full((n_embd,), float(w0)))
        self.time_bonus = nn.Parameter(torch.zeros(n_embd))   # u

        self.out = nn.Linear(n_embd, n_embd, bias=False)
        self.drop = nn.Dropout(dropout)

    @torch.no_grad()
    def debug_memory_half_life(self) -> float:
        """
        Rough diagnostic: how many steps until the decay halves the memory?
        If decay per step is d in (0,1), half-life h satisfies d^h = 0.5 => h = log(0.5)/log(d).

        We compute an average across channels.
        """
        w = F.softplus(self.time_decay)                 # >= 0
        d = torch.exp(-w).clamp(1e-6, 0.999999)        # (C,)
        h = torch.log(torch.tensor(0.5, device=d.device)) / torch.log(d)
        return float(h.mean().item())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C)
        returns: (B, T, C)
        """
        B, T, C = x.shape
        x = self.ln(x)

        # recurrent states (B, C)
        a = torch.zeros((B, C), device=x.device, dtype=x.dtype)
        b = torch.zeros((B, C), device=x.device, dtype=x.dtype)

        # Precompute per-channel decay in a stable range:
        # We want exp(-w) in (0,1]. Use softplus to ensure w >= 0.
        w = F.softplus(self.time_decay)  # (C,)
        decay = torch.exp(-w).unsqueeze(0)  # (1, C)

        # bonus u (can be pos/neg). We'll use it inside exp weighting.
        u = self.time_bonus.unsqueeze(0)  # (1, C)

        eps = 1e-6
        outs = []

        for t in range(T):
            xt = x[:, t, :]  # (B, C)

            k = self.key(xt)        # (B, C)
            v = self.value(xt)      # (B, C)
            r = torch.sigmoid(self.receptance(xt))  # (B, C)

            # exp(k + u) is always positive; acts like an "attention weight" without attention
            ek = torch.exp(torch.clamp(k + u, min=-10.0, max=10.0))  # clamp prevents overflow

            # decay old memory, add new content
            a = decay * a + ek * v
            b = decay * b + ek

            mem = a / (b + eps)
            yt = r * mem
            outs.append(yt)

        y = torch.stack(outs, dim=1)  # (B, T, C)
        y = self.out(y)
        y = self.drop(y)
        return y


class ChannelMix(nn.Module):
    """
    Per-token MLP like Transformer FFN.
    """
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.ln = nn.LayerNorm(n_embd)
        self.fc1 = nn.Linear(n_embd, 4 * n_embd)
        self.fc2 = nn.Linear(4 * n_embd, n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class RWKVLiteBlock(nn.Module):
    """
    Block = TimeMixKV + ChannelMix, each with residual.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.time_mix = TimeMixKV(cfg.n_embd, cfg.dropout)
        self.chan_mix = ChannelMix(cfg.n_embd, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.time_mix(x)
        x = x + self.chan_mix(x)
        return x


class RWKVLiteLM(nn.Module):
    """
    RWKV-lite language model (no attention).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_embedding = nn.Embedding(cfg.block_size, cfg.n_embd)

        self.blocks = nn.Sequential(*[RWKVLiteBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size)

    # ---- BaseLM interface ----
    def get_block_size(self) -> int:
        return self.cfg.block_size

    def configure_optimizers(self, train_cfg):
        lr = float(getattr(train_cfg, "learning_rate", 3e-4))
        return torch.optim.AdamW(self.parameters(), lr=lr)
    # --------------------------

    def forward(self, idx, targets=None):
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"Sequence length {T} > block_size {self.cfg.block_size}")

        tok_emb = self.token_embedding(idx)  # (B, T, C)
        pos = torch.arange(0, T, device=idx.device)
        pos_emb = self.pos_embedding(pos)    # (T, C)
        x = tok_emb + pos_emb

        # Debug print once: average half-life (in tokens) of layer0 time decay
        if not hasattr(self, "_printed_half_life"):
            hl = self.blocks[0].time_mix.debug_memory_half_life()
            print(f"[rwkv_lite] avg decay half-life (layer0): ~{hl:.1f} tokens")
            self._printed_half_life = True

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)                # (B, T, V)

        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))

            # ---- Day5: optional decay regularizer (prevents memory collapsing to ~1 token) ----
            # We penalize "too short" half-life in layer0 time_mix.
            # This is a *lab knob*, not a claim about optimal RWKV training.
            min_half_life = 6.0
            reg_weight = 1e-4  # small; won't dominate CE loss

            tm0 = self.blocks[0].time_mix
            # compute mean half-life in a differentiable-ish way
            # (we approximate; good enough for nudging)
            w = F.softplus(tm0.time_decay)                     # (C,)
            d = torch.exp(-w).clamp(1e-6, 0.999999)           # (C,)
            hl = torch.log(torch.tensor(0.5, device=d.device)) / torch.log(d)  # (C,)
            hl_mean = hl.mean()

            shortfall = F.relu(torch.tensor(min_half_life, device=hl_mean.device) - hl_mean)
            loss = loss + reg_weight * shortfall
            # -------------------------------------------------------------------------------

        return logits, loss