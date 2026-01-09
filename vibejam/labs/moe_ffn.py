# vibejam/labs/moe_ffn.py
"""
MoE-FFN: Minimal Mixture-of-Experts feed-forward layer.

Design goals:
- Extremely small expert count (4–8)
- Token-level top-k routing
- No distributed tricks, no fancy kernels
- Pure learning lab
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from vibejam.config import ModelConfig


class MoEFFN(nn.Module):
    """
    MoE FFN with routing diagnostics and optional load-balancing aux loss.

    Diagnostics (per forward, cached):
      - last_usage: (E,) token counts per expert (top-k votes normalized to tokens)
      - last_top1_frac: fraction routed to most-used expert (collapse indicator)
      - last_entropy: mean router entropy (higher = more diverse routing)
      - last_aux_loss_tensor: tensor aux loss (or None)
    """

    def __init__(
        self,
        n_embd: int,
        n_experts: int = 4,
        top_k: int = 2,
        dropout: float = 0.1,
        lb_weight: float = 0.0,
    ):
        super().__init__()
        assert top_k <= n_experts

        self.n_experts = n_experts
        self.top_k = top_k
        self.lb_weight = float(lb_weight)

        self.router = nn.Linear(n_embd, n_experts, bias=False)

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd),
                nn.ReLU(),
                nn.Linear(4 * n_embd, n_embd),
            )
            for _ in range(n_experts)
        ])

        self.dropout = nn.Dropout(dropout)

        # Cached diagnostics
        self.last_usage = None
        self.last_top1_frac = None
        self.last_entropy = None
        self.last_aux_loss_tensor = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        x_flat = x.view(B * T, C)            # N = B*T

        router_logits = self.router(x_flat)  # (N, E)
        router_probs = F.softmax(router_logits, dim=-1)

        # entropy diagnostic
        ent = -(router_probs * (router_probs + 1e-9).log()).sum(dim=-1).mean()

        # Top-k selection
        topk_probs, topk_idx = torch.topk(router_probs, self.top_k, dim=-1)  # (N, k)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        # Usage counts across top-k votes
        usage = torch.zeros(self.n_experts, device=x.device, dtype=torch.float32)
        flat_idx = topk_idx.reshape(-1)  # (N*k,)
        usage.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32))
        usage_tokens = usage / float(self.top_k)

        top1_frac = (usage_tokens.max() / (usage_tokens.sum() + 1e-9)).clamp(0.0, 1.0)

        # Cache diagnostics (CPU for printing)
        self.last_usage = usage_tokens.detach().cpu()
        self.last_top1_frac = float(top1_frac.detach().cpu().item())
        self.last_entropy = float(ent.detach().cpu().item())

        # Optional load-balancing aux loss (simple + stable)
        if self.lb_weight > 0:
            p = usage_tokens / (usage_tokens.sum() + 1e-9)  # (E,)
            uniform = torch.full_like(p, 1.0 / self.n_experts)
            aux = ((p - uniform) ** 2).mean()
            self.last_aux_loss_tensor = self.lb_weight * aux
        else:
            self.last_aux_loss_tensor = None

        # Compute expert outputs
        out = torch.zeros_like(x_flat)

        for expert_id, expert in enumerate(self.experts):
            mask = topk_idx == expert_id
            if not mask.any():
                continue

            token_idx, which_k = mask.nonzero(as_tuple=True)
            expert_inp = x_flat[token_idx]
            expert_out = expert(expert_inp)

            weights = topk_probs[token_idx, which_k].unsqueeze(-1)
            out[token_idx] += expert_out * weights

        out = self.dropout(out)
        return out.view(B, T, C)


class MoEBlock(nn.Module):
    """
    Transformer block with Attention + MoE-FFN.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        # IMPORTANT: vibejam.model attention uses globals; must sync them to this cfg.
        from vibejam.model import MultiHeadAttention, set_head_globals
        set_head_globals(cfg)

        head_size = cfg.n_embd // cfg.n_head
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)

        self.attn = MultiHeadAttention(cfg.n_head, head_size)

        # ---- Day 9 sweep toggles ----
        n_experts = 4
        top_k = 1          # try 1, then 2
        lb_weight = 0.0    # try 0.0, then 1e-2
        # -----------------------------

        self.moe_ffn = MoEFFN(
            n_embd=cfg.n_embd,
            n_experts=n_experts,
            top_k=top_k,
            dropout=cfg.dropout,
            lb_weight=lb_weight,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.moe_ffn(self.ln2(x))
        return x


class MoEFFNLM(nn.Module):
    """
    GPT-style LM where FFN is replaced by MoE-FFN.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_embedding = nn.Embedding(cfg.block_size, cfg.n_embd)

        self.blocks = nn.Sequential(*[
            MoEBlock(cfg) for _ in range(cfg.n_layer)
        ])

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
            raise ValueError("Sequence too long")

        tok = self.token_embedding(idx)
        pos = torch.arange(0, T, device=idx.device)
        x = tok + self.pos_embedding(pos)

        x = self.blocks(x)

        # Debug print once: routing stats from the first block
        if not hasattr(self, "_printed_moe_stats"):
            moe = self.blocks[0].moe_ffn
            if moe.last_usage is not None:
                print(
                    f"[moe_ffn] lb={moe.lb_weight} usage={moe.last_usage.numpy().round(1).tolist()} "
                    f"top1_frac={moe.last_top1_frac:.2f} entropy={moe.last_entropy:.2f} "
                    f"aux={'on' if moe.last_aux_loss_tensor is not None else 'off'}"
                )
            self._printed_moe_stats = True

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(B * T, -1),
                targets.view(B * T),
            )

            # Add load-balancing aux loss from each block (if enabled)
            aux_total = 0.0
            for blk in self.blocks:
                aux = blk.moe_ffn.last_aux_loss_tensor
                if aux is not None:
                    aux_total = aux_total + aux
            loss = loss + aux_total

        return logits, loss