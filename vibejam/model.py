# vibejam/model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig

# You can copy this helper:
def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

class Head(nn.Module): 
    """One head of causal self-attention."""

    def __init__(self, head_size: int):
        super().__init__()
        self.key    = nn.Linear(cfg_n_embd, head_size, bias=False)
        self.query  = nn.Linear(cfg_n_embd, head_size, bias=False)
        self.value  = nn.Linear(cfg_n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(cfg_block_size, cfg_block_size)))
        self.dropout = nn.Dropout(cfg_dropout)
    
    def forward(self, x): 
        # x: (B, T, C)
        B, T, C = x.shape
        k = self.key(x)         # (B, T, hs)
        q = self.query(x)       # (B, T, hs)

        wei = q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))   # (B, T, T)

        mask = self.tril[:T, :T]
        wei = wei.masked_fill(mask == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)       # (B, T, hs)
        out = wei @ v           # (B, T, hs)
        return out 

# We need these globals for simplicity in this file. 
# In a later refactor we'll avoid globals; for now it keeps typing short.
cfg_n_embd = 64
cfg_block_size = 64
cfg_dropout = 0.1

def set_head_globals(cfg: ModelConfig):
    global cfg_n_embd, cfg_block_size, cfg_dropout
    cfg_n_embd = cfg.n_embd
    cfg_block_size = cfg.block_size
    cfg_dropout = cfg.dropout

class MultiHeadAttention(nn.Module): 
    """Multiple attention heads in parallel"""

    def __init__(self, num_heads: int, head_size: int):
        super().__init__()
        self.heads  = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj   = nn.Linear(num_heads * head_size, cfg_n_embd)
        self.dropout = nn.Dropout(cfg_dropout)
    
    def forward(self, x): 
        out = torch.cat([h(x) for h in self.heads], dim=-1)     # (B, T, num_heads*hs)
        out = self.proj(out)
        out = self.dropout(out)
        return out
    
class FeedForward(nn.Module): 
    """Simple MLP applied at each position"""

    def __init__(self, n_embd:int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(cfg_dropout),
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module): 
    """Transformer block: attention + MLP with pre-LN"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        head_size = cfg.n_embd // cfg.n_head
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.sa = MultiHeadAttention(cfg.n_head, head_size)
        self.ffn = FeedForward(cfg.n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))    # residual 1
        x = x + self.ffn(self.ln2(x))   # residual 2
        return x 
    
class GPTModel(nn.Module):
    """
    Minimal GPT-style LM used in vibejam Layer 1
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        set_head_globals(cfg)
        self.cfg = cfg

        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_embedding   = nn.Embedding(cfg.block_size, cfg.n_embd)

        self.blocks = nn.Sequential(*[
            Block(cfg) for _ in range(cfg.n_layer)
        ])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size)

    def forward(self, idx, targets=None):
        """
        idx: (B, T) int64 token ids
        targets: (B, T) or None
        """
        B, T = idx.shape
        if T > self.cfg.block_size: 
            raise ValueError(f"Sequence length {T} > block_size {self.cfg.block_size}")
        
        tok_emb = self.token_embedding(idx)     # (B, T, C)
        pos = torch.arange(0, T, device=idx.device)
        pos_emb = self.pos_embedding(pos)
        x = tok_emb + pos_emb

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        if targets is None: 
            loss = None
        else: 
            B, T, V = logits.shape
            logits_flat = logits.view(B * T, V)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)
        
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int):
        """
        Autoregressive generation.
        idx: (B, T) initial context
        """
        for _ in range(max_new_tokens): 
            idx_cond = idx[:, -self.cfg.block_size:] # crop to blocksize
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]               # (B, V)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx