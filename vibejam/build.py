# vibejam/build.py
from __future__ import annotations

import importlib
from vibejam.config import ModelConfig
from vibejam.lm_interface import BaseLM
from vibejam.model import GPTModel


def build_model(arch: str, cfg: ModelConfig) -> BaseLM:
    """
    Factory to build a language model by architecture name.

    Registered: 
      - "gpt" (baseline)
      - "rwkv_lite" (layer 3 lab)
      - "moe_ffn" (soon)
    """
    arch = arch.lower().strip()

    if arch == "gpt":
        return GPTModel(cfg)
    
    if arch == "rwkv_lite":
        from vibejam.labs.rwkv_lite import RWKVLiteLM
        return RWKVLiteLM(cfg)

    if arch == "moe_ffn":
        from vibejam.labs.moe_ffn import MoEFFNLM
        return MoEFFNLM(cfg)

    raise ValueError(f"Unknown arch='{arch}'. Supported: ['gpt', 'rwkv_lite', 'moe_ffn']")