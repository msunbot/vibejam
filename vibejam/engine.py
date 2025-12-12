# vibejam/engine.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import torch

from vibejam.config import ModelConfig, DataConfig
from vibejam.train import load_dataset_from_file
from vibejam.model import GPTModel
from vibejam.sample import generate_text
from vibejam.rewrite import rewrite_text


@dataclass
class EngineConfig:
    # Sampling defaults tuned for rewrite
    temperature: float = 0.7
    top_k: int | None = 50
    max_new_tokens: int = 250


class VibejamEngine:
    """
    Minimal "style engine" wrapper:
    - loads a vibejam checkpoint
    - rebuilds dataset/vocab from a corpus file
    - provides rewrite() and generate() convenience APIs
    """

    def __init__(
        self,
        ckpt_path: str,
        corpus_path: str,
        block_size: int = 128,
        device: str | None = None,
        engine_cfg: EngineConfig | None = None,
    ):
        self.ckpt_path = ckpt_path
        self.corpus_path = corpus_path
        self.engine_cfg = engine_cfg or EngineConfig()

        # device selection
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # load checkpoint (weights + configs)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.model_cfg = ModelConfig(**ckpt["model_cfg"])

        # rebuild dataset vocab from corpus
        # IMPORTANT: same corpus used in training (or at least same character set)
        self.data_cfg = DataConfig(block_size=block_size, train_frac=0.95)
        self.dataset = load_dataset_from_file(corpus_path, self.data_cfg)

        # build model + load weights
        self.model = GPTModel(self.model_cfg).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def rewrite(
        self,
        draft: str,
        temperature: float | None = None,
        top_k: int | None = None,
        max_new_tokens: int | None = None,
    ) -> str:
        return rewrite_text(
            model=self.model,
            dataset=self.dataset,
            draft=draft,
            temperature=temperature if temperature is not None else self.engine_cfg.temperature,
            top_k=top_k if top_k is not None else self.engine_cfg.top_k,
            max_new_tokens=max_new_tokens if max_new_tokens is not None else self.engine_cfg.max_new_tokens,
        )

    def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        top_k: int | None = None,
        max_new_tokens: int | None = None,
    ) -> str:
        return generate_text(
            model=self.model,
            dataset=self.dataset,
            prompt=prompt,
            temperature=temperature if temperature is not None else self.engine_cfg.temperature,
            top_k=top_k if top_k is not None else self.engine_cfg.top_k,
            max_new_tokens=max_new_tokens if max_new_tokens is not None else self.engine_cfg.max_new_tokens,
        )