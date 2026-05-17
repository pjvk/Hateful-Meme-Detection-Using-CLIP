"""CLIP + MLP classifier with optional top-layer CLIP fine-tuning."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn

from src.clip_finetune import (
    clip_is_partially_trainable,
    freeze_clip,
    unfreeze_clip_top_layers,
)

try:
    import clip
except ImportError:
    clip = None  # type: ignore


class CLIPMLPClassifier(nn.Module):
    """
    CLIP image + text embeddings -> concat -> MLP classifier.

    If unfreeze_clip_layers > 0, the last N visual transformer blocks
    (and visual ln_post / proj) are trainable with a small learning rate.
    """

    def __init__(
        self,
        clip_model_name: str = "ViT-B/32",
        hidden_dim: int = 512,
        dropout: float = 0.3,
        num_classes: int = 2,
        unfreeze_clip_layers: int = 0,
        unfreeze_text_layers: int = 0,
    ) -> None:
        super().__init__()
        if clip is None:
            raise ImportError(
                "Install openai-clip: pip install git+https://github.com/openai/CLIP.git"
            )

        self.clip_model, self.preprocess = clip.load(clip_model_name, device="cpu")
        self.embed_dim = self.clip_model.visual.output_dim
        self.unfreeze_clip_layers = unfreeze_clip_layers
        self.unfreeze_text_layers = unfreeze_text_layers

        if unfreeze_clip_layers > 0 or unfreeze_text_layers > 0:
            unfreeze_clip_top_layers(
                self.clip_model,
                n_visual=unfreeze_clip_layers,
                n_text=unfreeze_text_layers,
            )
        else:
            freeze_clip(self.clip_model)
            self.clip_model.eval()

        input_dim = self.embed_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        ctx = nullcontext() if clip_is_partially_trainable(self.clip_model) else torch.no_grad()
        with ctx:
            feats = self.clip_model.encode_image(image).float()
        return feats / feats.norm(dim=-1, keepdim=True)

    def encode_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        ctx = nullcontext() if clip_is_partially_trainable(self.clip_model) else torch.no_grad()
        with ctx:
            feats = self.clip_model.encode_text(text_tokens).float()
        return feats / feats.norm(dim=-1, keepdim=True)

    def forward(self, image: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([self.encode_image(image), self.encode_text(text_tokens)], dim=-1)
        return self.classifier(fused)


def load_clip_mlp(
    checkpoint_path: str | Path | None = None,
    device: torch.device | None = None,
    clip_model_name: str = "ViT-B/32",
    hidden_dim: int = 512,
    dropout: float = 0.3,
) -> CLIPMLPClassifier:
    """Build CLIP+MLP and load checkpoint (uses registry for metadata)."""
    from src.models_registry import load_model_for_eval
    from src.utils import default_checkpoint_path, get_device

    path = checkpoint_path or default_checkpoint_path()
    return load_model_for_eval(
        "clip_mlp",
        path,
        clip_model_name=clip_model_name,
        hidden_dim=hidden_dim,
        dropout=dropout,
        device=device or get_device(),
    )
