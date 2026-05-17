"""Frozen CLIP feature extractor + trainable MLP classifier."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

try:
    import clip
except ImportError:
    clip = None  # type: ignore


class CLIPMLPClassifier(nn.Module):
    """
    Multimodal classifier: frozen CLIP embeddings -> concat -> MLP -> 2 classes.

    CLIP parameters are frozen; only the MLP head is trained.
    """

    def __init__(
        self,
        clip_model_name: str = "ViT-B/32",
        hidden_dim: int = 512,
        dropout: float = 0.3,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        if clip is None:
            raise ImportError(
                "Install openai-clip: pip install git+https://github.com/openai/CLIP.git"
            )

        self.clip_model, self.preprocess = clip.load(clip_model_name, device="cpu")
        self.embed_dim = self.clip_model.visual.output_dim

        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clip_model.eval()

        input_dim = self.embed_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.clip_model.encode_image(image).float()

    def encode_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.clip_model.encode_text(text_tokens).float()

    def forward(self, image: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        image_emb = self.encode_image(image)
        text_emb = self.encode_text(text_tokens)
        image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        fused = torch.cat([image_emb, text_emb], dim=-1)
        return self.classifier(fused)


def load_clip_mlp(
    checkpoint_path: str | Path | None = None,
    device: torch.device | None = None,
    clip_model_name: str = "ViT-B/32",
    hidden_dim: int = 512,
    dropout: float = 0.3,
) -> CLIPMLPClassifier:
    """Build model and optionally load trained MLP weights."""
    from src.utils import (
        default_checkpoint_path,
        get_device,
        load_checkpoint,
        read_checkpoint_config,
    )

    if device is None:
        device = get_device()

    path = Path(checkpoint_path) if checkpoint_path else default_checkpoint_path()
    if path.exists():
        cfg = read_checkpoint_config(path)
        hidden_dim = cfg["hidden_dim"]
        dropout = cfg["dropout"]

    model = CLIPMLPClassifier(
        clip_model_name=clip_model_name,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )
    if path.exists():
        from src.models_registry import load_checkpoint_into_model

        load_checkpoint_into_model(path, model, device=device)
    model.to(device)
    model.eval()
    return model
