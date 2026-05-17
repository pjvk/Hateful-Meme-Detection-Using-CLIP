"""Model factory and checkpoint helpers for all architectures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.model import CLIPMLPClassifier
from src.model_bert import CLIPBERTFusion
from src.model_coattention import CLIPBERTCoAttention
from src.model_cross_attention import CLIPBERTCrossAttention

MODEL_CHOICES = ("clip_mlp", "clip_bert", "clip_bert_cross", "clip_bert_coattn")

DEFAULT_CHECKPOINTS = {
    "clip_mlp": "checkpoints/clip_mlp.pt",
    "clip_bert": "checkpoints/clip_bert.pt",
    "clip_bert_cross": "checkpoints/clip_bert_cross.pt",
    "clip_bert_coattn": "checkpoints/clip_bert_coattn.pt",
}

DEFAULT_EPOCHS = {
    "clip_mlp": 10,
    "clip_bert": 10,
    "clip_bert_cross": 12,
    "clip_bert_coattn": 15,
}

DEFAULT_LR = {
    "clip_mlp": 1e-3,
    "clip_bert": 5e-5,
    "clip_bert_cross": 2e-5,
    "clip_bert_coattn": 2e-5,
}

DEFAULT_BATCH_SIZE = {
    "clip_mlp": 32,
    "clip_bert": 16,
    "clip_bert_cross": 16,
    "clip_bert_coattn": 16,
}

CONTRASTIVE_LOSS_WEIGHT = 0.2


def build_model(
    model_name: str,
    clip_model_name: str = "ViT-B/32",
    hidden_dim: int = 512,
    dropout: float = 0.3,
) -> nn.Module:
    """Instantiate one of the supported models."""
    if model_name == "clip_mlp":
        return CLIPMLPClassifier(
            clip_model_name=clip_model_name,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    if model_name == "clip_bert":
        return CLIPBERTFusion(
            clip_model_name=clip_model_name,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    if model_name == "clip_bert_cross":
        return CLIPBERTCrossAttention(
            clip_model_name=clip_model_name,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    if model_name == "clip_bert_coattn":
        return CLIPBERTCoAttention(
            clip_model_name=clip_model_name,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    raise ValueError(f"Unknown model: {model_name}. Choose from {MODEL_CHOICES}")


def uses_bert(model_name: str) -> bool:
    return model_name in ("clip_bert", "clip_bert_cross", "clip_bert_coattn")


def uses_contrastive_loss(model_name: str) -> bool:
    return model_name == "clip_bert_coattn"


def get_trainable_parameters(model: nn.Module, model_name: str) -> list[torch.nn.Parameter]:
    """Return parameters that should receive gradients."""
    if model_name == "clip_mlp":
        return list(model.classifier.parameters())
    return [p for p in model.parameters() if p.requires_grad]


def save_checkpoint(
    path: Path,
    model: nn.Module,
    model_name: str,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_accuracy: float,
    clip_model_name: str,
    hidden_dim: int,
    dropout: float,
    val_auroc: float | None = None,
) -> None:
    """Save trainable weights only (compact checkpoints)."""
    trainable = {k: v for k, v in model.state_dict().items() if k in _trainable_keys(model)}
    payload: dict[str, Any] = {
        "model_name": model_name,
        "trainable_state_dict": trainable,
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_accuracy": val_accuracy,
        "clip_model_name": clip_model_name,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
    }
    if val_auroc is not None:
        payload["val_auroc"] = val_auroc
    torch.save(payload, path)


def _trainable_keys(model: nn.Module) -> set[str]:
    return {k for k, p in model.named_parameters() if p.requires_grad}


def load_checkpoint_into_model(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Load trainable weights; supports legacy classifier-only checkpoints."""
    if device is None:
        from src.utils import get_device

        device = get_device()

    checkpoint = torch.load(path, map_location=device)

    if "trainable_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["trainable_state_dict"], strict=False)
    elif "classifier_state_dict" in checkpoint:
        if hasattr(model, "classifier"):
            model.classifier.load_state_dict(checkpoint["classifier_state_dict"])
    else:
        raise KeyError("Checkpoint missing trainable_state_dict or classifier_state_dict")

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def load_model_for_eval(
    model_name: str,
    checkpoint_path: str | Path,
    clip_model_name: str | None = None,
    hidden_dim: int = 512,
    dropout: float = 0.3,
    device: torch.device | None = None,
) -> nn.Module:
    """Build model and load weights for evaluation or inference."""
    from src.utils import get_device

    if device is None:
        device = get_device()

    path = Path(checkpoint_path)
    if path.exists():
        ckpt = torch.load(path, map_location="cpu")
        clip_model_name = clip_model_name or ckpt.get("clip_model_name", "ViT-B/32")
        hidden_dim = ckpt.get("hidden_dim", hidden_dim)
        dropout = ckpt.get("dropout", dropout)

    model = build_model(model_name, clip_model_name or "ViT-B/32", hidden_dim, dropout)
    if path.exists():
        load_checkpoint_into_model(path, model, device=device)
    model.to(device)
    model.eval()
    return model
