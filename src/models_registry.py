"""Model factory and checkpoint helpers for all architectures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.model import CLIPMLPClassifier
from src.model_bert import CLIPBERTFusion
from src.model_cross_attention import CLIPBERTCrossAttention

MODEL_CHOICES = ("clip_mlp", "clip_mlp_ft", "clip_bert", "clip_bert_cross")

DEFAULT_CHECKPOINTS = {
    "clip_mlp": "checkpoints/clip_mlp.pt",
    "clip_mlp_ft": "checkpoints/clip_mlp_ft.pt",
    "clip_bert": "checkpoints/clip_bert.pt",
    "clip_bert_cross": "checkpoints/clip_bert_cross.pt",
}

DEFAULT_EPOCHS = {
    "clip_mlp": 10,
    "clip_mlp_ft": 12,
    "clip_bert": 10,
    "clip_bert_cross": 12,
}

DEFAULT_LR = {
    "clip_mlp": 1e-3,
    "clip_mlp_ft": 1e-4,
    "clip_bert": 5e-5,
    "clip_bert_cross": 2e-5,
}

DEFAULT_CLIP_LR = {
    "clip_mlp": 0.0,
    "clip_mlp_ft": 5e-6,
    "clip_bert": 5e-6,
    "clip_bert_cross": 5e-6,
}

DEFAULT_UNFREEZE_CLIP = {
    "clip_mlp": 0,
    "clip_mlp_ft": 2,
    "clip_bert": 0,
    "clip_bert_cross": 0,
}

DEFAULT_BATCH_SIZE = {
    "clip_mlp": 32,
    "clip_mlp_ft": 16,
    "clip_bert": 16,
    "clip_bert_cross": 16,
}

DEFAULT_CLIP_BACKBONE = {
    "clip_mlp": "ViT-B/32",
    "clip_mlp_ft": "ViT-L/14",
    "clip_bert": "ViT-B/32",
    "clip_bert_cross": "ViT-B/32",
}


def build_model(
    model_name: str,
    clip_model_name: str = "ViT-B/32",
    hidden_dim: int = 512,
    dropout: float = 0.3,
    unfreeze_clip_layers: int = 0,
    unfreeze_text_layers: int = 0,
) -> nn.Module:
    """Instantiate one of the supported models."""
    if model_name in ("clip_mlp", "clip_mlp_ft"):
        if model_name == "clip_mlp_ft" and unfreeze_clip_layers == 0:
            unfreeze_clip_layers = DEFAULT_UNFREEZE_CLIP["clip_mlp_ft"]
        return CLIPMLPClassifier(
            clip_model_name=clip_model_name,
            hidden_dim=hidden_dim,
            dropout=dropout,
            unfreeze_clip_layers=unfreeze_clip_layers,
            unfreeze_text_layers=unfreeze_text_layers,
        )
    if model_name == "clip_bert":
        return CLIPBERTFusion(
            clip_model_name=clip_model_name,
            hidden_dim=hidden_dim,
            dropout=dropout,
            unfreeze_clip_layers=unfreeze_clip_layers,
            unfreeze_text_layers=unfreeze_text_layers,
        )
    if model_name == "clip_bert_cross":
        return CLIPBERTCrossAttention(
            clip_model_name=clip_model_name,
            hidden_dim=hidden_dim,
            dropout=dropout,
            unfreeze_clip_layers=unfreeze_clip_layers,
            unfreeze_text_layers=unfreeze_text_layers,
        )
    raise ValueError(f"Unknown model: {model_name}. Choose from {MODEL_CHOICES}")


def uses_bert(model_name: str) -> bool:
    return model_name in ("clip_bert", "clip_bert_cross")


def build_optimizer(
    model: nn.Module,
    lr: float,
    clip_lr: float,
) -> torch.optim.Adam:
    """Adam with separate learning rates for CLIP vs head when CLIP is partially trainable."""
    clip_params: list[nn.Parameter] = []
    head_params: list[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("clip_model."):
            clip_params.append(param)
        else:
            head_params.append(param)

    param_groups: list[dict] = []
    if clip_params and clip_lr > 0:
        param_groups.append({"params": clip_params, "lr": clip_lr})
    if head_params:
        param_groups.append({"params": head_params, "lr": lr})

    if not param_groups:
        raise ValueError("No trainable parameters found.")
    return torch.optim.Adam(param_groups)


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
    unfreeze_clip_layers: int = 0,
    unfreeze_text_layers: int = 0,
) -> None:
    """Save trainable weights only."""
    trainable = {k: v for k, v in model.state_dict().items() if k in _trainable_keys(model)}
    torch.save(
        {
            "model_name": model_name,
            "trainable_state_dict": trainable,
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_accuracy": val_accuracy,
            "clip_model_name": clip_model_name,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "unfreeze_clip_layers": unfreeze_clip_layers,
            "unfreeze_text_layers": unfreeze_text_layers,
        },
        path,
    )


def _trainable_keys(model: nn.Module) -> set[str]:
    return {k for k, p in model.named_parameters() if p.requires_grad}


def load_checkpoint_into_model(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
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
    from src.utils import get_device

    if device is None:
        device = get_device()

    path = Path(checkpoint_path)
    unfreeze_clip_layers = 0
    unfreeze_text_layers = 0

    if path.exists():
        ckpt = torch.load(path, map_location="cpu")
        clip_model_name = clip_model_name or ckpt.get("clip_model_name", "ViT-B/32")
        hidden_dim = ckpt.get("hidden_dim", hidden_dim)
        dropout = ckpt.get("dropout", dropout)
        unfreeze_clip_layers = ckpt.get("unfreeze_clip_layers", 0)
        unfreeze_text_layers = ckpt.get("unfreeze_text_layers", 0)
        if model_name == "clip_mlp_ft" and unfreeze_clip_layers == 0:
            unfreeze_clip_layers = DEFAULT_UNFREEZE_CLIP["clip_mlp_ft"]

    model = build_model(
        model_name if model_name != "clip_mlp_ft" else "clip_mlp",
        clip_model_name or "ViT-B/32",
        hidden_dim,
        dropout,
        unfreeze_clip_layers=unfreeze_clip_layers,
        unfreeze_text_layers=unfreeze_text_layers,
    )
    if path.exists():
        load_checkpoint_into_model(path, model, device=device)
    model.to(device)
    model.eval()
    return model
