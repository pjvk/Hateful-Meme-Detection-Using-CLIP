"""Helpers for partial (top-layer) CLIP fine-tuning."""

from __future__ import annotations

import torch.nn as nn


def freeze_clip(clip_model: nn.Module) -> None:
    for param in clip_model.parameters():
        param.requires_grad = False


def unfreeze_clip_top_layers(
    clip_model: nn.Module,
    n_visual: int = 0,
    n_text: int = 0,
) -> None:
    """Freeze all CLIP weights, then unfreeze the last n transformer blocks."""
    freeze_clip(clip_model)

    if n_visual > 0:
        blocks = clip_model.visual.transformer.resblocks
        n_visual = min(n_visual, len(blocks))
        for block in blocks[-n_visual:]:
            for param in block.parameters():
                param.requires_grad = True
        for param in clip_model.visual.ln_post.parameters():
            param.requires_grad = True
        if clip_model.visual.proj is not None:
            clip_model.visual.proj.requires_grad = True

    if n_text > 0:
        blocks = clip_model.transformer.resblocks
        n_text = min(n_text, len(blocks))
        for block in blocks[-n_text:]:
            for param in block.parameters():
                param.requires_grad = True


def clip_is_partially_trainable(clip_model: nn.Module) -> bool:
    return any(p.requires_grad for p in clip_model.parameters())


def set_clip_train_mode(clip_model: nn.Module, training: bool) -> None:
    if clip_is_partially_trainable(clip_model):
        clip_model.train(training)
    else:
        clip_model.eval()
