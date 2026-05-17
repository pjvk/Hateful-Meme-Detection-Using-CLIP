"""CLIP (image) + BERT (text) with cross-attention fusion."""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn
from transformers import BertModel

from src.clip_finetune import (
    clip_is_partially_trainable,
    freeze_clip,
    unfreeze_clip_top_layers,
)

try:
    import clip
except ImportError:
    clip = None  # type: ignore


class CLIPBERTCrossAttention(nn.Module):
    """CLIP image query attends to BERT tokens, then classifies."""

    def __init__(
        self,
        clip_model_name: str = "ViT-B/32",
        bert_model_name: str = "bert-base-uncased",
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.3,
        num_classes: int = 2,
        freeze_bert: bool = False,
        unfreeze_clip_layers: int = 0,
        unfreeze_text_layers: int = 0,
    ) -> None:
        super().__init__()
        if clip is None:
            raise ImportError("Install CLIP: pip install git+https://github.com/openai/CLIP.git")

        self.clip_model, self.preprocess = clip.load(clip_model_name, device="cpu")
        if unfreeze_clip_layers > 0 or unfreeze_text_layers > 0:
            unfreeze_clip_top_layers(
                self.clip_model, n_visual=unfreeze_clip_layers, n_text=unfreeze_text_layers
            )
        else:
            freeze_clip(self.clip_model)
            self.clip_model.eval()

        self.bert = BertModel.from_pretrained(bert_model_name)
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

        clip_dim = self.clip_model.visual.output_dim
        bert_dim = self.bert.config.hidden_size

        self.image_proj = nn.Linear(clip_dim, hidden_dim)
        self.text_proj = nn.Linear(bert_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        ctx = nullcontext() if clip_is_partially_trainable(self.clip_model) else torch.no_grad()
        with ctx:
            feats = self.clip_model.encode_image(image).float()
        return feats / feats.norm(dim=-1, keepdim=True)

    def forward(
        self,
        image: torch.Tensor,
        bert_input_ids: torch.Tensor,
        bert_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        img_token = self.image_proj(self.encode_image(image)).unsqueeze(1)
        bert_out = self.bert(input_ids=bert_input_ids, attention_mask=bert_attention_mask)
        text_tokens = self.text_proj(bert_out.last_hidden_state)
        key_padding_mask = bert_attention_mask == 0
        attn_out, _ = self.cross_attn(
            img_token, text_tokens, text_tokens, key_padding_mask=key_padding_mask
        )
        fused = self.norm(attn_out + img_token).squeeze(1)
        return self.classifier(fused)
