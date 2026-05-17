"""Frozen CLIP (image) + BERT (text) with concat fusion and MLP classifier."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import BertModel

try:
    import clip
except ImportError:
    clip = None  # type: ignore


class CLIPBERTFusion(nn.Module):
    """
    Multimodal fusion without cross-attention.

    - CLIP ViT: frozen image encoder
    - BERT: trainable text encoder (bert-base-uncased)
    - Project both to hidden_dim, concat, MLP classifier
    """

    def __init__(
        self,
        clip_model_name: str = "ViT-B/32",
        bert_model_name: str = "bert-base-uncased",
        hidden_dim: int = 512,
        dropout: float = 0.3,
        num_classes: int = 2,
        freeze_bert: bool = False,
    ) -> None:
        super().__init__()
        if clip is None:
            raise ImportError("Install CLIP: pip install git+https://github.com/openai/CLIP.git")

        self.clip_model, self.preprocess = clip.load(clip_model_name, device="cpu")
        for p in self.clip_model.parameters():
            p.requires_grad = False
        self.clip_model.eval()

        self.bert = BertModel.from_pretrained(bert_model_name)
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

        clip_dim = self.clip_model.visual.output_dim
        bert_dim = self.bert.config.hidden_size

        self.image_proj = nn.Linear(clip_dim, hidden_dim)
        self.text_proj = nn.Linear(bert_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = self.clip_model.encode_image(image).float()
        return feats / feats.norm(dim=-1, keepdim=True)

    def forward(
        self,
        image: torch.Tensor,
        bert_input_ids: torch.Tensor,
        bert_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        img_feat = self.encode_image(image)
        img_proj = self.image_proj(img_feat)

        bert_out = self.bert(input_ids=bert_input_ids, attention_mask=bert_attention_mask)
        text_feat = bert_out.pooler_output
        text_proj = self.text_proj(text_feat)

        fused = torch.cat([img_proj, text_proj], dim=-1)
        return self.classifier(fused)
