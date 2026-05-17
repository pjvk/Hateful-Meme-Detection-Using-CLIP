"""
Frozen CLIP + BERT with bidirectional cross-attention and rich fusion.

Image attends to text tokens; text [CLS] attends to image. Fused with
product and difference features, then a deep classifier head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

try:
    import clip
except ImportError:
    clip = None  # type: ignore


class CLIPBERTCoAttention(nn.Module):
    """
    Bidirectional cross-attention between CLIP image and BERT text.

    Stronger than single-direction cross-attention for multimodal memes where
    hate depends on image–text interaction (contrast, sarcasm, etc.).
    """

    def __init__(
        self,
        clip_model_name: str = "ViT-B/32",
        bert_model_name: str = "bert-base-uncased",
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.3,
        num_classes: int = 2,
        freeze_bert: bool = False,
    ) -> None:
        super().__init__()
        if clip is None:
            raise ImportError("Install CLIP: pip install git+https://github.com/openai/CLIP.git")

        self.hidden_dim = hidden_dim

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

        self.img2txt_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.txt2img_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_img = nn.LayerNorm(hidden_dim)
        self.norm_txt = nn.LayerNorm(hidden_dim)

        fusion_dim = hidden_dim * 4
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = self.clip_model.encode_image(image).float()
        return feats / feats.norm(dim=-1, keepdim=True)

    def encode_multimodal(
        self,
        image: torch.Tensor,
        bert_input_ids: torch.Tensor,
        bert_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Fused embedding before classifier (for contrastive auxiliary loss)."""
        return self._fuse(image, bert_input_ids, bert_attention_mask)

    def _fuse(
        self,
        image: torch.Tensor,
        bert_input_ids: torch.Tensor,
        bert_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        img_feat = self.encode_image(image)
        img_token = self.image_proj(img_feat).unsqueeze(1)

        bert_out = self.bert(input_ids=bert_input_ids, attention_mask=bert_attention_mask)
        text_tokens = self.text_proj(bert_out.last_hidden_state)
        key_padding_mask = bert_attention_mask == 0

        i2t_out, _ = self.img2txt_attn(
            img_token,
            text_tokens,
            text_tokens,
            key_padding_mask=key_padding_mask,
        )
        img_side = self.norm_img((i2t_out + img_token).squeeze(1))

        cls_token = text_tokens[:, 0, :].unsqueeze(1)
        t2i_out, _ = self.txt2img_attn(cls_token, img_token, img_token)
        text_side = self.norm_txt((t2i_out.squeeze(1) + text_tokens[:, 0, :]))

        interaction = img_side * text_side
        contrast = torch.abs(img_side - text_side)
        return torch.cat([img_side, text_side, interaction, contrast], dim=-1)

    def forward(
        self,
        image: torch.Tensor,
        bert_input_ids: torch.Tensor,
        bert_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        fused = self._fuse(image, bert_input_ids, bert_attention_mask)
        return self.classifier(fused)


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Supervised contrastive loss on L2-normalized embeddings.

    Pulls same-class samples together and pushes different classes apart.
    """
    features = F.normalize(features, dim=1)
    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(features.device)

    logits = torch.matmul(features, features.T) / temperature
    logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=mask.device)
    mask = mask * logits_mask

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

    mask_sum = mask.sum(dim=1)
    mean_log_prob = (mask * log_prob).sum(dim=1) / torch.clamp(mask_sum, min=1.0)
    return -mean_log_prob.mean()
