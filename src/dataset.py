"""Facebook Hateful Memes dataset loader with CLIP and optional BERT preprocessing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.utils import resolve_data_path, verify_data_dir

try:
    import clip
except ImportError:
    clip = None  # type: ignore


class HatefulMemesDataset(Dataset):
    """
    Load memes from JSONL (train/dev/test).

    Returns (per sample):
        - image: CLIP-preprocessed tensor
        - text_tokens: CLIP token ids (always, for zero-shot / CLIP+MLP)
        - bert_input_ids, bert_attention_mask: when use_bert=True
        - label: 0 non-hateful, 1 hateful (if present in JSONL)
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        data_dir: str | Path,
        preprocess=None,
        clip_tokenizer=None,
        tokenizer=None,  # backward compatibility alias
        clip_model_name: str = "ViT-B/32",
        use_bert: bool = False,
        bert_tokenizer=None,
        max_text_length: int = 128,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.jsonl_path = Path(jsonl_path)
        self.use_bert = use_bert
        self.max_text_length = max_text_length
        self.samples: list[dict[str, Any]] = []

        if not self.jsonl_path.is_file():
            split = self.jsonl_path.stem
            verify_data_dir(self.data_dir, split=split)

        with open(self.jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

        if clip_tokenizer is None and tokenizer is not None:
            clip_tokenizer = tokenizer
        if preprocess is None or clip_tokenizer is None:
            if clip is None:
                raise ImportError(
                    "Install CLIP: pip install git+https://github.com/openai/CLIP.git"
                )
            _clip_model, preprocess = clip.load(clip_model_name, device="cpu")
            del _clip_model
            clip_tokenizer = clip.tokenize

        self.preprocess = preprocess
        self.clip_tokenizer = clip_tokenizer

        if use_bert and bert_tokenizer is None:
            from transformers import BertTokenizer

            bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_tokenizer = bert_tokenizer

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.samples[idx]
        img_path = resolve_data_path(self.data_dir, row["img"])
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.preprocess(image)

        text = row.get("text", "") or ""
        text_tokens = self.clip_tokenizer([text], truncate=True).squeeze(0)

        sample: dict[str, Any] = {
            "image": image_tensor,
            "text_tokens": text_tokens,
            "id": row.get("id", idx),
        }

        if self.use_bert and self.bert_tokenizer is not None:
            encoded = self.bert_tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_text_length,
                return_tensors="pt",
            )
            sample["bert_input_ids"] = encoded["input_ids"].squeeze(0)
            sample["bert_attention_mask"] = encoded["attention_mask"].squeeze(0)

        if "label" in row:
            sample["label"] = int(row["label"])
        return sample


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack batch tensors for DataLoader."""
    out: dict[str, Any] = {
        "image": torch.stack([b["image"] for b in batch]),
        "text_tokens": torch.stack([b["text_tokens"] for b in batch]),
        "id": [b["id"] for b in batch],
    }
    if "bert_input_ids" in batch[0]:
        out["bert_input_ids"] = torch.stack([b["bert_input_ids"] for b in batch])
        out["bert_attention_mask"] = torch.stack([b["bert_attention_mask"] for b in batch])
    if "label" in batch[0]:
        out["label"] = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return out
