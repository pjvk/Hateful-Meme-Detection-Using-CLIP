"""Facebook Hateful Memes dataset loader with CLIP preprocessing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.utils import resolve_data_path

try:
    import clip
except ImportError:
    clip = None  # type: ignore


class HatefulMemesDataset(Dataset):
    """
    Load memes from JSONL (train/dev/test).

    Each sample returns:
        - image: CLIP-preprocessed image tensor [3, H, W]
        - text_tokens: tokenized caption [context_length]
        - label: 0 (non-hateful) or 1 (hateful)
        - id: meme id (for logging)
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        data_dir: str | Path,
        preprocess=None,
        tokenizer=None,
        clip_model_name: str = "ViT-B/32",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.jsonl_path = Path(jsonl_path)
        self.samples: list[dict[str, Any]] = []

        with open(self.jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

        if preprocess is None or tokenizer is None:
            if clip is None:
                raise ImportError("Install openai-clip: pip install git+https://github.com/openai/CLIP.git")
            _clip_model, preprocess = clip.load(clip_model_name, device="cpu")
            del _clip_model
            tokenizer = clip.tokenize

        self.preprocess = preprocess
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.samples[idx]
        img_path = resolve_data_path(self.data_dir, row["img"])
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.preprocess(image)

        text = row.get("text", "") or ""
        text_tokens = self.tokenizer([text], truncate=True).squeeze(0)

        meme_id = row.get("id", idx)
        sample = {
            "image": image_tensor,
            "text_tokens": text_tokens,
            "id": meme_id,
        }
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
    if "label" in batch[0]:
        out["label"] = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return out
