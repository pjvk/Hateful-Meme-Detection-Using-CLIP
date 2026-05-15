"""Inference for a single meme image (+ optional caption)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import clip
import torch
from PIL import Image

from src.model import load_clip_mlp
from src.utils import get_device

LABEL_MAP = {0: "Non-Hateful", 1: "Hateful"}


@torch.no_grad()
def predict_meme(
    image_path: str | Path,
    text: str = "",
    checkpoint_path: str | Path = "checkpoints/clip_mlp.pt",
    device: torch.device | None = None,
) -> dict[str, Any]:
    """
    Predict hateful vs non-hateful for one meme.

    Returns:
        label: "Hateful" or "Non-Hateful"
        confidence: softmax probability of predicted class
        class_id: 0 or 1
    """
    if device is None:
        device = get_device()

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Train first or pass --checkpoint."
        )

    model = load_clip_mlp(checkpoint_path=checkpoint_path, device=device)

    image = Image.open(image_path).convert("RGB")
    image_tensor = model.preprocess(image).unsqueeze(0).to(device)
    text_tokens = clip.tokenize([text or ""], truncate=True).to(device)

    logits = model(image_tensor, text_tokens)
    probs = torch.softmax(logits, dim=1)[0]
    class_id = int(probs.argmax().item())
    confidence = float(probs[class_id].item())

    return {
        "label": LABEL_MAP[class_id],
        "confidence": round(confidence, 4),
        "class_id": class_id,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Predict single meme")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--text", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/clip_mlp.pt")
    args = parser.parse_args()

    result = predict_meme(args.image, args.text, args.checkpoint)
    print(result)


if __name__ == "__main__":
    main()
