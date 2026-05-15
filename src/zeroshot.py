"""Zero-shot CLIP baseline for hateful vs non-hateful meme classification."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import clip
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import HatefulMemesDataset, collate_batch
from src.utils import get_device, setup_logging

# Prompt templates for zero-shot classification
HATEFUL_PROMPTS = [
    "a hateful meme",
    "an offensive hateful meme",
    "a meme that promotes hate",
]
NON_HATEFUL_PROMPTS = [
    "a non-hateful meme",
    "a harmless funny meme",
    "a meme that is not offensive",
]


def build_text_features(model, device: torch.device) -> torch.Tensor:
    """Encode all class prompts and average per class."""
    hateful_tokens = clip.tokenize(HATEFUL_PROMPTS).to(device)
    safe_tokens = clip.tokenize(NON_HATEFUL_PROMPTS).to(device)

    with torch.no_grad():
        hateful_emb = model.encode_text(hateful_tokens)
        safe_emb = model.encode_text(safe_tokens)
        hateful_emb = hateful_emb / hateful_emb.norm(dim=-1, keepdim=True)
        safe_emb = safe_emb / safe_emb.norm(dim=-1, keepdim=True)
        hateful_proto = hateful_emb.mean(dim=0, keepdim=True)
        safe_proto = safe_emb.mean(dim=0, keepdim=True)
        text_features = torch.cat([safe_proto, hateful_proto], dim=0)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features


@torch.no_grad()
def evaluate_zeroshot(
    model,
    dataloader: DataLoader,
    text_features: torch.Tensor,
    device: torch.device,
) -> tuple[float, list[int], list[int]]:
    """Zero-shot: compare image+text joint similarity to class prototypes."""
    model.eval()
    correct = 0
    total = 0
    all_preds: list[int] = []
    all_labels: list[int] = []

    for batch in tqdm(dataloader, desc="Zero-shot CLIP"):
        images = batch["image"].to(device)
        text_tokens = batch["text_tokens"].to(device)
        labels = batch["label"]

        image_features = model.encode_image(images)
        meme_text_features = model.encode_text(text_tokens)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        meme_text_features = meme_text_features / meme_text_features.norm(dim=-1, keepdim=True)
        # Simple multimodal fusion: average normalized embeddings
        meme_features = (image_features + meme_text_features) / 2
        meme_features = meme_features / meme_features.norm(dim=-1, keepdim=True)

        logits = 100.0 * meme_features @ text_features.T
        preds = logits.argmax(dim=1).cpu()

        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    accuracy = correct / total if total else 0.0
    return accuracy, all_preds, all_labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot CLIP baseline")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--split", type=str, default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    setup_logging()
    device = get_device()
    logging.info("Using device: %s", device)

    data_dir = Path(args.data_dir)
    jsonl = data_dir / f"{args.split}.jsonl"

    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    dataset = HatefulMemesDataset(
        jsonl_path=jsonl,
        data_dir=data_dir,
        preprocess=preprocess,
        tokenizer=clip.tokenize,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
    )

    text_features = build_text_features(clip_model, device)
    accuracy, _, _ = evaluate_zeroshot(clip_model, loader, text_features, device)
    logging.info("Zero-shot CLIP accuracy on %s: %.4f", args.split, accuracy)


if __name__ == "__main__":
    main()
