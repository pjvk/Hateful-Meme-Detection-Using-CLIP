"""Evaluate trained CLIP + MLP model with classification metrics."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import clip
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import HatefulMemesDataset, collate_batch
from src.model import load_clip_mlp
from src.utils import get_device, setup_logging

LABEL_NAMES = ["Non-Hateful", "Hateful"]


@torch.no_grad()
def collect_predictions(
    model,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run model on loader; return labels, preds, probabilities."""
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []
    all_probs: list[list[float]] = []

    for batch in tqdm(loader, desc="Evaluating"):
        images = batch["image"].to(device)
        text_tokens = batch["text_tokens"].to(device)
        labels = batch["label"]

        logits = model(images, text_tokens)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

        all_labels.extend(labels.tolist())
        all_preds.extend(preds.tolist())
        all_probs.extend(probs.tolist())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )


def print_evaluation_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Print metrics and confusion matrix; return metric dict."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="binary", zero_division=0)
    rec = recall_score(y_true, y_pred, average="binary", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print("\n" + "=" * 50)
    print("EVALUATION REPORT — CLIP + MLP")
    print("=" * 50)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(f"              Pred {LABEL_NAMES[0]}  Pred {LABEL_NAMES[1]}")
    print(f"True {LABEL_NAMES[0]:12} {cm[0, 0]:6d}  {cm[0, 1]:6d}")
    print(f"True {LABEL_NAMES[1]:12} {cm[1, 0]:6d}  {cm[1, 1]:6d}")
    print("\nDetailed classification report:")
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES, digits=4))
    print("=" * 50 + "\n")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def evaluate_split(
    data_dir: str | Path,
    split: str = "dev",
    checkpoint: str | Path = "checkpoints/clip_mlp.pt",
    batch_size: int = 32,
) -> dict[str, float]:
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}. Train first with: python -m src.train"
        )

    device = get_device()
    model = load_clip_mlp(checkpoint_path=checkpoint, device=device)

    data_dir = Path(data_dir)
    dataset = HatefulMemesDataset(
        jsonl_path=data_dir / f"{split}.jsonl",
        data_dir=data_dir,
        preprocess=model.preprocess,
        tokenizer=clip.tokenize,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
    )

    if len(dataset) > 0 and "label" not in dataset.samples[0]:
        raise ValueError(
            f"Split '{split}' has no labels (e.g. official test.jsonl). "
            "Use --split dev or --split train for evaluation."
        )

    y_true, y_pred, _ = collect_predictions(model, loader, device)
    return print_evaluation_report(y_true, y_pred)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CLIP + MLP model")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--split", type=str, default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--checkpoint", type=str, default="checkpoints/clip_mlp.pt")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    setup_logging()
    logging.info("Device: %s", get_device())
    evaluate_split(
        data_dir=args.data_dir,
        split=args.split,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
