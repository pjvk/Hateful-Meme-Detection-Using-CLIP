"""Evaluate trained models with classification metrics."""

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
from src.models_registry import (
    DEFAULT_CHECKPOINTS,
    MODEL_CHOICES,
    load_model_for_eval,
    uses_bert,
)

# clip_mlp_ft checkpoints store architecture as clip_mlp with unfreeze metadata
from src.train import forward_batch
from src.utils import get_device, setup_logging

LABEL_NAMES = ["Non-Hateful", "Hateful"]

MODEL_TITLES = {
    "clip_mlp": "Frozen CLIP + MLP",
    "clip_mlp_ft": "CLIP + MLP (top-layer CLIP fine-tune)",
    "clip_bert": "CLIP + BERT Fusion",
    "clip_bert_cross": "CLIP + BERT + Cross-Attention",
}


@torch.no_grad()
def collect_predictions(
    model,
    model_name: str,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []

    for batch in tqdm(loader, desc="Evaluating"):
        images = batch["image"].to(device)
        text_tokens = batch["text_tokens"].to(device)
        labels = batch["label"]

        model_batch = {"image": images, "text_tokens": text_tokens}
        if uses_bert(model_name):
            model_batch["bert_input_ids"] = batch["bert_input_ids"].to(device)
            model_batch["bert_attention_mask"] = batch["bert_attention_mask"].to(device)

        logits = forward_batch(model, model_batch, model_name)
        preds = logits.argmax(dim=1).cpu().numpy()

        all_labels.extend(labels.tolist())
        all_preds.extend(preds.tolist())

    return np.array(all_labels), np.array(all_preds)


def print_evaluation_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
) -> dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="binary", zero_division=0)
    rec = recall_score(y_true, y_pred, average="binary", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    title = MODEL_TITLES.get(model_name, model_name)

    print("\n" + "=" * 55)
    print(f"EVALUATION — {title}")
    print("=" * 55)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(f"              Pred {LABEL_NAMES[0]}  Pred {LABEL_NAMES[1]}")
    print(f"True {LABEL_NAMES[0]:12} {cm[0, 0]:6d}  {cm[0, 1]:6d}")
    print(f"True {LABEL_NAMES[1]:12} {cm[1, 0]:6d}  {cm[1, 1]:6d}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES, digits=4))
    print("=" * 55 + "\n")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def evaluate_split(
    model_name: str,
    data_dir: str | Path,
    split: str = "dev",
    checkpoint: str | Path | None = None,
    batch_size: int = 32,
    clip_model_name: str | None = None,
) -> dict[str, float]:
    checkpoint = Path(checkpoint or DEFAULT_CHECKPOINTS[model_name])
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}. Train with --model {model_name}")

    device = get_device()
    model = load_model_for_eval(
        model_name,
        checkpoint,
        clip_model_name=clip_model_name,
        device=device,
    )
    if clip_model_name is None:
        ckpt = torch.load(checkpoint, map_location="cpu")
        clip_model_name = ckpt.get("clip_model_name", "ViT-B/32")

    if uses_bert(model_name):
        batch_size = min(batch_size, 16)

    data_dir = Path(data_dir)
    dataset = HatefulMemesDataset(
        jsonl_path=data_dir / f"{split}.jsonl",
        data_dir=data_dir,
        preprocess=model.preprocess,
        clip_tokenizer=clip.tokenize,
        clip_model_name=clip_model_name,
        use_bert=uses_bert(model_name),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
    )

    if len(dataset) > 0 and "label" not in dataset.samples[0]:
        raise ValueError(f"Split '{split}' has no labels. Use dev or train.")

    y_true, y_pred = collect_predictions(model, model_name, loader, device)
    return print_evaluation_report(y_true, y_pred, model_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate hateful meme models")
    parser.add_argument("--model", type=str, default="clip_mlp", choices=MODEL_CHOICES)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--split", type=str, default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--clip-model", type=str, default=None)
    args = parser.parse_args()

    setup_logging()
    evaluate_split(
        model_name=args.model,
        data_dir=args.data_dir,
        split=args.split,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        clip_model_name=args.clip_model,
    )


if __name__ == "__main__":
    main()
