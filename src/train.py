"""Train multimodal hateful meme classifiers (CLIP+MLP, CLIP+BERT, cross-attention)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import clip
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import HatefulMemesDataset, collate_batch
from src.model_coattention import supervised_contrastive_loss
from src.models_registry import (
    CONTRASTIVE_LOSS_WEIGHT,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHECKPOINTS,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    MODEL_CHOICES,
    build_model,
    get_trainable_parameters,
    save_checkpoint,
    uses_bert,
    uses_contrastive_loss,
)
from src.utils import ensure_dir, get_device, set_seed, setup_logging


def forward_batch(model: nn.Module, batch: dict, model_name: str) -> torch.Tensor:
    """Run model forward pass for the selected architecture."""
    if model_name == "clip_mlp":
        return model(batch["image"], batch["text_tokens"])
    return model(batch["image"], batch["bert_input_ids"], batch["bert_attention_mask"])


def compute_batch_loss(
    model: nn.Module,
    model_name: str,
    batch: dict,
    labels: torch.Tensor,
    criterion: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Classification loss; co-attention model adds supervised contrastive term."""
    if uses_contrastive_loss(model_name):
        fused = model.encode_multimodal(
            batch["image"],
            batch["bert_input_ids"],
            batch["bert_attention_mask"],
        )
        logits = model.classifier(fused)
        ce = criterion(logits, labels)
        contrastive = supervised_contrastive_loss(fused, labels)
        return ce + CONTRASTIVE_LOSS_WEIGHT * contrastive, logits

    logits = forward_batch(model, batch, model_name)
    return criterion(logits, labels), logits


def run_epoch(
    model: nn.Module,
    model_name: str,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool = True,
    collect_auroc: bool = False,
) -> tuple[float, float, float | None]:
    if train:
        model.train()
        if hasattr(model, "clip_model"):
            model.clip_model.eval()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    all_labels: list[int] = []
    all_probs: list[float] = []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in tqdm(loader, desc="Train" if train else "Val", leave=False):
            images = batch["image"].to(device)
            text_tokens = batch["text_tokens"].to(device)
            labels = batch["label"].to(device)

            model_batch = {
                "image": images,
                "text_tokens": text_tokens,
            }
            if uses_bert(model_name):
                model_batch["bert_input_ids"] = batch["bert_input_ids"].to(device)
                model_batch["bert_attention_mask"] = batch["bert_attention_mask"].to(device)

            loss, logits = compute_batch_loss(model, model_name, model_batch, labels, criterion)

            if train and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if collect_auroc:
                prob_hateful = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
                all_labels.extend(labels.cpu().tolist())
                all_probs.extend(prob_hateful.tolist())

    avg_loss = total_loss / total if total else 0.0
    accuracy = correct / total if total else 0.0

    val_auroc = None
    if collect_auroc and all_labels and len(np.unique(all_labels)) > 1:
        val_auroc = float(roc_auc_score(np.array(all_labels), np.array(all_probs)))

    return avg_loss, accuracy, val_auroc


def train(
    model_name: str,
    data_dir: str | Path,
    epochs: int,
    batch_size: int,
    lr: float,
    clip_model_name: str = "ViT-B/32",
    hidden_dim: int = 512,
    dropout: float = 0.3,
    checkpoint_path: str | Path | None = None,
    seed: int = 42,
) -> Path:
    set_seed(seed)
    device = get_device()
    data_dir = Path(data_dir)

    model = build_model(model_name, clip_model_name, hidden_dim, dropout)
    model.to(device)

    use_bert = uses_bert(model_name)
    train_ds = HatefulMemesDataset(
        jsonl_path=data_dir / "train.jsonl",
        data_dir=data_dir,
        preprocess=model.preprocess,
        clip_tokenizer=clip.tokenize,
        clip_model_name=clip_model_name,
        use_bert=use_bert,
    )
    val_ds = HatefulMemesDataset(
        jsonl_path=data_dir / "dev.jsonl",
        data_dir=data_dir,
        preprocess=model.preprocess,
        clip_tokenizer=clip.tokenize,
        clip_model_name=clip_model_name,
        use_bert=use_bert,
        bert_tokenizer=train_ds.bert_tokenizer if use_bert else None,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(get_trainable_parameters(model, model_name), lr=lr)

    out_path = Path(checkpoint_path or DEFAULT_CHECKPOINTS[model_name])
    ensure_dir(out_path.parent)

    best_val_acc = 0.0
    best_val_auroc = 0.0
    select_by_auroc = model_name == "clip_bert_coattn"

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, _ = run_epoch(
            model, model_name, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_acc, val_auroc = run_epoch(
            model,
            model_name,
            val_loader,
            criterion,
            None,
            device,
            train=False,
            collect_auroc=select_by_auroc,
        )
        auroc_str = f" val_auroc={val_auroc:.4f}" if val_auroc is not None else ""
        logging.info(
            "Epoch %d/%d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f%s",
            epoch,
            epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            auroc_str,
        )

        improved = (
            val_auroc is not None and val_auroc > best_val_auroc
            if select_by_auroc
            else val_acc >= best_val_acc
        )
        if improved:
            if select_by_auroc and val_auroc is not None:
                best_val_auroc = val_auroc
            else:
                best_val_acc = val_acc
            save_checkpoint(
                out_path,
                model,
                model_name,
                optimizer,
                epoch,
                val_acc,
                clip_model_name,
                hidden_dim,
                dropout,
                val_auroc=val_auroc,
            )
            metric = val_auroc if select_by_auroc else val_acc
            logging.info("Saved checkpoint → %s (metric=%.4f)", out_path, metric)

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train hateful meme detection models")
    parser.add_argument("--model", type=str, default="clip_mlp", choices=MODEL_CHOICES)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--clip-model", type=str, default="ViT-B/32")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model_name = args.model
    epochs = args.epochs or DEFAULT_EPOCHS[model_name]
    batch_size = args.batch_size or DEFAULT_BATCH_SIZE[model_name]
    lr = args.lr or DEFAULT_LR[model_name]
    checkpoint = args.checkpoint or DEFAULT_CHECKPOINTS[model_name]

    setup_logging()
    logging.info("Device     : %s", get_device())
    logging.info("Model      : %s", model_name)
    logging.info("CLIP       : %s", args.clip_model)
    logging.info("Epochs/LR  : %d / %g", epochs, lr)

    train(
        model_name=model_name,
        data_dir=args.data_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        clip_model_name=args.clip_model,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        checkpoint_path=checkpoint,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
