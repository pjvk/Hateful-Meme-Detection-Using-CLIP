"""Train multimodal hateful meme classifiers."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import clip
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.clip_finetune import set_clip_train_mode
from src.dataset import HatefulMemesDataset, collate_batch
from src.models_registry import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHECKPOINTS,
    DEFAULT_CLIP_BACKBONE,
    DEFAULT_CLIP_LR,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_UNFREEZE_CLIP,
    MODEL_CHOICES,
    build_model,
    build_optimizer,
    save_checkpoint,
    uses_bert,
)
from src.utils import ensure_dir, get_device, set_seed, setup_logging


def forward_batch(model: nn.Module, batch: dict, model_name: str) -> torch.Tensor:
    base = "clip_mlp" if model_name == "clip_mlp_ft" else model_name
    if base == "clip_mlp":
        return model(batch["image"], batch["text_tokens"])
    return model(batch["image"], batch["bert_input_ids"], batch["bert_attention_mask"])


def run_epoch(
    model: nn.Module,
    model_name: str,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool = True,
) -> tuple[float, float]:
    if train:
        model.train()
        if hasattr(model, "clip_model"):
            set_clip_train_mode(model.clip_model, True)
    else:
        model.eval()
        if hasattr(model, "clip_model"):
            set_clip_train_mode(model.clip_model, False)

    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in tqdm(loader, desc="Train" if train else "Val", leave=False):
            images = batch["image"].to(device)
            text_tokens = batch["text_tokens"].to(device)
            labels = batch["label"].to(device)

            model_batch = {"image": images, "text_tokens": text_tokens}
            if uses_bert(model_name):
                model_batch["bert_input_ids"] = batch["bert_input_ids"].to(device)
                model_batch["bert_attention_mask"] = batch["bert_attention_mask"].to(device)

            logits = forward_batch(model, model_batch, model_name)
            loss = criterion(logits, labels)

            if train and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total if total else 0.0
    return avg_loss, correct / total if total else 0.0


def compute_class_weights(data_dir: Path) -> torch.Tensor:
    """Inverse-frequency weights for binary labels in train.jsonl."""
    counts = [0, 0]
    with open(data_dir / "train.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                counts[int(json.loads(line)["label"])] += 1
    total = sum(counts)
    return torch.tensor(
        [total / (2 * counts[0]), total / (2 * counts[1])],
        dtype=torch.float32,
    )


def train(
    model_name: str,
    data_dir: str | Path,
    epochs: int,
    batch_size: int,
    lr: float,
    clip_lr: float,
    clip_model_name: str = "ViT-B/32",
    hidden_dim: int = 512,
    dropout: float = 0.3,
    unfreeze_clip_layers: int = 0,
    unfreeze_text_layers: int = 0,
    class_weights: bool = False,
    checkpoint_path: str | Path | None = None,
    seed: int = 42,
) -> Path:
    set_seed(seed)
    device = get_device()
    data_dir = Path(data_dir)

    registry_name = model_name
    build_name = "clip_mlp" if model_name == "clip_mlp_ft" else model_name

    model = build_model(
        build_name,
        clip_model_name,
        hidden_dim,
        dropout,
        unfreeze_clip_layers=unfreeze_clip_layers,
        unfreeze_text_layers=unfreeze_text_layers,
    )
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
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_batch
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch
    )

    if class_weights:
        weights = compute_class_weights(data_dir).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        logging.info("Using class weights: %s", weights.tolist())
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = build_optimizer(model, lr=lr, clip_lr=clip_lr)

    out_path = Path(checkpoint_path or DEFAULT_CHECKPOINTS[model_name])
    ensure_dir(out_path.parent)

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(
            model, model_name, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_acc = run_epoch(
            model, model_name, val_loader, criterion, None, device, train=False
        )
        logging.info(
            "Epoch %d/%d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f",
            epoch, epochs, train_loss, train_acc, val_loss, val_acc,
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                out_path,
                model,
                registry_name,
                optimizer,
                epoch,
                val_acc,
                clip_model_name,
                hidden_dim,
                dropout,
                unfreeze_clip_layers,
                unfreeze_text_layers,
            )
            logging.info("Saved checkpoint → %s (val_acc=%.4f)", out_path, val_acc)

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train hateful meme detection models")
    parser.add_argument("--model", type=str, default="clip_mlp", choices=MODEL_CHOICES)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--clip-model", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--clip-lr", type=float, default=None)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--unfreeze-clip-layers", type=int, default=None)
    parser.add_argument("--unfreeze-text-layers", type=int, default=0)
    parser.add_argument("--class-weights", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model_name = args.model
    clip_model = args.clip_model or DEFAULT_CLIP_BACKBONE.get(model_name, "ViT-B/32")
    epochs = args.epochs or DEFAULT_EPOCHS[model_name]
    batch_size = args.batch_size or DEFAULT_BATCH_SIZE[model_name]
    lr = args.lr if args.lr is not None else DEFAULT_LR[model_name]
    clip_lr = args.clip_lr if args.clip_lr is not None else DEFAULT_CLIP_LR[model_name]
    unfreeze = (
        args.unfreeze_clip_layers
        if args.unfreeze_clip_layers is not None
        else DEFAULT_UNFREEZE_CLIP[model_name]
    )
    checkpoint = args.checkpoint or DEFAULT_CHECKPOINTS[model_name]

    setup_logging()
    logging.info("Device              : %s", get_device())
    logging.info("Model               : %s", model_name)
    logging.info("CLIP backbone       : %s", clip_model)
    logging.info("Unfreeze CLIP layers: %d", unfreeze)
    logging.info("LR (head / CLIP)    : %g / %g", lr, clip_lr)

    train(
        model_name=model_name,
        data_dir=args.data_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        clip_lr=clip_lr,
        clip_model_name=clip_model,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        unfreeze_clip_layers=unfreeze,
        unfreeze_text_layers=args.unfreeze_text_layers,
        class_weights=args.class_weights,
        checkpoint_path=checkpoint,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
