"""Train frozen CLIP + MLP classifier on Facebook Hateful Memes."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import clip
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import HatefulMemesDataset, collate_batch
from src.model import CLIPMLPClassifier
from src.utils import ensure_dir, get_device, set_seed, setup_logging


def run_epoch(
    model: CLIPMLPClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool = True,
) -> tuple[float, float]:
    """One pass over the dataloader; returns (avg_loss, accuracy)."""
    if train:
        model.train()
        model.clip_model.eval()  # CLIP stays in eval mode (frozen)
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in tqdm(loader, desc="Train" if train else "Val", leave=False):
            images = batch["image"].to(device)
            text_tokens = batch["text_tokens"].to(device)
            labels = batch["label"].to(device)

            logits = model(images, text_tokens)
            loss = criterion(logits, labels)

            if train and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total if total else 0.0
    accuracy = correct / total if total else 0.0
    return avg_loss, accuracy


def train(
    data_dir: str | Path,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    hidden_dim: int = 512,
    dropout: float = 0.3,
    checkpoint_path: str | Path | None = None,
    seed: int = 42,
) -> Path:
    """Full training loop with validation tracking."""
    set_seed(seed)
    device = get_device()
    data_dir = Path(data_dir)

    model = CLIPMLPClassifier(hidden_dim=hidden_dim, dropout=dropout)
    model.to(device)

    train_ds = HatefulMemesDataset(
        jsonl_path=data_dir / "train.jsonl",
        data_dir=data_dir,
        preprocess=model.preprocess,
        tokenizer=clip.tokenize,
    )
    val_ds = HatefulMemesDataset(
        jsonl_path=data_dir / "dev.jsonl",
        data_dir=data_dir,
        preprocess=model.preprocess,
        tokenizer=clip.tokenize,
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
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr)

    out_path = Path(checkpoint_path) if checkpoint_path else ensure_dir("checkpoints") / "clip_mlp.pt"
    ensure_dir(out_path.parent)

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, None, device, train=False
        )
        logging.info(
            "Epoch %d/%d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f",
            epoch,
            epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    # Save MLP only (~2MB); CLIP weights are re-downloaded at load time
                    "classifier_state_dict": model.classifier.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_accuracy": val_acc,
                    "hidden_dim": hidden_dim,
                    "dropout": dropout,
                },
                out_path,
            )
            logging.info("Saved best checkpoint to %s", out_path)

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CLIP + MLP classifier")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/clip_mlp.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging()
    logging.info("Device: %s", get_device())
    train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        checkpoint_path=args.checkpoint,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
