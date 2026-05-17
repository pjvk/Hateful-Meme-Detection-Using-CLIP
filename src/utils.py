"""Shared utilities for training, evaluation, and inference."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import torch


def get_project_root() -> Path:
    """Return project root (parent of src/)."""
    return Path(__file__).resolve().parent.parent


def get_device() -> torch.device:
    """Use CUDA when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure basic logging for scripts."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def verify_data_dir(data_dir: str | Path, split: str = "dev") -> Path:
    """
    Ensure Facebook Hateful Memes layout exists before training/eval.

    Raises FileNotFoundError with setup instructions if files are missing.
    """
    data_dir = Path(data_dir).resolve()
    jsonl = data_dir / f"{split}.jsonl"
    img_dir = data_dir / "img"

    missing: list[str] = []
    if not data_dir.is_dir():
        missing.append(f"directory not found: {data_dir}")
    if not jsonl.is_file():
        missing.append(f"file not found: {jsonl}")
    if not img_dir.is_dir():
        missing.append(f"directory not found: {img_dir}")

    if missing:
        raise FileNotFoundError(
            "Dataset not found. Expected layout:\n"
            "  data/\n"
            "  ├── img/\n"
            "  ├── train.jsonl\n"
            "  ├── dev.jsonl\n"
            "  └── test.jsonl\n\n"
            "Colab fixes:\n"
            "  1. Kaggle: python -m src.kaggle_data (see notebooks/training.ipynb)\n"
            "  2. Google Drive: !ln -sf /content/drive/MyDrive/<PATH>/data data\n"
            "  3. Run: !python -m src.verify_data --data-dir data\n\n"
            "Missing:\n  - " + "\n  - ".join(missing)
        )
    return data_dir


def resolve_data_path(data_dir: str | Path, relative_img: str) -> Path:
    """
    Resolve image path from JSONL 'img' field.

    JSONL entries use paths like 'img/42953.png' relative to data/.
    """
    data_dir = Path(data_dir)
    rel = Path(relative_img)
    if rel.parts and rel.parts[0] == "img":
        return data_dir / rel
    return data_dir / "img" / rel.name


def default_checkpoint_path() -> Path:
    return get_project_root() / "checkpoints" / "clip_mlp.pt"


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device | None = None,
) -> dict:
    """
    Load checkpoint into model.

    Supports compact checkpoints (classifier only, ~2MB) and legacy full-model saves.
    """
    if map_location is None:
        map_location = get_device()
    checkpoint = torch.load(path, map_location=map_location)

    if "classifier_state_dict" in checkpoint:
        model.classifier.load_state_dict(checkpoint["classifier_state_dict"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        raise KeyError("Checkpoint must contain 'classifier_state_dict' or 'model_state_dict'")

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def read_checkpoint_config(path: str | Path) -> dict:
    """Read training hyperparameters stored in a checkpoint (no full load)."""
    checkpoint = torch.load(path, map_location="cpu")
    return {
        "hidden_dim": checkpoint.get("hidden_dim", 512),
        "dropout": checkpoint.get("dropout", 0.3),
    }
