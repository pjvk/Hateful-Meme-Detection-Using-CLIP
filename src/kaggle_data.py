"""
Download Facebook Hateful Memes from Kaggle and link as ./data

Requires Kaggle API token (https://www.kaggle.com/settings -> API):
  export KAGGLE_API_TOKEN=your_token

Colab:
  import os
  os.environ["KAGGLE_API_TOKEN"] = "your_token"
  !python -m src.kaggle_data
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from src.utils import verify_data_dir

KAGGLE_DATASET = "parthplc/facebook-hateful-meme-dataset"


def find_dataset_root(download_path: Path) -> Path:
    """Locate folder containing img/ and *.jsonl inside a Kaggle download."""
    download_path = Path(download_path)
    for jsonl in download_path.rglob("dev.jsonl"):
        root = jsonl.parent
        if (root / "img").is_dir():
            return root
    for jsonl in download_path.rglob("train.jsonl"):
        root = jsonl.parent
        if (root / "img").is_dir():
            return root
    raise FileNotFoundError(
        f"Could not find img/ + dev.jsonl under {download_path}. "
        "Check the Kaggle dataset layout."
    )


def link_data_dir(source: Path, target: Path = Path("data")) -> Path:
    """Symlink Kaggle dataset folder to ./data for training scripts."""
    source = source.resolve()
    target = Path(target)

    if target.is_symlink():
        target.unlink()
    elif target.is_dir() and not any(target.iterdir()):
        target.rmdir()
    elif target.exists():
        raise FileExistsError(
            f"{target} already exists. Remove it or use another --target name."
        )

    target.symlink_to(source, target_is_directory=True)
    return target.resolve()


def download_and_link(target: str = "data") -> Path:
    if not os.environ.get("KAGGLE_API_TOKEN"):
        print(
            "Warning: KAGGLE_API_TOKEN not set. "
            "Create a token at https://www.kaggle.com/settings -> API",
            file=sys.stderr,
        )

    try:
        import kagglehub
    except ImportError:
        raise ImportError("Install kagglehub: pip install kagglehub") from None

    print(f"Downloading {KAGGLE_DATASET} from Kaggle (first run may take several minutes)...")
    cache_path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    print("Download cache:", cache_path)

    data_root = find_dataset_root(cache_path)
    linked = link_data_dir(data_root, Path(target))
    print("Linked data folder:", linked)
    verify_data_dir(linked, split="dev")
    return linked


def main() -> None:
    parser = argparse.ArgumentParser(description="Download dataset from Kaggle into ./data")
    parser.add_argument("--target", type=str, default="data", help="Symlink path (default: data)")
    args = parser.parse_args()
    download_and_link(args.target)


if __name__ == "__main__":
    main()
