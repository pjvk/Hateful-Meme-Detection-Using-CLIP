"""Verify dataset layout (run before train/eval in Colab or locally)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.utils import verify_data_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Facebook Hateful Memes data layout")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--split", type=str, default="dev")
    args = parser.parse_args()

    cwd = Path.cwd()
    print(f"Working directory: {cwd}")

    try:
        verify_data_dir(args.data_dir, split=args.split)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    data = Path(args.data_dir)
    n_lines = sum(1 for _ in open(data / f"{args.split}.jsonl", encoding="utf-8"))
    n_imgs = len(list((data / "img").glob("*.png")))
    print(f"OK: {data.resolve()}")
    print(f"  {args.split}.jsonl lines: {n_lines}")
    print(f"  img/*.png count: {n_imgs}")


if __name__ == "__main__":
    main()
