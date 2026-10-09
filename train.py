from __future__ import annotations

import argparse
from pathlib import Path

from preprocessing.transforms import DEFAULT_COLOR_MODE, DEFAULT_IMAGE_SIZE
from training.loop import train_model
from training.split import DEFAULT_SEED


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CNN image classifier")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("models/best_model.pth"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--color-mode", choices=("rgb", "grayscale"), default=DEFAULT_COLOR_MODE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default=None, help="cpu, cuda, or omit to auto-select")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_model(
        args.dataset,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        image_size=args.image_size,
        color_mode=args.color_mode,
        seed=args.seed,
        num_workers=args.workers,
        device=args.device,
    )


if __name__ == "__main__":
    main()
