from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.run import evaluate_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the CNN on the test split")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/best_model.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default=None, help="cpu, cuda, or omit to auto-select")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_model(
        args.dataset,
        args.checkpoint,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
        device=args.device,
    )


if __name__ == "__main__":
    main()
