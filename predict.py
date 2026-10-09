from __future__ import annotations

import argparse
from pathlib import Path

from inference.predict import predict_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict the class of one image")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/best_model.pth"))
    parser.add_argument("--device", default=None, help="cpu, cuda, or omit to auto-select")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = predict_image(args.image, args.checkpoint, device=args.device)
    print(f"Image: {result['image']}")
    print(f"Class: {result['class_name']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print("Probabilities:")
    for name, value in result["probabilities"].items():
        print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    main()
