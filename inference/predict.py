from __future__ import annotations

from pathlib import Path

import torch

from models.checkpoint import class_names_from_checkpoint, load_model_from_checkpoint
from preprocessing import preprocess_image
from training.device import select_device


def predict_image(
    image_path: str | Path,
    checkpoint_path: str | Path = Path("models/best_model.pth"),
    *,
    device: str | None = None,
) -> dict:
    torch_device = select_device(device)
    model, payload = load_model_from_checkpoint(checkpoint_path, map_location=torch_device)
    model.to(torch_device)
    model.eval()

    preprocess = payload["preprocess"]
    image_size = int(preprocess["image_size"])
    color_mode = str(preprocess["color_mode"])
    in_channels = 1 if color_mode == "grayscale" else 3
    if model.in_channels != in_channels:
        raise ValueError(
            f"Checkpoint in_channels={model.in_channels} does not match color_mode={color_mode}"
        )

    class_names = class_names_from_checkpoint(payload)
    # Same resize/normalize as evaluation. augment=False is required for inference.
    tensor = preprocess_image(
        image_path,
        image_size=image_size,
        color_mode=color_mode,
        augment=False,
    )
    batch = tensor.unsqueeze(0).to(torch_device)

    with torch.no_grad():
        probabilities = model.probabilities(batch)[0].cpu()

    values = probabilities.tolist()
    if abs(sum(values) - 1.0) > 1e-3:
        raise ValueError("Class probabilities do not sum to 1")
    if len(values) != len(class_names):
        raise ValueError("Probability vector length does not match class map")

    class_index = int(probabilities.argmax().item())
    class_name = class_names[class_index]
    confidence = float(values[class_index])
    return {
        "image": str(Path(image_path)),
        "checkpoint": str(Path(checkpoint_path)),
        "class_name": class_name,
        "class_index": class_index,
        "confidence": confidence,
        "probabilities": {name: float(value) for name, value in zip(class_names, values)},
    }
