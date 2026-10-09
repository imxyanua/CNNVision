from __future__ import annotations

from pathlib import Path

import torch

from models.cnn import SimpleCNN, build_cnn_from_settings


def save_checkpoint(path: str | Path, payload: dict) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return output.resolve()


def load_checkpoint(path: str | Path, map_location: str | torch.device | None = None) -> dict:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    # Full payload includes class maps and preprocess settings, not just weights.
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint is not a dict: {checkpoint_path}")
    return payload


def load_model_from_checkpoint(
    path: str | Path,
    map_location: str | torch.device | None = None,
) -> tuple[SimpleCNN, dict]:
    payload = load_checkpoint(path, map_location=map_location)
    model = build_cnn_from_settings(payload["model"])
    model.load_state_dict(payload["model_state_dict"])
    return model, payload


def class_names_from_checkpoint(payload: dict) -> list[str]:
    class_to_idx = payload["class_to_idx"]
    names: list[str | None] = [None] * len(class_to_idx)
    for name, idx in class_to_idx.items():
        names[int(idx)] = str(name)
    if any(name is None for name in names):
        raise ValueError("Checkpoint class_to_idx is missing an index")
    return [str(name) for name in names]
