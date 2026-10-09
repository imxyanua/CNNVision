from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from evaluation.metrics import compute_classification_metrics
from models.checkpoint import load_model_from_checkpoint
from training.dataset import build_dataloaders
from training.device import select_device
from visualization.confusion import save_confusion_matrix


def class_names_from_checkpoint(payload: dict) -> list[str]:
    class_to_idx = payload["class_to_idx"]
    names: list[str | None] = [None] * len(class_to_idx)
    for name, idx in class_to_idx.items():
        names[int(idx)] = str(name)
    if any(name is None for name in names):
        raise ValueError("Checkpoint class_to_idx is missing an index")
    return [str(name) for name in names]


def evaluate_model(
    dataset_root: str | Path,
    checkpoint_path: str | Path = Path("models/best_model.pth"),
    *,
    output_dir: str | Path = Path("outputs"),
    batch_size: int = 32,
    num_workers: int = 0,
    device: str | None = None,
) -> dict:
    torch_device = select_device(device)
    model, payload = load_model_from_checkpoint(checkpoint_path, map_location=torch_device)
    model.to(torch_device)
    model.eval()

    preprocess = payload["preprocess"]
    image_size = int(preprocess["image_size"])
    color_mode = str(preprocess["color_mode"])
    seed = int(payload.get("seed", 42))
    class_names = class_names_from_checkpoint(payload)

    loaders, dataset_split = build_dataloaders(
        dataset_root,
        image_size=image_size,
        color_mode=color_mode,
        batch_size=batch_size,
        seed=seed,
        num_workers=num_workers,
        pin_memory=torch_device.type == "cuda",
    )
    if dataset_split.class_to_idx != payload["class_to_idx"]:
        raise ValueError(
            "Dataset class map does not match the checkpoint. "
            "Use the same dataset and split.json that produced this model."
        )
    if len(loaders["test"].dataset) == 0:
        raise ValueError("Test split is empty")

    y_true, y_pred = _predict_loader(model, loaders["test"], torch_device)
    metrics = compute_classification_metrics(y_true, y_pred, class_names)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "classification_report.txt"
    metrics_path = out / "metrics.json"
    matrix_path = save_confusion_matrix(
        metrics["confusion_matrix"],
        class_names,
        out / "confusion_matrix.png",
    )
    report_path.write_text(metrics["classification_report"], encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Device: {torch_device}")
    print(f"Samples: {metrics['num_samples']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (macro): {metrics['recall_macro']:.4f}")
    print(f"F1 (macro): {metrics['f1_macro']:.4f}")
    print(metrics["classification_report"])
    print(f"Wrote {report_path}")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {matrix_path}")

    metrics["report_path"] = str(report_path.resolve())
    metrics["metrics_path"] = str(metrics_path.resolve())
    metrics["confusion_matrix_path"] = str(matrix_path)
    return metrics


def _predict_loader(model, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    true_labels: list[int] = []
    pred_labels: list[int] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            logits = model(images)
            true_labels.extend(labels.cpu().tolist())
            pred_labels.extend(logits.argmax(dim=1).cpu().tolist())
    return np.asarray(true_labels), np.asarray(pred_labels)
