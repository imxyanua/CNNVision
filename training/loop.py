from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from models.checkpoint import save_checkpoint
from models.cnn import build_cnn
from preprocessing.transforms import (
    DEFAULT_COLOR_MODE,
    DEFAULT_IMAGE_SIZE,
    preprocess_settings,
)
from training.dataset import build_dataloaders
from training.device import select_device
from training.split import DEFAULT_SEED


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(
    dataset_root: str | Path,
    *,
    output_path: str | Path = Path("models/best_model.pth"),
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    image_size: int = DEFAULT_IMAGE_SIZE,
    color_mode: str = DEFAULT_COLOR_MODE,
    seed: int = DEFAULT_SEED,
    num_workers: int = 0,
    device: str | None = None,
) -> dict:
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")

    set_seed(seed)
    torch_device = select_device(device)
    in_channels = 1 if color_mode == "grayscale" else 3
    pin_memory = torch_device.type == "cuda"

    loaders, dataset_split = build_dataloaders(
        dataset_root,
        image_size=image_size,
        color_mode=color_mode,
        batch_size=batch_size,
        seed=seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if len(loaders["val"].dataset) == 0:
        raise ValueError("Validation split is empty; add more images or use a pre-split dataset")

    print("Device:", torch_device)
    print("Classes:", dataset_split.class_to_idx)
    print("Counts:", dataset_split.counts())

    model = build_cnn(num_classes=dataset_split.num_classes, in_channels=in_channels)
    model.to(torch_device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history: list[dict[str, float | int]] = []
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    saved_path: Path | None = None

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _run_epoch(
            model, loaders["train"], criterion, optimizer, torch_device, train=True
        )
        val_loss, val_acc = _run_epoch(
            model, loaders["val"], criterion, optimizer, torch_device, train=False
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        print(
            f"Epoch {epoch}/{epochs}  "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        # Pick the checkpoint from validation accuracy. Test is never used here.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            saved_path = save_checkpoint(
                output_path,
                {
                    "model_state_dict": model.state_dict(),
                    "model": model.settings(),
                    "class_to_idx": dataset_split.class_to_idx,
                    "idx_to_class": dataset_split.idx_to_class,
                    "preprocess": preprocess_settings(image_size=image_size, color_mode=color_mode),
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "history": history,
                    "seed": seed,
                },
            )

    if saved_path is None:
        raise RuntimeError("No checkpoint was saved")

    print(f"Best val_acc={best_val_acc:.4f} at epoch {best_epoch}")
    print(f"Saved {saved_path}")
    return {
        "checkpoint": saved_path,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "history": history,
        "class_to_idx": dataset_split.class_to_idx,
        "device": str(torch_device),
        "num_classes": dataset_split.num_classes,
    }


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    train: bool,
) -> tuple[float, float]:
    model.train(train)
    total_loss = 0.0
    total_correct = 0
    total = 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            if train:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += batch_size
    if total == 0:
        raise ValueError("Loader produced no samples")
    return total_loss / total, total_correct / total
