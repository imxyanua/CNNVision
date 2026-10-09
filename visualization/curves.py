from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_training_curves(
    history: list[dict[str, float | int]],
    path: str | Path,
) -> Path:
    if not history:
        raise ValueError("Training history is empty")

    epochs = [int(row["epoch"]) for row in history]
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))
    ax_loss.plot(epochs, [float(row["train_loss"]) for row in history], label="train")
    ax_loss.plot(epochs, [float(row["val_loss"]) for row in history], label="val")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    ax_acc.plot(epochs, [float(row["train_acc"]) for row in history], label="train")
    ax_acc.plot(epochs, [float(row["val_acc"]) for row in history], label="val")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Accuracy")
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output.resolve()
