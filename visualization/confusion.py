from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


def save_confusion_matrix(
    matrix: list[list[int]] | np.ndarray,
    class_names: list[str],
    path: str | Path,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    display = ConfusionMatrixDisplay(
        confusion_matrix=np.asarray(matrix),
        display_labels=class_names,
    )
    display.plot(ax=ax, colorbar=True)
    ax.set_title("Confusion matrix (test)")
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output.resolve()
