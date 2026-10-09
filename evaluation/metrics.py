from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict:
    if len(y_true) == 0:
        raise ValueError("No samples to evaluate")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    labels = list(range(len(class_names)))
    precision, recall, f1, _support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
        "class_names": class_names,
        "num_samples": int(len(y_true)),
    }
