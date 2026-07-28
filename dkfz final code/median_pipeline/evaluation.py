from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "dkfz_matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(matrix, (y_true.astype(int), y_pred.astype(int)), 1)
    return matrix


def per_class_metrics(matrix: np.ndarray, labels: list[str]) -> pd.DataFrame:
    rows = []
    for index, label in enumerate(labels):
        tp = int(matrix[index, index])
        support = int(matrix[index].sum())
        predicted = int(matrix[:, index].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        errors = matrix[index].copy()
        errors[index] = 0
        confused_index = int(errors.argmax()) if errors.sum() else None
        rows.append({
            "label_index": index, "label": label, "support": support, "correct": tp,
            "precision": precision, "recall": recall, "f1": f1,
            "most_confused_with": labels[confused_index] if confused_index is not None else "",
            "most_confused_count": int(errors[confused_index]) if confused_index is not None else 0,
        })
    return pd.DataFrame(rows)


def _plot(matrix: np.ndarray, labels: list[str], title: str, output_base: Path, formats: list[str], dpi: int, normalized: bool) -> None:
    size = max(7.0, len(labels) * 0.7)
    fig, ax = plt.subplots(figsize=(size + 2, size))
    vmax = 100 if normalized else max(1, int(matrix.max()))
    image = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    for row in range(len(labels)):
        for col in range(len(labels)):
            value = float(matrix[row, col])
            if value:
                text = f"{value:.1f}%" if normalized else str(int(value))
                ax.text(col, row, text, ha="center", va="center", fontsize=7,
                        color="white" if value >= vmax * 0.5 else "black")
    fig.colorbar(image, ax=ax, label="Percentage" if normalized else "Count")
    fig.tight_layout()
    for fmt in formats:
        if fmt in {"png", "pdf"}:
            folder = output_base.parent / fmt
            folder.mkdir(parents=True, exist_ok=True)
            fig.savefig(folder / f"{output_base.name}.{fmt}", dpi=dpi)
    plt.close(fig)


def write_evaluation(predictions: pd.DataFrame, labels: list[str], run_dir: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    y_true = predictions["label_index"].to_numpy(dtype=int)
    y_pred = predictions["prediction_index"].to_numpy(dtype=int)
    matrix = confusion_matrix(y_true, y_pred, len(labels))
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0) * 100

    matrix_root = run_dir / "matrices"
    csv_dir = matrix_root / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    raw_frame = pd.DataFrame(matrix, index=labels, columns=labels)
    norm_frame = pd.DataFrame(normalized, index=labels, columns=labels)
    raw_frame.index.name = "true_label"
    norm_frame.index.name = "true_label"
    raw_frame.to_csv(csv_dir / "confusion_matrix_raw.csv")
    norm_frame.to_csv(csv_dir / "confusion_matrix_normalized.csv", float_format="%.4f")

    formats = list(cfg["evaluation"]["formats"])
    dpi = int(cfg["evaluation"]["dpi"])
    _plot(matrix, labels, "Confusion matrix (counts)", matrix_root / "confusion_matrix_raw", formats, dpi, False)
    _plot(normalized, labels, "Confusion matrix (row-normalized)", matrix_root / "confusion_matrix_normalized", formats, dpi, True)

    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    class_frame = per_class_metrics(matrix, labels)
    class_frame.to_csv(metrics_dir / "per_class_metrics.csv", index=False)
    accuracy = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    balanced_accuracy = float(class_frame["recall"].mean())
    macro_f1 = float(class_frame["f1"].mean())
    worst = class_frame.sort_values(["recall", "label"]).iloc[0]
    summary = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_recall": balanced_accuracy,
        "macro_f1": macro_f1,
        "n_test_rows": int(len(y_true)),
        "worst_class": str(worst["label"]),
        "worst_class_recall": float(worst["recall"]),
    }
    (metrics_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
