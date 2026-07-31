from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "dkfz_matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from median_pipeline.excel_outputs import write_matrix_workbook


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


def _plot(
    matrix: np.ndarray,
    labels: list[str],
    title: str,
    output_base: Path,
    formats: list[str],
    dpi: int,
    normalized: bool,
    theme: str,
    accuracy: float,
) -> None:
    dark = theme == "dark"
    background = "#111827" if dark else "white"
    foreground = "white" if dark else "black"
    size = max(7.0, len(labels) * 0.7)
    fig, ax = plt.subplots(figsize=(size + 2, size), facecolor=background)
    ax.set_facecolor(background)
    vmax = 1.0 if normalized else max(1, int(matrix.max()))
    image = ax.imshow(matrix, cmap="plasma" if dark else "YlGnBu", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"{title} ({theme}) | accuracy={accuracy:.4f}")
    ax.tick_params(colors=foreground)
    ax.xaxis.label.set_color(foreground)
    ax.yaxis.label.set_color(foreground)
    ax.title.set_color(foreground)
    for spine in ax.spines.values():
        spine.set_color(foreground)
    for row in range(len(labels)):
        for col in range(len(labels)):
            value = float(matrix[row, col])
            if value:
                text = f"{value:.3f}" if normalized else str(int(value))
                red, green, blue, _ = image.cmap(image.norm(value))
                luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                ax.text(col, row, text, ha="center", va="center", fontsize=7,
                        color="white" if luminance < 0.5 else "black")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Fraction (0–1)" if normalized else "Count", color=foreground)
    colorbar.ax.tick_params(colors=foreground)
    colorbar.outline.set_edgecolor(foreground)
    fig.tight_layout()
    for fmt in formats:
        if fmt in {"png", "pdf"}:
            folder = output_base.parent / fmt
            folder.mkdir(parents=True, exist_ok=True)
            fig.savefig(folder / f"{output_base.name}.{fmt}", dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_evaluation(
    predictions: pd.DataFrame,
    labels: list[str],
    run_dir: Path,
    cfg: dict[str, Any],
    split_name: str = "testing",
) -> dict[str, Any]:
    if split_name not in {"validation", "testing"}:
        raise ValueError("split_name must be 'validation' or 'testing'")
    y_true = predictions["label_index"].to_numpy(dtype=int)
    y_pred = predictions["prediction_index"].to_numpy(dtype=int)
    matrix = confusion_matrix(y_true, y_pred, len(labels))
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0)
    accuracy = float((y_true == y_pred).mean()) if len(y_true) else 0.0

    matrix_root = run_dir / f"{split_name}_matrices"
    csv_dir = matrix_root / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    raw_frame = pd.DataFrame(matrix, index=labels, columns=labels)
    norm_frame = pd.DataFrame(normalized, index=labels, columns=labels)
    raw_frame.index.name = "true_label"
    norm_frame.index.name = "true_label"
    raw_frame.to_csv(csv_dir / "confusion_matrix_raw.csv")
    norm_frame.to_csv(csv_dir / "confusion_matrix_normalized.csv", float_format="%.3f")
    excel_path = matrix_root / "confusion_matrix.xlsx"
    write_matrix_workbook(raw_frame, norm_frame, excel_path)
    print(f"  Excel confusion matrices written: {excel_path}", flush=True)

    formats = list(cfg["evaluation"]["formats"])
    dpi = int(cfg["evaluation"]["dpi"])
    for theme in ("light", "dark"):
        _plot(
            matrix,
            labels,
            f"{split_name.capitalize()} confusion matrix (counts)",
            matrix_root / f"confusion_matrix_raw_{theme}",
            formats,
            dpi,
            False,
            theme,
            accuracy,
        )
        _plot(
            normalized,
            labels,
            f"{split_name.capitalize()} confusion matrix (row-normalized)",
            matrix_root / f"confusion_matrix_normalized_{theme}",
            formats,
            dpi,
            True,
            theme,
            accuracy,
        )

    # Only testing normalized PDFs are copied to the root of the named run folder.
    if split_name == "testing" and "pdf" in formats:
        pdf_dir = matrix_root / "pdf"
        for suffix in ("_light", "_dark"):
            filename = f"confusion_matrix_normalized{suffix}.pdf"
            shutil.copy2(pdf_dir / filename, run_dir / filename)

    print(f"  {split_name.capitalize()} confusion-matrix CSV and light/dark PNG and PDF outputs written", flush=True)

    class_frame = per_class_metrics(matrix, labels)
    class_frame.to_csv(csv_dir / "per_class_metrics.csv", index=False)
    balanced_accuracy = float(class_frame["recall"].mean())
    macro_f1 = float(class_frame["f1"].mean())
    worst = class_frame.sort_values(["recall", "label"]).iloc[0]
    summary = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_recall": balanced_accuracy,
        "macro_f1": macro_f1,
        "n_rows": int(len(y_true)),
        "worst_class": str(worst["label"]),
        "worst_class_recall": float(worst["recall"]),
    }
    if split_name == "testing":
        summary["n_test_rows"] = int(len(y_true))
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    else:
        summary["n_validation_rows"] = int(len(y_true))
    pd.DataFrame([summary]).to_csv(
        run_dir / f"{split_name}_metrics.csv",
        index=False,
        float_format="%.6f",
    )
    return summary
