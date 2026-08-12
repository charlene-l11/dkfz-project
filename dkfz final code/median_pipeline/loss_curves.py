from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "dkfz_matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _epoch_metric(frame: pd.DataFrame, candidates: tuple[str, ...], output_name: str) -> pd.DataFrame:
    column = next((name for name in candidates if name in frame.columns), None)
    if column is None:
        raise RuntimeError(
            f"Lightning metrics do not contain {output_name}; checked columns {list(candidates)}"
        )
    values = frame.loc[frame[column].notna(), ["epoch", column]].copy()
    if values.empty:
        raise RuntimeError(f"Lightning did not record any values for {column}")
    values["epoch"] = pd.to_numeric(values["epoch"], errors="raise").astype(int) + 1
    values[column] = pd.to_numeric(values[column], errors="raise")
    return values.groupby("epoch", as_index=False)[column].last().rename(columns={column: output_name})


def _plot_loss(
    history: pd.DataFrame,
    column: str,
    title: str,
    output_dir: Path,
    dpi: int,
) -> None:
    values = history.loc[history[column].notna(), ["epoch", column]]
    if values.empty:
        raise RuntimeError(f"No values are available for {column}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(values["epoch"], values[column], color="#2563eb", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(left=1)
    fig.tight_layout()
    fig.savefig(output_dir / f"{column}_curve.png", dpi=dpi)
    plt.close(fig)


def write_loss_curves(metrics_path: Path, run_dir: Path, cfg: dict[str, Any]) -> Path:
    """Write separate training and validation loss curves for one run."""
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Lightning metrics were not found: {metrics_path}")
    frame = pd.read_csv(metrics_path)
    if "epoch" not in frame.columns:
        raise RuntimeError(f"Lightning metrics do not contain an epoch column: {metrics_path}")

    training = _epoch_metric(
        frame,
        ("train/ce_loss_epoch", "train/ce_loss"),
        "training_loss",
    )
    validation = _epoch_metric(
        frame,
        ("validation/ce_loss", "validation/ce_loss_epoch"),
        "validation_loss",
    )
    history = training.merge(validation, on="epoch", how="outer").sort_values("epoch")

    output_dir = run_dir / "loss_curves"
    output_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(output_dir / "loss_history.csv", index=False, float_format="%.8f")

    dpi = int(cfg.get("evaluation", {}).get("dpi", 300))
    _plot_loss(history, "training_loss", "Training loss", output_dir, dpi)
    _plot_loss(history, "validation_loss", "Validation loss", output_dir, dpi)
    print(f"  Training and validation loss curves written: {output_dir}", flush=True)
    return output_dir
