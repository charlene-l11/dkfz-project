from __future__ import annotations

import copy
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "dkfz_matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from median_pipeline.config import ConfigurationError, load_config
from median_pipeline.excel_outputs import write_split_workbooks
from median_pipeline.preparation import discover, split_manifest
from median_pipeline.wavelengths import resolve_wavelength_selection


@dataclass(frozen=True)
class StepwiseRun:
    direction: str
    range_name: str
    lower_wavelength: float
    upper_wavelength: float
    selected_indices: tuple[int, ...]

    @property
    def n_selected_channels(self) -> int:
        return len(self.selected_indices)


def _format_wavelength(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def _grid_index(available: np.ndarray, requested: float, setting_name: str) -> int:
    spacing = float(np.median(np.diff(available))) if len(available) > 1 else 1.0
    tolerance = max(1e-6, abs(spacing) * 0.01)
    matches = np.flatnonzero(np.isclose(available, requested, atol=tolerance, rtol=0))
    if len(matches) != 1:
        nearest = float(available[np.abs(available - requested).argmin()])
        raise ConfigurationError(
            f"{setting_name}={requested:g} is not on the wavelength grid; "
            f"nearest available wavelength is {nearest:g}"
        )
    return int(matches[0])


def build_stepwise_runs(cfg: dict[str, Any], wavelength_info: dict[str, Any]) -> list[StepwiseRun]:
    available = np.asarray(wavelength_info["available_wavelengths"], dtype=np.float64)
    if len(available) < 2:
        raise ConfigurationError("Stepwise analysis requires at least two wavelength channels")

    runs: list[StepwiseRun] = []
    if cfg["stepwise_analysis"]:
        stop_index = _grid_index(
            available,
            float(cfg["forward_stop_wavelength"]),
            "forward_stop_wavelength",
        )
        if stop_index < 1:
            raise ConfigurationError("Forward stepwise analysis must include at least two wavelength channels")
        for end_index in range(1, stop_index + 1):
            lower, upper = float(available[0]), float(available[end_index])
            runs.append(
                StepwiseRun(
                    direction="forward",
                    range_name=f"{_format_wavelength(lower)}-{_format_wavelength(upper)}",
                    lower_wavelength=lower,
                    upper_wavelength=upper,
                    selected_indices=tuple(range(0, end_index + 1)),
                )
            )

    if cfg["reverse_stepwise"]:
        stop_index = _grid_index(
            available,
            float(cfg["reverse_stop_wavelength"]),
            "reverse_stop_wavelength",
        )
        if stop_index > len(available) - 2:
            raise ConfigurationError("Reverse stepwise analysis must include at least two wavelength channels")
        for start_index in range(len(available) - 2, stop_index - 1, -1):
            lower, upper = float(available[start_index]), float(available[-1])
            runs.append(
                StepwiseRun(
                    direction="reverse",
                    range_name=f"{_format_wavelength(lower)}-{_format_wavelength(upper)}",
                    lower_wavelength=lower,
                    upper_wavelength=upper,
                    selected_indices=tuple(range(start_index, len(available))),
                )
            )
    return runs


def _child_root(master_root: Path, run: StepwiseRun) -> Path:
    direction_folder = "forward run" if run.direction == "forward" else "reverse run"
    return master_root / direction_folder / run.range_name


def _child_yaml_text(master_raw: dict[str, Any], master_config: Path, run: StepwiseRun) -> str:
    child = copy.deepcopy(master_raw)
    child["stepwise_analysis"] = False
    child["reverse_stepwise"] = False
    # Stop values control how many children the master creates; they do not
    # change an individual child's training configuration.
    child.pop("forward_stop_wavelength", None)
    child.pop("reverse_stop_wavelength", None)
    child["wavelength"]["selective"] = run.range_name
    child["stepwise_run"] = {
        "direction": run.direction,
        "range": run.range_name,
        "n_selected_channels": run.n_selected_channels,
        "master_config": str(master_config.resolve()),
    }
    return yaml.safe_dump(child, sort_keys=False, allow_unicode=True)


def _write_child_yaml(master_raw: dict[str, Any], master_config: Path, run: StepwiseRun) -> Path:
    child_root = _child_root(master_config.parent, run)
    child_root.mkdir(parents=True, exist_ok=True)
    child_path = child_root / "manifold_settings.yaml"
    expected = _child_yaml_text(master_raw, master_config, run)
    if child_path.exists() and child_path.read_text(encoding="utf-8-sig") != expected:
        raise FileExistsError(
            f"Generated child YAML differs from the existing file: {child_path}. "
            "Move the existing child folder before changing master stepwise settings."
        )
    child_path.write_text(expected, encoding="utf-8")
    return child_path


def _status_record(master_root: Path, run: StepwiseRun) -> dict[str, Any]:
    child_root = _child_root(master_root, run)
    completed = (child_root / "run_config" / "run_parameters.json").is_file()
    prepared = (child_root / "output" / "data" / "wavelengths.json").is_file()
    status = "completed" if completed else "prepared" if prepared else "pending"
    return {
        **asdict(run),
        "selected_indices": list(run.selected_indices),
        "n_selected_channels": run.n_selected_channels,
        "path": str(child_root.resolve()),
        "status": status,
    }


def _write_status(results_root: Path, master_config: Path, records: list[dict[str, Any]]) -> None:
    payload = {
        "master_config": str(master_config.resolve()),
        "runs": records,
        "counts": {
            status: sum(record["status"] == status for record in records)
            for status in ("pending", "prepared", "running", "completed", "failed")
        },
    }
    (results_root / "status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _metric_row(run: StepwiseRun, child_root: Path) -> dict[str, Any] | None:
    validation_path = child_root / "output" / "validation_metrics.csv"
    testing_path = child_root / "output" / "testing_metrics.csv"
    if not validation_path.is_file() or not testing_path.is_file():
        return None
    validation = pd.read_csv(validation_path).iloc[0].to_dict()
    testing = pd.read_csv(testing_path).iloc[0].to_dict()
    row: dict[str, Any] = {
        "direction": run.direction,
        "wavelength_range": run.range_name,
        "lower_wavelength": run.lower_wavelength,
        "upper_wavelength": run.upper_wavelength,
        "n_selected_channels": run.n_selected_channels,
        "run_path": str(child_root.resolve()),
    }
    row.update({f"validation_{key}": value for key, value in validation.items()})
    row.update({f"testing_{key}": value for key, value in testing.items()})
    return row


def _plot_results(frame: pd.DataFrame, results_root: Path) -> None:
    if frame.empty:
        return
    for x_column, filename, xlabel in (
        ("n_selected_channels", "accuracy_by_channel_count.png", "Selected wavelength channels"),
        ("boundary_wavelength", "accuracy_by_wavelength.png", "Expanding boundary wavelength (nm)"),
    ):
        fig, ax = plt.subplots(figsize=(10, 6))
        for direction, group in frame.groupby("direction", sort=False):
            ordered = group.sort_values("n_selected_channels")
            x_values = (
                ordered["n_selected_channels"]
                if x_column == "n_selected_channels"
                else np.where(
                    ordered["direction"].eq("forward"),
                    ordered["upper_wavelength"],
                    ordered["lower_wavelength"],
                )
            )
            ax.plot(x_values, ordered["validation_accuracy"], marker="o", markersize=3,
                    label=f"{direction} validation")
            ax.plot(x_values, ordered["testing_accuracy"], marker="o", markersize=3,
                    label=f"{direction} testing")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(results_root / filename, dpi=200)
        plt.close(fig)


def write_aggregate_results(master_root: Path, runs: list[StepwiseRun]) -> None:
    results_root = master_root / "stepwise results"
    rows = []
    for run in runs:
        row = _metric_row(run, _child_root(master_root, run))
        if row is not None:
            rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return
    forward = frame[frame["direction"].eq("forward")].copy()
    reverse = frame[frame["direction"].eq("reverse")].copy()
    if not forward.empty:
        forward.to_csv(results_root / "forward_results.csv", index=False)
    if not reverse.empty:
        reverse.to_csv(results_root / "reverse_results.csv", index=False)
    with pd.ExcelWriter(results_root / "combined_results.xlsx", engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="Combined", index=False)
        if not forward.empty:
            forward.to_excel(writer, sheet_name="Forward", index=False)
        if not reverse.empty:
            reverse.to_excel(writer, sheet_name="Reverse", index=False)
    _plot_results(frame, results_root)


def run_stepwise(
    master_config: Path,
    cfg: dict[str, Any],
    prepare_only: bool,
    run_child: Callable[..., Path],
) -> Path:
    master_config = master_config.resolve()
    master_root = master_config.parent
    results_root = master_root / "stepwise results"
    shared_splits = master_root / "splits"
    results_root.mkdir(parents=True, exist_ok=True)
    shared_splits.mkdir(parents=True, exist_ok=True)

    print("[Stepwise] Discovering data and creating one shared subject split", flush=True)
    manifest, mapping, base_wavelength_info = discover(cfg)
    manifest, split_summary = split_manifest(manifest, cfg)
    write_split_workbooks(manifest, shared_splits)

    runs = build_stepwise_runs(cfg, base_wavelength_info)
    if not runs:
        raise ConfigurationError("No forward or reverse stepwise runs were generated")
    master_raw = yaml.safe_load(master_config.read_text(encoding="utf-8-sig"))
    child_paths = [_write_child_yaml(master_raw, master_config, run) for run in runs]
    records = [_status_record(master_root, run) for run in runs]
    _write_status(results_root, master_config, records)
    print(
        f"[Stepwise] Generated {len(runs)} child run(s): "
        f"{sum(run.direction == 'forward' for run in runs)} forward, "
        f"{sum(run.direction == 'reverse' for run in runs)} reverse",
        flush=True,
    )

    available = np.asarray(base_wavelength_info["available_wavelengths"], dtype=np.float64)
    for number, (run, child_path, record) in enumerate(zip(runs, child_paths, records, strict=True), start=1):
        if record["status"] == "completed":
            print(f"[Stepwise {number}/{len(runs)}] Skipping completed {run.direction} {run.range_name}", flush=True)
            continue
        print(f"[Stepwise {number}/{len(runs)}] {run.direction.capitalize()} {run.range_name}", flush=True)
        record["status"] = "running"
        record.pop("error", None)
        _write_status(results_root, master_config, records)
        try:
            child_cfg = load_config(child_path)
            wavelength_info = resolve_wavelength_selection(
                child_cfg["wavelength"],
                len(available),
                available,
            )
            prepared = (manifest, mapping, split_summary, wavelength_info)
            run_child(
                child_path,
                child_cfg,
                prepare_only=prepare_only,
                prepared=prepared,
                shared_splits=shared_splits,
            )
            record["status"] = "prepared" if prepare_only else "completed"
            _write_status(results_root, master_config, records)
            if not prepare_only:
                write_aggregate_results(master_root, runs)
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
            _write_status(results_root, master_config, records)
            if not prepare_only:
                write_aggregate_results(master_root, runs)
            raise

    if not prepare_only:
        write_aggregate_results(master_root, runs)
    print(f"Stepwise analysis directory: {master_root}", flush=True)
    return master_root
