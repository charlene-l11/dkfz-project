from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from median_pipeline.config import ConfigurationError
from median_pipeline.excel_outputs import write_split_workbooks
from median_pipeline.preparation import prepare_data
from median_pipeline.wavelengths import resolve_wavelength_selection


def _format_wavelength(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


@dataclass(frozen=True)
class CombinationRun:
    wavelengths: tuple[float, ...]

    @property
    def selection_name(self) -> str:
        return "+".join(_format_wavelength(value) for value in self.wavelengths)

    @property
    def selection(self) -> str:
        return "; ".join(_format_wavelength(value) for value in self.wavelengths)


def build_combination_runs(
    cfg: dict[str, Any],
    wavelength_info: dict[str, Any],
) -> list[CombinationRun]:
    settings = cfg["combination_analysis"]
    if not settings.get("enabled", False):
        raise ConfigurationError("Combination analysis is not enabled")
    available = np.asarray(wavelength_info["available_wavelengths"], dtype=np.float64)
    if available.ndim != 1 or len(available) == 0:
        raise ConfigurationError("No available wavelengths were discovered")
    spacing = float(np.median(np.diff(available))) if len(available) > 1 else 1.0
    tolerance = max(1e-6, abs(spacing) * 0.01)
    minimum = float(settings["min"])
    maximum = float(settings["max"])
    variable_candidates = available[
        (available >= minimum - tolerance) & (available <= maximum + tolerance)
    ]
    count = int(settings["n_wavelengths"])
    fixed: list[float] = []
    for configured_value in settings.get("fixed_wavelengths", []):
        matches = np.flatnonzero(
            np.isclose(available, float(configured_value), atol=tolerance, rtol=0)
        )
        if len(matches) == 0:
            nearest = float(available[np.abs(available - float(configured_value)).argmin()])
            raise ConfigurationError(
                f"Fixed wavelength {float(configured_value):g} is not on the available grid; "
                f"nearest available value is {nearest:g}"
            )
        fixed.append(float(available[int(matches[0])]))
    if len(set(fixed)) != len(fixed):
        raise ConfigurationError("Fixed wavelengths resolve to duplicate channels on the available grid")

    if fixed:
        variable_candidates = np.asarray(
            [
                value
                for value in variable_candidates
                if not any(np.isclose(value, fixed_value, atol=tolerance, rtol=0) for fixed_value in fixed)
            ],
            dtype=np.float64,
        )
    variable_count = count - len(fixed)
    if variable_count < 0:
        raise ConfigurationError(
            "combination_analysis cannot define more fixed_wavelengths than n_wavelengths"
        )
    if variable_count > len(variable_candidates):
        raise ConfigurationError(
            f"combination_analysis needs {variable_count} variable wavelengths per run, but only "
            f"{len(variable_candidates)} eligible channels lie between {minimum:g} and {maximum:g}"
        )
    runs = [
        CombinationRun(tuple([*fixed, *(float(value) for value in values)]))
        for values in combinations(variable_candidates, variable_count)
    ]
    if runs and len(f"pending; {runs[-1].selection_name}") > 120:
        raise ConfigurationError(
            "The requested number of wavelengths produces folder names longer than 120 characters; "
            "reduce combination_analysis.n_wavelengths"
        )
    return runs


def _temporary_folder_name(run: CombinationRun) -> str:
    return f"pending; {run.selection_name}"


def _completed_folder_name(run: CombinationRun, accuracy: float) -> str:
    if not np.isfinite(accuracy) or not 0 <= accuracy <= 1:
        raise ConfigurationError(f"Validation accuracy must lie between 0 and 1, got {accuracy!r}")
    return f"{accuracy * 100:05.2f}%; {run.selection_name}"


def _selection_from_folder_name(name: str) -> str | None:
    if "; " not in name:
        return None
    prefix, selection = name.split("; ", 1)
    if prefix == "pending" or re.fullmatch(r"(?:\d{2}|100)\.\d{2}%", prefix):
        return selection
    return None


def _discover_roots(runs_root: Path) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for path in runs_root.iterdir():
        if not path.is_dir():
            continue
        selection = _selection_from_folder_name(path.name)
        if selection is None:
            continue
        if selection in roots:
            raise ConfigurationError(
                f"Multiple combination folders represent {selection!r}: {roots[selection]} and {path}"
            )
        roots[selection] = path
    return roots


def _read_split_metrics(child_root: Path) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for split in ("validation", "testing"):
        path = child_root / "output" / f"{split}_metrics.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Completed combination run is missing {path}")
        row = pd.read_csv(path).iloc[0].to_dict()
        values.update({f"{split}_{key}": value for key, value in row.items()})
    return values


def _replace_path_strings(value: Any, old_path: str, new_path: str) -> Any:
    if isinstance(value, dict):
        return {key: _replace_path_strings(item, old_path, new_path) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_path_strings(item, old_path, new_path) for item in value]
    if isinstance(value, str):
        return value.replace(old_path, new_path)
    return value


def _update_run_parameters_path(child_root: Path, old_path: str, new_path: str) -> None:
    parameters_path = child_root / "run_config" / "run_parameters.json"
    if not parameters_path.is_file():
        return
    payload = json.loads(parameters_path.read_text(encoding="utf-8"))
    payload = _replace_path_strings(payload, old_path, new_path)
    parameters_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _finalize_folder(runs_root: Path, run: CombinationRun, child_root: Path) -> tuple[Path, dict[str, Any]]:
    metrics = _read_split_metrics(child_root)
    accuracy = float(metrics["validation_accuracy"])
    final_root = runs_root / _completed_folder_name(run, accuracy)
    if child_root != final_root:
        if final_root.exists():
            raise FileExistsError(f"Combination result folder already exists: {final_root}")
        old_path = str(child_root.resolve())
        child_root.rename(final_root)
        new_path = str(final_root.resolve())
        _update_run_parameters_path(final_root, old_path, new_path)
    return final_root, metrics


def _child_config(master_cfg: dict[str, Any], master_config: Path, run: CombinationRun) -> dict[str, Any]:
    child = copy.deepcopy(master_cfg)
    child["combination_analysis"]["enabled"] = False
    child["wavelength"]["selective"] = run.selection
    child["combination_run"] = {
        "wavelengths": list(run.wavelengths),
        "selection": run.selection,
        "master_config": str(master_config.resolve()),
    }
    return child


def _status_record(run: CombinationRun, child_root: Path) -> dict[str, Any]:
    completed = (child_root / "run_config" / "run_parameters.json").is_file()
    prepared = (child_root / "output" / "data" / "wavelengths.json").is_file()
    record: dict[str, Any] = {
        "selection": run.selection_name,
        "selected_wavelengths": list(run.wavelengths),
        "n_selected_channels": len(run.wavelengths),
        "path": str(child_root.resolve()),
        "folder_name": child_root.name,
        "status": "completed" if completed else "prepared" if prepared else "pending",
    }
    if completed:
        record.update(_read_split_metrics(child_root))
    return record


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


def write_aggregate_results(results_root: Path, records: list[dict[str, Any]]) -> None:
    rows = [record.copy() for record in records if record.get("validation_accuracy") is not None]
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame = frame.sort_values(
        ["validation_accuracy", "selection"], ascending=[False, True], kind="stable"
    )
    frame.insert(0, "rank", range(1, len(frame) + 1))
    frame.to_csv(results_root / "combined_results.csv", index=False)
    with pd.ExcelWriter(results_root / "combined_results.xlsx", engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="Combination results", index=False)


def run_combinations(
    master_config: Path,
    cfg: dict[str, Any],
    prepare_only: bool,
    run_child: Callable[..., Path],
) -> Path:
    master_config = master_config.resolve()
    master_root = master_config.parent
    runs_root = master_root / "combination runs"
    results_root = master_root / "combination results"
    shared_splits = master_root / "splits"
    runs_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)
    shared_splits.mkdir(parents=True, exist_ok=True)

    print("[Combinations] Discovering data and creating one shared subject split", flush=True)
    discovery_cfg = copy.deepcopy(cfg)
    discovery_cfg["combination_analysis"]["enabled"] = False
    discovery_cfg["wavelength"]["selective"] = None
    manifest, mapping, split_summary, base_wavelength_info = prepare_data(discovery_cfg)
    write_split_workbooks(manifest, shared_splits)

    runs = build_combination_runs(cfg, base_wavelength_info)
    if not runs:
        raise ConfigurationError("No wavelength combinations were generated")
    candidate_count = len({value for run in runs for value in run.wavelengths})
    fixed_count = len(cfg["combination_analysis"].get("fixed_wavelengths", []))
    print(
        f"[Combinations] Generated {len(runs):,} mathematical combinations from "
        f"{candidate_count} participating wavelengths ({fixed_count} fixed)",
        flush=True,
    )

    roots = _discover_roots(runs_root)
    for run in runs:
        existing = roots.get(run.selection_name)
        if existing is None:
            continue
        if existing.name.startswith("pending; ") and (
            existing / "run_config" / "run_parameters.json"
        ).is_file():
            finalized, _ = _finalize_folder(runs_root, run, existing)
            roots[run.selection_name] = finalized

    records = [
        _status_record(
            run,
            roots.get(run.selection_name, runs_root / _temporary_folder_name(run)),
        )
        for run in runs
    ]
    _write_status(results_root, master_config, records)
    if not prepare_only:
        write_aggregate_results(results_root, records)

    available = np.asarray(base_wavelength_info["available_wavelengths"], dtype=np.float64)
    for number, (run, record) in enumerate(zip(runs, records, strict=True), start=1):
        if record["status"] == "completed":
            print(
                f"[Combination {number:,}/{len(runs):,}] Skipping completed {record['folder_name']}",
                flush=True,
            )
            continue
        print(
            f"[Combination {number:,}/{len(runs):,}] {run.selection_name}",
            flush=True,
        )
        record["status"] = "running"
        record.pop("error", None)
        _write_status(results_root, master_config, records)
        child_root = Path(record["path"])
        try:
            child_cfg = _child_config(cfg, master_config, run)
            wavelength_info = resolve_wavelength_selection(
                child_cfg["wavelength"],
                len(available),
                available,
            )
            prepared = (manifest, mapping, split_summary, wavelength_info)
            run_child(
                master_config,
                child_cfg,
                prepare_only=prepare_only,
                prepared=prepared,
                shared_splits=shared_splits,
                scenario_root=child_root,
                copy_source_yaml=True,
            )
            if prepare_only:
                record["status"] = "prepared"
            else:
                child_root, metrics = _finalize_folder(runs_root, run, child_root)
                record.update(metrics)
                record["status"] = "completed"
                record["path"] = str(child_root.resolve())
                record["folder_name"] = child_root.name
            _write_status(results_root, master_config, records)
            if not prepare_only:
                write_aggregate_results(results_root, records)
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
            _write_status(results_root, master_config, records)
            if not prepare_only:
                write_aggregate_results(results_root, records)
            raise

    if not prepare_only:
        write_aggregate_results(results_root, records)
    print(f"Combination analysis directory: {master_root}", flush=True)
    return master_root
