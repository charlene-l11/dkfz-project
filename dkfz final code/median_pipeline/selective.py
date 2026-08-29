from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from median_pipeline.config import ConfigurationError
from median_pipeline.excel_outputs import write_split_workbooks
from median_pipeline.preparation import prepare_data
from median_pipeline.wavelengths import resolve_wavelength_selection


@dataclass(frozen=True)
class SelectiveRun:
    name: str
    selection: Any

    @property
    def folder_name(self) -> str:
        """Use the test-case identifier instead of a potentially long selection."""
        name = self.name.strip()
        windows_unsafe = re.search(r'[<>:"/\\|?*\x00-\x1f]', name)
        if not name or name in {".", ".."} or windows_unsafe or name.endswith((" ", ".")):
            raise ConfigurationError(
                f"Selective test-case name {self.name!r} cannot be used as a "
                "Windows-compatible folder name"
            )
        if len(name) > 120:
            raise ConfigurationError(
                f"Selective test-case name {self.name!r} is too long for a portable child-folder name; "
                "use a name of at most 120 characters"
            )
        return name


def build_selective_runs(cfg: dict[str, Any]) -> list[SelectiveRun]:
    configured = cfg["wavelength"].get("selective")
    if not isinstance(configured, dict):
        raise ConfigurationError("Selective batch mode requires a mapping of test-case names to selections")
    runs = [SelectiveRun(str(name), selection) for name, selection in configured.items()]
    folder_names: dict[str, str] = {}
    for run in runs:
        normalized = run.folder_name.casefold()
        if normalized in folder_names:
            raise ConfigurationError(
                f"Selective test cases {folder_names[normalized]!r} and {run.name!r} "
                f"would both use folder {run.folder_name!r}"
            )
        folder_names[normalized] = run.name
    return runs


def _child_root(master_root: Path, run: SelectiveRun) -> Path:
    return master_root / run.folder_name


def _child_config(master_cfg: dict[str, Any], master_config: Path, run: SelectiveRun) -> dict[str, Any]:
    child = copy.deepcopy(master_cfg)
    child["wavelength"]["selective"] = run.selection
    child["selective_run"] = {
        "name": run.name,
        "selection": run.selection,
        "folder_name": run.folder_name,
        "master_config": str(master_config.resolve()),
    }
    return child


def _status_record(master_root: Path, run: SelectiveRun) -> dict[str, Any]:
    child_root = _child_root(master_root, run)
    completed = (child_root / "run_config" / "run_parameters.json").is_file()
    prepared = (child_root / "output" / "data" / "wavelengths.json").is_file()
    return {
        "name": run.name,
        "selection": run.selection,
        "folder_name": run.folder_name,
        "path": str(child_root.resolve()),
        "status": "completed" if completed else "prepared" if prepared else "pending",
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


def _metric_row(run: SelectiveRun, child_root: Path) -> dict[str, Any] | None:
    validation_path = child_root / "output" / "validation_metrics.csv"
    testing_path = child_root / "output" / "testing_metrics.csv"
    if not validation_path.is_file() or not testing_path.is_file():
        return None
    validation = pd.read_csv(validation_path).iloc[0].to_dict()
    testing = pd.read_csv(testing_path).iloc[0].to_dict()
    wavelengths_path = child_root / "output" / "data" / "wavelengths.json"
    wavelengths = json.loads(wavelengths_path.read_text(encoding="utf-8"))
    row: dict[str, Any] = {
        "test_case": run.name,
        "selection": str(run.selection),
        "n_selected_channels": wavelengths["n_selected_channels"],
        "selected_wavelengths": "; ".join(f"{value:g}" for value in wavelengths["selected_wavelengths"]),
        "run_path": str(child_root.resolve()),
    }
    row.update({f"validation_{key}": value for key, value in validation.items()})
    row.update({f"testing_{key}": value for key, value in testing.items()})
    return row


def write_aggregate_results(master_root: Path, runs: list[SelectiveRun]) -> None:
    rows = [
        row
        for run in runs
        if (row := _metric_row(run, _child_root(master_root, run))) is not None
    ]
    if not rows:
        return
    results_root = master_root / "selective results"
    frame = pd.DataFrame(rows)
    frame.to_csv(results_root / "combined_results.csv", index=False)
    with pd.ExcelWriter(results_root / "combined_results.xlsx", engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="Selective runs", index=False)


def run_selective(
    master_config: Path,
    cfg: dict[str, Any],
    prepare_only: bool,
    run_child: Callable[..., Path],
) -> Path:
    master_config = master_config.resolve()
    master_root = master_config.parent
    results_root = master_root / "selective results"
    shared_splits = master_root / "splits"
    results_root.mkdir(parents=True, exist_ok=True)
    shared_splits.mkdir(parents=True, exist_ok=True)

    runs = build_selective_runs(cfg)
    if not runs:
        raise ConfigurationError("No selective test cases were configured")

    print("[Selective] Discovering data and creating one shared subject split", flush=True)
    discovery_cfg = copy.deepcopy(cfg)
    discovery_cfg["wavelength"]["selective"] = None
    manifest, mapping, split_summary, base_wavelength_info = prepare_data(discovery_cfg)
    write_split_workbooks(manifest, shared_splits)

    records = [_status_record(master_root, run) for run in runs]
    _write_status(results_root, master_config, records)
    print(f"[Selective] Generated {len(runs)} child run(s)", flush=True)

    available = np.asarray(base_wavelength_info["available_wavelengths"], dtype=np.float64)
    for number, (run, record) in enumerate(zip(runs, records, strict=True), start=1):
        if record["status"] == "completed":
            print(
                f"[Selective {number}/{len(runs)}] Skipping completed wavelengths {run.folder_name}",
                flush=True,
            )
            continue
        print(
            f"[Selective {number}/{len(runs)}] Test case {run.name}: {run.folder_name}",
            flush=True,
        )
        record["status"] = "running"
        record.pop("error", None)
        _write_status(results_root, master_config, records)
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
                scenario_root=_child_root(master_root, run),
                copy_source_yaml=True,
            )
            record["status"] = "prepared" if prepare_only else "completed"
            record["n_selected_channels"] = wavelength_info["n_selected_channels"]
            record["selected_wavelengths"] = wavelength_info["selected_wavelengths"]
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
    print(f"Selective analysis directory: {master_root}", flush=True)
    return master_root
