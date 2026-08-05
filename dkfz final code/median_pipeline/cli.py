from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from median_pipeline.config import load_config
from median_pipeline.excel_outputs import write_split_workbooks
from median_pipeline.preparation import prepare_data, write_prepared


def scenario_layout(
    config_path: Path | None = None,
    *,
    scenario_root: Path | None = None,
    include_splits: bool = True,
) -> dict[str, Path]:
    if scenario_root is None:
        if config_path is None:
            raise ValueError("config_path or scenario_root is required")
        config_path = config_path.resolve()
        if not config_path.is_file() or config_path.suffix.lower() not in {".yaml", ".yml"}:
            raise FileNotFoundError(f"Scenario YAML was not found: {config_path}")
        scenario_root = config_path.parent
    else:
        scenario_root = scenario_root.resolve()
        scenario_root.mkdir(parents=True, exist_ok=True)
    layout = {
        "scenario_root": scenario_root,
        "output": scenario_root / "output",
        "run_config": scenario_root / "run_config",
    }
    if include_splits:
        layout["splits"] = scenario_root / "splits"
    for name, path in layout.items():
        if name != "scenario_root":
            path.mkdir(parents=False, exist_ok=True)
    completed_marker = layout["run_config"] / "run_parameters.json"
    if completed_marker.exists():
        raise FileExistsError(
            f"This scenario already contains a completed run: {completed_marker}. "
            "Move the previous outputs before running it again."
        )
    return layout


def run_single(
    config_path: Path,
    cfg: dict[str, Any],
    prepare_only: bool = False,
    prepared: tuple[pd.DataFrame, dict[str, int], dict[str, Any], dict[str, Any]] | None = None,
    shared_splits: Path | None = None,
    scenario_root: Path | None = None,
    copy_source_yaml: bool = True,
) -> Path:
    layout = scenario_layout(
        config_path,
        scenario_root=scenario_root,
        include_splits=shared_splits is None,
    )
    htc_root = Path(cfg["paths"]["htc_root"])
    if not (htc_root / "htc" / "__init__.py").exists():
        raise FileNotFoundError(f"Original HTC source was not found at {htc_root}")
    run_dir = layout["output"]
    run_config_dir = layout["run_config"]
    if copy_source_yaml:
        shutil.copy2(config_path, run_config_dir / "source_config.yaml")
    (run_config_dir / "resolved_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    print(f"[1/3] Preparing median-spectrum CSV data in {run_dir}", flush=True)
    if prepared is None:
        manifest, mapping, split_summary, wavelength_info = prepare_data(cfg)
    else:
        manifest, mapping, split_summary, wavelength_info = prepared
        print("  Reusing the master stepwise subject split", flush=True)
    print("  Split assignment finished; writing manifests and workbooks...", flush=True)
    write_prepared(manifest, mapping, split_summary, wavelength_info, run_dir / "data")
    if shared_splits is None:
        write_split_workbooks(manifest, layout["splits"])
    if prepare_only:
        print(f"Preparation complete: {layout['scenario_root']}", flush=True)
        return layout["scenario_root"]

    print("[2/3] Training the HTC median-pixel model", flush=True)
    from median_pipeline.training import run_training
    run_training(cfg, run_dir, run_config_dir)
    print("[3/3] Evaluation and confusion matrices complete", flush=True)
    print(f"Scenario directory: {layout['scenario_root']}", flush=True)
    return layout["scenario_root"]


def run(config_path: Path, prepare_only: bool = False) -> Path:
    cfg = load_config(config_path)
    if isinstance(cfg["wavelength"].get("selective"), dict):
        from median_pipeline.selective import run_selective

        return run_selective(config_path, cfg, prepare_only, run_single)
    if cfg["stepwise_analysis"] or cfg["reverse_stepwise"]:
        from median_pipeline.stepwise import run_stepwise

        return run_stepwise(config_path, cfg, prepare_only, run_single)
    return run_single(config_path, cfg, prepare_only=prepare_only)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare, train, and evaluate the DKFZ median-spectrum HTC pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Run preparation, HTC training, and evaluation")
    run_parser.add_argument("--config", type=Path, required=True)
    prepare_parser = subparsers.add_parser("prepare", help="Prepare and validate splits without training")
    prepare_parser.add_argument("--config", type=Path, required=True)
    validate_parser = subparsers.add_parser("validate", help="Validate YAML structure and HTC path")
    validate_parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "validate":
        cfg = load_config(args.config)
        htc_root = Path(cfg["paths"]["htc_root"])
        if not (htc_root / "htc" / "__init__.py").exists():
            raise FileNotFoundError(f"Original HTC source was not found at {htc_root}")
        print("Configuration is valid")
        return
    run(args.config, prepare_only=args.command == "prepare")


if __name__ == "__main__":
    main()
