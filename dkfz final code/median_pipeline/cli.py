from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from median_pipeline.config import load_config
from median_pipeline.excel_outputs import write_split_workbooks
from median_pipeline.preparation import discover, split_manifest, write_prepared


def scenario_layout(config_path: Path) -> dict[str, Path]:
    config_path = config_path.resolve()
    if not config_path.is_file() or config_path.suffix.lower() not in {".yaml", ".yml"}:
        raise FileNotFoundError(f"Scenario YAML was not found: {config_path}")
    scenario_root = config_path.parent
    layout = {
        "scenario_root": scenario_root,
        "output": scenario_root / "output",
        "run_config": scenario_root / "run_config",
        "splits": scenario_root / "splits",
    }
    missing = [str(path) for name, path in layout.items() if name != "scenario_root" and not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            "The scenario must contain premade output, run_config, and splits folders. Missing: "
            + ", ".join(missing)
        )
    completed_marker = layout["run_config"] / "run_parameters.json"
    if completed_marker.exists():
        raise FileExistsError(
            f"This scenario already contains a completed run: {completed_marker}. "
            "Move the previous outputs before running it again."
        )
    return layout


def run(config_path: Path, prepare_only: bool = False) -> Path:
    layout = scenario_layout(config_path)
    cfg = load_config(config_path)
    htc_root = Path(cfg["paths"]["htc_root"])
    if not (htc_root / "htc" / "__init__.py").exists():
        raise FileNotFoundError(f"Original HTC source was not found at {htc_root}")
    run_dir = layout["output"]
    run_config_dir = layout["run_config"]
    shutil.copy2(config_path, run_config_dir / "source_config.yaml")
    (run_config_dir / "resolved_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    print(f"[1/3] Preparing median-spectrum CSV data in {run_dir}", flush=True)
    manifest, mapping, wavelength_info = discover(cfg)
    print("  Spectrum discovery finished; finding a subject-separated split...", flush=True)
    manifest, split_summary = split_manifest(manifest, cfg)
    print("  Split assignment finished; writing manifests and workbooks...", flush=True)
    write_prepared(manifest, mapping, split_summary, wavelength_info, run_dir / "data")
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
