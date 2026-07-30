from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
from pathlib import Path

from median_pipeline.config import load_config
from median_pipeline.excel_outputs import write_split_workbooks
from median_pipeline.preparation import discover, split_manifest, write_prepared


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "experiment"


def create_run_dir(cfg: dict) -> Path:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(cfg["paths"]["runs_root"]) / f"{timestamp}_{safe_name(str(cfg['experiment']['name']))}"
    path.mkdir(parents=True, exist_ok=False)
    for folder in ("config", "data", "splits", "checkpoints", "logs", "predictions", "matrices/csv", "matrices/png", "matrices/pdf", "metrics"):
        (path / folder).mkdir(parents=True, exist_ok=True)
    return path


def run(config_path: Path, prepare_only: bool = False) -> Path:
    cfg = load_config(config_path)
    htc_root = Path(cfg["paths"]["htc_root"])
    if not (htc_root / "htc" / "__init__.py").exists():
        raise FileNotFoundError(f"Original HTC source was not found at {htc_root}")
    run_dir = create_run_dir(cfg)
    shutil.copy2(config_path, run_dir / "config" / "source_config.yaml")
    (run_dir / "config" / "resolved_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    print(f"[1/3] Preparing median-spectrum CSV data in {run_dir}", flush=True)
    manifest, mapping, wavelength_info = discover(cfg)
    print("  Spectrum discovery finished; finding a subject-separated split...", flush=True)
    manifest, split_summary = split_manifest(manifest, cfg)
    print("  Split assignment finished; writing manifests and workbooks...", flush=True)
    write_prepared(manifest, mapping, split_summary, wavelength_info, run_dir / "data")
    write_split_workbooks(manifest, run_dir / "splits")
    if prepare_only:
        print(f"Preparation complete: {run_dir}", flush=True)
        return run_dir

    print("[2/3] Training the HTC median-pixel model", flush=True)
    from median_pipeline.training import run_training
    run_training(cfg, run_dir)
    print("[3/3] Evaluation and confusion matrices complete", flush=True)
    print(f"Run directory: {run_dir}", flush=True)
    return run_dir


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
