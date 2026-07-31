from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from median_pipeline.wavelengths import resolve_wavelength_selection


def _status(message: str) -> None:
    print(message, flush=True)


def read_spectrum_with_wavelengths(path: Path) -> tuple[np.ndarray | None, np.ndarray]:
    """Read a one-column spectrum or a two-column wavelength/value CSV."""
    try:
        frame = pd.read_csv(path, sep=None, engine="python", header=None)
    except Exception:
        frame = pd.read_csv(path, header=None)
    numeric = frame.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="all").dropna(axis=1, how="all")
    if numeric.empty:
        raise ValueError(f"No numeric spectrum found in {path}")
    numeric = numeric.dropna(axis=0, how="any")
    if numeric.empty:
        raise ValueError(f"No complete numeric spectrum rows found in {path}")
    if numeric.shape[1] >= 2:
        candidate = numeric.iloc[:, 0].to_numpy(dtype=np.float64, copy=True)
        values = numeric.iloc[:, 1].to_numpy(dtype=np.float32, copy=True)
        wavelengths = candidate if len(candidate) < 2 or np.all(np.diff(candidate) > 0) else None
        return wavelengths, values
    return None, numeric.iloc[:, 0].to_numpy(dtype=np.float32, copy=True)


def read_spectrum(path: Path) -> np.ndarray:
    return read_spectrum_with_wavelengths(path)[1]


def subject_from_path(path: Path) -> str:
    for part in path.parts:
        match = re.match(r"^([PM]\d+)", part)
        if match:
            return match.group(1)
    for part in path.parts:
        if re.match(r"^Cat_\d+_", part):
            return part
    raise ValueError(f"Could not identify a subject from {path}")


def timestamp_from_path(path: Path) -> str:
    return next((p for p in path.parts if re.match(r"^\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}$", p)), path.parent.name)


def individual_from_path(path: Path) -> str:
    return next((part for part in path.parts if re.match(r"^[PM]\d+_.*Experiment\d+$", part)), subject_from_path(path))


def discover(cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, int], dict[str, Any]]:
    patterns = list(cfg["data"]["patterns"])
    expected = int(cfg["data"]["expected_channels"])
    annotation = str(cfg["data"]["annotation_name"])
    rows: list[dict[str, Any]] = []
    owner: dict[str, str] = {}
    wavelength_info: dict[str, Any] | None = None

    sources = list(cfg["experiment_folders"].items())
    _status(f"  Discovering spectra for {len(sources)} configured organs")
    for organ_number, (label, folder_value) in enumerate(sources, start=1):
        folder = Path(str(folder_value))
        hypergui_pattern = str(cfg["hyperguis"][label])
        labelling_pattern = str(cfg["labelling_file"][label])
        _status(f"  [{organ_number}/{len(sources)}] Scanning {label}: {folder}")
        if not folder.exists():
            raise FileNotFoundError(f"Source folder for {label!r} does not exist: {folder}")
        hypergui_dirs = sorted({
            path.resolve()
            for path in folder.rglob(hypergui_pattern)
            if path.is_dir()
        })
        if not hypergui_dirs:
            raise FileNotFoundError(
                f"No HyperGUI folders matching {hypergui_pattern!r} found for {label!r} below {folder}"
            )
        paths = sorted({
            path.resolve()
            for hypergui_dir in hypergui_dirs
            for pattern in patterns
            for path in hypergui_dir.rglob(pattern)
            if path.is_file()
        })
        if not paths:
            raise FileNotFoundError(
                f"No configured median-spectrum CSV files found for {label!r} in HyperGUI folders "
                f"matching {hypergui_pattern!r}"
            )
        _status(f"      Found {len(paths):,} files; validating spectra and wavelengths...")
        for file_number, path in enumerate(paths, start=1):
            key = str(path)
            if key in owner and owner[key] != str(label):
                raise ValueError(f"Spectrum belongs to multiple classes: {path} ({owner[key]}, {label})")
            owner[key] = str(label)
            csv_wavelengths, spectrum = read_spectrum_with_wavelengths(path)
            if len(spectrum) != expected:
                raise ValueError(f"Expected {expected} input channels but found {len(spectrum)} in {path}")
            if not np.isfinite(spectrum).all():
                raise ValueError(f"Non-finite values in {path}")
            current = resolve_wavelength_selection(cfg["wavelength"], len(spectrum), csv_wavelengths)
            if wavelength_info is None:
                wavelength_info = current
            else:
                if current["selected_indices"] != wavelength_info["selected_indices"]:
                    raise ValueError(f"Inconsistent wavelength grid in {path}")
                if not np.allclose(current["available_wavelengths"], wavelength_info["available_wavelengths"]):
                    raise ValueError(f"Inconsistent available wavelengths in {path}")
            subject = subject_from_path(path)
            individual = individual_from_path(path)
            timestamp = timestamp_from_path(path)
            sample_dir = path.parent.parent
            labelling_matches = sorted(candidate.resolve() for candidate in sample_dir.glob(labelling_pattern) if candidate.is_file())
            if not labelling_matches:
                raise FileNotFoundError(
                    f"No labelling file matching {labelling_pattern!r} beside HyperGUI folder {path.parent}"
                )
            rows.append({
                "file_path": key,
                "label": str(label),
                "subject_name": subject,
                "individual_name": individual,
                "timestamp": timestamp,
                "image_name": f"{subject}#{timestamp}#{path.stem}",
                "annotation_name": annotation,
                "sample_dir": str(sample_dir),
                "hypergui_dir": str(path.parent),
                "labelling_file": str(labelling_matches[0]),
            })
            if file_number % 100 == 0 or file_number == len(paths):
                _status(f"      Validated {file_number:,}/{len(paths):,} files for {label}")

    if wavelength_info is None:
        raise ValueError("No spectra were discovered")
    frame = pd.DataFrame(rows).drop_duplicates("file_path").reset_index(drop=True)
    labels = list(dict.fromkeys(frame["label"].tolist()))
    _status(f"  Discovery complete: {len(frame):,} spectra, {frame['subject_name'].nunique():,} subjects, {len(labels)} organs")
    return frame, {label: index for index, label in enumerate(labels)}, wavelength_info


def split_manifest(frame: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    split_cfg = cfg["splitting"]
    ratios = {"train": float(split_cfg["train_ratio"]), "val": float(split_cfg["val_ratio"]), "test": float(split_cfg["test_ratio"])}
    required = list(split_cfg["require_all_classes_in"])
    subjects = sorted(frame["subject_name"].astype(str).unique())
    labels = sorted(frame["label"].astype(str).unique())
    label_subjects = {label: sorted(frame.loc[frame["label"].eq(label), "subject_name"].astype(str).unique()) for label in labels}
    for label, available in label_subjects.items():
        if len(available) < len(required):
            raise ValueError(f"Class {label!r} occurs in only {len(available)} subjects; it cannot cover required splits {required}. Subjects: {available}")
    if len(subjects) < 3:
        raise ValueError("At least three subjects are required")
    n_val = max(1, round(len(subjects) * ratios["val"]))
    n_test = max(1, round(len(subjects) * ratios["test"]))
    n_train = len(subjects) - n_val - n_test
    if n_train < 1:
        raise ValueError("Split ratios leave no training subjects")

    attempts = int(split_cfg["search_attempts"])
    report_every = max(1, attempts // 10)
    _status(f"  Searching up to {attempts:,} subject-level assignments ({n_train} train, {n_val} validation, {n_test} test subjects)")
    rng = random.Random(int(split_cfg["seed"]))
    best: tuple[float, dict[str, str]] | None = None
    for attempt in range(1, attempts + 1):
        shuffled = subjects.copy()
        rng.shuffle(shuffled)
        assignment = {s: "train" for s in shuffled[:n_train]}
        assignment.update({s: "val" for s in shuffled[n_train:n_train + n_val]})
        assignment.update({s: "test" for s in shuffled[n_train + n_val:]})
        missing = 0
        imbalance = 0.0
        for name in ("train", "val", "test"):
            subset = frame[frame["subject_name"].map(assignment).eq(name)]
            if name in required:
                missing += len(set(labels) - set(subset["label"].astype(str)))
            counts = subset["label"].value_counts()
            if len(counts):
                imbalance += float(counts.std(ddof=0) / max(counts.mean(), 1))
        score = missing * 1_000_000 + imbalance
        if best is None or score < best[0]:
            best = (score, assignment)
            if score == 0:
                break
        if attempt % report_every == 0:
            best_missing = int(best[0] // 1_000_000) if best is not None else -1
            _status(f"      Checked {attempt:,}/{attempts:,} assignments; best missing-class count: {best_missing}")
    assert best is not None
    if best[0] >= 1_000_000:
        detail = {label: available for label, available in label_subjects.items()}
        raise ValueError(f"Could not find a subject-separated split containing every class in all required splits. Class subjects: {detail}")

    result = frame.copy()
    result["split"] = result["subject_name"].map(best[1])
    summary: dict[str, Any] = {"seed": int(split_cfg["seed"]), "ratios": ratios, "splits": {}}
    for name in ("train", "val", "test"):
        subset = result[result["split"].eq(name)]
        summary["splits"][name] = {
            "subjects": sorted(subset["subject_name"].astype(str).unique()),
            "rows": int(len(subset)),
            "class_counts": {str(k): int(v) for k, v in subset["label"].value_counts().sort_index().items()},
        }
        _status(f"  {name.capitalize()}: {subset['subject_name'].nunique():,} subjects, {len(subset):,} spectra")
    return result, summary


def write_prepared(
    frame: pd.DataFrame,
    mapping: dict[str, int],
    summary: dict[str, Any],
    wavelength_info: dict[str, Any],
    data_dir: Path,
) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    out = frame.copy()
    out["label_index"] = out["label"].map(mapping).astype(int)
    columns = ["file_path", "label", "label_index", "subject_name", "timestamp", "image_name", "annotation_name"]
    out[columns].to_csv(data_dir / "manifest.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    for name, filename in (("train", "train.csv"), ("val", "validation.csv"), ("test", "test.csv")):
        out.loc[out["split"].eq(name), columns].to_csv(data_dir / filename, index=False)
    labels = [label for label, _ in sorted(mapping.items(), key=lambda item: item[1])]
    (data_dir / "labels.json").write_text(json.dumps({"labels": labels, "mapping": mapping}, indent=2), encoding="utf-8")
    (data_dir / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (data_dir / "wavelengths.json").write_text(json.dumps(wavelength_info, indent=2), encoding="utf-8")
    digest = hashlib.sha256((data_dir / "manifest.csv").read_bytes()).hexdigest()
    (data_dir / "manifest.sha256").write_text(digest + "\n", encoding="utf-8")
    _status("  CSV manifests and split metadata written")
