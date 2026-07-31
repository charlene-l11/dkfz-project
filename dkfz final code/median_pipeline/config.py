from __future__ import annotations

import copy
import os
import platform
import re
from pathlib import Path
from typing import Any

import yaml

from median_pipeline.wavelengths import WavelengthSelectionError, parse_selective


class ConfigurationError(ValueError):
    pass


def _expand(value: str) -> str:
    for name in re.findall(r"\$\{([^}]+)\}", value):
        if name not in os.environ:
            raise ConfigurationError(f"Environment variable {name!r} is required by the YAML configuration")
    return os.path.expandvars(os.path.expanduser(value))


def platform_value(value: Any) -> Any:
    if isinstance(value, dict) and ("windows" in value or "linux" in value):
        key = "windows" if os.name == "nt" else "linux"
        if key not in value:
            raise ConfigurationError(f"Missing paths.{key} value for {platform.system()}")
        value = value[key]
    return _expand(value) if isinstance(value, str) else value


def load_config(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigurationError("The YAML root must be a mapping")

    cfg = copy.deepcopy(raw)
    required = {
        "experiment",
        "paths",
        "experiment_folders",
        "labelling_file",
        "hyperguis",
        "data",
        "splitting",
        "training",
        "evaluation",
    }
    missing = required - set(cfg)
    if missing:
        raise ConfigurationError(f"Missing top-level YAML sections: {sorted(missing)}")

    for key in ("htc_root", "runs_root"):
        if key not in cfg["paths"]:
            raise ConfigurationError(f"Missing paths.{key}")
        cfg["paths"][key] = platform_value(cfg["paths"][key])

    experiment_name = str(cfg["experiment"].get("name", "")).strip()
    if not experiment_name:
        raise ConfigurationError("experiment.name must not be empty")
    cfg["experiment"]["name"] = experiment_name

    input_sections = ("experiment_folders", "labelling_file", "hyperguis")
    for section in input_sections:
        if not isinstance(cfg[section], dict) or len(cfg[section]) < 2:
            raise ConfigurationError(f"{section} must define at least two organ mappings")
    expected_keys = set(cfg["experiment_folders"])
    for section in input_sections[1:]:
        section_keys = set(cfg[section])
        if section_keys != expected_keys:
            missing_keys = sorted(expected_keys - section_keys)
            extra_keys = sorted(section_keys - expected_keys)
            raise ConfigurationError(
                f"{section} keys must exactly match experiment_folders; "
                f"missing={missing_keys}, extra={extra_keys}"
            )
    cfg["experiment_folders"] = {
        label: platform_value(folder) for label, folder in cfg["experiment_folders"].items()
    }
    for section in ("labelling_file", "hyperguis"):
        for label, value in cfg[section].items():
            if not isinstance(value, str) or not value.strip():
                raise ConfigurationError(f"{section}.{label} must be a non-empty filename or glob pattern")
            cfg[section][label] = value.strip()
    cfg["data"].setdefault("patterns", ["spectrum_fromCSV1_(500.0-995.0)_masked_data_0_derivative.csv"])
    cfg["data"].setdefault("expected_channels", 100)
    cfg["data"].setdefault("annotation_name", "csv_spectrum")

    wavelength = cfg.setdefault("wavelength", {"min": 500, "max": 995, "selective": None})
    if "min" not in wavelength or "max" not in wavelength:
        raise ConfigurationError("wavelength.min and wavelength.max are required")
    minimum, maximum = float(wavelength["min"]), float(wavelength["max"])
    if minimum >= maximum:
        raise ConfigurationError("wavelength.min must be smaller than wavelength.max")
    wavelength.setdefault("selective", None)
    try:
        intervals = parse_selective(wavelength["selective"])
    except WavelengthSelectionError as exc:
        raise ConfigurationError(str(exc)) from exc
    for lower, upper in intervals:
        if lower < minimum or upper > maximum:
            raise ConfigurationError(
                f"Requested wavelength interval {lower:g}-{upper:g} lies outside {minimum:g}-{maximum:g}"
            )

    split = cfg["splitting"]
    ratios = [float(split.get(k, 0)) for k in ("train_ratio", "val_ratio", "test_ratio")]
    if any(r <= 0 for r in ratios) or abs(sum(ratios) - 1.0) > 1e-8:
        raise ConfigurationError("Split ratios must be positive and sum to 1.0")
    split.setdefault("seed", cfg["training"].get("seed", 0))
    split.setdefault("search_attempts", 10000)
    split.setdefault("require_all_classes_in", ["train", "val", "test"])

    training = cfg["training"]
    for key in ("seed", "max_epochs", "batch_size", "learning_rate"):
        if key not in training:
            raise ConfigurationError(f"Missing training.{key}")
    imbalance = training.setdefault("imbalance", {"strategy": "none"})
    strategy = str(imbalance.get("strategy", "none"))
    allowed = {"none", "class_weighting", "balanced_oversampling"}
    if strategy not in allowed:
        raise ConfigurationError(f"training.imbalance.strategy must be one of {sorted(allowed)}")
    if strategy == "class_weighting":
        method = str(imbalance.get("method", "inverse"))
        if method not in {"inverse", "balanced", "softmin", "nll"}:
            raise ConfigurationError("class-weighting method must be inverse, balanced, softmin, or nll")
    if strategy == "balanced_oversampling" and "method" in imbalance:
        raise ConfigurationError("Do not set a class-weighting method with balanced_oversampling")

    training.setdefault("accelerator", "auto")
    training.setdefault("devices", 1)
    training.setdefault("precision", "32-true")
    training.setdefault("num_workers", 0)
    training.setdefault("standardize", False)
    training.setdefault("checkpoint_metric", "accuracy")
    if training["checkpoint_metric"] != "accuracy":
        raise ConfigurationError("The original HTC median-pixel module exposes accuracy as its checkpoint metric")
    cfg["evaluation"].setdefault("formats", ["csv", "png", "pdf"])
    cfg["evaluation"].setdefault("dpi", 300)
    cfg["_source_yaml"] = str(path.resolve())
    return cfg


