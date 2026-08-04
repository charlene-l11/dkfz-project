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

    if "htc_root" not in cfg["paths"]:
        raise ConfigurationError("Missing paths.htc_root")
    cfg["paths"]["htc_root"] = platform_value(cfg["paths"]["htc_root"])

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
    resolved_experiment_folders = {}
    for label, folder in cfg["experiment_folders"].items():
        folder_path = Path(platform_value(folder))
        if not folder_path.is_absolute():
            raise ConfigurationError(
                f"experiment_folders.{label} must be an absolute path; paths.data_root is no longer used"
            )
        resolved_experiment_folders[label] = str(folder_path)
    cfg["experiment_folders"] = resolved_experiment_folders
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

    # Accept both the original internal names and the clearer public YAML names.
    # Normalize both pairs so existing master files remain compatible.
    stepwise_names = (
        ("stepwise_analysis", "forward_stepwise_analysis"),
        ("reverse_stepwise", "reverse_stepwise_analysis"),
    )
    for internal_name, public_name in stepwise_names:
        if internal_name in cfg and public_name in cfg and cfg[internal_name] != cfg[public_name]:
            raise ConfigurationError(
                f"{internal_name} and {public_name} disagree; keep only one name or give both the same value"
            )
        value = cfg.get(public_name, cfg.get(internal_name, False))
        if not isinstance(value, bool):
            raise ConfigurationError(f"{public_name} must be true or false")
        cfg[internal_name] = value
        cfg[public_name] = value
    if cfg["stepwise_analysis"] or cfg["reverse_stepwise"]:
        if cfg["stepwise_analysis"]:
            cfg.setdefault("forward_stop_wavelength", maximum)
            forward_stop = float(cfg["forward_stop_wavelength"])
            if not minimum < forward_stop <= maximum:
                raise ConfigurationError(
                    f"forward_stop_wavelength must be greater than {minimum:g} and at most {maximum:g}"
                )
            cfg["forward_stop_wavelength"] = forward_stop
        if cfg["reverse_stepwise"]:
            cfg.setdefault("reverse_stop_wavelength", minimum)
            reverse_stop = float(cfg["reverse_stop_wavelength"])
            if not minimum <= reverse_stop < maximum:
                raise ConfigurationError(
                    f"reverse_stop_wavelength must be at least {minimum:g} and below {maximum:g}"
                )
            cfg["reverse_stop_wavelength"] = reverse_stop

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
