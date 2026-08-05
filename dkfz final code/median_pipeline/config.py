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

    external = cfg.setdefault("external_testing", {"enabled": False})
    if not isinstance(external, dict):
        raise ConfigurationError("external_testing must be a YAML mapping")
    enabled = external.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ConfigurationError("external_testing.enabled must be true or false")
    external["enabled"] = enabled
    if enabled:
        required_external = {"experiment_folders", "labelling_file", "hyperguis"}
        missing_external = required_external - set(external)
        if missing_external:
            raise ConfigurationError(f"Missing external_testing sections: {sorted(missing_external)}")
        for section in required_external:
            if not isinstance(external[section], dict) or not external[section]:
                raise ConfigurationError(f"external_testing.{section} must be a non-empty mapping")
        external_keys = set(external["experiment_folders"])
        for section in ("labelling_file", "hyperguis"):
            section_keys = set(external[section])
            if section_keys != external_keys:
                raise ConfigurationError(
                    "The three external_testing class mappings must use identical keys; "
                    f"external_testing.{section} missing={sorted(external_keys - section_keys)}, "
                    f"extra={sorted(section_keys - external_keys)}"
                )

        resolved_external_folders = {}
        for label, folder in external["experiment_folders"].items():
            folder_path = Path(platform_value(folder))
            if not folder_path.is_absolute():
                raise ConfigurationError(
                    f"external_testing.experiment_folders.{label} must be an absolute path"
                )
            resolved_external_folders[label] = str(folder_path)
        external["experiment_folders"] = resolved_external_folders

        for section in ("labelling_file", "hyperguis"):
            for label, value in external[section].items():
                if not isinstance(value, str) or not value.strip():
                    raise ConfigurationError(
                        f"external_testing.{section}.{label} must be a non-empty filename or glob pattern"
                    )
                external[section][label] = value.strip()
    data = cfg["data"]
    if "spectrum_source" in data:
        raise ConfigurationError(
            "data.spectrum_source was replaced by the true/false data.use_l1pixel switch"
        )
    use_l1pixel = data.get("use_l1pixel")
    if "use_l1pixel" in data and not isinstance(use_l1pixel, bool):
        raise ConfigurationError("data.use_l1pixel must be true or false")
    if use_l1pixel is True:
        data["patterns"] = [
            "_L1pixel/spectrum_fromCSV1_masked_data_L1pixel_0_derivative.csv"
        ]
    elif use_l1pixel is False or "patterns" not in data:
        data["patterns"] = [
            "spectrum_fromCSV1_(500.0-995.0)_masked_data_0_derivative.csv",
            "spectrum_fromCSV1_(500.0-995.0)*_masked_data_0_derivative.csv",
        ]
    else:
        patterns = data.get("patterns")
        if not isinstance(patterns, list) or not patterns or not all(
            isinstance(pattern, str) and pattern.strip() for pattern in patterns
        ):
            raise ConfigurationError("data.patterns must be a non-empty list")
        data["patterns"] = [pattern.strip() for pattern in patterns]
    data["use_l1pixel"] = bool(use_l1pixel)
    data.setdefault("expected_channels", 100)
    data.setdefault("annotation_name", "csv_spectrum")

    wavelength = cfg.setdefault("wavelength", {"min": 500, "max": 995, "selective": None})
    if "min" not in wavelength or "max" not in wavelength:
        raise ConfigurationError("wavelength.min and wavelength.max are required")
    minimum, maximum = float(wavelength["min"]), float(wavelength["max"])
    if minimum >= maximum:
        raise ConfigurationError("wavelength.min must be smaller than wavelength.max")
    wavelength.setdefault("selective", None)
    selective = wavelength["selective"]
    if isinstance(selective, dict):
        if not selective:
            raise ConfigurationError("wavelength.selective batch mode must define at least one test case")
        selections = []
        normalized: dict[str, Any] = {}
        for raw_name, selection in selective.items():
            name = str(raw_name).strip()
            windows_unsafe = re.search(r'[<>:"/\\|?*\x00-\x1f]', name)
            if (
                not name
                or name in {".", ".."}
                or Path(name).name != name
                or windows_unsafe
                or name.endswith((" ", "."))
            ):
                raise ConfigurationError(
                    f"Invalid selective test-case name {raw_name!r}; use a single folder-safe name"
                )
            if name in normalized:
                raise ConfigurationError(f"Duplicate selective test-case name after normalization: {name!r}")
            normalized[name] = selection
            selections.append((name, selection))
        wavelength["selective"] = normalized
    else:
        selections = [(None, selective)]

    for case_name, selection in selections:
        try:
            intervals = parse_selective(selection)
        except WavelengthSelectionError as exc:
            prefix = f"Selective test case {case_name!r}: " if case_name is not None else ""
            raise ConfigurationError(prefix + str(exc)) from exc
        for lower, upper in intervals:
            if lower < minimum or upper > maximum:
                prefix = f"Selective test case {case_name!r}: " if case_name is not None else ""
                raise ConfigurationError(
                    f"{prefix}requested wavelength interval {lower:g}-{upper:g} "
                    f"lies outside {minimum:g}-{maximum:g}"
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
    if isinstance(wavelength["selective"], dict) and (cfg["stepwise_analysis"] or cfg["reverse_stepwise"]):
        raise ConfigurationError(
            "A wavelength.selective test-case mapping cannot be combined with forward or reverse stepwise analysis"
        )
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
