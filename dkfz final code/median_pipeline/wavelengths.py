from __future__ import annotations

import re
from typing import Any

import numpy as np


class WavelengthSelectionError(ValueError):
    pass


def scatter_selected_features(
    features: np.ndarray,
    selected_indices: list[int] | np.ndarray,
    n_input_channels: int,
) -> np.ndarray:
    """Scatter selected features into their original positions in a fixed-width tensor."""
    values = np.asarray(features)
    indices = np.asarray(selected_indices, dtype=np.int64)
    if values.ndim != 2:
        raise ValueError("Selected features must be a two-dimensional sample-by-channel array")
    if indices.ndim != 1 or len(indices) != values.shape[1]:
        raise ValueError("Selected indices must contain one entry per selected feature channel")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("Selected wavelength indices must be unique")
    if n_input_channels <= 0 or (len(indices) > 0 and (indices.min() < 0 or indices.max() >= n_input_channels)):
        raise ValueError("A selected wavelength index lies outside the fixed-width model input")

    masked = np.zeros((values.shape[0], n_input_channels), dtype=values.dtype)
    masked[:, indices] = values
    return masked


def parse_selective(value: Any) -> list[tuple[float, float]]:
    """Parse values such as ``500; 560-720; 875; 900-930``."""
    if value is None or value == "":
        return []
    if isinstance(value, (int, float)):
        number = float(value)
        return [(number, number)]
    if isinstance(value, list):
        tokens = value
    elif isinstance(value, str):
        if value.strip().lower() in {"all", "full", "none"}:
            return []
        tokens = [token.strip() for token in re.split(r"[;,]", value) if token.strip()]
    else:
        raise WavelengthSelectionError("wavelength.selective must be a number, string, list, or null")

    intervals: list[tuple[float, float]] = []
    number_pattern = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
    for token in tokens:
        if isinstance(token, (int, float)):
            number = float(token)
            intervals.append((number, number))
            continue
        text = str(token).strip()
        single = re.fullmatch(fr"({number_pattern})", text)
        if single:
            number = float(single.group(1))
            intervals.append((number, number))
            continue
        range_match = re.fullmatch(fr"({number_pattern})\s*-\s*({number_pattern})", text)
        if not range_match:
            raise WavelengthSelectionError(
                f"Invalid wavelength token {text!r}; use values/ranges such as '500; 560-720; 875'"
            )
        lower, upper = float(range_match.group(1)), float(range_match.group(2))
        if lower > upper:
            raise WavelengthSelectionError(f"Wavelength range starts above its end: {text!r}")
        intervals.append((lower, upper))
    return intervals


def resolve_wavelength_selection(
    configured: dict[str, Any],
    n_channels: int,
    csv_wavelengths: np.ndarray | None = None,
) -> dict[str, Any]:
    minimum = float(configured["min"])
    maximum = float(configured["max"])
    if minimum >= maximum:
        raise WavelengthSelectionError("wavelength.min must be smaller than wavelength.max")

    if csv_wavelengths is None:
        available = np.linspace(minimum, maximum, n_channels, dtype=np.float64)
        source = "inferred_from_min_max_and_channel_count"
    else:
        available = np.asarray(csv_wavelengths, dtype=np.float64)
        source = "csv_wavelength_column"
        if len(available) != n_channels:
            raise WavelengthSelectionError("The CSV wavelength and feature columns have different lengths")
        if not np.all(np.diff(available) > 0):
            raise WavelengthSelectionError("CSV wavelengths must be strictly increasing")
        tolerance = max(1e-6, float(np.median(np.diff(available))) * 0.01)
        if available.max() < minimum - tolerance or available.min() > maximum + tolerance:
            available = np.linspace(minimum, maximum, n_channels, dtype=np.float64)
            source = "inferred_from_min_max_and_channel_count"

    base = (available >= minimum - 1e-8) & (available <= maximum + 1e-8)
    intervals = parse_selective(configured.get("selective"))
    if intervals:
        selected_mask = np.zeros(n_channels, dtype=bool)
        spacing = float(np.median(np.diff(available))) if n_channels > 1 else 1.0
        tolerance = max(1e-6, spacing * 0.01)
        for lower, upper in intervals:
            if lower == upper:
                matches = np.isclose(available, lower, atol=tolerance, rtol=0)
                if not matches.any():
                    nearest = float(available[np.abs(available - lower).argmin()])
                    raise WavelengthSelectionError(
                        f"Requested wavelength {lower:g} is not on the CSV grid; nearest available value is {nearest:g}"
                    )
            else:
                matches = (available >= lower - tolerance) & (available <= upper + tolerance)
                if not matches.any():
                    raise WavelengthSelectionError(f"Requested range {lower:g}-{upper:g} contains no available wavelengths")
            selected_mask |= matches
        selected_mask &= base
    else:
        selected_mask = base

    indices = np.flatnonzero(selected_mask)
    if len(indices) == 0:
        raise WavelengthSelectionError("The configured wavelength selection produced zero channels")
    selected = available[indices]
    return {
        "configured_min": minimum,
        "configured_max": maximum,
        "selective": configured.get("selective"),
        "grid_source": source,
        "available_wavelengths": [float(value) for value in available],
        "selected_indices": [int(index) for index in indices],
        "selected_wavelengths": [float(value) for value in selected],
        "n_input_channels": int(n_channels),
        "n_selected_channels": int(len(indices)),
    }
