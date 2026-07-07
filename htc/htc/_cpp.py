"""Small Python fallback for the optional HTC C++ extension.

This keeps lightweight tabular/spectral training paths usable on machines
without Microsoft C++ Build Tools. Functions which need the real extension
still raise a clear error.
"""

from __future__ import annotations

import torch


def _missing_extension(name: str):
    raise ImportError(
        f"htc._cpp.{name} requires the compiled HTC extension. "
        "Install Microsoft C++ Build Tools and run `python -m pip install -e htc` "
        "to use this function."
    )


def tensor_mapping_integer(tensor: torch.Tensor, mapping: dict[int, int]) -> torch.Tensor:
    for old, new in mapping.items():
        tensor[tensor == old] = new
    return tensor


def tensor_mapping_floating(tensor: torch.Tensor, mapping: dict[float, float]) -> torch.Tensor:
    for old, new in mapping.items():
        tensor[tensor == old] = new
    return tensor


def nunique(inp: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    if dim is None:
        return torch.as_tensor(inp.unique().numel(), dtype=torch.int64, device=inp.device)

    return torch.stack([row.unique().numel() for row in inp.transpose(0, dim).reshape(inp.shape[dim], -1)])


def spxs_predictions(*args, **kwargs):
    _missing_extension("spxs_predictions")


def segmentation_mask(*args, **kwargs):
    _missing_extension("segmentation_mask")


def kfold_combinations(*args, **kwargs):
    _missing_extension("kfold_combinations")


def map_label_image(*args, **kwargs):
    _missing_extension("map_label_image")


def hierarchical_bootstrapping(*args, **kwargs):
    _missing_extension("hierarchical_bootstrapping")


def hierarchical_bootstrapping_labels(*args, **kwargs):
    _missing_extension("hierarchical_bootstrapping_labels")


def colorchecker_automask(*args, **kwargs):
    _missing_extension("colorchecker_automask")


def colorchecker_automask_search_area(*args, **kwargs):
    _missing_extension("colorchecker_automask_search_area")
