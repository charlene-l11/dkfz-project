# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import copy
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

from htc.cpp import map_label_image
from htc.models.image.DatasetImage import DatasetImage
from htc.settings import settings
from htc.settings_seg import settings_seg
from htc.tivita.DataPath import DataPath
from htc.utils.Config import Config
from htc.utils.helper_functions import sort_labels
from htc.utils.LabelMapping import LabelMapping


class LocationMap:
    def __init__(self, paths: list[DataPath], label_mapping: LabelMapping):
        """
        Helper class to create location maps for a set of paths.

        Args:
            paths: List of paths which should be considered for the location maps.
            label_mapping: Label mapping which defines which labels should be included and which colors should be used.
        """
        self.paths = paths

        self.label_mapping = label_mapping
        config = Config({
            "label_mapping": self.label_mapping,
            "input/no_features": True,
        })
        self.dataset = DatasetImage(self.paths, train=None, config=config)
        self.counts = None
        self.overlaps = None

    @staticmethod
    def _single_dim_sample(sample: dict[str, Any]) -> dict[str, Any]:
        if sample["labels"].ndim == 3:
            settings.log_once.info(
                "Multi-dimensional labels detected. This is currently not supported by LocationMap. Only the first dimension will be used."
            )
            sample["labels"] = sample["labels"][0]
            sample["valid_pixels"] = sample["valid_pixels"][0]

        return sample

    def compute_map(self) -> None:
        dsettings = self.paths[0].dataset_settings
        self.counts = torch.zeros(len(self.label_mapping), *dsettings["spatial_shape"], dtype=torch.int64)

        for sample in self.dataset:
            sample = self._single_dim_sample(sample)
            for label in sample["labels"][sample["valid_pixels"]].unique():
                self.counts[label, sample["labels"] == label] += 1

        # Find the image with the biggest overlap between the segmentation and the global counts (used to show an example image)
        self.overlaps = torch.zeros(len(self.label_mapping), len(self.paths), dtype=torch.int64)
        for i, sample in enumerate(self.dataset):
            sample = self._single_dim_sample(sample)
            for label in sample["labels"][sample["valid_pixels"]].unique():
                self.overlaps[label, i] = self.counts[label, sample["labels"] == label].sum()

        self.overlaps = self.overlaps.argmax(dim=1)

    def create_figure(self, figheigt_factor: float = 1.2) -> plt.Figure:
        n_outer_cols = 2
        n_rows = int(np.ceil(len(self.label_mapping) / n_outer_cols))
        n_cols = 3 * n_outer_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * figheigt_factor), constrained_layout=True)
        if axes.ndim == 1:
            axes = np.expand_dims(axes, axis=0)

        label_indices = {self.label_mapping.index_to_name(i): i for i in range(len(self.label_mapping))}
        label_indices = sort_labels(label_indices)
        for i, (label_name, label_index) in enumerate(label_indices.items()):
            col_idx = int(i // (len(self.label_mapping) / n_outer_cols))
            col_shift = col_idx * 3
            row = int(i % (len(self.label_mapping) / n_outer_cols))

            cmap = LinearSegmentedColormap.from_list(
                label_name, ["#ffffff", self.label_mapping.index_to_color(label_index)]
            )
            axes[row, col_shift].imshow(self.counts[label_index].numpy(), cmap=cmap)
            axes[row, col_shift].axis("off")

            current_label_mapping = copy.deepcopy(self.label_mapping)
            for name, color in current_label_mapping.label_colors.items():
                color = color[:7]  # In case the color already contains alpha --> ignore it
                if name != label_name:
                    current_label_mapping.label_colors[name] = f"{color}22"
                else:
                    current_label_mapping.label_colors[name] = f"{color}ff"

            sample = self.dataset[self.overlaps[label_index]]
            sample = self._single_dim_sample(sample)
            seg = map_label_image(sample["labels"], current_label_mapping)
            axes[row, col_shift + 1].imshow(seg.numpy(), cmap=cmap)
            axes[row, col_shift + 1].axis("off")
            y_shift = 0.71 if label_name != "kidney_with_Gerotas_fascia" else 0.51
            axes[row, col_shift + 1].set_title(
                settings_seg.labels_paper_renaming.get(label_name, label_name).replace("<br>", "\n"),
                fontsize=9.3,
                y=y_shift,
            )

            rgb = self.paths[self.overlaps[label_index]].read_rgb_reconstructed()
            axes[row, col_shift + 2].imshow(rgb, cmap=cmap)
            axes[row, col_shift + 2].axis("off")

        # Disable unused label slots
        max_labels = n_rows * n_outer_cols
        for label_index in range(len(self.label_mapping), max_labels):
            col_idx = int(label_index // (max_labels / n_outer_cols))
            col_shift = col_idx * 3
            row = int(label_index % (max_labels / n_outer_cols))

            axes[row, col_shift].axis("off")
            axes[row, col_shift + 1].axis("off")
            axes[row, col_shift + 2].axis("off")

        return fig
