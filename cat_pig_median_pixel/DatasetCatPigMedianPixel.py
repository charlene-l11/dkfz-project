from __future__ import annotations

import re
from dataclasses import dataclass
from math import isnan
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch

from htc.models.common.HTCDataset import HTCDataset
from htc.tivita.DataPath import DataPath
from htc.utils.Config import Config
from htc.utils.Task import Task


@dataclass(frozen=True)
class CatPigSpectrumPath(DataPath):
    file_path: Path
    image_name_value: str
    annotation_name_value: str = "cat_pig_csv"

    def image_name(self) -> str:
        return self.image_name_value

    def image_name_annotations(self) -> str:
        return f"{self.image_name_value}@{self.annotation_name_value}"


class DatasetCatPigMedianPixel(HTCDataset):
    def __init__(
        self,
        paths: list[CatPigSpectrumPath | str | Path] | None = None,
        train: bool = False,
        config: Config | None = None,
        fold_name: str | None = None,
    ):
        if config is None:
            config = Config({})

        self.manifest = self._load_manifest(paths, config)
        self.spectrum_paths = [
            CatPigSpectrumPath(
                file_path=self._path_from_value(row.get("file_path", f"row_{index}.csv")),
                image_name_value=str(row["image_name"]),
                annotation_name_value=str(row.get("annotation_name", "cat_pig_csv")),
            )
            for index, row in enumerate(self.manifest.to_dict("records"))
        ]

        base_paths: list[DataPath] = list(self.spectrum_paths)
        super().__init__(paths=base_paths, train=train, config=config, fold_name=fold_name or "cat_pig")

        feature_columns = config.get("input/feature_columns")
        if feature_columns:
            missing = set(feature_columns) - set(self.manifest.columns)
            if missing:
                raise ValueError(f"Manifest is missing feature columns: {sorted(missing)}")
            features = self.manifest[feature_columns].to_numpy(dtype=np.float32, copy=True)
        else:
            features = np.stack([self._read_spectrum(p.file_path) for p in self.spectrum_paths])

        self.features = torch.from_numpy(features)

        feature_mean = config.get("input/feature_mean")
        feature_std = config.get("input/feature_std")
        if feature_mean is not None and feature_std is not None:
            mean = torch.as_tensor(feature_mean, dtype=self.features.dtype)
            std = torch.as_tensor(feature_std, dtype=self.features.dtype)
            std = torch.where(std == 0, torch.ones_like(std), std)
            self.features = (self.features - mean) / std

        self.features = cast(torch.Tensor, self.apply_transforms(self.features))

        self.labels = self._read_labels(self.manifest, config)
        self.image_labels = None
        self.meta = None

        if self.labels is not None:
            assert len(self.labels) == len(self.features), "Labels and features must have the same length"
        assert len(self.features) == len(self.paths), "Features and paths must have the same length"

    @staticmethod
    def _load_manifest(paths: list[CatPigSpectrumPath | str | Path] | None, config: Config) -> pd.DataFrame:
        manifest_path = config.get("input/cat_pig_manifest")
        if manifest_path:
            df = pd.read_csv(manifest_path)
        elif paths is not None:
            rows = []
            for item in paths:
                if isinstance(item, CatPigSpectrumPath):
                    file_path = item.file_path
                    image_name = item.image_name()
                    annotation_name = item.annotation_name_value
                else:
                    file_path = DatasetCatPigMedianPixel._path_from_value(item)
                    image_name = DatasetCatPigMedianPixel.image_name_from_file(file_path)
                    annotation_name = "cat_pig_csv"

                rows.append({
                    "file_path": str(file_path),
                    "image_name": image_name,
                    "annotation_name": annotation_name,
                })
            df = pd.DataFrame(rows)
        else:
            raise ValueError("Provide either paths or config['input/cat_pig_manifest']")

        feature_columns = config.get("input/feature_columns")
        required = {"image_name"}
        if not feature_columns:
            required.add("file_path")

        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

        if "annotation_name" not in df.columns:
            df["annotation_name"] = "cat_pig_csv"

        if "file_path" in df.columns:
            df["file_path"] = df["file_path"].map(
                lambda p: str(DatasetCatPigMedianPixel._path_from_value(p).expanduser())
            )

        if not feature_columns:
            missing_files = [p for p in df["file_path"] if not Path(p).exists()]
            if missing_files:
                preview = "\n".join(missing_files[:10])
                raise FileNotFoundError(f"Missing spectrum files ({len(missing_files)} total), first entries:\n{preview}")

        return df.reset_index(drop=True)

    @staticmethod
    def _path_from_value(value: object) -> Path:
        if isinstance(value, Path):
            return value
        if isinstance(value, str):
            if not value:
                raise ValueError("Expected a file path, got an empty string")
            return Path(value)
        if value is None or (isinstance(value, float) and isnan(value)):
            raise ValueError("Expected a file path, got an empty value")
        return Path(str(value))

    @staticmethod
    def image_name_from_file(file_path: Path) -> str:
        parts = file_path.parts
        subject_name = next((p.split("_")[0] for p in parts if re.match(r"^[PM]\d+_.*Experiment\d+$", p)), None)
        if subject_name is None:
            subject_name = next((p for p in parts if re.match(r"^Cat_\d+_", p)), "unknown")
        timestamp = next((p for p in parts if re.match(r"^\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}$", p)), file_path.parent.name)
        stem = file_path.stem.replace("spectrum_from", "")
        return f"{subject_name}#{timestamp}#{stem}"

    @staticmethod
    def _read_spectrum(file_path: Path) -> np.ndarray:
        """
        Reads spectrum CSV files with either:
          - no header, two numeric columns: wavelength,value
          - a header row with at least two numeric columns
          - one numeric column of intensity values
        The returned feature vector is always the intensity/value column.
        """
        try:
            data = np.loadtxt(file_path, delimiter=",", dtype=np.float32)
            if data.ndim == 1:
                return data.astype(np.float32)
            if data.ndim == 2 and data.shape[1] >= 2:
                return data[:, 1].astype(np.float32)
        except Exception:
            pass

        df = pd.read_csv(file_path)
        numeric = df.select_dtypes(include=[np.number])
        if numeric.empty:
            # Try files with no useful header but mixed parsing.
            df = pd.read_csv(file_path, header=None)
            numeric = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")

        if numeric.empty:
            raise ValueError(f"No numeric spectrum columns found in {file_path}")

        if numeric.shape[1] >= 2:
            return numeric.iloc[:, 1].to_numpy(dtype=np.float32, copy=True)

        return numeric.iloc[:, 0].to_numpy(dtype=np.float32, copy=True)

    @staticmethod
    def _read_labels(df: pd.DataFrame, config: Config) -> torch.Tensor | None:
        if config.get("input/no_labels", False):
            return None

        label_index_column = config.get("input/cat_pig_label_index_column", "label_index")
        if label_index_column in df.columns:
            return torch.as_tensor(df[label_index_column].astype(int).to_numpy(copy=True), dtype=torch.long)

        label_column = config.get("input/cat_pig_label_column", "label")
        if label_column not in df.columns:
            return None

        label_mapping = config.get("input/cat_pig_label_mapping")
        labels = df[label_column].astype(str)
        if label_mapping is None:
            label_mapping = {label: index for index, label in enumerate(sorted(labels.unique()))}

        unknown = sorted(set(labels) - set(label_mapping))
        if unknown:
            raise ValueError(f"Labels missing from input/cat_pig_label_mapping: {unknown}")

        return torch.as_tensor(labels.map(label_mapping).to_numpy(), dtype=torch.long)

    def label_counts(self) -> tuple[torch.Tensor, torch.Tensor]:
        task = Task.from_config(self.config)
        labels = getattr(self, task.labels_name())
        if not isinstance(labels, torch.Tensor):
            raise ValueError("Cannot calculate label counts without labels in the manifest")
        return labels.unique(return_counts=True)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        sample: dict[str, torch.Tensor | str | int] = {"features": self.features[index, :]}

        if self.labels is not None:
            sample["labels"] = self.labels[index]
        if self.image_labels is not None:
            sample["image_labels"] = self.image_labels[index]
        if self.meta is not None:
            sample["meta"] = self.meta[index, :]

        if not self.train:
            path = self.spectrum_paths[index]
            sample["image_name"] = path.image_name()
            sample["image_name_annotations"] = path.image_name_annotations()
            sample["file_path"] = str(path.file_path)
            sample["image_index"] = index

        return sample
