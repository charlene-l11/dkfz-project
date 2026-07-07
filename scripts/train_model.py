#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HTC_PROJECT_ROOT = PROJECT_ROOT / "htc"

for import_root in (PROJECT_ROOT, HTC_PROJECT_ROOT):
    import_path = str(import_root)
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "dkfz_matplotlib"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lightning import Trainer, seed_everything  # noqa: E402
from lightning.pytorch.callbacks import ModelCheckpoint  # noqa: E402
from lightning.pytorch.loggers import CSVLogger  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler  # noqa: E402

from cat_pig_median_pixel.DatasetCatPigMedianPixel import DatasetCatPigMedianPixel  # noqa: E402
from cat_pig_median_pixel.LightningCatPigMedianPixel import LightningCatPigMedianPixel  # noqa: E402
from htc.utils.Config import Config  # noqa: E402


def dataset_config(config: Config, manifest: Path) -> Config:
    result = Config(copy.deepcopy(config.data))
    result["input/cat_pig_manifest"] = str(manifest.resolve())
    return result


def build_run_output_dir(base_dir: Path) -> Path:
    timestamp = dt.datetime.now().strftime("%d%m%Y_%H%M")
    output_dir = base_dir / f"{timestamp}_matrix"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def make_output_subfolders(run_output_dir: Path) -> dict[str, Path]:
    """Create organized output folders inside the timestamped _matrix folder."""
    output_dirs = {
        "pngs": run_output_dir / "pngs",
        "csvs": run_output_dir / "csvs",
        "checkpoints": run_output_dir / "checkpoints",
    }
    for folder in output_dirs.values():
        folder.mkdir(parents=True, exist_ok=True)
    return output_dirs


def write_run_parameters(run_output_dir: Path, run_parameters: dict[str, Any]) -> None:
    """Write a human-readable record of the parameters used for this run."""
    (run_output_dir / "run_parameters.json").write_text(
        json.dumps(run_parameters, indent=2),
        encoding="utf-8",
    )


def load_pipeline_config(data_dir: Path) -> dict[str, Any]:
    path = data_dir / "pipeline_config.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def config_training_value(pipeline_config: dict[str, Any], key: str, default: Any = None) -> Any:
    training = pipeline_config.get("training", {})
    if isinstance(training, dict) and key in training:
        return training[key]
    return default


def cli_or_config(cli_value: Any, pipeline_config: dict[str, Any], key: str, default: Any = None) -> Any:
    return cli_value if cli_value is not None else config_training_value(pipeline_config, key, default)


def write_confusion_matrix(predictions: pd.DataFrame, labels: list[str], output_dirs: dict[str, Path]) -> None:
    y_true = predictions["label_index"].astype(int).to_numpy()
    y_pred = predictions["prediction_index"].astype(int).to_numpy()
    n_classes = len(labels)

    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)

    with np.errstate(divide="ignore", invalid="ignore"):
        percentages = cm / cm.sum(axis=1, keepdims=True) * 100
    percentages = np.nan_to_num(percentages, nan=0.0)

    csv_dir = output_dirs["csvs"]
    png_dir = output_dirs["pngs"]

    pd.DataFrame(cm, index=labels, columns=labels).to_csv(csv_dir / "confusion_matrix_raw.csv")
    pd.DataFrame(percentages, index=labels, columns=labels).to_csv(csv_dir / "confusion_matrix_normalized.csv")

    size = max(6, n_classes * 0.75)
    fig, ax = plt.subplots(figsize=(size + 2, size))
    image = ax.imshow(percentages, cmap="YlGnBu", vmin=0, vmax=100)

    ax.set_xticks(range(n_classes), labels, rotation=45, ha="right")
    ax.set_yticks(range(n_classes), labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Confusion matrix (row-wise percentages)")

    for row in range(n_classes):
        for column in range(n_classes):
            if cm[row, column] > 0:
                ax.text(
                    column,
                    row,
                    f"{percentages[row, column]:.1f}%\n({cm[row, column]})",
                    ha="center",
                    va="center",
                    color="white" if percentages[row, column] >= 50 else "black",
                )

    fig.colorbar(image, ax=ax, label="Percentage")
    fig.tight_layout()
    fig.savefig(png_dir / "confusion_matrix.png", dpi=300)
    plt.close(fig)


def dataset_accuracy(module: LightningCatPigMedianPixel, dataset: DatasetCatPigMedianPixel, batch_size: int) -> dict:
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, persistent_workers=False)
    correct = 0
    total = 0

    module.eval()
    with torch.no_grad():
        for batch in loader:
            logits = module(batch)
            predictions = logits.argmax(dim=1)
            labels = batch["labels"]
            correct += int((predictions == labels).sum())
            total += int(labels.numel())

    return {"correct": correct, "total": total, "accuracy": correct / total if total else 0.0}


def train_sampler(dataset: DatasetCatPigMedianPixel, config: Config):
    if not config["input/oversampling"]:
        return RandomSampler(dataset, replacement=True, num_samples=config["input/epoch_size"])

    labels = dataset.labels
    if not isinstance(labels, torch.Tensor):
        raise ValueError("Cannot use oversampling because the training dataset has no labels.")

    label_indices, label_counts = labels.unique(return_counts=True)
    class_weights = torch.zeros(config["input/n_classes"], dtype=torch.float32)
    class_weights[label_indices] = 1.0 / label_counts.float()
    sample_weights = class_weights[labels].tolist()

    return WeightedRandomSampler(sample_weights, num_samples=config["input/epoch_size"], replacement=True)


def apply_class_weight_settings(config: Config, class_weight_method: str, class_weights: str | None, labels: list[str]) -> list[float] | None:
    if class_weight_method == "inverse":
        config["model/class_weight_method"] = "1∕m"
    elif class_weight_method == "balanced":
        config["model/class_weight_method"] = "(n-m)∕n"
    elif class_weight_method == "none":
        config["model/class_weight_method"] = None
    else:
        config["model/class_weight_method"] = class_weight_method

    if class_weights:
        explicit_weights = [float(weight.strip()) for weight in str(class_weights).split(",") if weight.strip()]
        if len(explicit_weights) != len(labels):
            raise ValueError(f"Expected {len(labels)} class weights for labels {labels}, got {len(explicit_weights)}")
        config["model/explicit_class_weights"] = explicit_weights
        return explicit_weights

    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and test an HTC median-pixel model on prepared spectra.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu", "mps"])
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--class-weight-method", choices=["none", "inverse", "balanced", "softmin", "nll"])
    parser.add_argument("--class-weights")
    parser.add_argument("--oversampling", action="store_true")
    parser.add_argument("--standardize", action="store_true")

    args = parser.parse_args()

    pipeline_config = load_pipeline_config(args.data_dir)

    seed = int(cli_or_config(args.seed, pipeline_config, "seed", 0))
    max_epochs = int(cli_or_config(args.max_epochs, pipeline_config, "max_epochs", 10))
    accelerator = str(cli_or_config(args.accelerator, pipeline_config, "accelerator", "auto"))
    batch_size = cli_or_config(args.batch_size, pipeline_config, "batch_size", None)
    learning_rate = cli_or_config(args.learning_rate, pipeline_config, "learning_rate", None)
    class_weight_method = str(cli_or_config(args.class_weight_method, pipeline_config, "class_weight_method", "none"))
    class_weights = cli_or_config(args.class_weights, pipeline_config, "class_weights", None)
    oversampling = bool(args.oversampling or config_training_value(pipeline_config, "oversampling", False))
    standardize = bool(args.standardize or config_training_value(pipeline_config, "standardize", False))

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(pipeline_config.get("output_dir", args.data_dir.parent if pipeline_config else args.data_dir.parent))

    config_path = args.config or PROJECT_ROOT / "cat_pig_median_pixel" / "configs" / "default.json"

    seed_everything(seed, workers=True)
    run_output_dir = build_run_output_dir(output_dir)
    output_dirs = make_output_subfolders(run_output_dir)
    config = Config(config_path)

    label_data = json.loads((args.data_dir / "labels.json").read_text())
    dataset_data_path = args.data_dir / "dataset.json"
    dataset_data = json.loads(dataset_data_path.read_text()) if dataset_data_path.exists() else {}

    labels = label_data["labels"]
    if len(labels) < 2:
        raise ValueError("Training a classifier requires at least two labels/classes.")

    config["input/n_classes"] = len(labels)
    config["input/cat_pig_label_mapping"] = label_data["mapping"]

    if dataset_data.get("feature_columns"):
        config["input/feature_columns"] = dataset_data["feature_columns"]
        config["input/n_channels"] = len(dataset_data["feature_columns"])

    config["trainer_kwargs/max_epochs"] = max_epochs
    config["trainer_kwargs/accelerator"] = accelerator

    if batch_size is not None:
        config["dataloader_kwargs/batch_size"] = int(batch_size)
    if learning_rate is not None:
        config["optimization/optimizer/lr"] = float(learning_rate)

    if standardize:
        train_manifest = pd.read_csv(args.data_dir / "train.csv")

        if any(column.startswith("feature_") for column in train_manifest.columns):
            feature_columns = [column for column in train_manifest.columns if column.startswith("feature_")]
            train_features = train_manifest[feature_columns].to_numpy(dtype=np.float32, copy=True)
        else:
            train_features = np.stack([
                DatasetCatPigMedianPixel._read_spectrum(Path(file_path))
                for file_path in train_manifest["file_path"]
            ])

        feature_std = train_features.std(axis=0)
        feature_std[feature_std == 0] = 1

        config["input/feature_mean"] = train_features.mean(axis=0).tolist()
        config["input/feature_std"] = feature_std.tolist()

    explicit_class_weights = apply_class_weight_settings(config, class_weight_method, class_weights, labels)
    config["input/oversampling"] = bool(oversampling)
    config["input/cat_pig_manifest"] = str((args.data_dir / "train.csv").resolve())

    train_config = dataset_config(config, args.data_dir / "train.csv")
    val_config = dataset_config(config, args.data_dir / "val.csv")
    test_config = dataset_config(config, args.data_dir / "test.csv")

    dataset_train = DatasetCatPigMedianPixel(paths=None, train=True, config=train_config, fold_name="cat_pig")
    dataset_val = DatasetCatPigMedianPixel(paths=None, train=False, config=val_config, fold_name="cat_pig")
    dataset_test = DatasetCatPigMedianPixel(paths=None, train=False, config=test_config, fold_name="cat_pig")

    if len(dataset_val) == 0:
        raise ValueError("Validation split is empty. Re-run prepare_data.py with more subjects or a different split.")
    if len(dataset_test) == 0:
        raise ValueError("Test split is empty. Re-run prepare_data.py with more subjects or a different split.")

    config["input/epoch_size"] = len(dataset_train)

    module = LightningCatPigMedianPixel(
        dataset_train=dataset_train,
        datasets_val=[dataset_val],
        dataset_test=dataset_test,
        config=config,
        fold_name="cat_pig",
    )

    if explicit_class_weights is not None:
        weights = torch.as_tensor(explicit_class_weights, dtype=torch.float32)
        module.ce_loss_weighted = torch.nn.CrossEntropyLoss(weight=weights)

    checkpoint = ModelCheckpoint(
        dirpath=output_dirs["checkpoints"],
        filename="{epoch:02d}-{accuracy:.4f}",
        monitor="accuracy",
        mode="max",
        save_top_k=1,
        save_last=True,
    )

    logger = CSVLogger(save_dir=run_output_dir, name="logs")
    trainer = Trainer(logger=logger, callbacks=[checkpoint], num_sanity_val_steps=0, **config["trainer_kwargs"])

    train_loader = DataLoader(
        dataset_train,
        batch_size=config["dataloader_kwargs/batch_size"],
        sampler=train_sampler(dataset_train, config),
        num_workers=0,
        persistent_workers=False,
    )
    val_loader = DataLoader(dataset_val, batch_size=config["dataloader_kwargs/batch_size"], num_workers=0, persistent_workers=False)
    test_loader = DataLoader(dataset_test, batch_size=config["dataloader_kwargs/batch_size"], num_workers=0, persistent_workers=False)

    print("Training settings:")
    print(f"  data_dir: {args.data_dir}")
    print(f"  output_dir: {output_dir}")
    print(f"  max_epochs: {max_epochs}")
    print(f"  seed: {seed}")
    print(f"  batch_size: {config['dataloader_kwargs/batch_size']}")
    print(f"  learning_rate: {config['optimization/optimizer/lr']}")
    print(f"  class_weight_method: {class_weight_method}")
    print(f"  oversampling: {oversampling}")
    print(f"  standardize: {standardize}")

    run_parameters = {
        "run_created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "data_dir": str(args.data_dir.resolve()),
        "output_dir": str(Path(output_dir).resolve()),
        "run_output_dir": str(run_output_dir.resolve()),
        "labels": labels,
        "label_mapping": label_data["mapping"],
        "seed": seed,
        "max_epochs": max_epochs,
        "accelerator": accelerator,
        "batch_size": config["dataloader_kwargs/batch_size"],
        "learning_rate": config["optimization/optimizer/lr"],
        "class_weight_method": class_weight_method,
        "explicit_class_weights": explicit_class_weights,
        "oversampling": bool(oversampling),
        "standardize": bool(standardize),
        "config_path": str(Path(config_path).resolve()),
        "pipeline_config": pipeline_config,
        "output_structure": {
            "pngs": str(output_dirs["pngs"].resolve()),
            "csvs": str(output_dirs["csvs"].resolve()),
            "checkpoints": str(output_dirs["checkpoints"].resolve()),
            "logs": str((run_output_dir / "logs").resolve()),
            "run_parameters_file": str((run_output_dir / "run_parameters.json").resolve()),
        },
    }
    write_run_parameters(run_output_dir, run_parameters)

    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(module, dataloaders=test_loader, ckpt_path="best")

    if checkpoint.best_model_path:
        best_state = torch.load(checkpoint.best_model_path, map_location="cpu", weights_only=False)["state_dict"]
        module.load_state_dict(best_state)

    results_path = Path(logger.save_dir) / "test_results.npz"
    results = np.load(results_path)

    test_manifest = pd.read_csv(args.data_dir / "test.csv")
    logits = results["logits"]
    predictions = logits.argmax(axis=1)

    output = test_manifest.copy()
    output["prediction_index"] = predictions
    for index in range(logits.shape[1]):
        output[f"logit_{index}"] = logits[:, index]

    output.to_csv(output_dirs["csvs"] / "test_predictions.csv", index=False)
    (run_output_dir / "labels.json").write_text(json.dumps(label_data, indent=2))
    write_confusion_matrix(output, labels, output_dirs)

    evaluation_summary = {
        "train": dataset_accuracy(module, dataset_train, config["dataloader_kwargs/batch_size"]),
        "val": dataset_accuracy(module, dataset_val, config["dataloader_kwargs/batch_size"]),
        "test": dataset_accuracy(module, dataset_test, config["dataloader_kwargs/batch_size"]),
        "best_checkpoint": checkpoint.best_model_path,
        "class_weight_method": class_weight_method,
        "explicit_class_weights": explicit_class_weights,
        "oversampling": bool(oversampling),
        "standardize": bool(standardize),
        "seed": seed,
        "max_epochs": max_epochs,
        "batch_size": config["dataloader_kwargs/batch_size"],
        "learning_rate": config["optimization/optimizer/lr"],
        "pipeline_config": pipeline_config,
        "output_structure": {
            "pngs": str(output_dirs["pngs"]),
            "csvs": str(output_dirs["csvs"]),
            "checkpoints": str(output_dirs["checkpoints"]),
            "logs": str(run_output_dir / "logs"),
            "run_parameters_file": str(run_output_dir / "run_parameters.json"),
        },
    }
    (run_output_dir / "evaluation_summary.json").write_text(json.dumps(evaluation_summary, indent=2))
    (run_output_dir / "config_used.json").write_text(json.dumps(config.data, indent=2))

    print(f"Run folder: {run_output_dir}")
    print(f"Best checkpoint: {checkpoint.best_model_path}")
    print(f"PNG outputs: {output_dirs['pngs']}")
    print(f"CSV outputs: {output_dirs['csvs']}")
    print(f"Test predictions: {output_dirs['csvs'] / 'test_predictions.csv'}")
    print(f"Confusion matrix: {output_dirs['pngs'] / 'confusion_matrix.png'}")
    print(f"Run parameters: {run_output_dir / 'run_parameters.json'}")


if __name__ == "__main__":
    main()
