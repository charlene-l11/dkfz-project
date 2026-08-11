from __future__ import annotations

import copy
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def bootstrap_htc(htc_root: Path) -> None:
    if not (htc_root / "htc" / "__init__.py").exists():
        raise FileNotFoundError(f"HTC root must contain htc/__init__.py: {htc_root}")
    value = str(htc_root.resolve())
    if value not in sys.path:
        sys.path.insert(0, value)


def _dataset_config(config, manifest: Path):
    from htc.utils.Config import Config
    result = Config(copy.deepcopy(config.data))
    result["input/cat_pig_manifest"] = str(manifest.resolve())
    return result


def _sampler(dataset, config):
    import torch
    from torch.utils.data import RandomSampler, WeightedRandomSampler
    if not config["input/oversampling"]:
        return RandomSampler(dataset, replacement=True, num_samples=int(config["input/epoch_size"]))
    labels = dataset.labels
    indices, counts = labels.unique(return_counts=True)
    weights = torch.zeros(int(config["input/n_classes"]), dtype=torch.float32)
    weights[indices] = 1.0 / counts.float()
    return WeightedRandomSampler(weights[labels].tolist(), num_samples=int(config["input/epoch_size"]), replacement=True)


def _class_weight_method(training: dict[str, Any]) -> str | None:
    imbalance = training["imbalance"]
    if imbalance["strategy"] != "class_weighting":
        return None
    method = imbalance.get("method", "inverse")
    return {"inverse": "1\u2215m", "balanced": "(n-m)\u2215n", "softmin": "softmin", "nll": "nll"}[method]


def _prediction_frame(manifest_path: Path, logits: np.ndarray, labels: list[str]) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    if len(manifest) != len(logits):
        raise RuntimeError(
            f"Manifest {manifest_path} has {len(manifest)} rows but HTC returned {len(logits)} predictions"
        )
    predictions = manifest.copy()
    predictions["prediction_index"] = logits.argmax(axis=1)
    predictions["prediction_label"] = predictions["prediction_index"].map(
        {index: label for index, label in enumerate(labels)}
    )
    for index in range(logits.shape[1]):
        predictions[f"logit_{index}"] = logits[:, index]
    return predictions


def run_training(cfg: dict[str, Any], run_dir: Path, run_config_dir: Path) -> dict[str, Any]:
    bootstrap_htc(Path(cfg["paths"]["htc_root"]))

    import htc
    import lightning
    import torch
    from lightning import Trainer, seed_everything
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    from torch.utils.data import DataLoader
    from htc.utils.Config import Config
    from median_pipeline.dataset import DatasetCatPigMedianPixel
    from median_pipeline.evaluation import write_evaluation, write_training_evaluation
    from median_pipeline.lightning_module import LightningCatPigMedianPixel
    from median_pipeline.loss_curves import write_loss_curves
    from median_pipeline.provenance import htc_source_fingerprint

    training = cfg["training"]
    seed_everything(int(training["seed"]), workers=True)
    data_dir = run_dir / "data"
    labels_data = json.loads((data_dir / "labels.json").read_text(encoding="utf-8"))
    labels = labels_data["labels"]
    external_testing = bool(cfg.get("external_testing", {}).get("enabled", False))
    testing_labels = labels_data.get("external_test_labels", labels)
    wavelength_info = json.loads((data_dir / "wavelengths.json").read_text(encoding="utf-8"))
    wavelength_indices = [int(index) for index in wavelength_info["selected_indices"]]

    base_config_path = Path(cfg["paths"]["htc_root"]) / "htc" / "models" / "median_pixel" / "configs" / "default.json"
    config = Config(base_config_path)
    config["lightning_class"] = "median_pipeline.lightning_module>LightningCatPigMedianPixel"
    config["label_mapping"] = labels_data["mapping"]
    config["input/n_classes"] = len(labels)
    # Keep the original spectral width for every wavelength-selection experiment.
    # The dataset scatters selected bands into their original positions and masks
    # all remaining positions to zero, so model capacity is identical across runs.
    config["input/n_channels"] = int(wavelength_info["n_input_channels"])
    config["input/n_selected_channels"] = len(wavelength_indices)
    config["input/wavelength_indices"] = wavelength_indices
    config["input/mask_unselected_wavelengths"] = True
    config["input/selected_wavelengths"] = wavelength_info["selected_wavelengths"]
    config["input/cat_pig_label_mapping"] = labels_data["mapping"]
    config["input/cat_pig_label_column"] = "label"
    config["input/cat_pig_label_index_column"] = "label_index"
    config["input/oversampling"] = training["imbalance"]["strategy"] == "balanced_oversampling"
    config["model/class_weight_method"] = _class_weight_method(training)
    config["optimization/optimizer/lr"] = float(training["learning_rate"])
    config["dataloader_kwargs/batch_size"] = int(training["batch_size"])
    config["dataloader_kwargs/num_workers"] = int(training["num_workers"])
    config["trainer_kwargs/max_epochs"] = int(training["max_epochs"])
    config["trainer_kwargs/accelerator"] = str(training["accelerator"])
    config["trainer_kwargs/devices"] = int(training["devices"])
    config["trainer_kwargs/precision"] = str(training["precision"])
    config["validation/checkpoint_metric"] = str(training["checkpoint_metric"])

    train_path = data_dir / "train.csv"
    val_path = data_dir / "validation.csv"
    test_path = data_dir / "test.csv"
    if bool(training["standardize"]):
        train_manifest = pd.read_csv(train_path)
        features = np.stack([DatasetCatPigMedianPixel._read_spectrum(Path(p)) for p in train_manifest["file_path"]])
        features = features[:, wavelength_indices]
        std = features.std(axis=0)
        std[std == 0] = 1
        config["input/feature_mean"] = features.mean(axis=0).tolist()
        config["input/feature_std"] = std.tolist()

    train_config = _dataset_config(config, train_path)
    val_config = _dataset_config(config, val_path)
    test_config = _dataset_config(config, test_path)
    if external_testing:
        # External labels define matrix rows, not model targets. Keeping them out
        # of HTC's test loss allows arbitrary external class sets and counts.
        test_config["input/no_labels"] = True
    dataset_train = DatasetCatPigMedianPixel(None, train=True, config=train_config, fold_name="dkfz")
    dataset_train_evaluation = DatasetCatPigMedianPixel(
        None, train=False, config=train_config, fold_name="dkfz"
    )
    dataset_val = DatasetCatPigMedianPixel(None, train=False, config=val_config, fold_name="dkfz")
    dataset_test = DatasetCatPigMedianPixel(None, train=False, config=test_config, fold_name="dkfz")
    config["input/epoch_size"] = int(training.get("epoch_size") or len(dataset_train))

    module = LightningCatPigMedianPixel(
        dataset_train=dataset_train, datasets_val=[dataset_val], dataset_test=dataset_test,
        config=config, fold_name="dkfz",
    )

    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    callback = ModelCheckpoint(
        dirpath=checkpoints, filename="{epoch:03d}-{accuracy:.4f}", monitor=str(training["checkpoint_metric"]),
        mode="max", save_top_k=1, save_last=True,
    )
    logger = CSVLogger(save_dir=run_dir, name="logs")
    trainer_kwargs = dict(config["trainer_kwargs"])
    trainer = Trainer(logger=logger, callbacks=[callback], num_sanity_val_steps=0, **trainer_kwargs)
    loader_kwargs = {
        "batch_size": int(training["batch_size"]),
        "num_workers": int(training["num_workers"]),
        "persistent_workers": int(training["num_workers"]) > 0,
    }
    train_loader = DataLoader(dataset_train, sampler=_sampler(dataset_train, config), **loader_kwargs)
    train_evaluation_loader = DataLoader(dataset_train_evaluation, shuffle=False, **loader_kwargs)
    val_loader = DataLoader(dataset_val, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(dataset_test, shuffle=False, **loader_kwargs)

    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    logger.save()
    write_loss_curves(Path(logger.log_dir) / "metrics.csv", run_dir, cfg)
    training_batches = trainer.predict(module, dataloaders=train_evaluation_loader, ckpt_path="best")
    if not training_batches:
        raise RuntimeError("HTC returned no training predictions")
    training_logits = torch.cat(
        [batch["class"].detach().cpu() for batch in training_batches]
    ).numpy()
    validation_batches = trainer.predict(module, dataloaders=val_loader, ckpt_path="best")
    if not validation_batches:
        raise RuntimeError("HTC returned no validation predictions")
    validation_logits = torch.cat([batch["class"].detach().cpu() for batch in validation_batches]).numpy()

    if external_testing:
        testing_batches = trainer.predict(module, dataloaders=test_loader, ckpt_path="best")
        if not testing_batches:
            raise RuntimeError("HTC returned no external testing predictions")
        logits = torch.cat([batch["class"].detach().cpu() for batch in testing_batches]).numpy()
    else:
        trainer.test(module, dataloaders=test_loader, ckpt_path="best")
        results_path = run_dir / "test_results.npz"
        if not results_path.exists():
            raise FileNotFoundError(f"HTC did not produce {results_path}")
        results = np.load(results_path)
        logits = results["logits"]
    predictions_dir = run_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    training_predictions = _prediction_frame(train_path, training_logits, labels)
    validation_predictions = _prediction_frame(val_path, validation_logits, labels)
    testing_predictions = _prediction_frame(test_path, logits, labels)
    validation_predictions.to_csv(predictions_dir / "validation_predictions.csv", index=False)
    testing_predictions.to_csv(predictions_dir / "test_predictions.csv", index=False)

    training_summary = write_training_evaluation(training_predictions, labels, run_dir, cfg)
    validation_summary = write_evaluation(
        validation_predictions, labels, run_dir, cfg, split_name="validation"
    )
    testing_summary = write_evaluation(
        testing_predictions,
        labels,
        run_dir,
        cfg,
        split_name="testing",
        true_labels=testing_labels,
    )
    (run_config_dir / "htc_config.json").write_text(json.dumps(config.data, indent=2), encoding="utf-8")
    run_parameters = {
        "pipeline_config": cfg,
        "htc_config": config.data,
        "wavelengths": wavelength_info,
        "best_checkpoint": callback.best_model_path,
        "best_checkpoint_score": float(callback.best_model_score) if callback.best_model_score is not None else None,
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "lightning": lightning.__version__,
            "htc_module": str(Path(htc.__file__).resolve()),
            "htc_source": htc_source_fingerprint(Path(cfg["paths"]["htc_root"])),
        },
        "evaluation": {
            "training": training_summary,
            "validation": validation_summary,
            "testing": testing_summary,
        },
    }
    (run_config_dir / "run_parameters.json").write_text(json.dumps(run_parameters, indent=2), encoding="utf-8")
    return run_parameters

