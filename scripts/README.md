# Cat/Pig Median-Pixel Training

This folder contains the two scripts needed to train the median-pixel HTC model and write a confusion matrix. By default, data preparation uses only `spectrum_fromCSV1_(500.0-995.0)_masked_data_0_derivative.csv`.

## 1. Prepare data

Create `manifest.csv`, `train.csv`, `val.csv`, `test.csv`, `labels.json`, and `dataset.json` from labeled spectrum folders:

```bash
python scripts/prepare_data.py \
  --source deox=C:/path/to/kidney_deox \
  --source ischem=C:/path/to/kidney_ischem \
  --source stas=C:/path/to/kidney_stas \
  --output-dir training_runs
```

You can also start from an existing manifest with `file_path`, `label`, and `subject_name` columns:

```bash
python scripts/prepare_data.py \
  --input-manifest C:/path/to/manifest.csv \
  --output-dir training_runs
```

## 2. Train and write the matrix

Pass the timestamped folder created by `prepare_data.py` as `--data-dir`:

```bash
python scripts/train_model.py \
  --data-dir training_runs/29062026_1105_prepCSV \
  --output-dir training_runs/stas_ischem_deox_matrix_runs \
  --max-epochs 10 \
  --standardize
```

The train script writes:

- `confusion_matrix.png`
- `confusion_matrix_raw.csv`
- `confusion_matrix_normalized.csv`
- `test_predictions.csv`
- `evaluation_summary.json`
- `config_used.json`
- Lightning checkpoints and logs
