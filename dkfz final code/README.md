# DKFZ median-spectrum HTC pipeline

This project merges the reusable pipeline components from `cluster scripts` with the original HTC codebases in `htc-main` and the cluster HTC tree. It reads existing median-spectrum CSV files and uses the original HTC median-pixel model without changing any file in the HTC repository.

## What the pipeline does

1. Loads and validates one YAML configuration.
2. Finds and validates median-spectrum CSV files.
3. Creates reproducible subject-separated train, validation, and test splits.
4. Requires every class in all configured splits.
5. Trains the original HTC median-pixel model.
6. Tests the best checkpoint automatically.
7. Writes raw and normalized confusion matrices as CSV, PNG, and PDF.
8. Writes test predictions, per-class metrics, split details, and all resolved run parameters.

Each complete experiment is stored in one timestamped folder under `runs_root`.

## Windows and VS Code setup

1. Open this folder in VS Code.
2. Select a complete Python environment containing the original HTC dependencies. The bundled `HSI_ML` folder currently has no `python.exe`; select or create a complete environment containing the HTC dependencies.
3. Edit `configs/manifold_settings.yaml` and verify the Windows paths and class folders.
4. Run **Terminal > Run Task > Validate configuration**.
5. Run **Terminal > Run Task > Run complete pipeline**.

The pipeline puts the configured original HTC repository at the front of Python's import path. It does not write to that repository.

## Cluster setup

Load Python and activate the existing HTC-capable environment, then run:

```bash
python -m median_pipeline validate --config configs/manifold_settings.yaml
python -m median_pipeline run --config configs/manifold_settings.yaml
```

The included `run_cluster.sh` is a template; adjust the module and environment paths for the cluster.

## Configuration

The same YAML has Windows and Linux values under `paths`. The active operating system selects the correct value automatically.

Manual wavelength selection is controlled by the same YAML. Exact wavelengths and inclusive ranges may be combined with semicolons:

```yaml
wavelength:
  min: 500
  max: 995
  selective: "500; 560-720; 730-735; 875; 900-930"
```

Set `selective: null` (or `selective: all`) to use the full configured range. For two-column CSV files, the first numeric column is treated as the wavelength grid. For one-column CSV files, the grid is inferred evenly from `min`, `max`, and `data.expected_channels`. Exact requested wavelengths must exist on that grid; ranges include every available channel within their bounds. The resolved indices and wavelengths are saved in `data/wavelengths.json` and `run_parameters.json`.
Imbalance handling is mutually exclusive:

```yaml
training:
  imbalance:
    strategy: none
```

```yaml
training:
  imbalance:
    strategy: class_weighting
    method: inverse
```

```yaml
training:
  imbalance:
    strategy: balanced_oversampling
```

Setting a class-weight method while balanced oversampling is active fails validation.

## Commands

```bash
python -m median_pipeline validate --config configs/manifold_settings.yaml
python -m median_pipeline prepare --config configs/manifold_settings.yaml
python -m median_pipeline run --config configs/manifold_settings.yaml
```

`prepare` creates and checks the manifests without starting training. `run` performs the complete pipeline.

## Run output

```text
runs/<timestamp>_<experiment>/
|-- config/
|   |-- source_config.yaml
|   |-- resolved_config.json
|   `-- htc_config.json
|-- data/
|   |-- manifest.csv
|   |-- train.csv
|   |-- validation.csv
|   |-- test.csv
|   |-- labels.json
|   |-- split_summary.json
|   `-- wavelengths.json
|-- checkpoints/
|-- logs/
|-- predictions/test_predictions.csv
|-- matrices/
|   |-- csv/
|   |-- png/
|   `-- pdf/
|-- metrics/
|   |-- per_class_metrics.csv
|   `-- summary.json
`-- run_parameters.json
```

## Important safeguards

- A subject can occur in only one split.
- Each class must occur in enough distinct subjects to cover the required splits.
- The source YAML is copied into every run.
- Generated JSON files record historical runs and are not meant to be edited.
- The original HTC repository is treated as read-only.

## Immutable HTC sources

The YAML selects `C:/Users/c758g/Downloads/htc-main` on Windows and `/omics/groups/OE0645/internal/HSI_RISE/htc` on Linux. The pipeline imports HTC from the selected location but never copies, patches, renames, or writes files there. `source_baseline.json` records SHA-256 hashes of the relevant source files, and `verify_source_integrity.py` checks the local Windows copies against that baseline.

## Microsoft Visual Studio

Open `dkfz_final_code.sln` in Microsoft Visual Studio with the Python development workload installed. Select a complete HTC-capable Python environment for the project. `run_pipeline.py` is the startup file and runs the complete pipeline with `configs/manifold_settings.yaml` when started without arguments. The same file can be given the normal `validate`, `prepare`, or `run` command-line arguments.

Before the first complete run, verify the Windows data root and every class folder in the YAML. The included HTC source folders do not currently contain a functional Python interpreter, so a separate environment with the dependencies in `requirements.txt` and the HTC dependencies is required.


