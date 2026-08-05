# DKFZ median-spectrum HTC pipeline

## What the pipeline does

1. Loads and validates one YAML configuration.
2. Finds and validates median-spectrum CSV files.
3. Creates reproducible subject-separated train, validation, and test splits.
4. Requires every class in all configured splits.
5. Trains the original HTC median-pixel model.
6. Evaluates the best checkpoint on validation and testing data.
7. Writes separate validation and testing confusion-matrix folders with CSV, PNG, PDF, and Excel outputs.
8. Writes validation/test predictions, per-class metrics, split details, and all resolved run parameters.

Each premade scenario folder is the run container. The pipeline derives that folder from the YAML location (`<scenario>/<file>.yaml`) and automatically creates `output/`, `run_config/`, and `splits/` beside the YAML. It does not create or rename the scenario folder itself. Progress is printed immediately during discovery, validation, splitting, workbook creation, training, and evaluation.

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
python run_pipeline.py --config "C:/path/to/scenario/manifold_settings.yaml"
```

The included `run_cluster.sh` is a template; adjust the module and environment paths for the cluster.

## Configuration

The same YAML has Windows and Linux values under `paths`. The active operating system selects the correct value automatically.

The organ inputs are user-controlled. The organ names must match exactly across `experiment_folders`, `labelling_file`, and `hyperguis`:

```yaml
paths:
  data_root:
    windows: "C:/path/to/Cat_Pig/Cat_atlas"
    linux: "/path/to/Cat_Pig/Cat_atlas"

experiment_folders:
  1_stom_pig: "Cat_0001_stomach/data"
  2_liv_pig: "Cat_0004_liver/data"

labelling_file:
  1_stom_pig: "_labelling_001.txt"
  2_liv_pig: "_labelling_*"

hyperguis:
  1_stom_pig: "_hypergui_1"
  2_liv_pig: "_hypergui_*"
```

Relative `experiment_folders` values are resolved below `paths.data_root`, so moving the dataset normally requires changing only `data_root`. Absolute organ paths remain supported. The corresponding `hyperguis` value selects which HyperGUI directory or directories are searched for spectra. The corresponding `labelling_file` value must match a TXT file beside each selected HyperGUI directory. Exact names and glob patterns such as `_*` are supported. Recording folders that do not have both configured matches are ignored and are not added to any manifest.

Manual wavelength selection is controlled by the same YAML. Exact wavelengths and inclusive ranges may be combined with semicolons:

```yaml
wavelength:
  min: 500
  max: 995
  selective: "500; 560-720; 730-735; 875; 900-930"
```

Set `selective: null` (or `selective: all`) to use the full configured range. For two-column CSV files, the first numeric column is treated as the wavelength grid. For one-column CSV files, the grid is inferred evenly from `min`, `max`, and `data.expected_channels`. Exact requested wavelengths must exist on that grid; ranges include every available channel within their bounds. The resolved indices and wavelengths are saved in `data/wavelengths.json` and `run_parameters.json`.

All wavelength-selection experiments retain the original fixed-width spectral input. Selected bands are standardized and placed at their original wavelength positions; unselected positions are set to zero. This allows small selections such as `500-505` to use the same HTC architecture and parameter count as the full-spectrum experiment.

### Train/validate on one animal and test on another

By default, the primary dataset is subject-split into training, validation, and testing as before. To reserve a second animal dataset exclusively for testing, enable `external_testing` and provide complete organ-folder paths, labelling TXT patterns, and HyperGUI patterns:

```yaml
external_testing:
  enabled: true

  experiment_folders:
    1_stom_pig:
      windows: "D:/datasets/rat_atlas/Rat_0001_stomach/data"
      linux: "/datasets/rat_atlas/Rat_0001_stomach/data"
    2_liv_pig:
      windows: "D:/datasets/rat_atlas/Rat_0002_liver/data"
      linux: "/datasets/rat_atlas/Rat_0002_liver/data"

  labelling_file:
    1_stom_pig: "_labelling_001.txt"
    2_liv_pig: "_labelling_001.txt"

  hyperguis:
    1_stom_pig: "_hypergui_1"
    2_liv_pig: "_hypergui_1"
```

The external mappings must contain exactly the same organ keys as the primary mappings because the trained model uses the primary class-to-index mapping. Values and file patterns may differ. Every external experiment folder must be an absolute path. A folder can be a single path for the current platform or contain separate `windows` and `linux` paths as shown above.

With external testing enabled, the configured primary `train_ratio` and `val_ratio` are normalized to use all primary subjects. For example, `0.70/0.15/0.15` becomes approximately `82.35%` training and `17.65%` validation. The primary dataset contributes no test rows; 100% of the external dataset becomes the test set. Feature standardization is still calculated only from primary training rows. The pipeline rejects missing/extra external classes or a different external wavelength grid. Set `enabled: false` to restore the original single-dataset train/validation/test split.

### Multiple selective test cases

To run several independent selections with one command, make `selective` a mapping. Each configured wavelength value becomes its child-folder name, and commas or semicolons may separate exact wavelengths:

```yaml
stepwise_analysis: false
reverse_stepwise: false

wavelength:
  min: 500
  max: 995
  selective:
    1: "630; 810"
    2: "575; 650"
    3: "630; 785"
    4: "865; 875"
    5: "595; 930"
```

The data are discovered once and all cases reuse the same subject split. Runs execute sequentially in folders named with the configured wavelength text directly below the master scenario. For example, `selective: {1: "450; 510"}` writes to `<master scenario>/450; 510/`. A case may contain any number of wavelengths, such as `"450; 510; 575; 630; 810"`. The numeric test-case keys are identifiers only. Each child contains its own `output/` and `run_config/`; the master `splits/` folder is shared. Progress is stored in `selective results/status.json`, and completed training runs are summarized in `combined_results.csv` and `combined_results.xlsx`. Rerunning the master skips children that already contain `run_config/run_parameters.json`.

Selective batch mode and stepwise mode are intentionally mutually exclusive. A scalar `selective` value still performs one normal run, and `selective: null` still performs one full-spectrum run.

## Stepwise wavelength analysis

Forward and reverse stepwise modes are independent and are controlled by the single master YAML:

```yaml
stepwise_analysis: true
reverse_stepwise: true
forward_stop_wavelength: 640
reverse_stop_wavelength: 850

wavelength:
  min: 500
  max: 995
  selective: null
```

Forward runs expand from the lower boundary (`500-505`, `500-510`, and so on) through `forward_stop_wavelength`. Reverse runs expand from the upper boundary (`990-995`, `985-995`, and so on) through `reverse_stop_wavelength`. Either direction can be enabled by itself. When both flags are false, the pipeline performs the existing single run using `wavelength.selective`.

The master command generates a `manifold_settings.yaml` inside every child folder and runs children sequentially under `forward run/<range>/` and `reverse run/<range>/`. Data discovery and the subject split occur once and are reused by every child. Completed children are detected by `run_config/run_parameters.json` and skipped when the master command is rerun. Progress is stored in `stepwise results/status.json`; aggregate CSV, Excel, and accuracy plots are updated after each completed child.
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
<scenario>/
|-- manifold_settings.yaml
|-- run_config/
|   |-- source_config.yaml
|   |-- resolved_config.json
|   |-- htc_config.json
|   `-- run_parameters.json
|-- splits/
|   `-- <organ>_splits.xlsx
`-- output/
    |-- confusion_matrix_normalized_light.pdf
    |-- confusion_matrix_normalized_dark.pdf
    |-- summary.json
    |-- validation_metrics.csv
    |-- testing_metrics.csv
    |-- data/
    |-- checkpoints/
    |-- logs/
    |-- predictions/
    |-- validation_matrices/
    `-- testing_matrices/
```

Every organ workbook contains `Train Details`, `Validation Details`, and `Test Details`. Each validation/testing matrix workbook contains `Matrix Normalized` and `Matrix Counts`. Normalized values are fractions from 0 to 1 displayed with three decimal places. PNG and PDF matrices are written in exactly two variants: `_light` and `_dark`; every plot title includes that split's accuracy. The run root contains one-row `validation_metrics.csv` and `testing_metrics.csv` files with the summary metrics for each split. Only the testing normalized light/dark PDFs are copied from `testing_matrices/pdf/` into the run root. The testing `summary.json` is also written in the run root.

## Important safeguards

- A subject can occur in only one split.
- Each class must occur in enough distinct subjects to cover the required splits.
- The source YAML is copied into every run.
- Generated JSON files record historical runs and are not meant to be edited.
- The original HTC repository is treated as read-only.

## Immutable HTC sources

The YAML selects `C:/Users/c758g/Documents/scripts/cluster scripts/htc` on Windows and `/omics/groups/OE0645/internal/HSI_RISE/htc` on Linux. These are the authoritative HTC runtime trees. The pipeline never copies, patches, renames, or writes files there. `htc-main` is reference-only. `source_baseline.json` records SHA-256 hashes of relevant files from both read-only sources, and `verify_source_integrity.py` checks the local copies against that baseline.

## Microsoft Visual Studio

Open `dkfz_final_code.sln` in Microsoft Visual Studio with the Python development workload installed. Select a complete HTC-capable Python environment for the project. `run_pipeline.py` is the startup file and runs the complete pipeline with `configs/manifold_settings.yaml` when started without arguments. The same file can be given the normal `validate`, `prepare`, or `run` command-line arguments.

Before the first complete run, verify every path in `experiment_folders` and ensure the names in `experiment_folders`, `labelling_file`, and `hyperguis` match exactly. The included HTC source folders do not currently contain a functional Python interpreter, so a separate environment with the dependencies in `requirements.txt` and the HTC dependencies is required.



