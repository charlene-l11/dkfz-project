# DKFZ median-spectrum classification pipeline

## Plain-language overview

This research pipeline trains a neural network to classify tissue or organ categories from **precomputed median hyperspectral spectra**. Each input file represents one median spectrum from one labelled recording. The software discovers the files, separates subjects into training, validation, and test groups, trains the inherited HTC median-pixel model, and creates tables and confusion matrices for review.

The main workflow is a basic internal experiment:

1. Identify the precomputed spectrum files for each tissue class.
2. Divide complete subjects into training, validation, and test groups.
3. Estimate optional per-wavelength standardization values from the training group only.
4. Train one neural network.
5. Use the validation group during training to select a checkpoint.
6. Evaluate the retained checkpoint on the held-out test group.

> **Research-use notice:** This software is an experimental analysis tool. It is not a medical device, does not provide a diagnosis or treatment recommendation, and has not been validated for clinical decision-making. Its results require review by appropriately qualified clinical and technical investigators.

## What the software does—and does not do

The pipeline:

- reads existing median-spectrum CSV files;
- checks the number and validity of spectral channels;
- creates subject-separated data partitions;
- trains the original, unmodified HTC median-pixel model;
- reports accuracy, balanced accuracy (macro recall), macro F1, class-level performance, and confusion matrices;
- records the configuration, data manifests, software versions, and HTC source fingerprints needed to audit a run.

The pipeline does **not**:

- create spectra directly from raw hyperspectral images;
- create or verify clinical annotations from image content;
- perform L1 normalization itself;
- estimate diagnostic sensitivity or specificity for clinical deployment;
- replace independent clinical validation.

The class assigned to a spectrum comes from the matching key in the YAML configuration, for example `stomach` or `liver`. A matching labelling TXT file must be present beside the selected HyperGUI folder, but this pipeline checks its presence only; it does not parse the TXT file to determine the class. Investigators should therefore verify the folder-to-class mapping before every run.

## What you need before starting

You need:

- Python 3.10 or later;
- a functioning Python environment containing the original HTC dependencies;
- the original HTC repository, including `htc/__init__.py`;
- at least two configured tissue or organ classes;
- precomputed median-spectrum CSV files with the same wavelength grid;
- subject identifiers in the source path in a supported form, such as `P12`, `M12`, or `Cat_0001_...`;
- one YAML configuration file for the scenario.

The package dependencies listed in `requirements.txt` can be installed after activating the HTC-capable environment:

```bash
python -m pip install -r requirements.txt
```

The pipeline adds the configured HTC repository to Python's import path. It treats that repository as read-only and does not patch or overwrite it.

## Quick start: basic internal run

### 1. Create a scenario folder

Place a copy of `configs/manifold_settings.yaml` in a folder dedicated to one analysis. The folder containing the YAML becomes the **scenario folder**. Results are written beside that YAML in `output/`, `run_config/`, and `splits/`.

Use a new scenario folder for every analysis. If `run_config/run_parameters.json` already exists, the software considers the run complete and will not overwrite it.

### 2. Set the HTC path

The same YAML can contain Windows and Linux paths. The current operating system selects the relevant value:

```yaml
paths:
  htc_root:
    windows: "D:/HTC_github_merged/htc"
    linux: "/path/to/htc"
```

`htc_root` must point to the repository folder that contains the `htc` Python package, not to the package's inner `htc/` folder.

### 3. Define the classes and source folders

The keys must match exactly in `experiment_folders`, `labelling_file`, and `hyperguis`. Use absolute paths for experiment folders.

```yaml
experiment_folders:
  stomach: "Z:/study/Cat_0001_stomach"
  liver: "Z:/study/Cat_0004_liver"

labelling_file:
  stomach: "_labelling_001.txt"
  liver: "_labelling_001.txt"

hyperguis:
  stomach: "_hypergui_1"
  liver: "_hypergui_1"
```

The keys (`stomach` and `liver` above) become the model's class names. Exact names or glob patterns may be used for `labelling_file` and `hyperguis`. A recording is ignored if the required HyperGUI directory, labelling TXT file, or spectrum CSV cannot be matched.

### 4. Choose the precomputed spectrum source

Use this exact, case-sensitive setting:

```yaml
data:
  use_L1pixel_normalized_values: true
  expected_channels: 100
  annotation_name: csv_spectrum
```

- `true` selects `_L1pixel/spectrum_fromCSV1_masked_data_L1pixel_0_derivative.csv`.
- `false` selects the original-reflectance median-spectrum files matching `spectrum_fromCSV1_(500.0-995.0)*_masked_data_0_derivative.csv`.

When `true`, the pipeline uses **precomputed median spectra derived from L1-normalized pixel spectra**. It does not calculate L1 normalization. Do not add a second `data:` block and do not use `use_l1pixel` in the source YAML.

The selected and resolved file pattern is recorded in `run_config/resolved_config.json`.

### 5. Configure a full-spectrum basic run

```yaml
external_testing:
  enabled: false

forward_stepwise_analysis: false
reverse_stepwise_analysis: false

combination_analysis:
  enabled: false
  min: 500
  max: 700
  n_wavelengths: 3
  fixed_wavelengths: []

wavelength:
  min: 500
  max: 995
  selective: null
```

With 100 input channels, the usual grid is 500–995 nm. `selective: null` uses all available wavelengths in the configured range.

### 6. Configure the split and training settings

Example reference settings for a basic run are:

```yaml
splitting:
  train_ratio: 0.60
  val_ratio: 0.20
  test_ratio: 0.20
  seed: 42
  search_attempts: 20000
  require_all_classes_in: [train, val, test]

training:
  seed: 42
  max_epochs: 150
  batch_size: 64
  learning_rate: 0.0002
  accelerator: auto
  devices: 1
  precision: 32-true
  num_workers: 0
  epoch_size: null
  standardize: true
  checkpoint_metric: accuracy
  imbalance:
    strategy: balanced_oversampling

evaluation:
  formats: [csv, png, pdf]
  dpi: 300
```

These are example run settings, not universal clinical defaults. The appropriate classes, sample size, imbalance strategy, and training settings depend on the study design.

### 7. Validate before training

Run these commands from the project folder:

```bash
python -m median_pipeline validate --config "/absolute/path/to/scenario/manifold_settings.yaml"
python -m median_pipeline prepare --config "/absolute/path/to/scenario/manifold_settings.yaml"
python -m median_pipeline run --config "/absolute/path/to/scenario/manifold_settings.yaml"
```

- `validate` checks the YAML structure and HTC path.
- `prepare` discovers the spectra and creates the subject split without training.
- `run` performs preparation, training, checkpoint selection, and evaluation.

The wrapper script is equivalent for a complete run:

```bash
python run_pipeline.py --config "/absolute/path/to/scenario/manifold_settings.yaml"
```

Review the split workbooks produced by `prepare` before starting a long run. In particular, confirm the class names, file paths, subjects, and number of spectra in each partition.

## How the three data partitions are used

| Partition | Purpose | What it can influence |
|---|---|---|
| Training | Fits the neural-network parameters | Per-channel standardization, gradient updates, and class-imbalance sampling or weighting |
| Validation | Monitors the trained model after every epoch | Checkpoint selection; it does not contribute to gradient updates or standardization estimates |
| Test | Provides the final held-out assessment | Final metrics only; it does not influence standardization, training, or checkpoint selection |

All spectra from one recognized subject are assigned to only one partition. The software searches up to the configured number of candidate subject assignments, requires all classes in the configured partitions, and retains the assignment with the lowest class-imbalance score. The requested ratios determine rounded numbers of validation and test subjects; the remaining subjects are assigned to training. Spectrum counts may therefore differ from the nominal percentages.

### Per-channel standardization

If `training.standardize: true`, the arithmetic mean and population standard deviation are calculated separately for each selected wavelength using **training spectra only**. A zero standard deviation is replaced by one. The same training-derived values are then applied unchanged to the training, validation, and test spectra.

This z-standardization occurs inside the pipeline and is different from the upstream L1 pixel normalization represented by `use_L1pixel_normalized_values`.

### Important checkpoint detail

The inherited HTC validation code logs the **running cumulative mean of subject-level validation accuracies from the first epoch through the current epoch**. The saved checkpoint with the highest value of this cumulative metric is retained.

It is therefore inaccurate to describe the selected checkpoint as “the checkpoint with maximum current-epoch validation accuracy.” Use this wording instead:

> The retained checkpoint maximized the running cumulative mean of subject-level validation accuracies accumulated from the first epoch through the current epoch.

Validation cross-entropy is recorded for the loss curve, but it does not select the checkpoint and does not control the exponential learning-rate schedule. Early stopping is not implemented; a normally completed run trains for the configured number of epochs.

## Training and class imbalance

The model is optimized with cross-entropy loss. The inherited HTC configuration supplies Adam optimization and an exponential learning-rate schedule; this pipeline sets the learning rate and other run-specific values from the YAML.

Exactly one imbalance strategy can be selected:

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

Balanced oversampling draws spectra with replacement using weights inversely related to class frequency and uses unweighted cross-entropy. Class weighting modifies the loss instead. These strategies are mutually exclusive.

When `epoch_size: null`, the number of samples drawn during an epoch equals the number of spectra in the training partition. Sampling uses replacement, so this does not guarantee that every training spectrum is seen exactly once in each epoch.

## Understanding the results

The most useful files for clinical and study review are:

- `splits/<class>_splits.xlsx`: subject and spectrum assignments for audit before training;
- `output/validation_metrics.csv`: development-set summary from the retained checkpoint;
- `output/testing_metrics.csv`: held-out test summary from the retained checkpoint;
- `output/testing_matrices/confusion_matrix.xlsx`: test confusion matrices as counts and row-normalized fractions;
- `output/testing_matrices/csv/per_class_metrics.csv`: support, precision, recall, F1, and dominant confusion for every test class;
- `output/predictions/test_predictions.csv`: the predicted class and raw model logits for every test spectrum;
- `output/loss_curves/`: training and validation cross-entropy histories;
- `run_config/run_parameters.json`: the resolved configuration, retained checkpoint, metrics, and software provenance.

The prediction files contain raw logits and the class selected by `argmax`. The pipeline does not explicitly convert logits into calibrated probabilities. A larger logit should not be presented as a clinical probability or confidence without a separate calibration study.

### Reported metrics

- **Accuracy:** proportion of spectra assigned the correct configured class.
- **Recall for one class:** proportion of spectra from that class that were correctly identified.
- **Precision for one class:** proportion of predictions for that class that were correct.
- **F1 score:** harmonic mean of precision and recall for one class.
- **Balanced accuracy / macro recall:** mean recall across classes, giving every class equal weight.
- **Macro F1:** mean F1 across classes.

The summary metrics are calculated at the **spectrum level**. The checkpoint-monitoring value during training is subject-level and cumulative, so it is not the same quantity as the final spectrum-level validation accuracy.

Always report the number of subjects and spectra per partition, the per-class support, and the confusion matrix alongside aggregate accuracy. Validation performance is development performance because validation labels influenced checkpoint selection. The held-out test partition provides the final internal estimate, but it is not a substitute for prospective or external clinical validation.

## Output structure

```text
<scenario>/
|-- manifold_settings.yaml
|-- run_config/
|   |-- source_config.yaml
|   |-- resolved_config.json
|   |-- htc_config.json
|   `-- run_parameters.json
|-- splits/
|   `-- <class>_splits.xlsx
`-- output/
    |-- data/
    |   |-- manifest.csv
    |   |-- train.csv
    |   |-- validation.csv
    |   |-- test.csv
    |   |-- split_summary.json
    |   `-- wavelengths.json
    |-- checkpoints/
    |-- logs/
    |-- loss_curves/
    |-- predictions/
    |   |-- validation_predictions.csv
    |   `-- test_predictions.csv
    |-- training_matrices/
    |-- validation_matrices/
    |-- testing_matrices/
    |-- validation_metrics.csv
    |-- testing_metrics.csv
    |-- wavelengths.xlsx
    `-- summary.json
```

Training confusion matrices are generated, but a separate training-predictions CSV is not written. Validation and testing folders contain raw and row-normalized confusion matrices in CSV and Excel formats, with configured PNG/PDF plots. Plot resolution is controlled by `evaluation.dpi`.

## Optional advanced workflows

The following modes are available but are not required for a basic run. Use them only when they are part of a prespecified study design.

### Manual wavelength selection

Exact wavelengths and inclusive ranges can be combined:

```yaml
wavelength:
  min: 500
  max: 995
  selective: "500; 560-720; 875; 900-930"
```

For every selection, the model retains the original input width. Selected channels are standardized and returned to their original positions; unselected positions are set to zero. This preserves model capacity across wavelength selections.

A mapping under `selective` runs several named selections sequentially while reusing one discovered dataset and subject split:

```yaml
wavelength:
  min: 500
  max: 995
  selective:
    case_1: "630; 810"
    case_2: "575; 650"
```

### Forward or reverse stepwise analysis

```yaml
forward_stepwise_analysis: true
forward_stop_wavelength: 700
reverse_stepwise_analysis: false
reverse_stop_wavelength: 850
```

Stepwise child runs reuse the same discovered data and subject split. They are exploratory wavelength analyses, not cross-validation folds.

### Exhaustive wavelength combinations

```yaml
combination_analysis:
  enabled: true
  min: 500
  max: 700
  n_wavelengths: 3
  fixed_wavelengths: []
```

This mode trains every unordered wavelength combination of the requested size and can create thousands of complete training runs. Estimate the number of combinations and computational cost before enabling it. Combination mode cannot be combined with manual selective, selective-batch, or stepwise modes.

### External test cohort

With `external_testing.enabled: true`, the primary subjects are divided into training and validation only, while every spectrum from the configured external cohort becomes test data. Training-derived standardization is applied unchanged to that cohort. Primary and external files must not overlap and must use the same wavelength grid.

The external true classes may differ from the trained classes. Confusion-matrix rows represent external true classes and columns represent trained prediction classes. An external class name is counted as correct only if it exactly matches a trained class name.

## Common problems

### “Original HTC source was not found”

Check that `paths.htc_root` points to a repository containing `htc/__init__.py` and that the correct Windows or Linux path is present.

### No spectra were discovered

Check all four items together:

1. the absolute experiment-folder path;
2. the `hyperguis` name or glob;
3. the labelling TXT name or glob beside that HyperGUI directory;
4. the selected spectrum source, especially `use_L1pixel_normalized_values`.

### A subject could not be identified

The source path must contain a supported subject token such as `P<number>`, `M<number>`, or `Cat_<number>_...`. Correct the source folder convention or extend `subject_from_path()` in `median_pipeline/preparation.py` deliberately and test the change.

### A class cannot cover all splits

Every required class must occur in enough distinct subjects to populate the configured partitions. Adding more spectra from the same subject does not solve a shortage of distinct subjects. Adjusting `require_all_classes_in` changes the study design and should be justified, not used merely to bypass the safeguard.

### The scenario has already completed

The pipeline will not overwrite a scenario containing `run_config/run_parameters.json`. Preserve the completed folder and create a new scenario folder for the new run.

### Results are unexpectedly different

Compare `run_config/source_config.yaml`, `resolved_config.json`, `htc_config.json`, and `run_parameters.json`. Also compare the saved manifest SHA-256 digest and the HTC source fingerprint. Reproducible seeds reduce random variation but do not guarantee identical results across all hardware, operating systems, and library versions.

## Reproducibility and safeguards

- Subject overlap between training, validation, and internal test partitions is prohibited.
- The same wavelength grid and selected indices are required for all spectra.
- Spectra with the wrong number of channels or non-finite values are rejected.
- A spectrum file cannot belong to more than one configured class.
- The source YAML and fully resolved configuration are saved for every run.
- Manifests, split summaries, wavelength selections, software versions, and HTC fingerprints are recorded.
- The original HTC repository is treated as immutable.

`source_baseline.json` and `verify_source_integrity.py` provide an additional integrity check for the recorded HTC source files. See `SOURCE_PROVENANCE.md` for details.

## Support handover checklist

When sharing a run with a clinician, statistician, or collaborator, provide:

- the clinical question and prespecified primary outcome;
- definitions of every configured class;
- subject and spectrum counts for all partitions;
- the split workbooks and test confusion matrix;
- aggregate and per-class metrics with uncertainty estimates calculated for the study, if applicable;
- the source YAML and `run_config/` provenance files;
- a clear statement that the outputs are research results and not diagnostic predictions for patient care.
