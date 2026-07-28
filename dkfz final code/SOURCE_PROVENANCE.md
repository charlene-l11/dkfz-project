# Source reuse and provenance

The merged project is intentionally separate from both HTC trees. Neither source tree is modified.

## Immutable HTC sources

- Windows: `C:/Users/c758g/Downloads/htc-main`
- Linux/cluster: `/omics/groups/OE0645/internal/HSI_RISE/htc`
- Local cluster reference: `C:/Users/c758g/Documents/scripts/cluster scripts/htc`

The relevant HTC APIs are almost identical. The Windows `htc-main` copy includes safer handling for zero-worker data loaders and Windows-locked training-result folders. The cluster copy remains the configured Linux implementation. The platform-specific `htc_root` in the YAML selects the correct source tree at runtime.

## Reused from cluster scripts

- `median_pipeline/dataset.py` retains the CSV-backed `DatasetCatPigMedianPixel` implementation.
- `median_pipeline/lightning_module.py` retains the minimal HTC median-pixel Lightning subclass.
- `median_pipeline/preparation.py` retains CSV discovery, spectrum parsing, subject extraction, grouped splitting, manifest creation, and label mapping.
- `median_pipeline/training.py` retains the HTC `Config`, original median-pixel model, Lightning trainer, checkpointing, standardization, weighting, oversampling, and test-result workflow.
- Timestamped run metadata and raw/normalized confusion matrices are retained and extended.

## New integration components

- One cross-platform YAML configuration.
- Strict subject and class-coverage validation.
- Mutually exclusive imbalance strategies.
- Manual exact-wavelength and inclusive-range selection with saved channel metadata.
- Automatic prepare, train, best-checkpoint test, and evaluation workflow.
- Per-class metrics and organized CSV, PNG, and PDF outputs.
- VS Code tasks/debug configuration and a Microsoft Visual Studio solution/project.
- SHA-256 source baselines and runtime HTC-source fingerprints.

## Intentionally excluded

- Modifications or patches to either HTC tree.
- Excel-to-adapter conversion, because the pipeline consumes existing median-spectrum CSV files.
- Hard-coded prepared-run directories.
- Separate manual confusion-matrix commands.

