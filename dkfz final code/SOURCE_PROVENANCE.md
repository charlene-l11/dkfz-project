# Source reuse and provenance

The merged project is intentionally separate from both HTC trees. Neither source tree is modified.

## Authoritative immutable HTC runtime

- Windows: `C:/Users/c758g/Documents/scripts/cluster scripts/htc`
- Linux/cluster: `/omics/groups/OE0645/internal/HSI_RISE/htc`

The platform-specific `htc_root` in the YAML always selects this original `htc` codebase at runtime. No installed or source HTC file is modified.

## Reference-only source

- `C:/Users/c758g/Downloads/htc-main`

`htc-main` is not a runtime base and is never placed on the pipeline import path. Only selected functionality and implementation ideas are reused in the separate merged-pipeline files where appropriate.

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



