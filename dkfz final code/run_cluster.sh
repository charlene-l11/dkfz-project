#!/bin/bash
set -euo pipefail

source /home/c758g/.bashrc
module load Python/3.12.4-GCCcore-14.1.0
cd "$(dirname "$0")"
source .venv/bin/activate
python -m median_pipeline run --config configs/manifold_settings.yaml
