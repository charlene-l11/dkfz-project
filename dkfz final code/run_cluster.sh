#!/bin/bash
set -euo pipefail

source /home/c758g/.bashrc
module load Python/3.12.4-GCCcore-14.1.0
cd "$(dirname "$0")"
source .venv/bin/activate
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /path/to/scenario/manifold_settings.yaml" >&2
  exit 2
fi
python -m median_pipeline run --config "$1"
