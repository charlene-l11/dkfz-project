#!/bin/bash

# Load shell settings
source /home/c758g/.bashrc

# Load the Python version used to create the virtual environment
module load Python/3.12.4-GCCcore-14.1.0

# Go to the project workspace
cd /omics/groups/OE0645/internal/HSI_RISE || exit 1

# Activate the cluster virtual environment
source .venv/bin/activate

# Run model training
python scripts/train_model.py \
  --data-dir /omics/groups/OE0645/internal/HSI_RISE/training_runs/16072026_1458_prepCSV