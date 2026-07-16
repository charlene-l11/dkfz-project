#!/bin/bash

# Load shell settings
source /home/c758g/.bashrc

# Load the same Python version used to create the virtual environment
module load Python/3.12.4-GCCcore-14.1.0

# Go to the project workspace
cd /omics/groups/OE0645/internal/HSI_RISE || exit 1

# Activate the cluster virtual environment
source .venv/bin/activate

# Run data preparation
python scripts/prepare_data.py \
  --config-file manifold_settings.yaml


######################################
# Cluster Wiki
# Bash scripting how to navigate etc
# LSF commands
######################################

