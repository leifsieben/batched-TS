#!/bin/bash

#SBATCH -J ts           # Job name
#SBATCH -c 8                      # Request CPU cores
#SBATCH --mem=128G                   # Request 64 GB of memory (adjust if needed)
#SBATCH -t 12:00:00                 # Set a time limit for the job (48 hours)
#SBATCH --gres=gpu

# Run the Python script
python -u ts_main.py runs/MiniMol_example_Andrew_ckpts.json

