#!/bin/bash -l
#SBATCH --job-name=meta_bo_generator
#SBATCH --output=logs/generator-%j.out
#SBATCH --error=logs/generator-%j.err
#SBATCH --mail-user=""
#SBATCH --mail-type=FAIL
#SBATCH --partition=cpu-short
#SBATCH --time=00:20:00
#SBATCH --mem=5G

conda activate pcabo

# Run meta-bo iteration, save candidates as a csv
python meta_bo.py
