#!/bin/bash
#SBATCH --job-name=test_python_simple
#SBATCH --partition=cpu-medium
#SBATCH --output=out/%j.out
#SBATCH --mail-user="adela.greganova@gmail.com"
#SBATCH --mail-type="ALL"
#SBATCH --mem=2G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --array=1-72
#SBATCH --time=24:00:00

conda activate myenv

# Define dimensions array
DIMS=(10 20 40)

# Calculate problem and dimension indices
PROBLEM_ID=$(( (($SLURM_ARRAY_TASK_ID - 1) / 3) + 1 ))
DIM_INDEX=$(( ($SLURM_ARRAY_TASK_ID - 1) % 3 ))
DIMENSION=${DIMS[$DIM_INDEX]}

python main.py --experiment_dir EXPERIMENT13-${PROBLEM_ID}-${DIMENSION}D --problems $PROBLEM_ID --dimensions $DIMENSION --runs 5

# zip -r experiments_$(date +%Y%m%d_%H%M%S).zip EXPERIMENT9-[0-9]*-[0-9]*D