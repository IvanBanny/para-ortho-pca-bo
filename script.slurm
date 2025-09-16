#!/bin/bash
#SBATCH --job-name=test_python_simple
#SBATCH --partition=cpu-medium
#SBATCH --output=out/%j.out
#SBATCH --mail-user="adela.greganova@gmail.com"
#SBATCH --mail-type="ALL"
#SBATCH --mem=1G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=1-72
#SBATCH --time=24:00:00

source ~/.bashrc

conda activate myenv

# Define dimensions array
DIMS=(40 20 10)

# Calculate problem and dimension indices
PROBLEM_ID=$(( (($SLURM_ARRAY_TASK_ID - 1) / 3) + 1 ))
DIM_INDEX=$(( ($SLURM_ARRAY_TASK_ID - 1) % 3 ))
DIMENSION=${DIMS[$DIM_INDEX]}

python main.py --experiment_dir EXPERIMENT37-${PROBLEM_ID}-${DIMENSION}D --problems $PROBLEM_ID --dimensions $DIMENSION

# 32: with extra doe
# 33: without extra doe
# 34: PCA
# 36: my-vanilla
# 37: lpca with extra doe and consider all points for gpr

# zip -r experiments_$(date +%Y%m%d_%H%M%S).zip clean-*pca-EXPERIMENT9-[0-9]*-[0-9]*D