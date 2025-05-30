#!/bin/bash
#SBATCH --job-name=test_python_simple
#SBATCH --partition=cpu-medium
#SBATCH --output=out/%j.out
#SBATCH --mail-user="adela.greganova@gmail.com"
#SBATCH --mail-type="ALL"
#SBATCH --mem=2G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --array=1-24
#SBATCH --time=24:00:00

conda activate myenv

python main.py --experiment_dir EXPERIMENT6-$SLURM_ARRAY_TASK_ID --verbose --problems $SLURM_ARRAY_TASK_ID --dimensions 10 20 40 --runs 10
