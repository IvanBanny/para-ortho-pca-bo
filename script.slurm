#!/bin/bash
#SBATCH --job-name=test_python_simple
#SBATCH --partition=cpu-short
#SBATCH --output=out/%j.out
#SBATCH --mail-user="adela.greganova@gmail.com"
#SBATCH --mail-type="ALL"
#SBATCH --mem=2G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --array=1-24
#SBATCH --time=04:00:00

conda activate myenv

python main.py --experiment_dir EXPERIMENT3-$SLURM_ARRAY_TASK_ID --verbose --problems $SLURM_ARRAY_TASK_ID --dimensions 5 20
