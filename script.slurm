#!/bin/bash
#SBATCH --job-name=test_python_simple
#SBATCH --output=out/%j.out
#SBATCH --mail-user="adela.greganova@gmail.com"
#SBATCH --mail-type="ALL"
#SBATCH --partition=testing
#SBATCH --mem=4G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

conda activate myenv

python main.py
