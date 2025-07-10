#!/bin/bash -l
#SBATCH --job-name=pcabo
#SBATCH --output=logs/pcabo_%A_%a.out
#SBATCH --error=logs/pcabo_%A_%a.err
#SBATCH --mail-user=""
#SBATCH --mail-type=FAIL
#SBATCH --partition=cpu-short
#SBATCH --time=00:45:00
#SBATCH --ntasks=1
#SBATCH --array=0-899
#SBATCH --constraint=AMD.Zen4
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

conda activate pcabo

algo="pca"
batch=10
dims=(10 20 40)
pids=({15..24})
iids=({0..29})

n_dims=${#dims[@]}
n_pids=${#pids[@]}
n_iids=${#iids[@]}

dim_idx=$((SLURM_ARRAY_TASK_ID / (n_pids * n_iids)))
pid_idx=$(((SLURM_ARRAY_TASK_ID % (n_pids * n_iids)) / n_iids))
iid_idx=$((SLURM_ARRAY_TASK_ID % n_iids))

dim=${dims[$dim_idx]}
pid=${pids[$pid_idx]}
iid=${iids[$iid_idx]}

echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Using algo=$algo batch=$batch dim=$dim pid=$pid iid=$iid"

python ../main.py \
  --algorithms $algo \
  --batch $batch \
  --dimensions $dim \
  --problems $pid \
  --instances $iid \
  --gpr_p 0.456144 \
  --gpr_val_factor 0.0 \
  --onorm_factor 6.875751
