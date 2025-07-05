#!/bin/bash -l
#SBATCH --job-name=opcabo
#SBATCH --output=logs/opcabo_%A_%a.out
#SBATCH --error=logs/opcabo_%A_%a.err
#SBATCH --mail-user=""
#SBATCH --mail-type=FAIL
#SBATCH --partition=cpu-short
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --array=0-95
#SBATCH --constraint=AMD.Zen4
#SBATCH --cpus-per-task=1
#SBATCH --mem=1536M

conda activate pcabo

algo="opca"
batch=5
dims=(10 20 40)
pids=(15 16 17 18 20 21 22 24)
iids=({0..3})

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

python main.py \
  --algorithms $algo \
  --batch $batch \
  --dimensions $dim \
  --problems $pid \
  --instances $iid \
  --experiment_dir "meta-bo-b$batch" \
  --gpr_p <GPRPTEAPOT> \
  --gpr_val_factor <GPRVALTEAPOT> \
  --onorm_factor <ONORMTEAPOT>
