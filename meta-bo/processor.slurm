#!/bin/bash -l
#SBATCH --job-name=meta_bo_processor
#SBATCH --output=logs/processor-%j.out
#SBATCH --error=logs/processor-%j.err
#SBATCH --mail-user=""
#SBATCH --mail-type=FAIL
#SBATCH --partition=cpu-short
#SBATCH --time=00:20:00
#SBATCH --mem=1G

conda activate pcabo

# Write slurms for candidates from the csv
python write_slurms.py

# Send candidate slurms
exp_job_ids=()
for slurm_file in ./cands/*.slurm; do
    if [ -f "$slurm_file" ]; then
        exp_job_id=$(sbatch --parsable "$slurm_file")
        exp_job_ids+=($exp_job_id)
    fi
done

# Exit if no jobs sent
if [ ${#exp_job_ids[@]} -e 0 ]; then
    echo "No experiment jobs to send. Stopping loop."
    exit 1
fi

job_id_list=$(IFS=:; echo "${exp_job_ids[*]}")
echo "All experiment jobs sent. IDs: $job_id_list"

# Restart loop (generator)
gen_job_id=$(sbatch --parsable --dependency=afterok:$job_id_list generator.slurm)
echo "Queued generator as job $gen_job_id"

# Restart loop (processor)
proc_job_id=$(sbatch --parsable --dependency=afterok:$gen_job_id processor.slurm)
echo "Queued processor as job $proc_job_id"
