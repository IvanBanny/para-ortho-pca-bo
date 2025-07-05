#!/bin/bash
# This script starts the meta-bo loop
echo -e "\nStarting the meta-bo loop\n"

gen_job_id=$(sbatch --parsable generator.slurm)
echo "Sent generator as job $gen_job_id"

proc_job_id=$(sbatch --parsable --dependency=afterok:$gen_job_id processor.slurm)
echo "Queued processor as job $proc_job_id"

echo -e "\nLoop initialized. Good.\n"
