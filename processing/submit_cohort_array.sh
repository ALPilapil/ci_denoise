#!/bin/bash
#SBATCH --job-name=cohort_sobi
#SBATCH --partition=high
#SBATCH --account=millerlmgrp
#SBATCH --qos=millerlmgrp-high-qos
#SBATCH --time=00:15:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-36%8
#SBATCH --output=logs/cohort_%A_%a.out
#SBATCH --error=logs/cohort_%A_%a.err

cd /home/evliang/cidenoise
export PYTHONPATH="/home/evliang/cidenoise/processing:${PYTHONPATH}"
mkdir -p logs benchmark_results/sobi_polarity_cohort

source "/cvmfs/hpc.ucdavis.edu/sw/conda/root/etc/profile.d/conda.sh"
conda activate ci_denoise_env

echo "=========================================================="
echo "RUNNING COHORT PIPELINE TASK ${SLURM_ARRAY_TASK_ID}"
echo "Host: $(hostname) | Date: $(date)"
echo "=========================================================="

python -u processing/run_cohort_pipeline.py \
    --cohort-type ci \
    --task-id     ${SLURM_ARRAY_TASK_ID} \
    --output-dir  ./benchmark_results/sobi_polarity_cohort

echo "Task ${SLURM_ARRAY_TASK_ID} complete!"
