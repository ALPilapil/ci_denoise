#!/bin/bash
#SBATCH --job-name=cohort_n2n
#SBATCH --partition=high
#SBATCH --account=millerlmgrp
#SBATCH --qos=millerlmgrp-high-qos
#SBATCH --time=00:15:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-36%8
#SBATCH --output=logs/cohort_n2n_%A_%a.out
#SBATCH --error=logs/cohort_n2n_%A_%a.err

cd /home/evliang/cidenoise
export PYTHONPATH="/home/evliang/cidenoise/track_a_deep_learning:${PYTHONPATH}"
mkdir -p logs benchmark_results/sobi_n2n_cohort

source "/cvmfs/hpc.ucdavis.edu/sw/conda/root/etc/profile.d/conda.sh"
conda activate ci_denoise_env

echo "=========================================================="
echo "RUNNING N2N COHORT EVALUATION TASK ${SLURM_ARRAY_TASK_ID}"
echo "Host: $(hostname) | Date: $(date)"
echo "=========================================================="

python -u track_a_deep_learning/evaluate_sobi_n2n_cohort.py \
    --task-id     ${SLURM_ARRAY_TASK_ID} \
    --checkpoint  ./benchmark_results/sobi_n2n_results/best_sobi_n2n_unet.pt \
    --output-dir  ./benchmark_results/sobi_n2n_cohort

echo "Task ${SLURM_ARRAY_TASK_ID} complete!"
