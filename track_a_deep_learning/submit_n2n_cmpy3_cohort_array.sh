#!/bin/bash
#SBATCH --job-name=cmpy3_found_n2n
#SBATCH --partition=high
#SBATCH --account=millerlmgrp
#SBATCH --qos=millerlmgrp-high-qos
#SBATCH --time=00:15:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-26%8
#SBATCH --output=logs/cohort_cmpy3_n2n_%A_%a.out
#SBATCH --error=logs/cohort_cmpy3_n2n_%A_%a.err

cd /home/evliang/cidenoise
export PYTHONPATH="/home/evliang/cidenoise/track_a_deep_learning:${PYTHONPATH}"
mkdir -p logs benchmark_results/sobi_n2n_cmpy3_cohort

source "/cvmfs/hpc.ucdavis.edu/sw/conda/root/etc/profile.d/conda.sh"
conda activate ci_denoise_env

echo "=========================================================="
echo "EVALUATING FOUNDATION N2N ON CMPY3 COHORT TASK ${SLURM_ARRAY_TASK_ID}"
echo "Host: $(hostname) | Date: $(date)"
echo "=========================================================="

python -u track_a_deep_learning/evaluate_sobi_n2n_cohort.py \
    --data-dir    /quobyte/millerlmgrp/annotated_data/CMPy3 \
    --sobi-dir    ./benchmark_results/sobi_cmpy3_cohort \
    --year-tag    y3 \
    --task-id     ${SLURM_ARRAY_TASK_ID} \
    --checkpoint  ./benchmark_results/sobi_n2n_foundation/best_sobi_n2n_foundation_unet.pt \
    --output-dir  ./benchmark_results/sobi_n2n_cmpy3_cohort

echo "CMPy3 N2N Task ${SLURM_ARRAY_TASK_ID} complete!"
