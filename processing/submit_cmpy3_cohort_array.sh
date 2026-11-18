#!/bin/bash
#SBATCH --job-name=cmpy3_sobi
#SBATCH --partition=high
#SBATCH --account=millerlmgrp
#SBATCH --qos=millerlmgrp-high-qos
#SBATCH --time=00:15:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-26%8
#SBATCH --output=logs/cmpy3_sobi_%A_%a.out
#SBATCH --error=logs/cmpy3_sobi_%A_%a.err

cd /home/evliang/cidenoise
export PYTHONPATH="/home/evliang/cidenoise/processing:${PYTHONPATH}"
mkdir -p logs benchmark_results/sobi_cmpy3_cohort

source "/cvmfs/hpc.ucdavis.edu/sw/conda/root/etc/profile.d/conda.sh"
conda activate ci_denoise_env

echo "=========================================================="
echo "RUNNING CMPY3 SOBI PIPELINE TASK ${SLURM_ARRAY_TASK_ID}"
echo "Host: $(hostname) | Date: $(date)"
echo "=========================================================="

python -u processing/run_cohort_pipeline.py \
    --cohort-type ci \
    --data-dir    /quobyte/millerlmgrp/annotated_data/CMPy3_extracted \
    --year-tag    y3 \
    --task-id     ${SLURM_ARRAY_TASK_ID} \
    --output-dir  ./benchmark_results/sobi_cmpy3_cohort

echo "CMPy3 Task ${SLURM_ARRAY_TASK_ID} complete!"
