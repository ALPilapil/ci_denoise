#!/bin/bash
#SBATCH --job-name=audit_trials
#SBATCH --partition=high
#SBATCH --account=millerlmgrp
#SBATCH --qos=millerlmgrp-high-qos
#SBATCH --time=00:10:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

cd /home/evliang/cidenoise
mkdir -p logs benchmark_results/pipeline_audit

source "/cvmfs/hpc.ucdavis.edu/sw/conda/root/etc/profile.d/conda.sh"
conda activate ci_denoise_env

python -u processing/audit_trial_structure_and_polarities.py \
    --data-dir /quobyte/millerlmgrp/annotated_data/CMPy2 \
    --subject 0901y2
