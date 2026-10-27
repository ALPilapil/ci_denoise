#!/bin/bash
#SBATCH --job-name=track_b_eval
#SBATCH --partition=high
#SBATCH --account=publicgrp
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Fast evaluation of real ERP preservation (Raw vs AATS vs Classical)
# Assumes AATS cleaned files already exist in /quobyte/millerlmgrp/aats_cleaned_data/
# Runtime: ~2-5 minutes on CPU.

cd /home/evliang/cidenoise
export PYTHONPATH="/home/evliang/cidenoise/track_b_adaptive_subtraction:${PYTHONPATH}"

mkdir -p logs
mkdir -p benchmark_results/track_b

source "/cvmfs/hpc.ucdavis.edu/sw/conda/root/etc/profile.d/conda.sh"
conda activate ci_denoise_env

echo "=========================================================="
echo "EVALUATING REAL ERP PRESERVATION (Raw vs AATS vs Classical)"
echo "=========================================================="

python track_b_adaptive_subtraction/evaluate_erp.py \
    --annotated-dir  /quobyte/millerlmgrp/annotated_data \
    --aats-dir       /quobyte/millerlmgrp/aats_cleaned_data \
    --classical-root /quobyte/millerlmgrp/ml_cleaned_data \
    --zarr-path      /quobyte/millerlmgrp/processed_data/raw_epoched_data.zarr \
    --output-dir     ./benchmark_results/track_b \
    --condition      A \
    --target-channel Cz \
    --year           2

echo ""
echo "Track B Evaluation Complete! Results saved to ./benchmark_results/track_b/"
