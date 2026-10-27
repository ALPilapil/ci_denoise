#!/bin/bash
#SBATCH --job-name=track_b_cpu
#SBATCH --partition=high
#SBATCH --account=publicgrp
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# NOTE: Track B DOES NOT REQUIRE A GPU (100% CPU-bound).
# The regularized FIR Wiener filter solves a tiny 30x30 matrix equation per recording.
# Inversion takes <1 millisecond in NumPy on CPU.
# Moving data to a GPU would actually be slower due to PCIe memory transfer overhead.

cd /home/evliang/cidenoise
export PYTHONPATH="/home/evliang/cidenoise/track_b_adaptive_subtraction:${PYTHONPATH}"

mkdir -p logs
mkdir -p /quobyte/millerlmgrp/aats_cleaned_data
mkdir -p /quobyte/millerlmgrp/aats_isolated_noise
mkdir -p benchmark_results/track_b

source "/cvmfs/hpc.ucdavis.edu/sw/conda/root/etc/profile.d/conda.sh"
conda activate ci_denoise_env

echo "=========================================================="
echo "STEP 1: Running AATS Adaptive Denoising on Annotated Files (CPU)"
echo "=========================================================="

python track_b_adaptive_subtraction/denoise_fif.py \
    --data-dir   /quobyte/millerlmgrp/annotated_data \
    --clean-dir  /quobyte/millerlmgrp/aats_cleaned_data \
    --noise-dir  /quobyte/millerlmgrp/aats_isolated_noise \
    --year       2 \
    --n-taps     6 \
    --reg-lambda 1e-3

echo ""
echo "=========================================================="
echo "STEP 2: Evaluating Real ERP Preservation (Raw vs AATS vs CCA)"
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
echo "Track B CPU Pipeline Finished! Results saved to ./benchmark_results/track_b/"
