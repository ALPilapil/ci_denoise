#!/bin/bash
#SBATCH --job-name=track_a_cpu
#SBATCH --partition=high
#SBATCH --account=cmb-grp
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# NOTE: CPU fallback for Track A on cmb-grp when GPUs are occupied.
# Allocates 16 cores and 128GB RAM.

cd /home/evliang/cidenoise
export PYTHONPATH="/home/evliang/cidenoise/track_a_deep_learning:${PYTHONPATH}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

mkdir -p logs
mkdir -p /quobyte/millerlmgrp/checkpoints/track_a
mkdir -p benchmark_results/track_a

source "/cvmfs/hpc.ucdavis.edu/sw/conda/root/etc/profile.d/conda.sh"
conda activate ci_denoise_env

echo "=========================================================="
echo "TRACK A: Training 1D U-Net with Composite Loss on CPU"
echo "=========================================================="

python track_a_deep_learning/train.py \
    --zarr-path       /quobyte/millerlmgrp/processed_data/raw_epoched_data.zarr \
    --checkpoint-dir  /quobyte/millerlmgrp/checkpoints/track_a \
    --clean-group     hearing_trial_data \
    --noise-group     ci_trial_data \
    --batch-size      2 \
    --epochs          50 \
    --lr              1e-3 \
    --loss-type       composite \
    --lambda-spec     0.5 \
    --lambda-corr     0.2 \
    --lambda-erp      0.3 \
    --split-strategy  participant \
    --device          cpu \
    --num-workers     0 \
    --resume

echo ""
echo "=========================================================="
echo "TRACK A: Evaluating Trained Model on Held-Out Test Split"
echo "=========================================================="

python track_a_deep_learning/evaluate_downstream.py \
    --zarr-path       /quobyte/millerlmgrp/processed_data/raw_epoched_data.zarr \
    --checkpoint-dir  /quobyte/millerlmgrp/checkpoints/track_a \
    --checkpoints     best.pt \
    --batch-size      16 \
    --device          cpu \
    --output-dir      ./benchmark_results/track_a

echo "Track A CPU Pipeline Finished!"
