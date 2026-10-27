#!/bin/bash
#SBATCH --job-name=track_a_gpu
#SBATCH --partition=high
#SBATCH --account=publicgrp
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a6000:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# NOTE: Track A Training REQUIRES / STRONGLY BENEFITS FROM A GPU.
# On an NVIDIA A6000 GPU, 50 epochs takes ~1-2 hours.
# On CPU, the same run takes 24-48 hours.

cd /home/evliang/cidenoise
export PYTHONPATH="/home/evliang/cidenoise/track_a_deep_learning:${PYTHONPATH}"

mkdir -p logs
mkdir -p /quobyte/millerlmgrp/checkpoints/track_a
mkdir -p benchmark_results/track_a

source "/cvmfs/hpc.ucdavis.edu/sw/conda/root/etc/profile.d/conda.sh"
conda activate ci_denoise_env

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "=========================================================="
echo "TRACK A: Training 1D U-Net with Composite Loss on GPU"
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
    --device          cuda \
    --num-workers     2 \
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
    --device          cuda \
    --output-dir      ./benchmark_results/track_a

echo "Track A GPU Pipeline Finished Successfully!"
