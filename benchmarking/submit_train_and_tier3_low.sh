#!/bin/bash
#SBATCH --job-name=train_tier3_low
#SBATCH --partition=low
#SBATCH --account=publicgrp
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

cd /home/evliang/cidenoise
export PYTHONPATH="/home/evliang/cidenoise:/home/evliang/cidenoise/modeling:${PYTHONPATH}"

mkdir -p logs
mkdir -p /quobyte/millerlmgrp/checkpoints
mkdir -p benchmark_results/tier3

source "/cvmfs/hpc.ucdavis.edu/sw/conda/root/etc/profile.d/conda.sh"
conda activate ci_denoise_env

echo "=========================================================="
echo "STEP 1: Training 1D U-Net on Low Partition (GPU Accelerated)"
echo "=========================================================="

python modeling/train.py \
    --zarr-path       /quobyte/millerlmgrp/processed_data/raw_epoched_data.zarr \
    --checkpoint-dir  /quobyte/millerlmgrp/checkpoints \
    --clean-group     hearing_trial_data \
    --noise-group     ci_trial_data \
    --batch-size      2 \
    --epochs          50 \
    --lr              1e-3 \
    --num-workers     2 \
    --device          cuda \
    --resume

echo ""
echo "=========================================================="
echo "STEP 2: Evaluating Trained Checkpoint with Tier 3 Benchmark"
echo "=========================================================="

python benchmarking/tier3_downstream_eval.py \
    --zarr-path       /quobyte/millerlmgrp/processed_data/raw_epoched_data.zarr \
    --checkpoint-dir  /quobyte/millerlmgrp/checkpoints \
    --checkpoints     best.pt \
    --batch-size      16 \
    --num-workers     0 \
    --device          cuda \
    --output-dir      ./benchmark_results/tier3

echo ""
echo "=========================================================="
echo "Job Completed Successfully! Results saved to ./benchmark_results/tier3/"
echo "=========================================================="
