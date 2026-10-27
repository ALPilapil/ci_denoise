#!/bin/bash
#SBATCH --job-name=train_tier3_cmb
#SBATCH --partition=high
#SBATCH --account=cmb-grp
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

cd /home/evliang/cidenoise
export PYTHONPATH="/home/evliang/cidenoise:/home/evliang/cidenoise/modeling:${PYTHONPATH}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

mkdir -p logs
mkdir -p /quobyte/millerlmgrp/checkpoints
mkdir -p benchmark_results/tier3

source "/cvmfs/hpc.ucdavis.edu/sw/conda/root/etc/profile.d/conda.sh"
conda activate ci_denoise_env

echo "=========================================================="
echo "STEP 1: Training 1D U-Net on cmb-grp (CPU, 16 Cores, 128G)"
echo "=========================================================="

python modeling/train.py \
    --zarr-path       /quobyte/millerlmgrp/processed_data/raw_epoched_data.zarr \
    --checkpoint-dir  /quobyte/millerlmgrp/checkpoints \
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
    --num-workers     0 \
    --device          cpu \
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
    --device          cpu \
    --output-dir      ./benchmark_results/tier3

echo ""
echo "=========================================================="
echo "Job Completed Successfully! Results saved to ./benchmark_results/tier3/"
echo "=========================================================="
