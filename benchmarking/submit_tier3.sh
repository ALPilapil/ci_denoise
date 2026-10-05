#!/bin/bash
#SBATCH --job-name=bench_tier3_downstream
#SBATCH --partition=high
#SBATCH --account=publicgrp
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a6000:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

cd /home/evliang/cidenoise
export PYTHONPATH="/home/evliang/cidenoise:/home/evliang/cidenoise/modeling:${PYTHONPATH}"

mkdir -p logs
mkdir -p benchmark_results/tier3

source "/cvmfs/hpc.ucdavis.edu/sw/conda/root/etc/profile.d/conda.sh"
conda activate ci_denoise_env

python benchmarking/tier3_downstream_eval.py \
    --zarr-path       /quobyte/millerlmgrp/processed_data/raw_epoched_data.zarr \
    --checkpoint-dir  /quobyte/millerlmgrp/checkpoints \
    --checkpoints     best.pt \
    --batch-size      8 \
    --num-workers     0 \
    --device          cuda \
    --output-dir      ./benchmark_results/tier3
