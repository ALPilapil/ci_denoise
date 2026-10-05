#!/bin/bash
#SBATCH --job-name=bench_tier1_screening
#SBATCH --partition=high
#SBATCH --account=publicgrp
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

cd /home/evliang/cidenoise
export PYTHONPATH="/home/evliang/cidenoise:/home/evliang/cidenoise/processing:${PYTHONPATH}"

mkdir -p logs
mkdir -p benchmark_results/tier1

source "/cvmfs/hpc.ucdavis.edu/sw/conda/root/etc/profile.d/conda.sh"
conda activate ci_denoise_env

python benchmarking/tier1_screening.py \
    --zarr-path /quobyte/millerlmgrp/processed_data/raw_epoched_data.zarr \
    --output-dir ./benchmark_results/tier1 \
    --methods ica pca ssp wavelet \
    --device cpu \
    --num-epochs 50 \
    --alpha 1.0
