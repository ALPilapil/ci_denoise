#!/bin/bash
#SBATCH --job-name=bench_tier2_erp
#SBATCH --partition=high
#SBATCH --account=publicgrp
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

cd /home/evliang/cidenoise
export PYTHONPATH="/home/evliang/cidenoise:/home/evliang/cidenoise/processing:${PYTHONPATH}"

mkdir -p logs
mkdir -p benchmark_results/tier2

source "/cvmfs/hpc.ucdavis.edu/sw/conda/root/etc/profile.d/conda.sh"
conda activate ci_denoise_env

python benchmarking/tier2_erp_eval.py \
    --annotated-dir /quobyte/millerlmgrp/annotated_data \
    --cleaned-root  /quobyte/millerlmgrp/ml_cleaned_data \
    --hearing-dir   /quobyte/millerlmgrp/processed_data/hearing \
    --zarr-path     /quobyte/millerlmgrp/processed_data/raw_epoched_data.zarr \
    --output-dir    ./benchmark_results/tier2 \
    --condition     A \
    --target-channel Cz \
    --methods       raw ica pca cca ssp wavelet \
    --year          2 \
    --save-plots
