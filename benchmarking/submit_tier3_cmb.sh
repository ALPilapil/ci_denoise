#!/bin/bash
#SBATCH --job-name=bench_tier3_cmb
#SBATCH --partition=high
#SBATCH --account=cmb-grp
#SBATCH --time=08:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

cd /home/evliang/cidenoise
export PYTHONPATH="/home/evliang/cidenoise:${PYTHONPATH}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

mkdir -p logs
mkdir -p benchmark_results/tier3

source "/cvmfs/hpc.ucdavis.edu/sw/conda/root/etc/profile.d/conda.sh"
conda activate ci_denoise_env

python benchmarking/tier3_downstream_eval.py \
    --zarr-path       /quobyte/millerlmgrp/processed_data/raw_epoched_data.zarr \
    --checkpoint-dir  /quobyte/millerlmgrp/checkpoints \
    --checkpoints     best.pt \
    --batch-size      16 \
    --num-workers     0 \
    --device          cpu \
    --output-dir      ./benchmark_results/tier3

