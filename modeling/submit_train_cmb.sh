#!/bin/bash
#SBATCH --job-name=train_unet_cpu
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

source "/cvmfs/hpc.ucdavis.edu/sw/conda/root/etc/profile.d/conda.sh"
conda activate ci_denoise_env

python modeling/train.py \
    --zarr-path       /quobyte/millerlmgrp/processed_data/raw_epoched_data.zarr \
    --checkpoint-dir  /quobyte/millerlmgrp/checkpoints \
    --clean-group     hearing_trial_data \
    --noise-group     ci_trial_data \
    --batch-size      2 \
    --epochs          50 \
    --lr              1e-3 \
    --num-workers     0 \
    --device          cpu \
    --resume

