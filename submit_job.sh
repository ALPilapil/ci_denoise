#!/bin/bash
#SBATCH --job-name=ml_denoise_all_years
#SBATCH --partition=high
#SBATCH --account=publicgrp
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a6000:1
#SBATCH --output=/quobyte/millerlmgrp/logs/%x_%j.out
#SBATCH --error=/quobyte/millerlmgrp/logs/%x_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=aarpila@gmail.com

mkdir -p /quobyte/millerlmgrp/logs

source "/cvmfs/hpc.ucdavis.edu/sw/conda/root/etc/profile.d/conda.sh"
conda activate ci_denoise_env
python processing/ml_denoise.py \
    --data-dir  /quobyte/millerlmgrp/annotated_data \
    --clean-dir /quobyte/millerlmgrp/ml_cleaned_data \
    --noise-dir /quobyte/millerlmgrp/ml_isolated_noise \
    --methods ica pca ssp \
    --n-jobs 1
