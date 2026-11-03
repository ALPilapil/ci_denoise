#!/bin/bash
#SBATCH --job-name=plot_ica_topomap
#SBATCH --partition=high
#SBATCH --account=cmb-grp
#SBATCH --time=00:15:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Generates 2D Scalp Brain Topographies and Component Maps for ICA Denoising
# Evaluates Subject 0901y2 (CI) vs. 0801y2 (Hearing Control) on Permutation 1

cd /home/evliang/cidenoise
export PYTHONPATH="/home/evliang/cidenoise/track_b_adaptive_subtraction:/home/evliang/cidenoise/processing:${PYTHONPATH}"

mkdir -p logs
mkdir -p benchmark_results/single_pair_demo

source /etc/profile.d/modules.sh
module load conda
conda activate ci_denoise_env

echo "=========================================================="
echo "GENERATING 2D SCALP TOPOMAPS AND BRAIN VOLTAGE MAPS"
echo "=========================================================="

python processing/plot_ica_topomaps.py \
    --data-dir        /quobyte/millerlmgrp/annotated_data \
    --ci-subject      0901y2 \
    --hearing-subject 0801y2 \
    --permutation     1 \
    --n-components    15 \
    --output-dir      ./benchmark_results/single_pair_demo

echo ""
echo "Visual aids generation complete! Plots saved to ./benchmark_results/single_pair_demo/"
