#!/bin/bash
#SBATCH --job-name=compare_sobi_ica
#SBATCH --partition=high
#SBATCH --account=cmb-grp
#SBATCH --time=00:15:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Head-to-Head Comparison: SOBI vs. FastICA for CI Denoising
# Evaluates Subject 0901y2 (CI) vs. 0801y2 (Hearing Control) on Permutation 1

cd /home/evliang/cidenoise
export PYTHONPATH="/home/evliang/cidenoise/track_b_adaptive_subtraction:/home/evliang/cidenoise/processing:${PYTHONPATH}"

mkdir -p logs
mkdir -p benchmark_results/single_pair_demo

source "/cvmfs/hpc.ucdavis.edu/sw/conda/root/etc/profile.d/conda.sh"
conda activate ci_denoise_env

echo "=========================================================="
echo "LAUNCHING BSS HEAD-TO-HEAD: SOBI vs. FASTICA"
echo "=========================================================="

python processing/compare_sobi_ica.py \
    --data-dir        /quobyte/millerlmgrp/annotated_data \
    --ci-subject      0901y2 \
    --hearing-subject 0801y2 \
    --permutation     1 \
    --n-components    15 \
    --output-dir      ./benchmark_results/single_pair_demo

echo ""
echo "Comparison complete! Figures and metrics saved to ./benchmark_results/single_pair_demo/"
