#!/bin/bash
#SBATCH --job-name=pair_single_demo
#SBATCH --partition=high
#SBATCH --account=cmb-grp
#SBATCH --time=00:20:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Runs the Single-Subject CI Noise Isolation & Pairing Demonstration
# Compares 1 CI (0901y2) and 1 Hearing (0801y2) on the same permutation (perm1)

cd /home/evliang/cidenoise
export PYTHONPATH="/home/evliang/cidenoise/track_b_adaptive_subtraction:/home/evliang/cidenoise/processing:${PYTHONPATH}"

mkdir -p logs
mkdir -p benchmark_results/single_pair_demo

source "/cvmfs/hpc.ucdavis.edu/sw/conda/root/etc/profile.d/conda.sh"
conda activate ci_denoise_env

echo "=========================================================="
echo "RUNNING NOISE ISOLATION & PAIRING ON MATCHING PERMUTATION"
echo "=========================================================="

python processing/isolate_and_pair_single_subject.py \
    --data-dir        /quobyte/millerlmgrp/annotated_data \
    --ci-subject      0901y2 \
    --hearing-subject 0801y2 \
    --permutation     1 \
    --method          all \
    --alpha           1.0 \
    --output-dir      ./benchmark_results/single_pair_demo \
    --num-block-trials 8 \
    --start-trial     1

echo ""
echo "Demonstration complete! Results, epoch plots, and multi-trial block comparison saved to ./benchmark_results/single_pair_demo/"
