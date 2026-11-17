#!/bin/bash
#SBATCH --job-name=sobi_multi
#SBATCH --partition=high
#SBATCH --account=millerlmgrp
#SBATCH --qos=millerlmgrp-high-qos
#SBATCH --time=00:20:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Multi-Pass SOBI Stress Test: Run SOBI 6 times consecutively to observe brain activity erosion
cd /home/evliang/cidenoise
export PYTHONPATH="/home/evliang/cidenoise/track_b_adaptive_subtraction:/home/evliang/cidenoise/processing:${PYTHONPATH}"

mkdir -p logs
mkdir -p benchmark_results/single_pair_demo

source "/cvmfs/hpc.ucdavis.edu/sw/conda/root/etc/profile.d/conda.sh"
conda activate ci_denoise_env

echo "=========================================================="
echo "LAUNCHING MULTI-PASS SOBI STRESS TEST (PASSES 1 TO 6)"
echo "Host: $(hostname) | Date: $(date)"
echo "=========================================================="

python -u processing/run_sobi_multi_pass_collapse.py \
    --data-dir        /quobyte/millerlmgrp/annotated_data \
    --ci-subject      0901y2 \
    --hearing-subject 0801y2 \
    --max-passes      6 \
    --n-components    15 \
    --output-dir      ./benchmark_results/single_pair_demo

echo ""
echo "Multi-pass SOBI run complete! Saved to ./benchmark_results/single_pair_demo/"
