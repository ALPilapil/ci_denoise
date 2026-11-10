#!/bin/bash
#SBATCH --job-name=export_viewprops
#SBATCH --partition=high
#SBATCH --account=cmb-grp
#SBATCH --time=00:15:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Generates EEGLAB viewprops (Topography + Time Series + PSD + ERP)
# and Quantifies Exact CI Artifact Removed across Channels

cd /home/evliang/cidenoise
export PYTHONPATH="/home/evliang/cidenoise/track_b_adaptive_subtraction:/home/evliang/cidenoise/processing:${PYTHONPATH}"

mkdir -p logs
mkdir -p benchmark_results/single_pair_demo/viewprops

source "/cvmfs/hpc.ucdavis.edu/sw/conda/root/etc/profile.d/conda.sh"
conda activate ci_denoise_env

echo "=========================================================="
echo "GENERATING EEGLAB VIEWPROPS & ARTIFACT QUANTIFICATION"
echo "=========================================================="

python processing/export_component_viewprops.py \
    --data-dir        /quobyte/millerlmgrp/annotated_data \
    --ci-subject      0901y2 \
    --permutation     1 \
    --n-components    15 \
    --output-dir      ./benchmark_results/single_pair_demo

echo ""
echo "Viewprops and Artifact Quantification Complete!"
