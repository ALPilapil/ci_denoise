#!/bin/bash
#SBATCH --job-name=sobi_n2n
#SBATCH --partition=high
#SBATCH --account=millerlmgrp
#SBATCH --qos=millerlmgrp-high-qos
#SBATCH --time=00:15:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

cd /home/evliang/cidenoise
export PYTHONPATH="/home/evliang/cidenoise/track_a_deep_learning:${PYTHONPATH}"
mkdir -p logs benchmark_results/sobi_n2n_results

source "/cvmfs/hpc.ucdavis.edu/sw/conda/root/etc/profile.d/conda.sh"
conda activate ci_denoise_env

echo "=========================================================="
echo "STAGE 2: TRAINING SYMMETRIC NOISE2NOISE ON SOBI 2-PASS EEG"
echo "Host: $(hostname) | Date: $(date)"
echo "=========================================================="

python -u track_a_deep_learning/train_sobi_n2n.py \
    --sobi-fif     ./benchmark_results/sobi_polarity_cohort/trial_onsets_0901y2_sobi2pass-raw.fif \
    --csv          /quobyte/millerlmgrp/annotated_data/CMPy2/trial_onsets_0901y2.csv \
    --summary-json ./benchmark_results/sobi_polarity_cohort/sobi2pass_summary_0901y2.json \
    --output-dir   ./benchmark_results/sobi_n2n_results \
    --epochs       25

echo ""
echo "Stage 2 SOBI-N2N training run completed!"
