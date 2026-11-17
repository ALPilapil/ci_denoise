#!/bin/bash
#SBATCH --job-name=eval_sobi_ep
#SBATCH --partition=high
#SBATCH --account=millerlmgrp
#SBATCH --qos=millerlmgrp-high-qos
#SBATCH --time=00:10:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

cd /home/evliang/cidenoise
mkdir -p logs benchmark_results/sobi2pass_cache

source "/cvmfs/hpc.ucdavis.edu/sw/conda/root/etc/profile.d/conda.sh"
conda activate ci_denoise_env

python -u processing/evaluate_sobi2pass_epochs.py \
    --raw-fif  /quobyte/millerlmgrp/annotated_data/CMPy2/trial_onsets_0901y2-raw.fif \
    --sobi-fif ./benchmark_results/sobi2pass_cache/trial_onsets_0901y2_sobi2pass-raw.fif \
    --csv      /quobyte/millerlmgrp/annotated_data/CMPy2/trial_onsets_0901y2.csv \
    --output-json ./benchmark_results/sobi2pass_cache/epoch_audit_0901y2.json
