#!/bin/bash
#SBATCH --job-name=eval_n2n
#SBATCH --partition=high
#SBATCH --account=millerlmgrp
#SBATCH --qos=millerlmgrp-high-qos
#SBATCH --time=00:10:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

cd /home/evliang/cidenoise
export PYTHONPATH="/home/evliang/cidenoise/track_a_deep_learning:${PYTHONPATH}"

source "/cvmfs/hpc.ucdavis.edu/sw/conda/root/etc/profile.d/conda.sh"
conda activate ci_denoise_env

python -u track_a_deep_learning/evaluate_sobi_n2n_downstream.py \
    --sobi-fif     ./benchmark_results/sobi2pass_cache/trial_onsets_0901y2_sobi2pass-raw.fif \
    --hearing-fif  /quobyte/millerlmgrp/annotated_data/CMPy2/trial_onsets_0801y2-raw.fif \
    --csv          /quobyte/millerlmgrp/annotated_data/CMPy2/trial_onsets_0901y2.csv \
    --summary-json ./benchmark_results/sobi2pass_cache/sobi2pass_summary_0901y2.json \
    --checkpoint   ./benchmark_results/sobi_n2n_results/best_sobi_n2n_unet.pt \
    --output-dir   ./benchmark_results/sobi_n2n_results
