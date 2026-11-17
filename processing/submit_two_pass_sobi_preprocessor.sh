#!/bin/bash
#SBATCH --job-name=sobi2pass
#SBATCH --partition=high
#SBATCH --account=millerlmgrp
#SBATCH --qos=millerlmgrp-high-qos
#SBATCH --time=00:15:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

cd /home/evliang/cidenoise
export PYTHONPATH="/home/evliang/cidenoise/processing:${PYTHONPATH}"
mkdir -p logs benchmark_results/sobi2pass_cache

source "/cvmfs/hpc.ucdavis.edu/sw/conda/root/etc/profile.d/conda.sh"
conda activate ci_denoise_env

echo "=========================================================="
echo "RUNNING TWO-PASS SOBI PREPROCESSOR WITH BIOPHYSICAL GATES"
echo "Host: $(hostname) | Date: $(date)"
echo "=========================================================="

python -u processing/two_pass_sobi_preprocessor.py \
    --input-fif   /quobyte/millerlmgrp/annotated_data/CMPy2/trial_onsets_0901y2-raw.fif \
    --output-fif  ./benchmark_results/sobi2pass_cache/trial_onsets_0901y2_sobi2pass-raw.fif \
    --target-fs   256.0 \
    --highpass    1.0 \
    --lowpass     45.0 \
    --max-passes  2 \
    --r-thresh    2.5 \
    --output-json ./benchmark_results/sobi2pass_cache/sobi2pass_summary_0901y2.json

echo ""
echo "Two-Pass SOBI run completed!"
