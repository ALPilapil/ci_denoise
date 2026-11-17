#!/bin/bash
#SBATCH --job-name=sobi_polar
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
mkdir -p logs benchmark_results/sobi_polarity_pipeline

source "/cvmfs/hpc.ucdavis.edu/sw/conda/root/etc/profile.d/conda.sh"
conda activate ci_denoise_env

echo "=========================================================="
echo "RUNNING END-TO-END SOBI + POLARITY PIPELINE"
echo "Host: $(hostname) | Date: $(date)"
echo "=========================================================="

python -u processing/run_sobi_polarity_pipeline.py \
    --raw-fif    /quobyte/millerlmgrp/annotated_data/CMPy2/trial_onsets_0901y2-raw.fif \
    --csv        /quobyte/millerlmgrp/annotated_data/CMPy2/trial_onsets_0901y2.csv \
    --output-dir ./benchmark_results/sobi_polarity_pipeline \
    --subject-id 0901y2 \
    --highpass   1.0 \
    --lowpass    45.0

echo ""
echo "SOBI + Polarity pipeline run completed!"
