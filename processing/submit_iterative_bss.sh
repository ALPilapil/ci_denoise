#!/bin/bash
#SBATCH --job-name=iter_bss
#SBATCH --partition=high
#SBATCH --account=millerlmgrp
#SBATCH --qos=millerlmgrp-high-qos
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Iterative BSS: Run ICA/SOBI, remove single main component, re-run ICA/SOBI on reconstructed data
# Evaluates Subject 0901y2 (CI) vs 0801y2 (Hearing Control) on Permutation 1

cd /home/evliang/cidenoise
export PYTHONPATH="/home/evliang/cidenoise/track_b_adaptive_subtraction:/home/evliang/cidenoise/processing:${PYTHONPATH}"

mkdir -p logs
mkdir -p benchmark_results/single_pair_demo

source "/cvmfs/hpc.ucdavis.edu/sw/conda/root/etc/profile.d/conda.sh"
conda activate ci_denoise_env

echo "=========================================================="
echo "LAUNCHING OPTIMIZED ITERATIVE BSS (MAIN COMPONENT PASS 1 -> PASS 2)"
echo "Host: $(hostname) | Date: $(date)"
echo "=========================================================="

python -u processing/run_iterative_bss_main_component.py \
    --data-dir        /quobyte/millerlmgrp/annotated_data \
    --ci-subject      0901y2 \
    --hearing-subject 0801y2 \
    --permutation     1 \
    --n-components    15 \
    --decim           16 \
    --output-dir      ./benchmark_results/single_pair_demo

echo ""
echo "Iterative BSS complete! Output figures and JSON metrics saved to ./benchmark_results/single_pair_demo/"
