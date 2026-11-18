#!/bin/bash
#SBATCH --job-name=foundation_n2n
#SBATCH --partition=high
#SBATCH --account=millerlmgrp
#SBATCH --qos=millerlmgrp-high-qos
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

cd /home/evliang/cidenoise
export PYTHONPATH="/home/evliang/cidenoise/track_a_deep_learning:${PYTHONPATH}"
mkdir -p logs benchmark_results/sobi_n2n_foundation

source "/cvmfs/hpc.ucdavis.edu/sw/conda/root/etc/profile.d/conda.sh"
conda activate ci_denoise_env

echo "=========================================================="
echo "RUNNING MULTI-SUBJECT FOUNDATION SOBI-N2N PRE-TRAINING"
echo "Host: $(hostname) | Date: $(date)"
echo "=========================================================="

python -u track_a_deep_learning/train_sobi_n2n_multisubject.py \
    --num-subjects 8 \
    --epochs       20 \
    --batch-size   16 \
    --lr           1e-4 \
    --output-dir   ./benchmark_results/sobi_n2n_foundation

echo "Multi-Subject Foundation Training Complete!"
