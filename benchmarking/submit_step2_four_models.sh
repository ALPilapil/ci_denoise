#!/bin/bash
#SBATCH --job-name=two_step_4m
#SBATCH --partition=high
#SBATCH --account=millerlmgrp
#SBATCH --qos=millerlmgrp-high-qos
#SBATCH --time=00:25:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Head-to-Head 2-Step Denoising: Two-Pass SOBI -> 4 Neural Networks in Parallel
cd /home/evliang/cidenoise
export PYTHONPATH="/home/evliang/cidenoise/track_a_deep_learning:/home/evliang/cidenoise/processing:/home/evliang/cidenoise/benchmarking:${PYTHONPATH}"

mkdir -p logs
mkdir -p benchmark_results/single_pair_demo

source "/cvmfs/hpc.ucdavis.edu/sw/conda/root/etc/profile.d/conda.sh"
conda activate ci_denoise_env

echo "=========================================================="
echo "LAUNCHING TWO-STEP 4-MODEL PARALLEL BENCHMARK"
echo "Host: $(hostname) | Date: $(date)"
echo "=========================================================="

python -u benchmarking/run_step2_four_models_benchmark.py \
    --data-dir        /quobyte/millerlmgrp/annotated_data \
    --checkpoint-dir  /quobyte/millerlmgrp/checkpoints/track_a_arch_bench \
    --ci-subject      0901y2 \
    --hearing-subject 0801y2 \
    --target-fs       256.0 \
    --output-dir      ./benchmark_results/single_pair_demo

echo ""
echo "Two-step 4-model benchmark complete! Saved to ./benchmark_results/single_pair_demo/"
