#!/bin/bash
#SBATCH --job-name=eval_panel_demands
#SBATCH --account=millerlmgrp
#SBATCH --partition=high
#SBATCH --qos=millerlmgrp-high-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/eval_panel_demands_%j.out
#SBATCH --error=logs/eval_panel_demands_%j.err

cd /home/evliang/cidenoise
export PYTHONPATH="/home/evliang/cidenoise/track_a_deep_learning:/home/evliang/cidenoise/processing:/home/evliang/cidenoise/benchmarking:${PYTHONPATH}"

export PATH=/cvmfs/hpc.ucdavis.edu/sw/spack/environments/core/view/generic/slurm/bin:$PATH
source /cvmfs/hpc.ucdavis.edu/sw/conda/root/etc/profile.d/conda.sh
conda activate ci_denoise_env

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python -u benchmarking/evaluate_panel_demands.py \
  --data-dir /quobyte/millerlmgrp/annotated_data \
  --checkpoint-path /quobyte/millerlmgrp/checkpoints/track_a_arch_bench/best_UNet1D_Baseline.pt \
  --output-dir ./benchmark_results/panel_demands_evaluation
