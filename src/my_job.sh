#!/bin/bash
#SBATCH --partition=high
#SBATCH --account=publicgrp
#SBATCH --mem=128G
#SBATCH --cpus-per-task=1
#SBATCH --time=20:00:00  
#SBATCH --job-name=misolate_noise
#SBATCH --output=my_job_%j.out
#SBATCH --error=my_job_%j.err

module load conda/latest
source activate ci_denoise_env

python src/isolate_noise.py
