#!/bin/bash
#SBATCH --job-name=extract_cmpy3
#SBATCH --partition=high
#SBATCH --account=millerlmgrp
#SBATCH --qos=millerlmgrp-high-qos
#SBATCH --time=00:20:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-26%4
#SBATCH --output=logs/extract_cmpy3_%A_%a.out
#SBATCH --error=logs/extract_cmpy3_%A_%a.err

cd /home/evliang/cidenoise
mkdir -p logs /quobyte/millerlmgrp/annotated_data/CMPy3_extracted

source "/cvmfs/hpc.ucdavis.edu/sw/conda/root/etc/profile.d/conda.sh"
conda activate ci_denoise_env

SUBJECTS=(
    0901y3 0902y3 0903y3 0904y3 0905y3 0908y3 0910y3 0913y3 0918y3 0919y3
    0921y3 0923y3 0924y3 0925y3 0929y3 0931y3 0932y3 0934y3 0935y3 0936y3
    0937y3 0938y3 0940y3 0941y3 0947y3 0949y3 0951y3
)

SUB=${SUBJECTS[${SLURM_ARRAY_TASK_ID}]}

echo "=========================================================="
echo "EXTRACTING TRIAL ONSETS FOR ${SUB} (Task ${SLURM_ARRAY_TASK_ID})"
echo "Host: $(hostname) | Date: $(date)"
echo "=========================================================="

python -u processing/extract_trial_onsets.py ${SUB} \
    --study-dir /quobyte/millerlmgrp/CMPy3 \
    --output-dir /quobyte/millerlmgrp/annotated_data/CMPy3_extracted

echo "Extraction for ${SUB} complete!"
