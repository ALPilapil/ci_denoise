# Hive Cluster Workflow Guide

A step-by-step reference for syncing code, connecting via SSH, managing environments, and running SLURM batch/interactive jobs on the UC Davis Hive HPC cluster.

---

## 1. Quick Reference Workflow

```
[Local Mac] ──rsync──> [Hive Login Node (evliang/cidenoise)]
                                │
                                ├── srun (interactive debug session)
                                └── sbatch (background batch job on compute node)
                                        │
                                        ▼
                               [Logs & Results in /quobyte/ & /mnt/]
```

---

## 2. Step 1: Sync Scripts to Hive (`rsync`)

Run these commands from your **local Mac terminal** (not inside an SSH session):

### Option A: Transfer All Project Scripts (Excluding `.venv` and datasets) (Recommended)
This syncs all Python scripts, SLURM shell scripts, and markdown docs across all subdirectories (including `track_a_deep_learning/` and `track_b_adaptive_subtraction/`) while **strictly excluding** `.venv`, raw datasets, git history, and caches:

> [!IMPORTANT]
> In `rsync`, `--exclude` patterns for `.venv` and `venv` **must come before** `--include='*/'`. Otherwise, `rsync` will traverse into `.venv` and sync thousands of site-packages files.

```bash
rsync -avzP \
  --exclude='.venv' \
  --exclude='venv' \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --include='*/' \
  --include='*.py' \
  --include='*.sh' \
  --include='*.md' \
  --exclude='*' \
  /Users/evanywliang/Documents/cidenoise/ci_denoise/ \
  evliang@hive.hpc.ucdavis.edu:~/cidenoise/
```

### Option B: Transfer Only Track A or Track B Folders
If you are developing one track at a time:

```bash
# Transfer Track A (Deep Learning):
rsync -avzP \
  /Users/evanywliang/Documents/cidenoise/ci_denoise/track_a_deep_learning/ \
  evliang@hive.hpc.ucdavis.edu:~/cidenoise/track_a_deep_learning/

# Transfer Track B (Adaptive Template Subtraction):
rsync -avzP \
  /Users/evanywliang/Documents/cidenoise/ci_denoise/track_b_adaptive_subtraction/ \
  evliang@hive.hpc.ucdavis.edu:~/cidenoise/track_b_adaptive_subtraction/
```

### Option C: Transfer a Single Script
```bash
rsync -avzP \
  /Users/evanywliang/Documents/cidenoise/ci_denoise/track_a_deep_learning/train.py \
  evliang@hive.hpc.ucdavis.edu:~/cidenoise/track_a_deep_learning/
```

### Useful `rsync` Flags:
- **`-a` (archive):** Preserves file permissions, timestamps, and directory structure.
- **`-v` (verbose):** Lists each file transferred.
- **`-z` (compression):** Compresses files during transfer for speed.
- **`-P` (progress):** Displays real-time transfer progress.
- **Dry-Run Preview (`-n`):** Add `-n` (e.g., `rsync -avzPn ...`) to simulate the command first and verify exactly which files will be copied without writing anything.

---

## 3. Step 2: SSH into Hive & Navigate to Project

Open an SSH session:

```bash
ssh evliang@hive.hpc.ucdavis.edu
```

Once connected, you will start at your home directory (`/home/evliang/` or `evliang/`). Navigate into your project repository:

```bash
cd ~/cidenoise
pwd
```

Ensure shell scripts have executable permissions:

```bash
chmod +x benchmarking/*.sh submit_job.sh
```

---

## 4. Step 3: Activate the Conda Environment

Hive uses CVMFS to distribute central software and Conda environments. Run:

```bash
# Source central Conda initialization
source "/cvmfs/hpc.ucdavis.edu/sw/conda/root/etc/profile.d/conda.sh"

# Activate the project environment
conda activate ci_denoise_env

# Verify environment and Python path
which python
python --version
```

*(Note: Every SLURM batch script includes these two lines so worker nodes automatically load the correct packages.)*

---

## 5. Step 4: Running Jobs on Hive (Interactive vs. SLURM)

> [!WARNING]
> **Never run heavy computation, model training, or data preprocessing directly on the login node.** It can be killed by the system administrator. Always use an interactive compute session (`srun`) or submit a batch job (`sbatch`).

### Option A: Interactive Compute Session (`srun`)
Use this for quick debugging, testing script syntax, or running short sanity checks (e.g. 5 epochs):

```bash
# Request an interactive worker node for 1 hour with 4 CPUs and 32 GB RAM
srun --partition=high --account=publicgrp --mem=32G --cpus-per-task=4 --time=01:00:00 --pty bash

# Once inside the worker node, activate conda:
source "/cvmfs/hpc.ucdavis.edu/sw/conda/root/etc/profile.d/conda.sh"
conda activate ci_denoise_env

# Run your test script:
python benchmarking/tier1_screening.py --num-epochs 5

# When finished, exit back to the login node:
exit
```

### Option B: Batch Job Submission (`sbatch`)
Use this for full experiments, training, or long benchmarks:

#### 1. Track A: Deep Learning (Composite Loss U-Net)
- **Requires GPU for training** (1–2 hrs on A6000 vs. 24–48 hrs on CPU):
  ```bash
  # Submit GPU training (high partition, 1x A6000 GPU):
  sbatch track_a_deep_learning/submit_train_gpu.sh

  # CPU fallback training (cmb-grp partition, 16 CPUs):
  sbatch track_a_deep_learning/submit_train_cpu.sh
  ```

#### 2. Track B: Adaptive Template Subtraction (AATS)
- **Does NOT need a GPU (100% CPU-bound)**:
  ```bash
  # Submit full pipeline (denoise + evaluate ERPs):
  sbatch track_b_adaptive_subtraction/submit_aats_cpu.sh

  # Or submit fast ERP evaluation only (when files are already cleaned, ~2 mins):
  sbatch track_b_adaptive_subtraction/submit_evaluate_erp.sh
  ```

#### 3. Original Benchmarking Suite (Tier 1–3)
```bash
# Tier 1 synthetic benchmark:
sbatch benchmarking/submit_tier1.sh

# Tier 2 real ERP benchmark:
sbatch benchmarking/submit_tier2.sh

# Tier 3 downstream benchmark:
sbatch benchmarking/submit_tier3.sh
```

---

## 6. Step 5: Monitoring & Managing SLURM Jobs

### Check Queue Status
```bash
# View all jobs running or queued under your username
squeue -u evliang

# Detailed job inspection
scontrol show job <job_id>
```

### View Output Logs in Real Time
Logs are configured to write to `/quobyte/millerlmgrp/logs/`:

```bash
# Stream the stdout log as it writes:
tail -f /quobyte/millerlmgrp/logs/<job_name>_<job_id>.out

# Check for errors:
cat /quobyte/millerlmgrp/logs/<job_name>_<job_id>.err
```

### Cancel a Job
```bash
# Cancel a specific job ID
scancel <job_id>

# Cancel all your running and queued jobs
scancel -u evliang
```

---

## 7. Step 6: Pull Results Back to Your Local Mac (`rsync`)

Run these commands from your **local Mac terminal** (not inside the SSH session) to copy CSV reports, summaries, plots, and logs back to your laptop:

### Option A: Pull ALL Benchmark Results (Track A, Track B, Tier 1–3) (Recommended)
This downloads all output CSVs, summaries, and figures from Hive into your local `benchmark_results/` folder:

```bash
rsync -avzP \
  evliang@hive.hpc.ucdavis.edu:~/cidenoise/benchmark_results/ \
  /Users/evanywliang/Documents/cidenoise/ci_denoise/benchmark_results/
```

### Option B: Pull Job Output Logs
To view full stdout/stderr execution logs from Hive SLURM jobs:

```bash
# Pull local project logs directory:
rsync -avzP \
  evliang@hive.hpc.ucdavis.edu:~/cidenoise/logs/ \
  /Users/evanywliang/Documents/cidenoise/ci_denoise/logs/

# Or pull Quobyte central logs (if using /quobyte/millerlmgrp/logs/):
rsync -avzP \
  evliang@hive.hpc.ucdavis.edu:/quobyte/millerlmgrp/logs/ \
  /Users/evanywliang/Documents/cidenoise/ci_denoise/logs/
```

### Option C: Pull Trained Model Checkpoints (Optional)
If you want to pull the trained Track A U-Net weights (`best.pt`, `latest.pt`) back locally:

```bash
rsync -avzP \
  evliang@hive.hpc.ucdavis.edu:/quobyte/millerlmgrp/checkpoints/track_a/ \
  /Users/evanywliang/Documents/cidenoise/ci_denoise/checkpoints/track_a/
```

---

## 8. SLURM Script Template (`.sh`) Reference

When creating a new SLURM script on Hive, follow this standard header:

```bash
#!/bin/bash
#SBATCH --job-name=my_experiment
#SBATCH --partition=high          # or 'low' depending on queue priorities
#SBATCH --account=publicgrp       # primary lab account
#SBATCH --time=12:00:00           # HH:MM:SS max runtime
#SBATCH --mem=64G                 # Node RAM limit
#SBATCH --cpus-per-task=4         # CPU threads
#SBATCH --gres=gpu:a6000:1        # Optional: request 1 GPU (e.g. a6000 or rtx8000)
#SBATCH --output=/quobyte/millerlmgrp/logs/%x_%j.out
#SBATCH --error=/quobyte/millerlmgrp/logs/%x_%j.err

# Create log directories if needed
mkdir -p /quobyte/millerlmgrp/logs

# Load environment
source "/cvmfs/hpc.ucdavis.edu/sw/conda/root/etc/profile.d/conda.sh"
conda activate ci_denoise_env

# Run python script with unbuffered output
python my_script.py
```
