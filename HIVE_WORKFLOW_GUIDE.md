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

### Option A: Transfer ONLY Python (`.py`) and Shell (`.sh`) Scripts (Recommended)
This syncs all Python scripts and SLURM shell scripts across your subdirectories (e.g. `benchmarking/`, `modeling/`, `processing/`, `src/`) while completely skipping datasets, `.venv`, notebooks, git history, and temporary files:

```bash
rsync -avzP \
  --include='*/' \
  --include='*.py' \
  --include='*.sh' \
  --exclude='*' \
  /Users/evanywliang/Documents/cidenoise/ci_denoise/ \
  evliang@<hive-hostname>:~/cidenoise/
```

```bash
rsync -avzP \
  /Users/evanywliang/Documents/cidenoise/ci_denoise/benchmarking/ \
  evliang@hive.hpc.ucdavis.edu:~/cidenoise/benchmarking/
```

### Option B: Transfer Only Specific Folders (e.g., Just `benchmarking/` or `modeling/`)
If you only want to push the newly created benchmarking suite or modeling code:

```bash
# Transfer just the benchmarking scripts:
rsync -avzP \
  /Users/evanywliang/Documents/cidenoise/ci_denoise/benchmarking/ \
  evliang@<hive-hostname>:~/cidenoise/benchmarking/

# Or transfer just the modeling training scripts:
rsync -avzP \
  /Users/evanywliang/Documents/cidenoise/ci_denoise/modeling/ \
  evliang@<hive-hostname>:~/cidenoise/modeling/
```

### Option C: Transfer a Single Script
```bash
rsync -avzP \
  /Users/evanywliang/Documents/cidenoise/ci_denoise/benchmarking/tier1_screening.py \
  evliang@<hive-hostname>:~/cidenoise/benchmarking/
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

```bash
# Submit Tier 1 screening:
sbatch benchmarking/submit_tier1.sh

# Submit Tier 2 ERP evaluation:
sbatch benchmarking/submit_tier2.sh

# Submit Tier 3 deep learning evaluation:
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

When jobs finish on Hive, run this from your **local Mac terminal** to copy CSV reports, summaries, or plots back to your laptop:

```bash
# Download Tier 1 benchmark results to your local Mac:
rsync -avzP \
  evliang@<hive-hostname>:/quobyte/millerlmgrp/benchmarking/ \
  /Users/evanywliang/Documents/cidenoise/ci_denoise/benchmark_results/
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
