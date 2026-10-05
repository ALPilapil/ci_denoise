# Benchmarking Suite for Cochlear Implant Denoising

This suite breaks down method evaluation into **three decoupled tiers** so you can run them independently on the Hive cluster via SSH or SLURM.

It builds directly on the design patterns from [`shyam/prepare_data.py`](file:///Users/evanywliang/Documents/cidenoise/ci_denoise/shyam/prepare_data.py) and [`shyam/reconstruct.py`](file:///Users/evanywliang/Documents/cidenoise/ci_denoise/shyam/reconstruct.py):
- **Unbuffered logging** (`sys.stdout.reconfigure(line_buffering=True, write_through=True)`) so SLURM `.out` log files stream in real-time.
- **Memory safety** with explicit garbage collection and batching to avoid OOM kills on high-memory cluster nodes.
- **Native Hive path defaults** (`/mnt/data/PilapilData/...` and `/quobyte/millerlmgrp/...`).

---

## Overview of Tiers

```
                       ┌────────────────────────────────────────┐
                       │  Tier 1: Synthetic Screening           │
                       │  (tier1_screening.py)                  │
                       │  • RRMSE, ΔSNR (dB), Suppression (dB)  │
                       │  • Filters out non-viable methods      │
                       └───────────────────┬────────────────────┘
                                           │ Top 3-4 Candidates
                                           ▼
                       ┌────────────────────────────────────────┐
                       │  Tier 2: Real ERP Preservation         │
                       │  (tier2_erp_eval.py)                   │
                       │  • N1/P2 recovery, Template Corr       │
                       │  • Validates actual neural signal      │
                       └───────────────────┬────────────────────┘
                                           │ Best Performing
                                           ▼
                       ┌────────────────────────────────────────┐
                       │  Tier 3: Downstream & DL Validation    │
                       │  (tier3_downstream_eval.py)            │
                       │  • Cross-subject test loss (MSE/MAE)   │
                       │  • Spectral fidelity (Delta..Gamma)    │
                       └────────────────────────────────────────┘
```

---

## 1. Tier 1: Fast Screening (`tier1_screening.py`)

Runs candidate methods on synthetic pairings (`clean + α·noise`) from `raw_epoched_data.zarr`.

### Interactive Run
```bash
python benchmarking/tier1_screening.py \
    --zarr-path /mnt/data/PilapilData/processed_data/raw_epoched_data.zarr \
    --output-dir ./benchmark_results/tier1 \
    --methods ica pca cca ssp wavelet \
    --num-epochs 50 \
    --alpha 1.0
```

### SLURM Submission
```bash
sbatch benchmarking/submit_tier1.sh
```

**Outputs:**
- `tier1_epoch_metrics.csv`: Per-pair metrics (RRMSE, ΔSNR dB, Suppression dB, Correlation, CI channel RRMSE, time ms).
- `tier1_summary.csv`: Aggregated leaderboard (mean ± std per method).

---

## 2. Tier 2: Real ERP Preservation (`tier2_erp_eval.py`)

Evaluates real CI EEG recordings with trial-onset annotations (`extract_trial_onsets.py` output) against hearing control grand-average templates.

### Interactive Run
```bash
python benchmarking/tier2_erp_eval.py \
    --annotated-dir /quobyte/millerlmgrp/annotated_data \
    --cleaned-root  /quobyte/millerlmgrp/ml_cleaned_data \
    --hearing-dir   /mnt/data/PilapilData/processed_data/hearing \
    --output-dir    ./benchmark_results/tier2 \
    --condition     A \
    --target-channel Cz \
    --methods       raw ica pca cca ssp wavelet
```

### SLURM Submission
```bash
sbatch benchmarking/submit_tier2.sh
```

**Outputs:**
- `tier2_subject_erp_metrics.csv`: Subject-level N1 amplitude/latency, P2 amplitude/latency, auditory SNR, and correlation with hearing template.
- `tier2_method_summary.csv`: Leaderboard ranked by ERP template preservation.

---

## 3. Tier 3: Downstream DL Validation (`tier3_downstream_eval.py`)

Evaluates deep learning models (UNet, etc.) on held-out test splits, measuring generalization error, temporal correlation, and spectral band power fidelity (Delta, Theta, Alpha, Beta, Gamma).

### Interactive Run
```bash
python benchmarking/tier3_downstream_eval.py \
    --zarr-path       /mnt/data/PilapilData/processed_data/raw_epoched_data.zarr \
    --checkpoint-dir  /mnt/data/PilapilData/checkpoints \
    --checkpoints     best.pt \
    --batch-size      16 \
    --output-dir      ./benchmark_results/tier3
```

### SLURM Submission
```bash
sbatch benchmarking/submit_tier3.sh
```

**Outputs:**
- `tier3_downstream_summary.csv`: Test MSE, MAE, waveform correlation, and spectral band error (dB) for each evaluated checkpoint.
