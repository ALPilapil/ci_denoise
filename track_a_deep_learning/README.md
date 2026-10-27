# Track A: Deep Learning with Multi-Objective Neuro Loss

Standalone supervised deep learning track for cochlear implant EEG denoising.

---

## 1. Hardware Requirements: GPU vs. CPU

| Task | Device Required | Hive Allocation Recommendation | Typical Runtime |
| :--- | :---: | :--- | :--- |
| **Model Training** | **GPU (Required)** | `#SBATCH --partition=high --gres=gpu:a6000:1` | ~1–2 hours for 50 epochs |
| *CPU Training (Fallback)* | *CPU Only* | `#SBATCH --partition=high --account=cmb-grp --cpus-per-task=16 --mem=128G` | ~24–48 hours |
| **Downstream Evaluation** | **CPU or GPU** | Can run on either; GPU takes ~1 min, CPU takes ~5 min | < 5 minutes |

> [!IMPORTANT]
> Deep learning backpropagation through a 5-layer 1D U-Net over multi-channel time-series data is intensely tensor-parallel. **Always prefer running training on an NVIDIA GPU** (`submit_train_gpu.sh`). Use the CPU fallback (`submit_train_cpu.sh`) only when cluster GPUs are completely booked.

---

## 2. Key Architecture & Loss Innovations

Unlike naive time-domain MSE (which acts like a low-pass filter and crushes the auditory N1/P2 ERP), Track A uses a composite objective:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{time}} + \lambda_{\text{spec}} \mathcal{L}_{\text{spec}} + \lambda_{\text{corr}} \mathcal{L}_{\text{corr}} + \lambda_{\text{erp}} \mathcal{L}_{\text{erp}}$$

1. **$\mathcal{L}_{\text{spec}}$ (Multi-Resolution STFT)**: Computes spectral convergence and log-magnitude distance across 64, 128, and 256 FFT bins to guarantee clinical frequency band preservation (Delta through Gamma).
2. **$\mathcal{L}_{\text{corr}}$ (Pearson Correlation)**: $1 - r(\hat{y}, y)$, directly optimizing waveform phase and peak alignment.
3. **$\mathcal{L}_{\text{erp}}$ (Auditory Window Weighting)**: Heavily penalizes reconstruction errors on vertex electrodes (`Cz`, `Fz`) in the $50\text{--}350\text{ ms}$ post-stimulus interval.
4. **Subject-Stratified Splitting**: Strictly partitions subjects so epochs from the same participant are never leaked between train, validation, and test splits.

---

## 3. How to Run on Hive

```bash
# 1. Run GPU training + downstream evaluation (Recommended):
sbatch track_a_deep_learning/submit_train_gpu.sh

# 2. Check output:
tail -f logs/track_a_gpu_*.out

# 3. View downstream evaluation metrics:
cat benchmark_results/track_a/track_a_downstream_summary.csv
```
