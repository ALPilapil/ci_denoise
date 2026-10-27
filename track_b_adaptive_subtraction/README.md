# Track B: Stimulus-Locked Adaptive Artifact Template Subtraction (AATS)

Standalone physiological adaptive filtering track for cochlear implant EEG denoising.

---

## 1. Hardware Requirements: GPU vs. CPU

| Task | Device Required | Hive Allocation Recommendation | Typical Runtime |
| :--- | :---: | :--- | :--- |
| **AATS Batch Denoising** | **CPU Only** (No GPU Needed) | `#SBATCH --partition=high --account=publicgrp --cpus-per-task=4 --mem=32G` | ~10–30 seconds for entire cohort |
| **ERP Preservation Eval** | **CPU Only** (No GPU Needed) | Same node allocation | ~1–2 minutes |

> [!NOTE]
> **Why Track B does NOT need a GPU:**
> The regularized multi-tap FIR Wiener transfer function solves normal equations: $(R^T R + \lambda I)^{-1} R^T X$.
> With $K=5$ reference channels and $M=6$ delay taps, $R^T R$ is a tiny **$30 \times 30$ matrix**.
> Solving a $30 \times 30$ linear system in NumPy takes **less than 1 millisecond** on a single CPU core.
> Moving the EEG data over PCIe to GPU VRAM would actually be *slower* due to bus transfer latency.

---

## 2. Core Mechanism

1. **Reference Extraction**: Uses CI-proximal retroauricular electrodes (`P7, T7, M2, M1, P8`) where the electrical stimulation artifact is $>20\text{ dB}$ higher than neural potentials.
2. **Transfer Function Fitting**: Fits an FIR transfer matrix across stimulus-active intervals (`A`, `AV`, `V`) to capture volume-conduction coupling to scalp electrodes (`Cz`, `Fz`).
3. **Artifact Subtraction**: Subtracts only coherent electrical artifact from scalp channels. Asynchronous neural brainwaves (auditory N1/P2 dipole) pass through unattenuated.

---

## 3. How to Run on Hive

```bash
# Option 1: Full pipeline (Denoise FIFs + Evaluate ERPs):
sbatch track_b_adaptive_subtraction/submit_aats_cpu.sh

# Option 2: Fast ERP evaluation only (~2 mins, when files are already cleaned):
sbatch track_b_adaptive_subtraction/submit_evaluate_erp.sh

# Monitor log:
tail -f logs/track_b_*.out

# View ERP preservation metrics:
cat benchmark_results/track_b/track_b_method_summary.csv
```
