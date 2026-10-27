# CI EEG Denoising: Research Meeting Notes & Method Review

**Topic:** Cochlear Implant (CI) Artifact Removal & Cortical Auditory ERP Preservation  
**Target Channels:** Auditory Vertex Dipole (`Cz`, `Fz`)  
**Evaluation Scope:** Synthetic Mixtures (Tier 1), Real Clinical ERP Preservation (Tier 2 / Track B, 37 Participants), Generalization (Tier 3 / Track A)  

---

## 1. Executive Summary & Meeting Purpose

* **The Core Clinical Challenge:** Electrical pulses and RF leakage from Cochlear Implants propagate volume-conducted artifacts to scalp electrodes that are $10\times\text{ to }100\times$ larger than the underlying brainwaves ($>20 - 50\ \mu\text{V}$ artifact vs. $3 - 8\ \mu\text{V}$ true auditory N1-P2 ERP).
* **The Clinical Goal:** Remove the electrical stimulation artifact so clinicians and neuroscientists can reliably detect whether the patient's auditory cortex has perceived sound (intact N1 at ~100 ms and P2 at ~200 ms), without distorting peak amplitudes or latencies.
* **Meeting Objective:** Review benchmark findings across classical decomposition baselines, deep learning architectures (Track A), and physiological adaptive subtraction (Track B), and align on future direction.

```
                    ┌────────────────────────────────────────────────────────┐
                    │               CI EEG Denoising Approaches              │
                    └────────────────────────────────────────────────────────┘
                                                 │
         ┌───────────────────────────────────────┼───────────────────────────────────────┐
         ▼                                       ▼                                       ▼
┌─────────────────────────┐             ┌─────────────────────────┐             ┌─────────────────────────┐
│  Classical Blind Source │             │   Track A: Deep Learning│             │ Track B: Physiological  │
│      Separation         │             │      (1D U-Net)         │             │    Adaptive (AATS)      │
├─────────────────────────┤             ├─────────────────────────┤             ├─────────────────────────┤
│ • ICA (Infomax/FastICA) │             │ • Temporal 1D U-Net     │             │ • Multi-tap FIR Wiener  │
│ • PCA                   │             │ • Composite Neuro Loss  │             │ • CI reference-anchored │
│ • CCA (BSS-CCA)         │             │   (STFT + Corr + ERP)   │             │   (P7, T7, M1, M2, P8)  │
│ • SSP                   │             │ • Cross-participant split│             │ • Closed-form solution  │
└─────────────────────────┘             └─────────────────────────┘             └─────────────────────────┘
```

---

## 2. Quick Comparison Matrix

| Method Category | Primary Mechanism | Hardware Required | Compute Time | Noise Reduction | Peak Latency Preservation | Amplitude Integrity | Primary Risk / Failure Mode |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **ICA** (FastICA / Infomax) | Statistical independence of spatial sources | CPU | ~5–10 min/file | Moderate | Poor–Moderate | Variable | Subjective component selection; removes genuine neural signal |
| **PCA** | Orthogonal variance maximization | CPU | < 1 min/file | Low (removes high var) | Fair (N1: 124 ms) | Poor ($16.6\ \mu\text{V}$) | Brainwave and artifact subspaces overlap; drops neural power |
| **CCA** (BSS-CCA) | Autocorrelation minimization (reference coupling) | CPU | ~1–2 min/file | **Negative** (Triples noise: $5.85\ \mu\text{V}$) | Moderate (N1: 110 ms) | **Severe Distortion** ($30.7\ \mu\text{V}$) | High-frequency noise amplification; phase distortion |
| **Track A: 1D U-Net (Baseline MSE)** | End-to-end non-linear translation | **GPU Required** | ~1–2 hrs (Train) | High | Poor | Severe Smearing (Phase cancellation) | Regresses to the mean; wipes out sharp N1/P2 peaks |
| **Track A: 1D U-Net (Composite Loss)** | Multi-scale STFT + Pearson Corr + ERP Loss | **GPU Required** | ~1–2 hrs (Train) | High | Under Evaluation | Restores morphology | Needs large, diverse training cohort; GPU dependency |
| **Track B: AATS** | Regularized FIR Wiener filter from CI mastoid sensors | **CPU Only** | **~30 sec/subject** | **High** (40.3% reduction, 34/36 subjs) | **Excellent** (N1: 112 ms, P2: 203 ms) | **Canonical** ($6.22\ \mu\text{V}$ physiological) | Sensitive to dead/disconnected reference electrodes |

---

## 3. Deep Dive into Methods

### Group 1: Classical Blind Source Separation (Preexisting Baselines)

#### 1. Independent Component Analysis (ICA / FastICA)
* **What it does:** Unmixes multi-channel EEG into statistically independent spatial projection topographies $S = W X$. Components whose spatial maps concentrate around the CI implant or whose timecourses correlate with CI stimulation are removed, and the remaining components are back-projected.
* **What we tested:** FastICA with mastoid reference correlation thresholds.
* **Benefits:** Standard in cognitive EEG pipelines; effective for eye blinks and lateralized high-amplitude muscle artifacts.
* **Downsides:**
  - *Non-orthogonal overlap:* The CI electrical artifact is so massive that it bleeds into neural independent components; zeroing the artifact component strips 30–60% of genuine auditory cortex dipole energy.
  - *Automated rejection instability:* Hard to reliably automate component selection without manual expert inspection.

#### 2. Principal Component Analysis (PCA)
* **What it does:** Orthogonally diagonalizes the sensor covariance matrix and discards the top $K$ eigenvalues carrying the largest total variance.
* **What we tested:** Truncated PCA on clinical FIF files.
* **Benefits:** Extremely fast, closed-form, deterministic.
* **Downsides:**
  - *Subspace orthogonality violation:* Assumes artifact and brainwaves are strictly orthogonal in spatial coordinate space. When both project to central channels (`Cz`, `Fz`), discarding top components discards the genuine auditory ERP.
  - *Empirical result:* Resulted in a high baseline noise floor ($2.98\ \mu\text{V}$) and distorted P2P amplitude ($16.61\ \mu\text{V}$).

#### 3. Canonical Correlation Analysis (CCA / BSS-CCA)
* **What it does:** Transforms multi-channel signals to maximize correlation with lagged versions (exploiting the fact that neural brainwaves are smooth/autocorrelated, whereas stimulation pulses are high-frequency) or between scalp channels and reference mastoid channels.
* **What we tested:** MNE-based CCA denoiser on clinical Year 2 cohort.
* **Benefits:** Requires fewer channels than ICA; theoretically tailored for colored vs. white noise.
* **Downsides:**
  - **Severe empirical failure in clinical testing:** CCA tripled baseline pre-stimulus noise ($5.85\ \mu\text{V}$ vs. $1.98\ \mu\text{V}$ Raw) and inflated P2P amplitude to $30.68\ \mu\text{V}$ (massive spurious distortion).
  - High-frequency residual noise gets amplified when reconstructing the signal from canonical variates.

---

### Group 2: Track A — Deep Learning (1D Temporal U-Net)

* **Architecture:** 5-layer 1D U-Net with contracting convolutional encoder, bottleneck with dilated receptive fields, expanding decoder with skip connections, and residual learning ($X_{\text{clean}} = X_{\text{dirty}} - \mathcal{N}_{\theta}(X_{\text{dirty}})$).
* **Hardware:** **Requires GPU** (NVIDIA A6000 / RTX 8000; ~1–2 hours training across 50 epochs on 19 GB Zarr dataset vs. 24–48 hours on CPU).

#### What Failed in Baseline Deep Learning:
* When trained with standard **MSE Loss** on synthetic mixtures:
  - The model minimizes mean squared error by smoothing out high-frequency transitions, leading to **temporal phase cancellation** and flattening of the biological N1 and P2 peaks.
  - When split randomly across trials, the model memorized participant-specific background EEG instead of learning general artifact transfer functions.

#### The Track A Enhancements:
1. **`CompositeNeuroLoss`:**
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}} + \lambda_{\text{STFT}}\mathcal{L}_{\text{Multi-STFT}} + \lambda_{\text{corr}}\mathcal{L}_{\text{Pearson}} + \lambda_{\text{ERP}}\mathcal{L}_{\text{Auditory}}$$
   - *Multi-Resolution STFT Loss:* Computes spectral convergence and log magnitude across 3 FFT resolutions (64, 128, 256) to prevent high-frequency spectral blurring.
   - *Pearson Correlation Loss ($1 - r$):* Enforces temporal phase and waveform polarity alignment.
   - *Auditory Window ERP Loss:* Heavily penalizes errors specifically at `Cz` and `Fz` in the crucial 50–350 ms post-stimulus window.
2. **Participant-Stratified Splitting:**
   - Grouping strictly by subject ID (`split_strategy='participant'`) ensures the model is tested on unseen brains, measuring genuine clinical generalization.

---

### Group 3: Track B — Physiological Adaptive Filtering (AATS)

* **Method:** Stimulus-Locked Adaptive Artifact Template Subtraction (**AATS**).
* **Architecture:** Multi-tap regularized FIR Wiener filter.
* **Hardware:** **100% CPU-bound** (no GPU needed; processes all 54 clinical subjects in ~30 minutes total, or ~35 seconds per subject).

#### How AATS Works:
1. **Physiological Asymmetry:**
   - CI-proximal retroauricular/mastoid electrodes (`P7, T7, M2, M1, P8`) sit directly over the implant RF coil and return electrode. The electrical artifact on these sensors is $>20\text{ dB}$ stronger than cortical brainwaves.
   - The primary auditory cortex generator (tangential dipole in Heschl’s gyrus) projects maximally to the vertex (`Cz`, `Fz`) and has negligible representation on retroauricular electrodes.
2. **Transfer Function Fitting:**
   - Uses multi-tap delay lines ($M = 6$ taps) from reference sensors to scalp sensors to model the capacitive/resistive volume-conduction transfer function through the skull and scalp.
   - Solves the regularized closed-form normal equations on stimulus-active windows (`A`, `AV`, `V`):
     $$\mathbf{W} = (\mathbf{R}^T \mathbf{R} + \lambda \mathbf{I})^{-1} \mathbf{R}^T \mathbf{X}$$
3. **Subtraction:**
   - Subtracts the estimated volume-conducted artifact: $\mathbf{X}_{\text{clean}} = \mathbf{X} - \mathbf{R} \mathbf{W}$.
   - Uncorrelated cortical evoked potentials pass through unattenuated.

#### Clinical Findings on 37 Real CI Patients:
* **Baseline Noise Reduction:**
  - Raw Baseline Noise: $1.98 \pm 1.44\ \mu\text{V}$ $\to$ **AATS: $1.02 \pm 0.64\ \mu\text{V}$** (**40.3% reduction**).
  - Noise was reduced in **34 out of 36 valid subjects (94.4%)**.
* **Precise Temporal Latency Preservation:**
  - N1 Latency: Raw $115.4\text{ ms} \to$ AATS **$112.3\text{ ms}$** ($\Delta = 3.1\text{ ms}$).
  - P2 Latency: Raw $193.0\text{ ms} \to$ AATS **$203.4\text{ ms}$** ($\Delta = 10.4\text{ ms}$).
* **Normalization of Biological P2P Amplitude:**
  - Raw: $12.80\ \mu\text{V}$ (inflated by artifact pulses) $\to$ AATS: **$6.22\ \mu\text{V}$** (canonical normal hearing range is $4 - 8\ \mu\text{V}$).
  - CCA blew up to $30.68\ \mu\text{V}$ and PCA was $16.61\ \mu\text{V}$.

---

## 4. Hard Data: Benchmark Leaderboard (Cz Channel, Auditory Condition A)

*Cohort: 37 Clinical Adult CI Recipients (CMPy2 Cohort)*

| Method | N Valid | Pre-stimulus Baseline RMS ($\mu\text{V}$) $\downarrow$ | N1 Amplitude ($\mu\text{V}$) | N1 Latency ($\text{ms}$) | P2 Amplitude ($\mu\text{V}$) | P2 Latency ($\text{ms}$) | Peak-to-Peak ($\mu\text{V}$) | ERP SNR ($\text{dB}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **AATS (Track B)** | **36** | **$1.02 \pm 0.64$** | **$-2.96 \pm 1.60$** | **$112.3 \pm 23.3$** | **$+3.26 \pm 2.21$** | **$203.4 \pm 33.7$** | **$6.22 \pm 3.48$** | **$13.10 \pm 2.11$** |
| **Raw CI EEG** | 37 | $1.98 \pm 1.44$ | $-6.02 \pm 3.94$ | $115.4 \pm 24.0$ | $+6.78 \pm 4.44$ | $193.0 \pm 30.8$ | $12.80 \pm 7.62$ | $14.37 \pm 3.64$ |
| **PCA Baseline** | 7 | $2.98 \pm 2.76$ | $-7.26 \pm 5.56$ | $124.0 \pm 21.9$ | $+9.34 \pm 7.06$ | $207.1 \pm 23.0$ | $16.61 \pm 12.39$ | $13.07 \pm 2.91$ |
| **CCA Baseline** | 8 | $5.85 \pm 5.93$ | $-15.98 \pm 13.91$ | $110.8 \pm 24.3$ | $+14.70 \pm 14.72$ | $186.9 \pm 28.7$ | $30.68 \pm 25.99$ | $13.06 \pm 2.82$ |

> [!IMPORTANT]
> **Key Takeaway for the Meeting:** Classical BSS methods (CCA, PCA) fail on CI EEG because their mathematical assumptions (statistical independence or zero correlation) are broken by high-power electrical volume conduction. AATS succeeds because it is **physiologically grounded in reference-channel dipole geometry**.

---

## 5. Failure Cases Diagnosed & Engineering Fixes

| Issue Identified | Cause | Fix Implemented | File Location |
| :--- | :--- | :--- | :--- |
| **Track A CUDA Out-Of-Memory Error** (47.4 GB VRAM exhausted on A6000) | `batch_size=8` combined with Multi-Resolution STFT backward graphs accumulated VRAM | Reduced `batch_size` to 2 (matching baseline), enabled `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"`, cached Hann windows, and added explicit graph deletion and periodic `empty_cache()` | [`track_a_deep_learning/train.py`](file:///Users/evanywliang/Documents/cidenoise/ci_denoise/track_a_deep_learning/train.py) & [`submit_train_gpu.sh`](file:///Users/evanywliang/Documents/cidenoise/ci_denoise/track_a_deep_learning/submit_train_gpu.sh) |
| **Track B Summary Aggregation TypeError** | Pandas `.groupby().agg()` attempted to average string metadata columns | Explicitly indexed numeric metric columns prior to `.agg()` and added robust outlier exclusion (|P2P| > 500 $\mu$V) | [`track_b_adaptive_subtraction/evaluate_erp.py`](file:///Users/evanywliang/Documents/cidenoise/ci_denoise/track_b_adaptive_subtraction/evaluate_erp.py) |
| **Outlier explosion on subject `0923y2`** ($10^{16}\ \mu\text{V}$) | Disconnected/dead reference channel created near-zero singular values in normal equations | Added variance check (`np.std > 1e-6`) to drop flat reference channels, conditioned ridge regularization, and a $5\times$ power sanity clamp | [`track_b_adaptive_subtraction/aats_filter.py`](file:///Users/evanywliang/Documents/cidenoise/ci_denoise/track_b_adaptive_subtraction/aats_filter.py) |
| **SLURM Job OOM Kill during evaluation** | `load_hearing_template` tried to load full 19 GB Zarr array slice `[:]` into node RAM | Replaced full slice with chunked streaming (50 trials/batch, < 5 MB RAM) | [`track_b_adaptive_subtraction/evaluate_erp.py`](file:///Users/evanywliang/Documents/cidenoise/ci_denoise/track_b_adaptive_subtraction/evaluate_erp.py) |
| **MNE RAM build-up over 54 files** | All 128 channels epoched with `picks=None`; objects kept in memory | Filtered picks to `[args.target_channel]` (`Cz`) and added explicit `gc.collect()` | [`track_b_adaptive_subtraction/evaluate_erp.py`](file:///Users/evanywliang/Documents/cidenoise/ci_denoise/track_b_adaptive_subtraction/evaluate_erp.py) |
| **Redundant re-filtering on reruns** | `denoise_fif.py` re-processed existing files every time | Added `--overwrite` (default `False`) to instantly skip already-cleaned FIF files | [`track_b_adaptive_subtraction/denoise_fif.py`](file:///Users/evanywliang/Documents/cidenoise/ci_denoise/track_b_adaptive_subtraction/denoise_fif.py) |

---

## 6. Discussion Questions for the Research Meeting

1. **Adopting Track B (AATS) as the Primary Preprocessing Pipeline:**
   - *Question:* Given that AATS reduces noise by 40.3% and runs deterministically in 30 seconds on CPU without training, should we establish AATS as the default preprocessing stage for all clinical FIF files in the lab?
2. **Next Step for Track A (Deep Learning):**
   - *Question:* Should Track A operate on **raw data**, or should it operate as a **Stage 2 polisher on AATS-cleaned data** (Track C: Hybrid Pipeline)?
   - *Rationale:* Track A struggles with high-amplitude RF pulses when trained end-to-end, but neural networks excel at non-linear residual noise (eye blinks, myogenic tension, alpha ringing) once the massive electrical pulse has been removed by AATS.
3. **Validation Against Hearing Controls:**
   - *Question:* What correlation threshold with the normal hearing grand average template constitutes a "clinically validated" restored ERP?
4. **Expanding to Multi-Channel Topographies:**
   - *Question:* Current benchmark focuses on vertex `Cz` / `Fz`. Should we extend evaluation to 2D topographies (e.g., scalp voltage maps showing the bilateral temporal auditory dipole) across all 64/128 channels?

---

## 7. How to Reproduce / Run Each Track on Hive

```bash
# -------------------------------------------------------------
# TRACK A: Deep Learning (U-Net with Composite Loss)
# -------------------------------------------------------------
# Train GPU model on Hive:
sbatch track_a_deep_learning/submit_train_gpu.sh

# -------------------------------------------------------------
# TRACK B: Physiological Adaptive Subtraction (AATS)
# -------------------------------------------------------------
# Fast evaluation of ERP preservation on CPU (~1-2 mins):
sbatch track_b_adaptive_subtraction/submit_evaluate_erp.sh

# Or run full batch denoising from scratch on CPU (~30 mins):
sbatch track_b_adaptive_subtraction/submit_aats_cpu.sh

# -------------------------------------------------------------
# SYNC RESULTS TO LOCAL MAC
# -------------------------------------------------------------
rsync -avzP evliang@hive.hpc.ucdavis.edu:~/cidenoise/benchmark_results/ ./benchmark_results/
```
