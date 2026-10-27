# Research Meeting Guideline & Progress Report
**Date:** October 27, 2026  
**Project:** Cochlear Implant (CI) EEG Denoising (`ci_denoise`)  
**Author:** Evan Liang  

---

## 1. Executive Summary & Major Work Done (Post-Oct 20, 2026)

Over the past week, we made substantial advancements across empirical pipeline development, artifact isolation methodologies, and deep learning model analysis:

1. **Developed Single-Pair Isolation Pipeline & Interactive Visualizations:** Built a matched-permutation extraction pipeline (`isolate_and_pair_single_subject.py`) and an interactive HTML5 waveform viewer (`visualization/single_pair_viewer.html`) to directly inspect and compare raw CI artifact extraction methods against hearing controls.
2. **Engineered the AATS + SOBI Synthetic Pairing Strategy:** Designed a principled two-stage hybrid framework (Adaptive Artifact Template Subtraction + Second-Order Blind Identification) to isolate pure CI stimulation artifacts without stripping underlying brain waves, solving the fundamental training contamination problem of synthetic mixing.
3. **Analyzed 1D U-Net Mechanics & Architecture:** Detailed how multi-channel 1D convolutions act as non-linear spatial filters across 21 scalp electrodes, and how horizontal skip connections prevent temporal smearing of clinical ERP latencies.
4. **Discovered & Analyzed Shyam's Noise2Noise Implementation:** Uncovered that Shyam implemented a polarity-paired Noise2Noise model (`shyam/pipeline.ipynb`), identified why it was restricted to discrete high-frequency clicks (>800 Hz), and formulated the exact roadmap to generalize N2N to low-frequency continuous speech (1–50 Hz).

---

### Detailed Technical Findings:

#### A. Single-Pair Matching & Interactive Visualizations (`single_pair_viewer.html`)
* **Matched Permutation Pairing:** In `processing/isolate_and_pair_single_subject.py`, we paired 1 CI participant (`0901y2`) with 1 Hearing control (`0801y2`) sharing the **exact same permutation (`perm1`)**, ensuring that both participants heard the identical continuous speech sentence stimuli.
* **Multi-Method Comparison:** Extracted artifact and denoised outputs across 5 distinct methods on the same recording:
  - **AATS** (Stimulus-Locked Adaptive Artifact Template Subtraction)
  - **SOBI** (Second-Order Blind Identification via lagged covariances)
  - **ICA** (FastICA mastoid-loading component rejection)
  - **AATS + SOBI** (Sequential hybrid cascade)
  - **AATS + ICA** (Sequential hybrid cascade)
* **Interactive Inspection Tool (`visualization/single_pair_viewer.html`):**
  - Exported continuous and single-epoch data (`export_single_pair_viewer.py`) to an interactive browser tool.
  - Enables real-time toggling across all 21 electrode channels, comparing Clean Hearing Ground Truth, Raw CI EEG, isolated noise, and denoised estimates.
  - Automatically computes pairwise Pearson correlations ($r(\text{Clean}, \text{AATS})$, $r(\text{Clean}, \text{SOBI})$, $r(\text{Clean}, \text{AATS+SOBI})$, $r(\text{AATS}, \text{ICA})$), visually validating that AATS+SOBI preserves Cz and auditory vertex potentials while eliminating massive temporal coil spikes.

#### B. The Plan: Using AATS + SOBI to Generate High-Fidelity Synthetic Pairs
* **The Fundamental Problem with Naive Synthetic Mixing:**
  - Standard training uses $\text{dirty} = \text{clean\_hearing} + \alpha \cdot \text{ci\_noise}$.
  - If we naively treat raw CI recordings as pure "noise", we inject the CI user's real auditory cortical responses into the noise matrix. The neural network then learns to subtract real brain activity!
* **The AATS + SOBI Solution:**
  1. **Stage 1 — AATS (Time-Domain Envelope Subtraction):** AATS models the deterministic, stimulus-locked electrical stimulation burst time-locked to speech onsets without distorting background EEG phase or amplitude.
  2. **Stage 2 — SOBI (Time-Lagged Autocovariance Blind Source Separation):** SOBI jointly diagonalizes multiple time-lagged cross-covariance matrices (lags $k \in [1, 2, 4, 8, 16, 25, 50]$). Unlike standard ICA (which relies on higher-order statistics/kurtosis and notoriously over-cleans frontal/temporal brain waves), SOBI exploits temporal auto-correlations to isolate non-stationary electrical stimulation dipoles while leaving uncorrelated Gaussian EEG untouched.
  3. **Stage 3 — Hybrid Cascade (AATS + SOBI):** AATS strips the massive stimulus transients first; SOBI then cleanly isolates the residual stationary spatial components.
  4. **Pristine Synthetic Dataset Generation:** The resulting isolated noise ($\text{noise}_{\text{AATS+SOBI}}$) represents true, isolated CI electrical interference. Mixing this with matched-permutation clean hearing EEG produces pristine, leak-free synthetic training pairs for the 1D U-Net:
     $$\text{dirty} = \text{clean\_hearing}_{\text{perm}_k} + \alpha \cdot \text{noise}_{\text{AATS+SOBI},\,\text{perm}_k}$$

#### C. Mechanics of the 1D U-Net for Continuous CI EEG
* **Multi-Scale Spatio-Temporal Filtering:** Unlike 2D computer vision U-Nets, our 1D U-Net slides convolutions across time ($T$) while processing all 21 scalp electrodes simultaneously as input feature channels.
* **Separating Artifact from Brain:** In CI users, electrical stimulation artifacts are $10\times$ to $100\times$ larger than microvolt-level neural signals. The U-Net acts as a non-linear spatial filter (similar to a deep, non-linear beamformer/ICA), learning that artifacts originate unilaterally from the implant coil while auditory brain activity distributes bilaterally.
* **Preserving Phase & Latency:** Horizontally linked skip connections (`torch.cat([x_enc, x_dec], dim=1)`) bypass the downsampling bottleneck, preventing the smearing of millisecond-precise ERP latencies (P1-N1-P2) and speech tracking phase dynamics.

#### D. Discovery: Shyam's Existing Noise2Noise Implementation
* **Finding:** In `shyam/pipeline.ipynb` (Stages 3, B, and C), Shyam **did implement a Noise2Noise (N2N) 1D U-Net** (`ConvBlock`, `UNet1D`, `make_loss`).
* **The Mechanism:** Shyam exploited stimulus polarity. In auditory evoked potentials, auditory brainstem responses are largely polarity-invariant, whereas electrical CI artifacts invert or vary between alternating positive and negative clicks (triggers `104` vs `105`).
* **The Critical Difference Between `shyam/` and `modeling/`:**

| Feature | Shyam's Pipeline (`shyam/`) | Our Target Pipeline (`modeling/` & `benchmarking/`) |
| :--- | :--- | :--- |
| **Learning Paradigm** | Self-Supervised Noise2Noise | Supervised Synthetic Mixtures (`clean + α·noise`) |
| **Noise Source** | Isolated from real CI data | Synthetic pairs via **AATS + SOBI** isolation |
| **Frequency Band** | **High-pass split at 800 Hz** (`HF_CUTOFF = 800.0 Hz`) | **Clinical EEG band (1–50 Hz)** |
| **Brain Waves (<800 Hz)** | **Untouched by N2N** (relied completely on standard ICA) | Directly denoises delta, theta, alpha, and ERPs |
| **Target Noise** | 1–8 kHz RF coil stimulation comb spikes | Low-frequency CI envelope modulation leaking into EEG |
| **Sampling Rate** | 16,384 Hz | 250 Hz (Nyquist = 125 Hz) |
| **Stimulus Type** | Cropped **62.5 ms click windows** (triggers 104/105) | **Continuous 60-second running speech** (`CMPy2`) |
| **Loss Target** | Multi-Scale STFT on 1–8 kHz comb lines + Time L1 | Time L1/MSE + Spectral fidelity on clinical bands |

---

## 2. Theoretical Blueprint: Generalizing N2N to Continuous Speech

Shyam proved that Noise2Noise works on discrete alternating-polarity clicks. To eliminate synthetic mixing and train a self-supervised model directly on real CI continuous speech, we have formulated a 4-part generalization plan:

1. **The Continuous Pairing Strategy:**
   * In `CMPy2_Stim_Matrix_SentenceSet{1,2,3}.mat` (annotated in `extract_trial_onsets.py`), continuous sentences repeat across experimental blocks.
   * **Pairing Option 1 (Repeated Sentences):** Pair the EEG of Sentence $i$ from Block $k$ with the EEG of Sentence $i$ from Block $m$ for the same subject. The underlying speech-locked brain response $s(t)$ is identical, while non-stationary CI artifacts and background sensor noise $n(t)$ vary.
   * **Pairing Option 2 (Polarity-Inverted Continuous Blocks):** Trigger codes 10 and 11 designate positive vs. negative audio polarity halves. Continuous speech played in opposite polarities cancels artifact envelope modulations while preserving polarity-invariant cortical tracking.
2. **Sub-Millisecond Temporal Synchronization:**
   * Unlike clicks, continuous speech cannot tolerate temporal jitter; a 15–20 ms offset causes destructive phase cancellation above 25 Hz. Continuous streams must be cross-correlated against trigger markers and stimulus envelopes down to sub-sample precision.
3. **Receptive Field Expansion:**
   * Expand from Shyam's 62.5 ms window to **1.5–2.0 second continuous temporal receptive fields** (using dilated convolutions or attention) to capture phonemic, syllabic (150–300 ms), and prosodic dynamics.
4. **Clinical Low-Frequency Spectral Loss:**
   * Replace Shyam's 1–8 kHz comb loss with a multi-scale STFT loss over clinical EEG bands (Delta 1–4 Hz, Theta 4–8 Hz, Alpha 8–13 Hz) and batch-consistency constraints.

---

## 3. High-Performance Cluster Infrastructure (Hive)

We established production-ready SLURM pipelines ready to execute on UC Davis Hive:
* **`submit_train_and_tier3_cmb.sh`:** 
  * Designed for the `cmb-grp` CPU allocation (16 cores, 128 GB RAM, 48-hour walltime limit).
  * Automatically sets `OMP_NUM_THREADS=16`, `MKL_NUM_THREADS=16`, and dynamic PyTorch thread utilization.
  * Auto-resumes from `latest.pt` / `best.pt` via `--resume` so interruptions never lose progress.
  * Chains model training directly into Tier 3 downstream benchmark evaluation in one script.
* **`submit_train_and_tier3_low.sh`:**
  * Configured for opportunistic GPU nodes on `publicgrp-low` (`--gres=gpu:1`, A6000/A100/H100).
  * Equipped with `#SBATCH --requeue` + `--resume` for robust preemption recovery.

---

## 4. Future Research Directions

### Direction 1: Self-Supervised Continuous Speech Noise2Noise (High Impact)
* Move beyond synthetic mixing (`clean hearing + α·CI noise`).
* Implement a continuous speech `N2NDataset` that pairs repeat presentations of sentences from `annotated_data/` and trains without requiring ground-truth clean EEG.

### Direction 2: Advanced Generative Architectures
* **Diffusion Models (*EEGDfus*, Huang et al. 2024):** Replace MSE regression (which causes temporal blurring) with conditional score-based diffusion to reconstruct high-fidelity EEG.
* **Attention / Self-Similarity (*EEGDNet*, Yi et al.):** Incorporate non-local self-attention blocks to capture long-range linguistic context in continuous speech.

### Direction 3: Cortical Speech Tracking as Primary Downstream Metric
* Adopt the neuroengineering standard from **Somers & Francart (2018–2020)**: evaluate denoised EEG by measuring the cross-correlation ($r_{\text{envelope}}$) between continuous theta-band EEG (4–8 Hz) and the acoustic speech envelope.
* Compute Temporal Response Functions (TRFs) using mTRF to confirm that real auditory cortex responses are preserved.

---

## 5. Clarifying Questions to Ask in Today's Meeting

To ensure research alignment with lab leadership and collaborators, ask the following targeted questions:

### A. Regarding Methodology & Long-Term Goal:
1. *"Shyam’s pipeline used N2N strictly above 800 Hz to clean click RF combs, leaving the brain waves (<800 Hz) to classical ICA. Is our research goal to **replace ICA entirely** with an end-to-end deep learning denoiser operating on the clinical band (1–50 Hz), or are we aiming for a two-stage hybrid (ICA first, then neural refinement)?"*
2. *"Are we satisfied keeping the synthetic supervised mixing paradigm (`clean hearing + α·CI noise`) as our primary conference paper baseline, or should we actively push into self-supervised Noise2Noise across repeated sentences as the central contribution?"*
3. *"For our synthetic baseline training, we developed an **AATS + SOBI** extraction cascade to isolate CI artifacts without stripping the patient's underlying brain waves (as demonstrated in our single-pair viewer). Should we re-generate the full synthetic training zarr dataset using this AATS + SOBI noise across all permutations?"*

### B. Regarding Continuous Speech & Downstream Validation:
3. *"Since our dataset consists of continuous running speech sentences rather than discrete clicks, isolated P1-N1-P2 ERP peaks naturally overlap. Do we have access to the raw stimulus audio files (`.wav`) so we can compute **acoustic speech envelope tracking ($r_{\text{envelope}}$)** and TRFs as our primary neuroscience validation metric?"*
4. *"In `CMPy2`, sentences were presented in alternating polarity blocks (triggers 10 and 11). Would you prefer we pair repeated identical sentences across blocks, or pair opposite polarity blocks for self-supervised training?"*

### C. Regarding Compute & Checkpoint Assets:
5. *"Has a prior 1D U-Net checkpoint (`best.pt`) already been trained by Aaron or Shyam on an external lab machine (e.g. `matterhorn`), or should we let our 48-hour automated training pipeline run on Hive to produce the official baseline?"*
