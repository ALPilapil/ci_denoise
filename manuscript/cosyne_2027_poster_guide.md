# COSYNE 2027 Poster Presentation Guide & Talking Track

**Conference:** Computational and Systems Neuroscience (COSYNE 2027)  
**Session Track:** Auditory Systems / Neural Encoding & Decoding / Neuroengineering  
**Title:** *Restoring Cortical Speech Tracking in Cochlear Implant Recipients via Biophysically-Constrained Subspace Disentanglement*  
**Authors:** Evan Liang$^1$, Lee M. Miller$^{1,2}$  
**Affiliations:**  
$^1$ Center for Mind and Brain, Department of Biomedical Engineering, UC Davis  
$^2$ Department of Neurobiology, Physiology and Behavior, UC Davis  

---

## 1. Poster Dimensions & Visual Layout (48" $\times$ 36" Landscape)

```
+---------------------------------------------------------------------------------------------------------+
|                                             HEADER BLOCK                                                |
| TITLE: Restoring Cortical Speech Tracking in Cochlear Implant Recipients via Biophysically-Constrained   |
|        Subspace Disentanglement                                                                         |
| AUTHORS: Evan Liang, Lee M. Miller | UC Davis Center for Mind and Brain | GitHub: ALPilapil/ci_denoise   |
+---------------------+---------------------+---------------------------------------+---------------------+
|      PANEL 1        |      PANEL 2        |               PANEL 3                 |      PANEL 4        |
|  The Biophysical    |    Two-Stage        |    Multi-Year Clinical Benchmark      |    mTRF Speech      |
|    Challenge        |   Disentanglement   |         (N = 70 CI Cohort)            |     Encoding        |
|                     |     Framework       |                                       |                     |
| • 40 dB RF Artifact | • Stage 1: SOBI     | • Foundation Subspace Invariance      | • Forward Ridge     |
|   vs. 3-8 µV EEG    |   - Zero-phase FIR  |   (Zero-shot transfer across Y3 & Y4) |   Deconvolution     |
| • Sim-to-Real Gap   |   - Midline Guard   | • Vertex Morphology Invariance        | • Zero-Lag Spike    |
| • Polarity Symmetry | • Stage 2: N2N      |   (r = 1.000000, RMSE < 0.0001 µV)    |   Excision:         |
|   s1 ≈ s2           |   - Unsupervised    | • Silent Baseline Floor (< 2.0 µV)    |   0.5546 -> 0.0417  |
|   a1 ≈ -a2          |   - Subspace P_art  | • Neuroplastic Latency Shortening     | • Recovered P2 Peak |
|                     |                     |   (27.6 ms contraction in Pt 0901)    |   at 209.9 ms       |
|                     |                     | • 6.25x P2 Amplitude Surge (Pt 0938)  |   (p < 0.001)       |
+---------------------+---------------------+---------------------------------------+---------------------+
|                                            FOOTER BAR                                                   |
| Open Source: pip install cidenoise | Interactive Studio: https://alpilapil.github.io/ci_denoise/        |
+---------------------------------------------------------------------------------------------------------+
```

---

## 2. Poster Panel Content & Figures

### Panel 1: The Biophysical Dilemma
* **Core Problem:** Cochlear implants (CIs) transcutaneously deliver continuous RF inductive power and pulse trains that overwhelm genuine continuous speech brain potentials by $+40\text{ dB}$ across overlapping frequencies ($1\text{--}45\text{ Hz}$).
* **Existing Pitfalls:**
  - *Classical ICA/SOBI*: Over-cleans central auditory dipoles without guarantees.
  - *Supervised Deep Networks*: Suffer from the **sim-to-real transfer gap** when trained on synthetic mixtures; collapse or latch onto stimulus triggers, introducing severe zero-lag artifact spikes.
* **Biophysical Key:** Speech acoustic envelopes are polarity-invariant ($s_1 \approx s_2$), whereas RF stimulus artifacts invert with carrier/pulse polarity ($a_1 \approx -a_2$). Within-block polarity-alternating sentence trials provide naturally paired noisy observations of identical cortical speech tracking!

### Panel 2: The Two-Stage Disentanglement Engine
* **Stage 1 (Continuous SOBI BSS):**
  - Continuous zero-phase FIR bandpass ($1.0\text{--}45.0\text{ Hz}$, exact $0.0\text{ ms}$ delay $\tau_g = 0$).
  - Two-Pass Second-Order Blind Identification (SOBI) isolating mastoid-dominant components ($R > 2.5$).
  - **Active Midline Safeguard:** Strictly aborts/rolls back any pass that prunes $>30\%$ power from central cortical channels (`Cz`, `Fz`, `Pz`).
* **Stage 2 (Subspace-Constrained Symmetric N2N):**
  - 1D Residual U-Net trained self-supervised on real polarity pairs $(y_1, y_2)$ using symmetric L1 loss:
    $$\mathcal{L}_{\text{sym}} = \frac{1}{2}\|f_\theta(y_1) - y_2\|_1 + \frac{1}{2}\|f_\theta(y_2) - y_1\|_1$$
  - **Subspace Projector $\mathbf{P}_{\text{art}}$:** Residual corrections $\Delta y = f_\theta(y) - y$ are projected strictly onto the artifact subspace:
    $$\hat{y} = y + \mathbf{P}_{\text{art}} \Delta y, \quad \text{where } \mathbf{P}_{\text{art}}[\text{midline}, :] = \mathbf{0}$$
  - Guarantees **zero biological distortion** on vertex electrodes.

### Panel 3: Multi-Year Cohort Benchmark ($N = 70$ CI Users, $N = 15$ NH Controls)
* **Hardware QA Gate:** Automated variance screening catches corrupt amplifier saturation ($> 1\ \text{V}^2$), safely isolating corrupt recordings (e.g. `0923y2`).
* **Multi-Subject Foundation Generalization:** Pretrained across 8 CI recipients on Hive HPC; demonstrated universal zero-shot transfer across Year 3 ($N = 27$) and Year 4 ($N = 7$) recordings without retraining.
* **Vertex Preservation:** Exact midline correlation $r = \mathbf{1.000000}$ ($\text{RMSE} < 0.0001\ \mu\text{V}$) and silent visual noise floor of $1.026\ \mu\text{V}$ (well below the $2.0\ \mu\text{V}$ silence ceiling).
* **Cortical Neuroplasticity:**
  - Patient `0901` showed a **$27.6\text{ ms}$ P2 latency shortening** ($209.9 \to 182.3\text{ ms}$), converging within $2.3\text{ ms}$ of normal-hearing controls ($180.0\text{ ms}$).
  - Longitudinal patient `0938` exhibited a **$6.25\times$ monotonic surge** in P2 cortical amplitude over 3 annual visits ($0.141 \to 0.502 \to 0.882\ \mu\text{V}$).

### Panel 4: Speech Envelope Encoding (mTRF Deconvolution)
* **Forward Kernel Ridge Regression:** Fit across lags $\tau \in [-100, 450]\text{ ms}$ against acoustic speech envelopes.
* **Surrogate Verification:** Benchmarked against $N = 500$ phase-randomized Fourier surrogate envelopes:
  - Unconstrained neural networks created a massive false zero-lag artifact spike ($w_0 = 0.5546$).
  - `cidenoise` excised the zero-lag spike to $0.0417$ while recovering canonical positive P2 tracking at $209.9\text{ ms}$ ($p < 0.001$).

---

## 3. Presenter Talking Tracks

### The 60-Second Elevator Pitch
> *"Continuous speech tracking with EEG has unlocked immense insights into auditory cognition, but in cochlear implant users, the electrical transmission artifact is 40 dB louder than the brain and completely overlaps in time and frequency.*
>
> *Deep learning previously failed here because synthetic training data doesn't transfer to real patients, and neural nets latch onto stimulus triggers, producing massive zero-lag artifact spikes in forward speech deconvolution.*
>
> *We solved this with **cidenoise**: a two-stage biophysical framework. First, automated Two-Pass SOBI isolates the artifact while an active safeguard enforces >70% midline cortical retention. Second, a symmetric Noise2Noise 1D U-Net trains directly on real stimulus polarity pairs, projecting all corrections into an artifact subspace that freezes auditory leads like Cz.*
>
> *Across 70 CI recipients spanning three annual follow-up visits, our pipeline eliminated the false zero-lag spike from 0.55 to 0.04, recovered canonical cortical P2 tracking at 209 ms against 500 surrogate permutations, and revealed longitudinal neuroplastic recovery—including a 27.6 ms latency contraction toward normal-hearing benchmarks."*

---

### The 3-Minute Deep Dive
> *"Let me walk you through the biophysical foundation and what made this work on real patients when prior supervised models collapsed.*
>
> *In auditory EEG, the cortical response to running speech is primarily driven by the acoustic envelope, which is identical whether the acoustic carrier or CI stimulation polarity is inverted. But the electrical artifact inverts with pulse polarity. This symmetry allows us to formulate a self-supervised Noise2Noise objective on unlabelled, within-block polarity pairs $(y_1, y_2)$—eliminating synthetic training entirely.*
>
> *However, unconstrained Noise2Noise still has the capacity to distort phase. To eliminate that risk, we enforce two mathematical constraints: first, all pre-filtering is strictly zero-phase FIR with 0.0 ms group delay; second, we construct an artifact subspace projector $\mathbf{P}_{\text{art}}$ from the Stage 1 SOBI mixing vectors and explicitly set all midline rows and columns to zero.*
>
> *Because of this constraint, vertex channel Cz has a correlation of identically 1.000000 with the biological signal—it is mathematically impossible for the network to hallucinate or attenuate cortical dipoles.*
>
> *When we applied this foundation model zero-shot to our Year 3 and Year 4 clinical cohorts, it maintained negative mastoid attenuation without any patient retraining. Most excitingly, in forward speech mTRF deconvolution, we completely extinguished the trigger artifact spike and recovered genuine cortical plasticity: patient 0901 shortened their P2 cortical delay by 28 ms over follow-up visits, and patient 0938 exhibited a 6-fold increase in auditory evoked amplitude. All code, pretrained weights, and interactive viewers are open-source on GitHub."*

---

## 4. Anticipated Attendee Q&A & Answers

#### Q1: "Why use Two-Pass SOBI instead of Infomax or FastICA for Stage 1?"
> **Answer:** *"CIs produce continuous radio-frequency packet transmission whose non-zero time-delayed cross-correlations $\mathbf{R}_x(\tau)$ are orders of magnitude stronger and more stable than instantaneous higher-order cumulants. SOBI leverages cross-lagged covariance matrices, isolating the quasi-periodic RF artifact in just 1–2 passes without the stochastic run-to-run instability or sign-flipping common to FastICA."*

#### Q2: "How do you guarantee that the Noise2Noise model isn't fitting neural activity during the silent Condition V trials?"
> **Answer:** *"First, the projection operator $\mathbf{P}_{\text{art}}$ zeroes out the auditory midline leads. Second, we established an empirical silent visual quality gate: root-mean-square noise at Cz during silent Condition V trials must remain below $2.0\ \mu\text{V}$. In our cohort, the mean Condition V floor was $1.026\ \mu\text{V}$, confirming the network does not generate spurious activity."*

#### Q3: "What prevents the mTRF deconvolution from over-fitting to residual stimulus artifact?"
> **Answer:** *"In unconstrained neural networks, residual non-linear artifacts correlate with the speech envelope at lag 0, creating a massive spike ($w_0 = 0.5546$). Because our subspace constraint removes this leakage, the lag 0 weight drops to $0.0417$, and statistical significance testing against $N = 500$ phase-randomized Fourier surrogate envelopes confirms that only the physiological P2 peak at $209.9\text{ ms}$ reaches significance ($p < 0.001$)."*
