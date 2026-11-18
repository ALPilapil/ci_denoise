# Editorial Submission Cover Letter

**To:** The Editorial Board, *Nature Communications* (or *IEEE Transactions on Biomedical Engineering*)  
**From:** Evan Liang & Lee M. Miller  
**Date:** November 17, 2026  
**Subject:** Submission of Research Article: *"Biophysical Disentanglement and Self-Supervised Subspace Learning for Continuous Speech EEG Denoising in Cochlear Implant Recipients"*  

---

Dear Editors,

We are pleased to submit our original research manuscript titled **"Biophysical Disentanglement and Self-Supervised Subspace Learning for Continuous Speech EEG Denoising in Cochlear Implant Recipients"** for consideration as an Article in *Nature Communications*.

### Significance & Scientific Breakthrough
Over one million deaf individuals worldwide rely on cochlear implants (CIs) to recover auditory sensation. However, measuring non-invasive auditory cortical processing during natural running speech has remained fundamentally obstructed for three decades. Transcutaneous radio-frequency (RF) inductive power transmission and continuous stimulus pulse trains propagate volume-conducted electrical artifacts that exceed neural evoked potentials ($3\text{--}8\ \mu\text{V}$) by over $30\text{--}40\text{ dB}$, completely overlapping with speech-relevant electrophysiological frequencies ($1\text{--}45\text{ Hz}$).

Existing signal processing methods exhibit critical failure modes: classical linear blind source separation (ICA/SOBI) routinely prunes genuine auditory dipoles, while deep neural networks trained on synthetic mixtures suffer from the *sim-to-real transfer gap* and artificially inflate zero-lag stimulus artifact spikes in forward temporal response function (mTRF) deconvolution.

To solve this intractable trade-off, we introduce **`cidenoise`**, a biophysically grounded disentanglement framework:
1. **Automated Two-Pass SOBI**: Continuous blind source separation with an active midline safeguard guaranteeing $>70\%$ biological power retention on auditory cortex channels (`Cz`, `Fz`, `Pz`) with exact $0.0\text{ ms}$ zero-phase delay.
2. **Subspace-Constrained Symmetric Noise2Noise (N2N)**: Trains directly on unlabelled, real within-block stimulus polarity pairs ($y_1, y_2$), eliminating the sim-to-real transfer gap while projecting residual corrections strictly into the artifact subspace $\mathbf{P}_{\text{art}}$.
3. **Cohort-Wide Multi-Year Clinical Validation ($N = 70$)**: Validated across the UC Davis CI cohort across three distinct follow-up years (Year 2, Year 3, and Year 4) alongside normal-hearing controls ($N = 15$). Downstream N2N delivered up to **$-23.09\text{ dB}$** coil suppression while preserving vertex morphology invariance ($r = 1.000000$, $\text{RMSE} < 0.0001\ \mu\text{V}$) across all patients.
4. **Longitudinal Cortical Neuroplasticity**: We demonstrate for the first time that recovered continuous EEG tracks long-term cortical neuroplastic adaptation, including a **$27.6\text{ ms}$ P2 latency contraction** toward normal-hearing benchmarks in repeat patient `0901`, and a **$6.25\times$ monotonic surge** in central P2 amplitude in patient `0938` across 3 consecutive annual visits.
5. **Continuous Speech Forward Encoding (mTRF)**: Tested against $N = 500$ phase-randomized surrogate envelopes, our pipeline suppressed the spurious zero-lag spike from $0.5546 \to 0.0417$ and recovered canonical cortical P2 speech tracking at $209.9\text{ ms}$ ($p < 0.001$).

### Open-Source Release & Reproducibility
The full framework is packaged into a turnkey Python package with automated biophysical unit tests passing on high-performance computing clusters, accompanied by an interactive browser studio loaded with 183 cohort evaluation records.

### Declarations
- This manuscript has not been published previously and is not under consideration elsewhere.
- All authors have approved the final manuscript and declare no competing financial interests.
- Data and code are openly available under the MIT license at `https://github.com/ALPilapil/ci_denoise`.

We suggest the following expert reviewers in auditory neuroscience and neural engineering:
1. Prof. Tom Francart, KU Leuven (Auditory speech tracking in CI users)
2. Prof. Jonathan Z. Simon, University of Maryland (mTRF cortical deconvolution)
3. Prof. Edmund C. Lalor, University of Rochester (Continuous speech electrophysiology)

Thank you for your time and consideration of our work.

Sincerely,

**Evan Liang**  
Center for Mind and Brain, Department of Biomedical Engineering  
University of California, Davis  
Email: `evliang@ucdavis.edu`  

**Lee M. Miller, Ph.D.**  
Professor of Neurobiology, Physiology, and Behavior  
Center for Mind and Brain, University of California, Davis  
