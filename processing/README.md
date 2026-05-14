# Overview
General data processing tools.

## extract_trial_onsets.py
Reads raw EEGLAB `.set` files and adds A, AV, and V-only condition codes as MNE annotations. Outputs one `trial_onsets_{subject}-raw.fif` and one companion `.csv` of metadata per subject. `.fif` files are organized by year (`CMPy2/`, `CMPy3/`, etc.).

## ml_denoise.py
Reads annotated `.fif` files from `/quobyte/millerlmgrp/annotated_data/CMPy*/` and applies classical signal-separation methods to produce `(cleaned_signal, isolated_noise)` pairs for model training. A 2 Hz high-pass filter is applied to every file before any method runs.

**Clean data** → `/quobyte/millerlmgrp/ml_cleaned_data/{method}/CMPy{N}/`  
**Isolated noise** → `/quobyte/millerlmgrp/ml_isolated_noise/{method}/CMPy{N}/`

Each output `.fif` has a companion `.csv` with metadata (participant ID, year, method, n_components, sfreq, paths, timestamp, status).

### Methods

| Key | Method | Description |
|---|---|---|
| `ica` | Independent Component Analysis | MNE ICA; CI noise components identified by highest weights at P7/T7/M2/M1/P8 |
| `iva` | Independent Vector Analysis | Sub-band FastICA across 4 frequency bands (2–8, 8–15, 15–30, 30+ Hz); CI components aggregated across bands |
| `cca` | Canonical Correlation Analysis | Self-CCA with 1-sample-delayed reference; removes canonical variates dominated by CI channels |
| `emd_ica` | EMD-ICA | PyEMD decomposes each channel into IMFs, stacked as virtual channels, then ICA selects CI components |
| `emd_cca` | EMD-CCA | Same IMF extraction as EMD-ICA, but self-CCA used instead of ICA |
| `pca` | Principal Component Analysis | Projects out PCs where CI channels account for >30% of the total loading |
| `ssp` | Signal Space Projection | MNE `compute_proj_raw`; removes top-variance spatial projectors |
| `wavelet` | Wavelet Thresholding | Per-channel DWT with MAD-based universal threshold (PyWavelets, `db4`) |

### Usage
```bash
# Install optional deps once
pip install EMD-signal PyWavelets

# Run specific methods on one year
python processing/ml_denoise.py --methods ica pca ssp --year 2

# Run all methods across all annotated years (CMPy2, CMPy3, CMPy4)
python processing/ml_denoise.py \
  --methods ica iva cca emd_ica emd_cca pca ssp wavelet \
  --data-dir /quobyte/millerlmgrp/annotated_data \
  --clean-dir /quobyte/millerlmgrp/ml_cleaned_data \
  --noise-dir /quobyte/millerlmgrp/ml_isolated_noise
```
