import mne
import numpy as np
import scipy.linalg

path = "/quobyte/millerlmgrp/annotated_data/CMPy2/trial_onsets_0923y2-raw.fif"
print(f"Loading {path}...")
raw = mne.io.read_raw_fif(path, preload=True, verbose=False)
raw.resample(256.0)
raw.filter(1.0, 45.0, phase='zero')

data = raw.get_data()
print("Data shape:", data.shape)
print("NaN count:", np.isnan(data).sum())
print("Inf count:", np.isinf(data).sum())

stds = np.std(data, axis=1)
print(f"Channel STDs: min={np.min(stds):.2e}, max={np.max(stds):.2e}")
zero_stds = np.where(stds == 0)[0]
print(f"Zero-std channels ({len(zero_stds)}):", [raw.ch_names[i] for i in zero_stds])

cov = np.cov(data)
print("Cov min/max:", np.min(cov), np.max(cov))
print("Cov is finite:", np.all(np.isfinite(cov)))

try:
    d, u = np.linalg.eigh(cov)
    print("np.linalg.eigh: SUCCESS")
except Exception as e:
    print("np.linalg.eigh FAILED:", type(e), e)

try:
    d, u = scipy.linalg.eigh(cov)
    print("scipy.linalg.eigh: SUCCESS")
except Exception as e:
    print("scipy.linalg.eigh FAILED:", type(e), e)

try:
    u, s, vt = scipy.linalg.svd(cov)
    print("scipy.linalg.svd: SUCCESS")
except Exception as e:
    print("scipy.linalg.svd FAILED:", type(e), e)
