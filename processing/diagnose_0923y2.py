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

nan_channels = [raw.ch_names[i] for i in range(len(raw.ch_names)) if np.isnan(data[i]).all()]
any_nan_channels = [raw.ch_names[i] for i in range(len(raw.ch_names)) if np.isnan(data[i]).any()]
print("All-NaN channels:", nan_channels)
print("Any-NaN channels:", any_nan_channels)
