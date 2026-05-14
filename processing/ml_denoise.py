"""
ml_denoise.py

Applies multiple classical signal-separation methods to annotated EEG .fif files
to produce (cleaned_signal, isolated_noise) pairs for model training.

Methods:
  ica     – mne.preprocessing.ICA, auto-select CI noise components by electrode weight
  iva     – Sub-band FastICA across 4 frequency-band views (IVA approximation)
  cca     – Self-CCA with time-lagged reference; project out CI-proximal variates
  emd_ica – PyEMD IMF decomposition then ICA on virtual IMF channels
  emd_cca – PyEMD IMF decomposition then self-CCA on IMF channels
  pca     – sklearn PCA; project out components dominated by CI channel loadings
  ssp     – MNE signal-space projection via mne.proj.compute_proj_raw
  wavelet – Per-channel DWT with MAD-based universal thresholding (pywt)

Usage:
  python ml_denoise.py --methods ica pca ssp \\
    --data-dir /quobyte/millerlmgrp/annotated_data \\
    --clean-dir /quobyte/millerlmgrp/ml_cleaned_data \\
    --noise-dir /quobyte/millerlmgrp/ml_isolated_noise \\
    [--year 2] [--n-jobs 1]

Requirements:
  pip install EMD-signal PyWavelets
  (mne, scikit-learn, scipy, numpy, pandas already in ci_denoise_env)

Example – all methods on all annotated years (CMPy2, CMPy3, CMPy4):
  python processing/ml_denoise.py \\
    --methods ica iva cca emd_ica emd_cca pca ssp wavelet \\
    --data-dir /quobyte/millerlmgrp/annotated_data \\
    --clean-dir /quobyte/millerlmgrp/ml_cleaned_data \\
    --noise-dir /quobyte/millerlmgrp/ml_isolated_noise

  Outputs per method and year:
    /quobyte/millerlmgrp/ml_cleaned_data/{method}/CMPy{N}/trial_onsets_{subject}-raw.fif
    /quobyte/millerlmgrp/ml_isolated_noise/{method}/CMPy{N}/trial_onsets_{subject}-raw.fif
    .../{method}/CMPy{N}/trial_onsets_{subject}.csv   (metadata)
"""

import abc
import argparse
import datetime
import gc
import glob
import json
import os
import re
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import mne
import numpy as np
import pandas as pd
from mne.preprocessing import ICA
from mne.proj import compute_proj_raw
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import FastICA, PCA

CI_CHANNELS = ['P7', 'T7', 'M2', 'M1', 'P8']
HPF_FREQ = 2.0  # Hz, matches isolate_noise.py


# ── Base class ────────────────────────────────────────────────────────────────

class DenoiseMethod(abc.ABC):
    """
    Abstract base for all EEG denoising methods.

    Contract: fit_transform(raw) returns (cleaned_raw, noise_raw) where
    cleaned_raw + noise_raw ≈ raw to floating-point precision.
    """

    name: str = "base"

    @abc.abstractmethod
    def fit_transform(self, raw: mne.io.Raw) -> Tuple[mne.io.Raw, mne.io.Raw]:
        """Return (cleaned_raw, noise_raw)."""

    @abc.abstractmethod
    def get_metadata(self) -> dict:
        """Return method-specific params for the companion CSV."""

    @staticmethod
    def _make_noise_raw(raw_original: mne.io.Raw, clean_data: np.ndarray) -> mne.io.Raw:
        noise_data = raw_original.get_data() - clean_data
        noise_raw = mne.io.RawArray(noise_data, raw_original.info.copy(), verbose=False)
        noise_raw.set_annotations(raw_original.annotations)
        return noise_raw

    @staticmethod
    def _copy_with_data(raw_original: mne.io.Raw, new_data: np.ndarray) -> mne.io.Raw:
        new_raw = mne.io.RawArray(new_data, raw_original.info.copy(), verbose=False)
        new_raw.set_annotations(raw_original.annotations)
        return new_raw

    @staticmethod
    def _pca_n_components(data: np.ndarray, variance_threshold: float = 0.99999) -> int:
        """Integer component count explaining variance_threshold of variance. data: (n_samples, n_features)."""
        pca = PCA(n_components=variance_threshold)
        pca.fit(data)
        return pca.n_components_

    @staticmethod
    def _ci_noise_cols(weight_matrix: np.ndarray, ch_names: List[str],
                       ci_channels: List[str], n_top: int = 3) -> List[int]:
        """Return column indices whose top-n_top weights include any CI channel."""
        noise_cols = []
        for i, col in enumerate(weight_matrix.T):
            ranked = np.argsort(np.abs(col))[::-1]
            ranked_ch = [ch_names[idx] for idx in ranked[:n_top]]
            if any(ch in ci_channels for ch in ranked_ch):
                noise_cols.append(i)
        return noise_cols


# ── ICA ───────────────────────────────────────────────────────────────────────

class ICADenoiseMethod(DenoiseMethod):
    """ICA via mne.preprocessing.ICA; mirrors isolate_noise.py component selection."""

    name = "ica"

    def __init__(
        self,
        n_components: float = 0.99999,
        ci_channels: List[str] = CI_CHANNELS,
        random_state: int = 97,
    ):
        self.n_components = n_components
        self.ci_channels = ci_channels
        self.random_state = random_state
        self._noise_indices: List[int] = []
        self._n_components_found: Optional[int] = None

    def fit_transform(self, raw: mne.io.Raw) -> Tuple[mne.io.Raw, mne.io.Raw]:
        ica = ICA(n_components=self.n_components, max_iter="auto",
                  random_state=self.random_state)
        ica.fit(raw)
        self._n_components_found = ica.n_components_

        component_matrix = ica.get_components()  # (n_channels, n_components)
        noise_indices = self._ci_noise_cols(component_matrix, raw.ch_names, self.ci_channels)

        if not noise_indices:
            raise ValueError("ICA: no CI-proximal components found.")

        self._noise_indices = noise_indices
        clean_raw = raw.copy()
        ica.apply(clean_raw, exclude=noise_indices)

        noise_raw = self._make_noise_raw(raw, clean_raw.get_data())
        return clean_raw, noise_raw

    def get_metadata(self) -> dict:
        return {
            "method": self.name,
            "n_components": self._n_components_found,
            "noise_component_indices": str(self._noise_indices),
            "filter_params": f"l_freq={HPF_FREQ}",
        }


# ── IVA (sub-band FastICA approximation) ─────────────────────────────────────

class IVADenoiseMethod(DenoiseMethod):
    """
    IVA approximation: run FastICA independently on 4 frequency-band views,
    aggregate CI-proximal components, and subtract their sensor-space projections.
    """

    name = "iva"

    def __init__(
        self,
        variance_threshold: float = 0.99999,
        ci_channels: List[str] = CI_CHANNELS,
        frequency_bands: Optional[List[Tuple[Optional[float], Optional[float]]]] = None,
        random_state: int = 97,
    ):
        self.variance_threshold = variance_threshold
        self.ci_channels = ci_channels
        self.frequency_bands = frequency_bands or [
            (2.0, 8.0),
            (8.0, 15.0),
            (15.0, 30.0),
            (30.0, None),
        ]
        self.random_state = random_state

    def fit_transform(self, raw: mne.io.Raw) -> Tuple[mne.io.Raw, mne.io.Raw]:
        original_data = raw.get_data()
        accumulated_noise = np.zeros_like(original_data)

        for l_freq, h_freq in self.frequency_bands:
            band_raw = raw.copy()
            band_raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
            band_data = band_raw.get_data().T  # (n_times, n_channels)
            del band_raw

            n_comp = self._pca_n_components(band_data, self.variance_threshold)
            fica = FastICA(n_components=n_comp, random_state=self.random_state,
                           max_iter=1000)
            sources = fica.fit_transform(band_data)  # (n_times, n_comp)
            mixing = fica.mixing_                     # (n_channels, n_comp)

            noise_cols = self._ci_noise_cols(mixing, raw.ch_names, self.ci_channels)
            if noise_cols:
                accumulated_noise += mixing[:, noise_cols] @ sources[:, noise_cols].T

            gc.collect()

        clean_data = original_data - accumulated_noise
        return self._copy_with_data(raw, clean_data), self._copy_with_data(raw, accumulated_noise)

    def get_metadata(self) -> dict:
        return {
            "method": self.name,
            "variance_threshold": self.variance_threshold,
            "frequency_bands": str(self.frequency_bands),
            "filter_params": f"l_freq={HPF_FREQ}",
        }


# ── CCA ───────────────────────────────────────────────────────────────────────

class CCADenoiseMethod(DenoiseMethod):
    """
    Self-CCA denoising: uses a time-delayed copy of the data as the reference.
    CI artifact is highly autocorrelated, so canonical variates that correlate
    strongly with the delayed signal tend to capture artifact components.
    """

    name = "cca"

    def __init__(
        self,
        variance_threshold: float = 0.99999,
        delay_samples: int = 1,
        ci_channels: List[str] = CI_CHANNELS,
    ):
        self.variance_threshold = variance_threshold
        self.delay_samples = delay_samples
        self.ci_channels = ci_channels
        self._removed_components: List[int] = []

    def fit_transform(self, raw: mne.io.Raw) -> Tuple[mne.io.Raw, mne.io.Raw]:
        data = raw.get_data()  # (n_channels, n_times)
        delayed = np.zeros_like(data)
        delayed[:, self.delay_samples:] = data[:, :-self.delay_samples]

        X = data.T      # (n_times, n_channels)
        Y = delayed.T

        n_comp = min(self._pca_n_components(X, self.variance_threshold), X.shape[1])
        cca = CCA(n_components=n_comp)
        cca.fit(X, Y)
        x_weights = cca.x_weights_  # (n_channels, n_components)

        noise_cols = self._ci_noise_cols(x_weights, raw.ch_names, self.ci_channels)
        self._removed_components = noise_cols

        if not noise_cols:
            warnings.warn("CCA: no CI-proximal components found; signal unchanged.")
            return raw.copy(), self._copy_with_data(raw, np.zeros_like(data))

        W_noise = x_weights[:, noise_cols]
        # Project noise-component subspace out via orthogonal projection
        proj = W_noise @ np.linalg.pinv(W_noise).T  # (n_channels, n_channels)
        noise_data = proj.T @ data
        clean_data = data - noise_data

        return self._copy_with_data(raw, clean_data), self._copy_with_data(raw, noise_data)

    def get_metadata(self) -> dict:
        return {
            "method": self.name,
            "variance_threshold": self.variance_threshold,
            "delay_samples": self.delay_samples,
            "removed_components": str(self._removed_components),
            "filter_params": f"l_freq={HPF_FREQ}",
        }


# ── EMD-ICA ───────────────────────────────────────────────────────────────────

class EMDICADenoiseMethod(DenoiseMethod):
    """
    EMD-ICA: decompose each channel into IMFs via PyEMD, stack IMFs as virtual
    channels, apply FastICA, identify CI components, reconstruct sensor-space noise.

    Requires: pip install EMD-signal
    """

    name = "emd_ica"

    def __init__(
        self,
        variance_threshold: float = 0.99999,
        ci_channels: List[str] = CI_CHANNELS,
        max_imfs: int = 8,
        random_state: int = 97,
    ):
        self.variance_threshold = variance_threshold
        self.ci_channels = ci_channels
        self.max_imfs = max_imfs
        self.random_state = random_state

    def _extract_imfs(self, data: np.ndarray) -> np.ndarray:
        """Return IMF matrix (n_channels * max_imfs, n_times)."""
        try:
            from PyEMD import EMD
        except ImportError:
            raise ImportError("PyEMD required: pip install EMD-signal")

        n_channels, n_times = data.shape
        emd = EMD()
        all_imfs = []
        for ch_idx in range(n_channels):
            imfs = emd(data[ch_idx], max_imf=self.max_imfs)
            if imfs.shape[0] < self.max_imfs:
                pad = np.zeros((self.max_imfs - imfs.shape[0], n_times))
                imfs = np.vstack([imfs, pad])
            else:
                imfs = imfs[:self.max_imfs]
            all_imfs.append(imfs)
        return np.vstack(all_imfs)  # (n_ch * max_imfs, n_times)

    def _imf_to_sensor_noise(self, imf_noise: np.ndarray, n_channels: int) -> np.ndarray:
        """Sum IMF contributions within each channel to get sensor-space noise."""
        n_times = imf_noise.shape[1]
        noise_data = np.zeros((n_channels, n_times))
        for ch_idx in range(n_channels):
            r0, r1 = ch_idx * self.max_imfs, (ch_idx + 1) * self.max_imfs
            noise_data[ch_idx] = imf_noise[r0:r1].sum(axis=0)
        return noise_data

    def fit_transform(self, raw: mne.io.Raw) -> Tuple[mne.io.Raw, mne.io.Raw]:
        data = raw.get_data()
        n_channels, n_times = data.shape

        imf_matrix = self._extract_imfs(data)  # (n_ch * max_imfs, n_times)

        n_comp = self._pca_n_components(imf_matrix.T, self.variance_threshold)
        fica = FastICA(n_components=n_comp, random_state=self.random_state, max_iter=2000)
        sources = fica.fit_transform(imf_matrix.T)  # (n_times, n_comp)
        mixing = fica.mixing_                        # (n_ch * max_imfs, n_comp)

        # Aggregate mixing weights back to per-channel level for component selection
        ch_importance = np.zeros((n_channels, n_comp))
        for ch_idx in range(n_channels):
            r0, r1 = ch_idx * self.max_imfs, (ch_idx + 1) * self.max_imfs
            ch_importance[ch_idx] = np.abs(mixing[r0:r1]).sum(axis=0)

        noise_cols = self._ci_noise_cols(ch_importance, raw.ch_names, self.ci_channels)

        if not noise_cols:
            warnings.warn("EMD-ICA: no CI-proximal components found.")
            return raw.copy(), self._copy_with_data(raw, np.zeros_like(data))

        noise_imf = mixing[:, noise_cols] @ sources[:, noise_cols].T  # (n_ch*max_imfs, n_times)
        noise_data = self._imf_to_sensor_noise(noise_imf, n_channels)
        clean_data = data - noise_data

        return self._copy_with_data(raw, clean_data), self._copy_with_data(raw, noise_data)

    def get_metadata(self) -> dict:
        return {
            "method": self.name,
            "variance_threshold": self.variance_threshold,
            "max_imfs": self.max_imfs,
            "filter_params": f"l_freq={HPF_FREQ}",
        }


# ── EMD-CCA ───────────────────────────────────────────────────────────────────

class EMDCCADenoiseMethod(DenoiseMethod):
    """
    EMD-CCA: same IMF extraction as EMD-ICA, but uses self-CCA instead of ICA
    on the virtual IMF channel matrix.

    Requires: pip install EMD-signal
    """

    name = "emd_cca"

    def __init__(
        self,
        variance_threshold: float = 0.99999,
        ci_channels: List[str] = CI_CHANNELS,
        max_imfs: int = 8,
        delay_samples: int = 1,
    ):
        self.variance_threshold = variance_threshold
        self.ci_channels = ci_channels
        self.max_imfs = max_imfs
        self.delay_samples = delay_samples

    def fit_transform(self, raw: mne.io.Raw) -> Tuple[mne.io.Raw, mne.io.Raw]:
        try:
            from PyEMD import EMD
        except ImportError:
            raise ImportError("PyEMD required: pip install EMD-signal")

        data = raw.get_data()
        n_channels, n_times = data.shape

        emd = EMD()
        all_imfs = []
        for ch_idx in range(n_channels):
            imfs = emd(data[ch_idx], max_imf=self.max_imfs)
            if imfs.shape[0] < self.max_imfs:
                pad = np.zeros((self.max_imfs - imfs.shape[0], n_times))
                imfs = np.vstack([imfs, pad])
            else:
                imfs = imfs[:self.max_imfs]
            all_imfs.append(imfs)
        imf_matrix = np.vstack(all_imfs)  # (n_ch * max_imfs, n_times)

        X = imf_matrix.T
        delayed = np.zeros_like(imf_matrix)
        delayed[:, self.delay_samples:] = imf_matrix[:, :-self.delay_samples]
        Y = delayed.T

        n_comp = min(self._pca_n_components(X, self.variance_threshold), X.shape[1])
        cca = CCA(n_components=n_comp)
        cca.fit(X, Y)
        x_weights = cca.x_weights_  # (n_ch * max_imfs, n_components)

        # Aggregate to channel level for component selection
        ch_importance = np.zeros((n_channels, self.n_components))
        for ch_idx in range(n_channels):
            r0, r1 = ch_idx * self.max_imfs, (ch_idx + 1) * self.max_imfs
            ch_importance[ch_idx] = np.abs(x_weights[r0:r1]).sum(axis=0)

        noise_cols = self._ci_noise_cols(ch_importance, raw.ch_names, self.ci_channels)

        if not noise_cols:
            warnings.warn("EMD-CCA: no CI-proximal components found.")
            return raw.copy(), self._copy_with_data(raw, np.zeros_like(data))

        sources_x = X @ x_weights
        noise_imf = (x_weights[:, noise_cols] @ sources_x[:, noise_cols].T)

        noise_data = np.zeros((n_channels, n_times))
        for ch_idx in range(n_channels):
            r0, r1 = ch_idx * self.max_imfs, (ch_idx + 1) * self.max_imfs
            noise_data[ch_idx] = noise_imf[r0:r1].sum(axis=0)

        clean_data = data - noise_data
        return self._copy_with_data(raw, clean_data), self._copy_with_data(raw, noise_data)

    def get_metadata(self) -> dict:
        return {
            "method": self.name,
            "variance_threshold": self.variance_threshold,
            "max_imfs": self.max_imfs,
            "delay_samples": self.delay_samples,
            "filter_params": f"l_freq={HPF_FREQ}",
        }


# ── PCA ───────────────────────────────────────────────────────────────────────

class PCADenoiseMethod(DenoiseMethod):
    """
    PCA denoising: project out principal components whose loadings are
    dominated by CI channel activity above a threshold fraction.
    """

    name = "pca"

    def __init__(
        self,
        n_components: float = 0.99999,
        ci_channels: List[str] = CI_CHANNELS,
        ci_loading_threshold: float = 0.3,
    ):
        self.n_components = n_components
        self.ci_channels = ci_channels
        self.ci_loading_threshold = ci_loading_threshold
        self._noise_indices: List[int] = []

    def fit_transform(self, raw: mne.io.Raw) -> Tuple[mne.io.Raw, mne.io.Raw]:
        data = raw.get_data()
        ci_indices = [raw.ch_names.index(ch) for ch in self.ci_channels
                      if ch in raw.ch_names]

        pca = PCA(n_components=self.n_components)
        pca.fit(data.T)
        components = pca.components_  # (n_components, n_channels)

        noise_indices = []
        for i, comp in enumerate(components):
            ci_loading = np.abs(comp[ci_indices]).sum()
            total_loading = np.abs(comp).sum()
            if total_loading > 0 and ci_loading / total_loading > self.ci_loading_threshold:
                noise_indices.append(i)

        self._noise_indices = noise_indices

        if not noise_indices:
            warnings.warn("PCA: no CI-dominated components found.")
            return raw.copy(), self._copy_with_data(raw, np.zeros_like(data))

        noise_basis = components[noise_indices]         # (k, n_channels)
        scores = data.T @ noise_basis.T                 # (n_times, k)
        noise_data = (noise_basis.T @ scores.T)         # (n_channels, n_times)
        clean_data = data - noise_data

        return self._copy_with_data(raw, clean_data), self._copy_with_data(raw, noise_data)

    def get_metadata(self) -> dict:
        return {
            "method": self.name,
            "n_components": self.n_components,
            "noise_component_indices": str(self._noise_indices),
            "ci_loading_threshold": self.ci_loading_threshold,
            "filter_params": f"l_freq={HPF_FREQ}",
        }


# ── SSP ───────────────────────────────────────────────────────────────────────

class SSPDenoiseMethod(DenoiseMethod):
    """
    Signal-Space Projection via mne.proj.compute_proj_raw.
    Removes the top-variance spatial patterns from the signal.
    """

    name = "ssp"

    def __init__(self, variance_threshold: float = 0.99999, duration: float = 1.0):
        self.variance_threshold = variance_threshold
        self.duration = duration

    def fit_transform(self, raw: mne.io.Raw) -> Tuple[mne.io.Raw, mne.io.Raw]:
        n_eeg = self._pca_n_components(raw.get_data().T, self.variance_threshold)
        projectors = compute_proj_raw(
            raw, n_grad=0, n_mag=0, n_eeg=n_eeg,
            duration=self.duration, verbose=False,
        )
        clean_raw = raw.copy()
        clean_raw.add_proj(projectors, remove_existing=True)
        clean_raw.apply_proj(verbose=False)

        noise_raw = self._make_noise_raw(raw, clean_raw.get_data())
        return clean_raw, noise_raw

    def get_metadata(self) -> dict:
        return {
            "method": self.name,
            "variance_threshold": self.variance_threshold,
            "duration_s": self.duration,
            "filter_params": f"l_freq={HPF_FREQ}",
        }


# ── Wavelet ───────────────────────────────────────────────────────────────────

class WaveletDenoiseMethod(DenoiseMethod):
    """
    Per-channel wavelet thresholding using pywt (PyWavelets).
    MAD-based universal threshold applied to all detail coefficient levels.

    Requires: pip install PyWavelets
    """

    name = "wavelet"

    def __init__(
        self,
        wavelet: str = "db4",
        level: Optional[int] = None,
        threshold_mode: str = "soft",
        sigma_multiplier: float = 3.0,
    ):
        self.wavelet = wavelet
        self.level = level
        self.threshold_mode = threshold_mode
        self.sigma_multiplier = sigma_multiplier

    def fit_transform(self, raw: mne.io.Raw) -> Tuple[mne.io.Raw, mne.io.Raw]:
        try:
            import pywt
        except ImportError:
            raise ImportError("pywt required: pip install PyWavelets")

        data = raw.get_data()
        n_channels, n_times = data.shape
        clean_data = np.zeros_like(data)

        for ch_idx in range(n_channels):
            signal = data[ch_idx]
            level = self.level or pywt.dwt_max_level(n_times, self.wavelet)
            coeffs = pywt.wavedec(signal, self.wavelet, level=level)

            # Donoho-Johnstone MAD noise floor estimate from finest detail
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            threshold = self.sigma_multiplier * sigma

            coeffs_thresh = [coeffs[0]]  # keep approximation coefficients intact
            for detail in coeffs[1:]:
                coeffs_thresh.append(
                    pywt.threshold(detail, threshold, mode=self.threshold_mode)
                )
            clean_data[ch_idx] = pywt.waverec(coeffs_thresh, self.wavelet)[:n_times]

        clean_raw = self._copy_with_data(raw, clean_data)
        noise_raw = self._make_noise_raw(raw, clean_data)
        return clean_raw, noise_raw

    def get_metadata(self) -> dict:
        return {
            "method": self.name,
            "n_components": None,
            "wavelet": self.wavelet,
            "level": self.level,
            "threshold_mode": self.threshold_mode,
            "sigma_multiplier": self.sigma_multiplier,
            "filter_params": f"l_freq={HPF_FREQ}",
        }


# ── Method registry ───────────────────────────────────────────────────────────

METHOD_REGISTRY = {
    "ica":     ICADenoiseMethod,
    "iva":     IVADenoiseMethod,
    "cca":     CCADenoiseMethod,
    "emd_ica": EMDICADenoiseMethod,
    "emd_cca": EMDCCADenoiseMethod,
    "pca":     PCADenoiseMethod,
    "ssp":     SSPDenoiseMethod,
    "wavelet": WaveletDenoiseMethod,
}


# ── File discovery ────────────────────────────────────────────────────────────

def find_fif_files(data_dir: str, year: Optional[int] = None) -> List[str]:
    """Glob all *-raw.fif files under data_dir/CMPy{year or *}/."""
    pattern = f"CMPy{year}" if year else "CMPy*"
    return sorted(glob.glob(os.path.join(data_dir, pattern, "*-raw.fif")))


def _parse_subject_year(fif_path: str) -> Tuple[str, Optional[int]]:
    """Extract (subject_id, year) from a path like .../CMPy2/trial_onsets_0801y2-raw.fif."""
    parts = Path(fif_path).parts
    cmp_dir = next((p for p in parts if p.startswith("CMPy")), None)
    year = int(re.search(r"CMPy(\d+)", cmp_dir).group(1)) if cmp_dir else None
    m = re.search(r"trial_onsets_(.+?)-raw\.fif", Path(fif_path).name)
    subject_id = m.group(1) if m else Path(fif_path).stem
    return subject_id, year


# ── Metadata helpers ──────────────────────────────────────────────────────────

def _build_metadata_row(
    subject_id: str,
    year: Optional[int],
    method: DenoiseMethod,
    raw: mne.io.Raw,
    input_path: str,
    clean_output_path: str,
    noise_output_path: str,
    status: str,
) -> dict:
    method_meta = method.get_metadata()
    return {
        "participant_id": subject_id,
        "year": year,
        "method": method.name,
        "n_components": method_meta.get("n_components"),
        "sfreq": raw.info["sfreq"],
        "n_channels": len(raw.ch_names),
        "filter_params": method_meta.get("filter_params"),
        "input_path": input_path,
        "clean_output_path": clean_output_path,
        "noise_output_path": noise_output_path,
        "timestamp": datetime.datetime.now().isoformat(),
        "method_params": json.dumps({k: v for k, v in method_meta.items()
                                     if k != "method"}),
        "ci_channels": ",".join(CI_CHANNELS),
        "status": status,
    }


def _save_error_csv(
    subject_id: str,
    year: Optional[int],
    method: DenoiseMethod,
    input_path: str,
    clean_dir: str,
    error: str,
) -> None:
    year_label = f"CMPy{year}" if year else "CMPyUnknown"
    out_dir = os.path.join(clean_dir, method.name, year_label)
    os.makedirs(out_dir, exist_ok=True)
    row = {
        "participant_id": subject_id,
        "year": year,
        "method": method.name,
        "n_components": None,
        "sfreq": None,
        "n_channels": None,
        "filter_params": None,
        "input_path": input_path,
        "clean_output_path": None,
        "noise_output_path": None,
        "timestamp": datetime.datetime.now().isoformat(),
        "method_params": None,
        "ci_channels": ",".join(CI_CHANNELS),
        "status": f"error: {error}",
    }
    csv_path = os.path.join(out_dir, f"trial_onsets_{subject_id}.csv")
    pd.DataFrame([row]).to_csv(csv_path, index=False)


# ── Per-file processing ───────────────────────────────────────────────────────

def process_file(
    fif_path: str,
    methods: List[DenoiseMethod],
    clean_dir: str,
    noise_dir: str,
) -> None:
    """Load one annotated .fif, apply 2Hz HPF, run all methods, save outputs."""
    subject_id, year = _parse_subject_year(fif_path)
    year_label = f"CMPy{year}" if year else "CMPyUnknown"
    fname = f"trial_onsets_{subject_id}-raw.fif"
    meta_fname = f"trial_onsets_{subject_id}.csv"

    print(f"\n[{subject_id}] Loading {fif_path}")
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)

    # Global 2 Hz high-pass filter applied once before all methods
    raw.filter(l_freq=HPF_FREQ, h_freq=None, verbose=False)

    for method in methods:
        try:
            print(f"  [{subject_id}] {method.name}...")
            clean_raw, noise_raw = method.fit_transform(raw)

            method_clean_dir = os.path.join(clean_dir, method.name, year_label)
            method_noise_dir = os.path.join(noise_dir, method.name, year_label)
            os.makedirs(method_clean_dir, exist_ok=True)
            os.makedirs(method_noise_dir, exist_ok=True)

            clean_fif = os.path.join(method_clean_dir, fname)
            noise_fif = os.path.join(method_noise_dir, fname)
            clean_raw.save(clean_fif, overwrite=True, verbose=False)
            noise_raw.save(noise_fif, overwrite=True, verbose=False)

            meta = _build_metadata_row(
                subject_id=subject_id,
                year=year,
                method=method,
                raw=raw,
                input_path=fif_path,
                clean_output_path=clean_fif,
                noise_output_path=noise_fif,
                status="ok",
            )
            pd.DataFrame([meta]).to_csv(
                os.path.join(method_clean_dir, meta_fname), index=False)
            pd.DataFrame([meta]).to_csv(
                os.path.join(method_noise_dir, meta_fname), index=False)

            print(f"  [{subject_id}] {method.name} done")

        except Exception as e:
            print(f"  [{subject_id}] ERROR in {method.name}: {e}")
            try:
                _save_error_csv(subject_id, year, method, fif_path, clean_dir, str(e))
            except Exception as csv_err:
                print(f"  [{subject_id}] WARNING: could not write error CSV: {csv_err}")

        finally:
            gc.collect()

    del raw
    gc.collect()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply classical ML denoising methods to annotated EEG .fif files."
    )
    parser.add_argument(
        "--methods", nargs="+",
        choices=list(METHOD_REGISTRY.keys()),
        default=["ica", "pca", "ssp"],
        help="Denoising methods to apply (default: ica pca ssp).",
    )
    parser.add_argument(
        "--data-dir", default="/quobyte/millerlmgrp/annotated_data",
        help="Root directory containing CMPy*/ annotated .fif files.",
    )
    parser.add_argument(
        "--clean-dir", default="/quobyte/millerlmgrp/ml_cleaned_data",
        help="Root directory for cleaned output .fif files.",
    )
    parser.add_argument(
        "--noise-dir", default="/quobyte/millerlmgrp/ml_isolated_noise",
        help="Root directory for noise output .fif files.",
    )
    parser.add_argument(
        "--year", type=int, default=None,
        help="Restrict to CMPy{year} only (e.g. 2, 3, 4).",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=1,
        help="Number of parallel jobs. Keep at 1 to stay within the 128GB RAM cap.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.n_jobs != 1:
        warnings.warn(
            "--n-jobs > 1 may exceed the 128GB RAM cap. Running sequentially.",
            RuntimeWarning,
        )

    methods = [METHOD_REGISTRY[name]() for name in args.methods]

    fif_files = find_fif_files(args.data_dir, year=args.year)
    if not fif_files:
        print(f"No .fif files found under {args.data_dir}. Check --data-dir and CMPy* subdirectories.")
        return

    print(f"Found {len(fif_files)} file(s) | methods: {[m.name for m in methods]}")

    for fif_path in fif_files:
        process_file(
            fif_path=fif_path,
            methods=methods,
            clean_dir=args.clean_dir,
            noise_dir=args.noise_dir,
        )

    print("\nDone.")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
