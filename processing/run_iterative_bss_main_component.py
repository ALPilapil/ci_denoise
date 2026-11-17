"""
run_iterative_bss_main_component.py

Executes iterative / two-pass Blind Source Separation (BSS):
1. FastICA Pass 1: Fit ICA -> isolate the single MAIN artifact component -> project it out -> reconstruct EEG.
2. FastICA Pass 2: Re-fit ICA on Pass 1 reconstructed EEG -> analyze secondary components -> isolate main secondary artifact -> reconstruct EEG.
3. SOBI Pass 1: Fit SOBI -> isolate the single MAIN artifact component -> project it out -> reconstruct EEG.
4. SOBI Pass 2: Re-fit SOBI on Pass 1 reconstructed EEG -> analyze secondary components -> isolate main secondary artifact -> reconstruct EEG.

Evaluates and compares:
- Scalp topographies & explained variance of main components in Pass 1 vs Pass 2
- Cortical ERP morphometry at Cz (N1 latency/amp, P2 latency/amp, peak-to-peak)
- Ground-truth fidelity against Normal Hearing Control (Pearson r, RMSE in uV, ERP SNR)
- Spectral artifact attenuation (20-50 Hz RF hash) vs neural band preservation (1-15 Hz)
"""

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne.preprocessing import ICA
from scipy import signal


CI_REFERENCE_CHANNELS = ["P7", "T7", "M1", "M2", "P8"]
CENTRAL_AUDITORY_CHANNELS = ["Cz", "Fz", "FC1", "FC2", "C3", "C4"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. SOBI Implementation
# ─────────────────────────────────────────────────────────────────────────────

def run_sobi(
    data: np.ndarray,
    n_components: Optional[int] = None,
    lags: Optional[List[int]] = None,
    eps: float = 1e-6,
    max_iter: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Second-Order Blind Identification (SOBI) via joint approximate diagonalization
    of multiple time-lagged cross-covariance matrices.
    """
    n_channels, n_times = data.shape
    if n_components is None or n_components > n_channels:
        n_components = n_channels
    if lags is None:
        lags = [1, 2, 4, 8, 16, 25, 50]

    # Pre-whitening via SVD / EVD with condition-number protection
    X = data - np.mean(data, axis=1, keepdims=True)
    try:
        cov0 = (X @ X.T) / (n_times - 1)
        cov0 = 0.5 * (cov0 + cov0.T)
        d, u = np.linalg.eigh(cov0)
    except np.linalg.LinAlgError:
        # Fallback to direct SVD on X avoiding condition squaring kappa(cov0) = kappa(X)^2
        u, s, _ = np.linalg.svd(X, full_matrices=False)
        d = (s ** 2) / (n_times - 1)
    idx = np.argsort(d)[::-1][:n_components]
    d = np.maximum(d[idx], 1e-12)
    u = u[:, idx]

    whitening = np.diag(1.0 / np.sqrt(d)) @ u.T
    dewhitening = u @ np.diag(np.sqrt(d))
    Z = whitening @ X

    # Compute lagged cross-covariance matrices
    K = len(lags)
    M = np.zeros((n_components, n_components * K))
    for k_idx, lag in enumerate(lags):
        if lag >= n_times:
            continue
        c_lag = (Z[:, :n_times - lag] @ Z[:, lag:].T) / (n_times - lag)
        c_sym = 0.5 * (c_lag + c_lag.T)
        M[:, k_idx * n_components : (k_idx + 1) * n_components] = c_sym

    # Jacobi joint approximate diagonalization
    V = np.eye(n_components)
    for _ in range(max_iter):
        delta = 0.0
        for p in range(n_components - 1):
            for q in range(p + 1, n_components):
                h_p = M[p, :].reshape(K, n_components)
                h_q = M[q, :].reshape(K, n_components)
                g11 = np.sum(h_p[:, p] ** 2)
                g22 = np.sum(h_q[:, q] ** 2)
                g12 = np.sum(h_p[:, q] ** 2)
                g0 = np.sum((h_p[:, p] - h_q[:, q]) * h_p[:, q])

                theta = 0.5 * np.arctan2(2.0 * g0, g11 + g22 - 2.0 * g12 + 1e-12)
                c = np.cos(theta)
                s = np.sin(theta)

                if np.abs(s) > eps:
                    delta = max(delta, np.abs(s))
                    V_p = V[:, p].copy()
                    V_q = V[:, q].copy()
                    V[:, p] = c * V_p + s * V_q
                    V[:, q] = -s * V_p + c * V_q

                    row_p = M[p, :].copy()
                    row_q = M[q, :].copy()
                    M[p, :] = c * row_p + s * row_q
                    M[q, :] = -s * row_p + c * row_q
                    for k in range(K):
                        col_p = M[:, k * n_components + p].copy()
                        col_q = M[:, k * n_components + q].copy()
                        M[:, k * n_components + p] = c * col_p + s * col_q
                        M[:, k * n_components + q] = -s * col_p + c * col_q

        if delta < eps:
            break

    W = V.T @ whitening
    A = dewhitening @ V
    return A, W


# ─────────────────────────────────────────────────────────────────────────────
# 2. Setup & Component Utilities
# ─────────────────────────────────────────────────────────────────────────────

def setup_montage(raw: mne.io.Raw) -> mne.io.Raw:
    """Applies standard 10-05 montage with alias renaming for 2D scalp plotting."""
    try:
        rename_map = {"LMas": "M1", "RMas": "M2"}
        actual_renames = {k: v for k, v in rename_map.items() if k in raw.ch_names and v not in raw.ch_names}
        if actual_renames:
            raw.rename_channels(actual_renames)
        montage = mne.channels.make_standard_montage("standard_1005")
        raw.set_montage(montage, on_missing="ignore", match_case=False, verbose=False)
    except Exception as e:
        print(f"Montage setup notice: {e}", flush=True)
    return raw


def identify_main_component(
    A: np.ndarray,
    S: np.ndarray,
    ch_names: List[str],
    ref_channels: List[str] = CI_REFERENCE_CHANNELS,
) -> Tuple[int, Dict]:
    """
    Identifies the single MAIN CI artifact component based on:
    1. Total backprojected variance: ||a_k * s_k||^2 = ||a_k||^2 * ||s_k||^2
    2. Spatial localization to ipsilateral CI coil/mastoid channels
    3. Mastoid-to-central power ratio

    Returns:
        main_idx: Index of main artifact component
        comp_info: Dictionary of diagnostic properties for all components
    """
    n_chs, n_comps = A.shape
    ref_idxs = [ch_names.index(c) for c in ref_channels if c in ch_names]
    central_idxs = [ch_names.index(c) for c in CENTRAL_AUDITORY_CHANNELS if c in ch_names]

    # Exact closed-form Frobenius norm of rank-1 outer product: ||a_k s_k^T||_F^2 = ||a_k||^2 * ||s_k||^2
    a_norms_sq = np.sum(A ** 2, axis=0) # (n_comps,)
    s_norms_sq = np.sum(S ** 2, axis=1) # (n_comps,)
    comp_vars = a_norms_sq * s_norms_sq
    var_total = float(np.sum(comp_vars))
    pvaf = (comp_vars / (var_total + 1e-12)) * 100.0

    ratios = []
    top_chs = []

    for k in range(n_comps):
        abs_w = np.abs(A[:, k])
        top_ch_idx = int(np.argmax(abs_w))
        top_chs.append(ch_names[top_ch_idx])

        ref_power = np.mean(abs_w[ref_idxs]) if ref_idxs else 0.0
        central_power = np.mean(abs_w[central_idxs]) if central_idxs else 0.0
        ratios.append(float(ref_power / (central_power + 1e-9)))

    # Score components: High variance + High ref_ratio + Top channel in ref_channels
    scores = []
    for k in range(n_comps):
        score = pvaf[k] * (1.0 + min(ratios[k], 10.0))
        if top_chs[k] in ref_channels:
            score *= 2.0
        scores.append(score)

    main_idx = int(np.argmax(scores))

    info = {
        "main_idx": main_idx,
        "pvaf": pvaf.tolist(),
        "top_channels": top_chs,
        "ref_ratios": ratios,
        "main_pvaf": float(pvaf[main_idx]),
        "main_top_channel": top_chs[main_idx],
        "main_ref_ratio": float(ratios[main_idx]),
    }
    return main_idx, info


def get_averaged_erp(
    raw: mne.io.Raw,
    conditions: List[str] = ["A", "AV"],
    tmin: float = -0.2,
    tmax: float = 0.8,
) -> mne.Evoked:
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    target_ids = [event_id[c] for c in conditions if c in event_id]
    if not target_ids:
        target_ids = list(event_id.values())

    epochs = mne.Epochs(
        raw,
        events,
        event_id=target_ids,
        tmin=tmin,
        tmax=tmax,
        baseline=(tmin, 0.0),
        preload=True,
        reject=None,
        verbose=False,
    )
    return epochs.average(picks="all")


def compute_metrics(
    evoked: mne.Evoked,
    evoked_ref: mne.Evoked,
    channel: str = "Cz",
) -> Dict[str, float]:
    times = evoked.times
    ch_idx = evoked.ch_names.index(channel)
    ref_ch_idx = evoked_ref.ch_names.index(channel)

    data_ch = evoked.data[ch_idx] * 1e6  # uV
    ref_ch = evoked_ref.data[ref_ch_idx] * 1e6  # uV

    # Baseline noise floor (-200ms to 0ms)
    baseline_mask = (times >= -0.2) & (times <= 0.0)
    baseline_rms = float(np.sqrt(np.mean(data_ch[baseline_mask] ** 2)))

    # N1 peak (70-140 ms)
    n1_mask = (times >= 0.070) & (times <= 0.140)
    n1_data = data_ch[n1_mask]
    n1_times = times[n1_mask]
    n1_idx = int(np.argmin(n1_data))
    n1_amp = float(n1_data[n1_idx])
    n1_lat = float(n1_times[n1_idx] * 1000.0)

    # P2 peak (140-230 ms)
    p2_mask = (times >= 0.140) & (times <= 0.230)
    p2_data = data_ch[p2_mask]
    p2_times = times[p2_mask]
    p2_idx = int(np.argmax(p2_data))
    p2_amp = float(p2_data[p2_idx])
    p2_lat = float(p2_times[p2_idx] * 1000.0)

    p2p_amp = float(p2_amp - n1_amp)

    # Evoked time window (0 to 500 ms) for ground-truth comparison
    eval_mask = (times >= 0.0) & (times <= 0.500)
    y_test = data_ch[eval_mask]
    y_true = ref_ch[eval_mask]

    corr = float(np.corrcoef(y_test, y_true)[0, 1]) if (np.std(y_test) > 1e-6 and np.std(y_true) > 1e-6) else 0.0
    rmse = float(np.sqrt(np.mean((y_test - y_true) ** 2)))

    # Post-stimulus ERP SNR
    evoked_power = np.mean(y_test ** 2)
    noise_power = np.mean(data_ch[baseline_mask] ** 2) + 1e-12
    erp_snr_db = float(10.0 * np.log10(evoked_power / noise_power))

    # Signal Preservation Index: Phase correlation weighted by inverse RMSE
    spi = float(max(0.0, corr) / (rmse + 1e-6))

    return {
        "baseline_rms_uv": baseline_rms,
        "n1_amp_uv": n1_amp,
        "n1_lat_ms": n1_lat,
        "p2_amp_uv": p2_amp,
        "p2_lat_ms": p2_lat,
        "p2p_amp_uv": p2p_amp,
        "corr_with_hearing": corr,
        "rmse_with_hearing": rmse,
        "erp_snr_db": erp_snr_db,
        "signal_preservation_index": spi,
    }


def compute_band_power(sig: np.ndarray, fs: float, fmin: float, fmax: float) -> float:
    """Computes Welch power in band [fmin, fmax]."""
    freqs, psd = signal.welch(sig, fs=fs, nperseg=min(len(sig), int(fs * 2)))
    mask = (freqs >= fmin) & (freqs <= fmax)
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(psd[mask], freqs[mask]))
    elif hasattr(np, "trapz"):
        return float(np.trapz(psd[mask], freqs[mask]))
    else:
        from scipy import integrate
        return float(integrate.trapezoid(psd[mask], freqs[mask]))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Main Experiment Execution
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Iterative BSS: Run ICA/SOBI, remove single main component, re-run ICA/SOBI")
    parser.add_argument("--data-dir", default="/quobyte/millerlmgrp/annotated_data", help="Directory of annotated FIF files")
    parser.add_argument("--ci-subject", default="0901y2", help="CI subject ID")
    parser.add_argument("--hearing-subject", default="0801y2", help="Hearing subject ID")
    parser.add_argument("--permutation", type=int, default=1, help="Permutation index")
    parser.add_argument("--n-components", type=int, default=15, help="Number of BSS components per pass")
    parser.add_argument("--decim", type=int, default=16, help="Decimation factor for FastICA fitting (applies to full signal)")
    parser.add_argument("--output-dir", default="./benchmark_results/single_pair_demo", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80, flush=True)
    print("  ITERATIVE BSS EXPERIMENT: ISOLATE & REMOVE MAIN COMPONENT -> RE-RUN", flush=True)
    print("=" * 80, flush=True)
    print(f"CI Subject:      {args.ci_subject} (Left CI, perm {args.permutation})", flush=True)
    print(f"Hearing Control: {args.hearing_subject} (Normal Hearing ground truth)", flush=True)
    print(f"Components:      {args.n_components} | Decimation: {args.decim}\n", flush=True)

    ci_files = sorted(glob.glob(os.path.join(args.data_dir, "**", f"*{args.ci_subject}*-raw.fif"), recursive=True))
    hearing_files = sorted(glob.glob(os.path.join(args.data_dir, "**", f"*{args.hearing_subject}*-raw.fif"), recursive=True))

    if not ci_files:
        raise FileNotFoundError(f"Could not find FIF file for CI subject {args.ci_subject}")
    if not hearing_files:
        raise FileNotFoundError(f"Could not find FIF file for Hearing subject {args.hearing_subject}")

    ci_path = ci_files[0]
    hearing_path = hearing_files[0]

    print(f"1. Loading CI Recording:      {os.path.basename(ci_path)}", flush=True)
    print(f"2. Loading Hearing Recording: {os.path.basename(hearing_path)}", flush=True)

    raw_ci = mne.io.read_raw_fif(ci_path, preload=True, verbose=False)
    raw_hearing = mne.io.read_raw_fif(hearing_path, preload=True, verbose=False)

    common_chs = [ch for ch in raw_hearing.ch_names if ch in raw_ci.ch_names]
    raw_ci.pick(common_chs)
    raw_hearing.pick(common_chs)

    raw_ci = setup_montage(raw_ci)
    raw_hearing = setup_montage(raw_hearing)

    fs = float(raw_ci.info["sfreq"])
    n_chs = len(common_chs)
    n_comps = min(n_chs, args.n_components)
    raw_data_ci = raw_ci.get_data()

    print(f"Sampling frequency: {fs} Hz | Channels: {n_chs} | Components to fit: {n_comps}", flush=True)

    # =========================================================================
    # PART 1: FASTICA ITERATIVE PIPELINE
    # =========================================================================
    print("\n" + "=" * 80, flush=True)
    print("  PART 1: FASTICA ITERATIVE (PASS 1 -> REMOVE MAIN -> PASS 2)", flush=True)
    print("=" * 80, flush=True)

    # 1.1 FastICA Pass 1
    print(f"\n[FastICA Pass 1/2] Fitting FastICA on raw CI EEG (decim={args.decim})...", flush=True)
    ica_p1 = ICA(n_components=n_comps, method="fastica", random_state=42, max_iter="auto")
    ica_p1.fit(raw_ci, decim=args.decim, verbose=False)

    A_ica_p1 = ica_p1.get_components() # (n_chs, n_comps)
    # Extract unmixing sources
    W_ica_p1 = ica_p1.unmixing_matrix_ @ ica_p1.pca_components_[:n_comps]
    S_ica_p1 = W_ica_p1 @ (raw_data_ci - ica_p1.pca_mean_[:, None])

    main_ica_p1_idx, info_ica_p1 = identify_main_component(A_ica_p1, S_ica_p1, common_chs)
    print(f"  -> Pass 1 Main Component: IC {main_ica_p1_idx}", flush=True)
    print(f"     Top Channel:           {info_ica_p1['main_top_channel']}", flush=True)
    print(f"     Explained Variance:    {info_ica_p1['main_pvaf']:.2f}%", flush=True)
    print(f"     Mastoid/Central Ratio: {info_ica_p1['main_ref_ratio']:.2f}x", flush=True)

    # Reconstruct data removing ONLY the single main component
    raw_ica_p1_clean = raw_ci.copy()
    ica_p1.apply(raw_ica_p1_clean, exclude=[main_ica_p1_idx], verbose=False)
    data_ica_p1_clean = raw_ica_p1_clean.get_data()

    # 1.2 FastICA Pass 2 (Re-fit on reconstructed data)
    print(f"\n[FastICA Pass 2/2] Re-fitting FastICA on Pass-1 cleaned EEG (decim={args.decim})...", flush=True)
    ica_p2 = ICA(n_components=n_comps, method="fastica", random_state=42, max_iter="auto")
    ica_p2.fit(raw_ica_p1_clean, decim=args.decim, verbose=False)

    A_ica_p2 = ica_p2.get_components()
    W_ica_p2 = ica_p2.unmixing_matrix_ @ ica_p2.pca_components_[:n_comps]
    S_ica_p2 = W_ica_p2 @ (data_ica_p1_clean - ica_p2.pca_mean_[:, None])

    main_ica_p2_idx, info_ica_p2 = identify_main_component(A_ica_p2, S_ica_p2, common_chs)
    print(f"  -> Pass 2 Main Component: IC {main_ica_p2_idx}", flush=True)
    print(f"     Top Channel:           {info_ica_p2['main_top_channel']}", flush=True)
    print(f"     Explained Variance:    {info_ica_p2['main_pvaf']:.2f}%", flush=True)
    print(f"     Mastoid/Central Ratio: {info_ica_p2['main_ref_ratio']:.2f}x", flush=True)

    # Reconstruct data removing the Pass 2 main component
    raw_ica_p2_clean = raw_ica_p1_clean.copy()
    ica_p2.apply(raw_ica_p2_clean, exclude=[main_ica_p2_idx], verbose=False)

    # =========================================================================
    # PART 2: SOBI ITERATIVE PIPELINE
    # =========================================================================
    print("\n" + "=" * 80, flush=True)
    print("  PART 2: SOBI ITERATIVE (PASS 1 -> REMOVE MAIN -> PASS 2)", flush=True)
    print("=" * 80, flush=True)

    # 2.1 SOBI Pass 1
    print("\n[SOBI Pass 1/2] Fitting SOBI (7 lags) on raw CI EEG...", flush=True)
    A_sobi_p1, W_sobi_p1 = run_sobi(raw_data_ci, n_components=n_comps)
    S_sobi_p1 = W_sobi_p1 @ raw_data_ci

    main_sobi_p1_idx, info_sobi_p1 = identify_main_component(A_sobi_p1, S_sobi_p1, common_chs)
    print(f"  -> Pass 1 Main Component: SOBI {main_sobi_p1_idx}", flush=True)
    print(f"     Top Channel:           {info_sobi_p1['main_top_channel']}", flush=True)
    print(f"     Explained Variance:    {info_sobi_p1['main_pvaf']:.2f}%", flush=True)
    print(f"     Mastoid/Central Ratio: {info_sobi_p1['main_ref_ratio']:.2f}x", flush=True)

    # Project out ONLY the single main component
    S_sobi_p1_clean = S_sobi_p1.copy()
    S_sobi_p1_clean[main_sobi_p1_idx, :] = 0.0
    data_sobi_p1_clean = A_sobi_p1 @ S_sobi_p1_clean

    raw_sobi_p1_clean = mne.io.RawArray(data_sobi_p1_clean, raw_ci.info.copy(), verbose=False)
    raw_sobi_p1_clean.set_annotations(raw_ci.annotations)

    # 2.2 SOBI Pass 2 (Re-fit on reconstructed data)
    print("\n[SOBI Pass 2/2] Re-fitting SOBI (7 lags) on Pass-1 cleaned EEG...", flush=True)
    A_sobi_p2, W_sobi_p2 = run_sobi(data_sobi_p1_clean, n_components=n_comps)
    S_sobi_p2 = W_sobi_p2 @ data_sobi_p1_clean

    main_sobi_p2_idx, info_sobi_p2 = identify_main_component(A_sobi_p2, S_sobi_p2, common_chs)
    print(f"  -> Pass 2 Main Component: SOBI {main_sobi_p2_idx}", flush=True)
    print(f"     Top Channel:           {info_sobi_p2['main_top_channel']}", flush=True)
    print(f"     Explained Variance:    {info_sobi_p2['main_pvaf']:.2f}%", flush=True)
    print(f"     Mastoid/Central Ratio: {info_sobi_p2['main_ref_ratio']:.2f}x", flush=True)

    # Project out Pass 2 main component
    S_sobi_p2_clean = S_sobi_p2.copy()
    S_sobi_p2_clean[main_sobi_p2_idx, :] = 0.0
    data_sobi_p2_clean = A_sobi_p2 @ S_sobi_p2_clean

    raw_sobi_p2_clean = mne.io.RawArray(data_sobi_p2_clean, raw_ci.info.copy(), verbose=False)
    raw_sobi_p2_clean.set_annotations(raw_ci.annotations)

    # =========================================================================
    # PART 3: QUANTITATIVE BENCHMARKING & ERP METRICS (Cz)
    # =========================================================================
    print("\n" + "=" * 80, flush=True)
    print("  PART 3: QUANTITATIVE ERP BENCHMARKING AT VERTEX (Cz)", flush=True)
    print("=" * 80, flush=True)

    print("Extracting trial-averaged ERPs across all pipelines...", flush=True)
    evoked_hearing = get_averaged_erp(raw_hearing)
    evoked_raw_ci = get_averaged_erp(raw_ci)
    evoked_ica_p1 = get_averaged_erp(raw_ica_p1_clean)
    evoked_ica_p2 = get_averaged_erp(raw_ica_p2_clean)
    evoked_sobi_p1 = get_averaged_erp(raw_sobi_p1_clean)
    evoked_sobi_p2 = get_averaged_erp(raw_sobi_p2_clean)

    cz_idx = common_chs.index("Cz")

    pipelines = [
        ("Clean Hearing Control", evoked_hearing, raw_hearing.get_data()[cz_idx]),
        ("Raw CI EEG (Contaminated)", evoked_raw_ci, raw_data_ci[cz_idx]),
        ("FastICA Pass 1 (Main Removed)", evoked_ica_p1, data_ica_p1_clean[cz_idx]),
        ("FastICA Pass 2 (Cascade Removed)", evoked_ica_p2, raw_ica_p2_clean.get_data()[cz_idx]),
        ("SOBI Pass 1 (Main Removed)", evoked_sobi_p1, data_sobi_p1_clean[cz_idx]),
        ("SOBI Pass 2 (Cascade Removed)", evoked_sobi_p2, data_sobi_p2_clean[cz_idx]),
    ]

    metric_rows = []
    for name, evk, sig_cz in pipelines:
        print(f"  Computing metrics for: {name}...", flush=True)
        m = compute_metrics(evk, evoked_hearing, "Cz")
        neural_power = compute_band_power(sig_cz, fs, 1.0, 15.0)
        rf_hash_power = compute_band_power(sig_cz, fs, 20.0, 50.0)
        metric_rows.append({
            "Pipeline": name,
            **m,
            "neural_power_1_15hz": neural_power,
            "rf_hash_power_20_50hz": rf_hash_power,
        })

    df_metrics = pd.DataFrame(metric_rows)

    display_cols = [
        "Pipeline", "corr_with_hearing", "rmse_with_hearing",
        "p2_lat_ms", "p2_amp_uv", "n1_lat_ms", "n1_amp_uv",
        "baseline_rms_uv", "erp_snr_db", "rf_hash_power_20_50hz"
    ]
    print("\n" + df_metrics[display_cols].to_string(index=False), flush=True)

    # Save metrics to disk
    csv_path = os.path.join(args.output_dir, f"iterative_bss_metrics_{args.ci_subject}.csv")
    df_metrics.to_csv(csv_path, index=False)
    json_path = os.path.join(args.output_dir, f"iterative_bss_metrics_{args.ci_subject}.json")
    with open(json_path, "w") as f:
        json.dump(df_metrics.to_dict(orient="records"), f, indent=2)

    # Save detailed component metadata
    comp_meta = {
        "FastICA": {
            "Pass_1": {
                "main_idx": main_ica_p1_idx,
                "top_channel": info_ica_p1["main_top_channel"],
                "pvaf": info_ica_p1["main_pvaf"],
                "ref_ratio": info_ica_p1["main_ref_ratio"],
            },
            "Pass_2": {
                "main_idx": main_ica_p2_idx,
                "top_channel": info_ica_p2["main_top_channel"],
                "pvaf": info_ica_p2["main_pvaf"],
                "ref_ratio": info_ica_p2["main_ref_ratio"],
            },
        },
        "SOBI": {
            "Pass_1": {
                "main_idx": main_sobi_p1_idx,
                "top_channel": info_sobi_p1["main_top_channel"],
                "pvaf": info_sobi_p1["main_pvaf"],
                "ref_ratio": info_sobi_p1["main_ref_ratio"],
            },
            "Pass_2": {
                "main_idx": main_sobi_p2_idx,
                "top_channel": info_sobi_p2["main_top_channel"],
                "pvaf": info_sobi_p2["main_pvaf"],
                "ref_ratio": info_sobi_p2["main_ref_ratio"],
            },
        },
    }
    with open(os.path.join(args.output_dir, f"iterative_bss_components_{args.ci_subject}.json"), "w") as f:
        json.dump(comp_meta, f, indent=2)

    # =========================================================================
    # PART 4: VISUALIZATIONS
    # =========================================================================
    print("\n" + "=" * 80, flush=True)
    print("  PART 4: GENERATING COMPREHENSIVE FIGURE PANELS", flush=True)
    print("=" * 80, flush=True)

    # 4.1 FIGURE 1: Component Topographies & Timecourses
    print("Rendering Figure 1: Component Topographies & Timecourses...", flush=True)
    fig1 = plt.figure(figsize=(16, 10), facecolor="#ffffff")
    fig1.suptitle("Iterative BSS: Main Component Isolation Across Pass 1 & Pass 2", fontsize=14, fontweight="bold", y=0.98)
    gs1 = fig1.add_gridspec(2, 4, width_ratios=[1, 1.6, 1, 1.6], hspace=0.35, wspace=0.25)

    times_s = np.arange(1000) / fs
    sample_slice = slice(int(fs * 10), int(fs * 10 + 1000))

    components_to_plot = [
        # (Row, Col_topo, Col_wave, Method_Title, Mixing_vec, Source_sig, info_dict, color)
        (0, 0, 1, f"FastICA Pass 1 (IC {main_ica_p1_idx})\nPVAF: {info_ica_p1['main_pvaf']:.1f}% | Top: {info_ica_p1['main_top_channel']}",
         A_ica_p1[:, main_ica_p1_idx], S_ica_p1[main_ica_p1_idx, sample_slice], "#d62728"),
        (0, 2, 3, f"FastICA Pass 2 (IC {main_ica_p2_idx})\nPVAF: {info_ica_p2['main_pvaf']:.1f}% | Top: {info_ica_p2['main_top_channel']}",
         A_ica_p2[:, main_ica_p2_idx], S_ica_p2[main_ica_p2_idx, sample_slice], "#ff7f0e"),
        (1, 0, 1, f"SOBI Pass 1 (SOBI {main_sobi_p1_idx})\nPVAF: {info_sobi_p1['main_pvaf']:.1f}% | Top: {info_sobi_p1['main_top_channel']}",
         A_sobi_p1[:, main_sobi_p1_idx], S_sobi_p1[main_sobi_p1_idx, sample_slice], "#1f77b4"),
        (1, 2, 3, f"SOBI Pass 2 (SOBI {main_sobi_p2_idx})\nPVAF: {info_sobi_p2['main_pvaf']:.1f}% | Top: {info_sobi_p2['main_top_channel']}",
         A_sobi_p2[:, main_sobi_p2_idx], S_sobi_p2[main_sobi_p2_idx, sample_slice], "#2ca02c"),
    ]

    for row, c_topo, c_wave, title, w, s_trace, col in components_to_plot:
        ax_t = fig1.add_subplot(gs1[row, c_topo])
        im, _ = mne.viz.plot_topomap(w, raw_ci.info, axes=ax_t, show=False, cmap="RdBu_r", sphere=0.095)
        ax_t.set_title(title, fontsize=10, fontweight="bold", pad=4)
        plt.colorbar(im, ax=ax_t, fraction=0.046, pad=0.04)

        ax_w = fig1.add_subplot(gs1[row, c_wave])
        ax_w.plot(times_s * 1000.0, s_trace, color=col, lw=1.2)
        ax_w.set_title(f"Time Course (1.0s sample)", fontsize=9.5)
        ax_w.set_xlabel("Time (ms)", fontsize=8.5)
        ax_w.set_ylabel("Source Amplitude", fontsize=8.5)
        ax_w.grid(True, linestyle=":", alpha=0.5)

    fig1_path = os.path.join(args.output_dir, f"iterative_bss_components_topomap_{args.ci_subject}.png")
    fig1.savefig(fig1_path, dpi=300)
    plt.close(fig1)
    print(f"Saved Component Figure -> {fig1_path}", flush=True)

    # 4.2 FIGURE 2: Cortical ERP Waveform Comparison at Cz
    print("Rendering Figure 2: Cortical ERP Waveforms at Cz...", flush=True)
    fig2, axes2 = plt.subplots(3, 1, figsize=(14, 11), sharex=True, facecolor="#ffffff")
    fig2.suptitle(f"Cortical Auditory Evoked Potential (Cz): Iterative BSS Performance (Subject {args.ci_subject})", fontsize=13, fontweight="bold", y=0.98)

    times_erp = evoked_hearing.times * 1000.0  # ms
    cz_hearing = evoked_hearing.data[cz_idx] * 1e6
    cz_raw = evoked_raw_ci.data[cz_idx] * 1e6

    # Subplot A: FastICA Pass 1 vs Pass 2
    ax2_0 = axes2[0]
    ax2_0.plot(times_erp, cz_raw, label="Raw CI EEG (Contaminated)", color="#888888", lw=1.0, alpha=0.6)
    ax2_0.plot(times_erp, cz_hearing, label="Clean Hearing Ground Truth", color="#000000", lw=2.2, ls="--")
    ax2_0.plot(times_erp, evoked_ica_p1.data[cz_idx] * 1e6, label=f"FastICA Pass 1 (r={df_metrics.loc[2, 'corr_with_hearing']:.2f}, RMSE={df_metrics.loc[2, 'rmse_with_hearing']:.2f} µV)", color="#ff7f0e", lw=1.8)
    ax2_0.plot(times_erp, evoked_ica_p2.data[cz_idx] * 1e6, label=f"FastICA Pass 2 (Cascade) (r={df_metrics.loc[3, 'corr_with_hearing']:.2f}, RMSE={df_metrics.loc[3, 'rmse_with_hearing']:.2f} µV)", color="#d62728", lw=2.2)
    ax2_0.axvspan(-200, 0, color="gray", alpha=0.08)
    ax2_0.axvline(100, color="darkred", ls=":", alpha=0.5, label="N1 Target (~100 ms)")
    ax2_0.axvline(200, color="navy", ls=":", alpha=0.5, label="P2 Target (~200 ms)")
    ax2_0.set_title("A. FastICA: Single Main Component Removal vs. Re-run Second Pass Removal", fontsize=11, fontweight="bold")
    ax2_0.set_ylabel("Amplitude (µV)", fontweight="bold")
    ax2_0.legend(loc="upper right", fontsize=8.5, frameon=True)
    ax2_0.grid(True, linestyle=":", alpha=0.4)

    # Subplot B: SOBI Pass 1 vs Pass 2
    ax2_1 = axes2[1]
    ax2_1.plot(times_erp, cz_raw, label="Raw CI EEG (Contaminated)", color="#888888", lw=1.0, alpha=0.6)
    ax2_1.plot(times_erp, cz_hearing, label="Clean Hearing Ground Truth", color="#000000", lw=2.2, ls="--")
    ax2_1.plot(times_erp, evoked_sobi_p1.data[cz_idx] * 1e6, label=f"SOBI Pass 1 (r={df_metrics.loc[4, 'corr_with_hearing']:.2f}, RMSE={df_metrics.loc[4, 'rmse_with_hearing']:.2f} µV)", color="#1f77b4", lw=1.8)
    ax2_1.plot(times_erp, evoked_sobi_p2.data[cz_idx] * 1e6, label=f"SOBI Pass 2 (Cascade) (r={df_metrics.loc[5, 'corr_with_hearing']:.2f}, RMSE={df_metrics.loc[5, 'rmse_with_hearing']:.2f} µV)", color="#2ca02c", lw=2.2)
    ax2_1.axvspan(-200, 0, color="gray", alpha=0.08)
    ax2_1.axvline(100, color="darkred", ls=":", alpha=0.5)
    ax2_1.axvline(200, color="navy", ls=":", alpha=0.5)
    ax2_1.set_title("B. SOBI: Single Main Component Removal vs. Re-run Second Pass Removal", fontsize=11, fontweight="bold")
    ax2_1.set_ylabel("Amplitude (µV)", fontweight="bold")
    ax2_1.legend(loc="upper right", fontsize=8.5, frameon=True)
    ax2_1.grid(True, linestyle=":", alpha=0.4)

    # Subplot C: Head-to-Head: FastICA Pass 2 vs SOBI Pass 2
    ax2_2 = axes2[2]
    ax2_2.plot(times_erp, cz_raw, label="Raw CI EEG", color="#888888", lw=1.0, alpha=0.5)
    ax2_2.plot(times_erp, cz_hearing, label="Clean Hearing Ground Truth", color="#000000", lw=2.2, ls="--")
    ax2_2.plot(times_erp, evoked_ica_p2.data[cz_idx] * 1e6, label=f"FastICA Pass 2 (r={df_metrics.loc[3, 'corr_with_hearing']:.2f}, RMSE={df_metrics.loc[3, 'rmse_with_hearing']:.2f} µV)", color="#d62728", lw=2.0)
    ax2_2.plot(times_erp, evoked_sobi_p2.data[cz_idx] * 1e6, label=f"SOBI Pass 2 (r={df_metrics.loc[5, 'corr_with_hearing']:.2f}, RMSE={df_metrics.loc[5, 'rmse_with_hearing']:.2f} µV)", color="#2ca02c", lw=2.0)
    ax2_2.axvspan(-200, 0, color="gray", alpha=0.08)
    ax2_2.axvline(100, color="darkred", ls=":", alpha=0.5)
    ax2_2.axvline(200, color="navy", ls=":", alpha=0.5)
    ax2_2.set_title("C. Head-to-Head Comparison: Two-Pass FastICA vs. Two-Pass SOBI", fontsize=11, fontweight="bold")
    ax2_2.set_xlabel("Time post-stimulus onset (ms)", fontweight="bold")
    ax2_2.set_ylabel("Amplitude (µV)", fontweight="bold")
    ax2_2.legend(loc="upper right", fontsize=8.5, frameon=True)
    ax2_2.grid(True, linestyle=":", alpha=0.4)

    fig2.tight_layout()
    fig2_path = os.path.join(args.output_dir, f"iterative_bss_erp_comparison_{args.ci_subject}.png")
    fig2.savefig(fig2_path, dpi=300)
    plt.close(fig2)
    print(f"Saved ERP Comparison Figure -> {fig2_path}", flush=True)

    # 4.3 FIGURE 3: Power Spectral Density (PSD)
    print("Rendering Figure 3: Power Spectral Density (PSD)...", flush=True)
    fig3, (ax3_ica, ax3_sobi) = plt.subplots(1, 2, figsize=(14, 5), facecolor="#ffffff")
    fig3.suptitle("Vertex (Cz) Power Spectral Density: High-Frequency Artifact Suppression", fontsize=13, fontweight="bold")

    cz_hearing_ts = raw_hearing.get_data()[cz_idx]
    f_h, p_h = signal.welch(cz_hearing_ts, fs=fs, nperseg=int(fs * 2))
    f_raw, p_raw = signal.welch(raw_data_ci[cz_idx], fs=fs, nperseg=int(fs * 2))

    # FastICA PSD
    f_i1, p_i1 = signal.welch(data_ica_p1_clean[cz_idx], fs=fs, nperseg=int(fs * 2))
    f_i2, p_i2 = signal.welch(raw_ica_p2_clean.get_data()[cz_idx], fs=fs, nperseg=int(fs * 2))

    ax3_ica.semilogy(f_h, p_h, label="Clean Hearing", color="#000000", lw=2.0, ls="--")
    ax3_ica.semilogy(f_raw, p_raw, label="Raw CI", color="#888888", lw=1.2, alpha=0.6)
    ax3_ica.semilogy(f_i1, p_i1, label="FastICA Pass 1", color="#ff7f0e", lw=1.8)
    ax3_ica.semilogy(f_i2, p_i2, label="FastICA Pass 2", color="#d62728", lw=2.0)
    ax3_ica.set_xlim(1, 60)
    ax3_ica.set_title("FastICA Spectral Evolution", fontweight="bold")
    ax3_ica.set_xlabel("Frequency (Hz)")
    ax3_ica.set_ylabel("Power Spectral Density (µV²/Hz)")
    ax3_ica.grid(True, linestyle=":", alpha=0.5)
    ax3_ica.legend(loc="upper right", fontsize=8.5)

    # SOBI PSD
    f_s1, p_s1 = signal.welch(data_sobi_p1_clean[cz_idx], fs=fs, nperseg=int(fs * 2))
    f_s2, p_s2 = signal.welch(data_sobi_p2_clean[cz_idx], fs=fs, nperseg=int(fs * 2))

    ax3_sobi.semilogy(f_h, p_h, label="Clean Hearing", color="#000000", lw=2.0, ls="--")
    ax3_sobi.semilogy(f_raw, p_raw, label="Raw CI", color="#888888", lw=1.2, alpha=0.6)
    ax3_sobi.semilogy(f_s1, p_s1, label="SOBI Pass 1", color="#1f77b4", lw=1.8)
    ax3_sobi.semilogy(f_s2, p_s2, label="SOBI Pass 2", color="#2ca02c", lw=2.0)
    ax3_sobi.set_xlim(1, 60)
    ax3_sobi.set_title("SOBI Spectral Evolution", fontweight="bold")
    ax3_sobi.set_xlabel("Frequency (Hz)")
    ax3_sobi.set_ylabel("Power Spectral Density (µV²/Hz)")
    ax3_sobi.grid(True, linestyle=":", alpha=0.5)
    ax3_sobi.legend(loc="upper right", fontsize=8.5)

    fig3.tight_layout()
    fig3_path = os.path.join(args.output_dir, f"iterative_bss_psd_{args.ci_subject}.png")
    fig3.savefig(fig3_path, dpi=300)
    plt.close(fig3)
    print(f"Saved PSD Figure -> {fig3_path}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("  EXPERIMENT COMPLETE! All outputs saved to:", args.output_dir, flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
