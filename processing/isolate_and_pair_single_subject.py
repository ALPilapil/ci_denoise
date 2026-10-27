"""
isolate_and_pair_single_subject.py

Direct implementation of the Research Meeting Next-Step Workflow:
1. Picks 1 CI participant and 1 Hearing control participant sharing the SAME permutation
   (perm1, perm2, or perm3) to ensure matching speech sentences.
2. Extracts the pure CI stimulation artifact (channel x time matrix) using:
   - AATS (Stimulus-Locked Adaptive Artifact Template Subtraction)
   - ICA (FastICA mastoid-loading component isolation)
   - SOBI (Second-Order Blind Identification via lagged covariances)
3. Analyzes Visual-only (V) trials as the "silence" / ambient CI baseline vs.
   Audio-only (A) and AudioVisual (AV) active speech stimulation trials.
4. Synthesizes ground-truth mixtures: dirty = clean_hearing + alpha * ci_noise.
5. Generates comparison plots and exports the channel-by-time artifact matrices.
"""

import argparse
import glob
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np


CI_REFERENCE_CHANNELS = ['P7', 'T7', 'M2', 'M1', 'P8']


# ─────────────────────────────────────────────────────────────────────────────
# 1. SOBI (Second-Order Blind Identification) Implementation
# ─────────────────────────────────────────────────────────────────────────────

def run_sobi(
    data: np.ndarray,
    n_components: Optional[int] = None,
    lags: Optional[List[int]] = None,
    eps: float = 1e-6,
    max_iter: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Second-Order Blind Identification (SOBI) via joint diagonalization of
    time-lagged covariance matrices.

    Args:
        data: (n_channels, n_times) zero-mean continuous or epoched EEG.
        n_components: Number of sources to extract (defaults to n_channels).
        lags: Sample lags for autocorrelation matrices (default: [1, 2, 4, 8, 16, 25, 50]).

    Returns:
        mixing_matrix A: (n_channels, n_components)
        unmixing_matrix W: (n_components, n_channels)
    """
    n_channels, n_times = data.shape
    if n_components is None or n_components > n_channels:
        n_components = n_channels
    if lags is None:
        lags = [1, 2, 4, 8, 16, 25, 50]

    # Pre-whiten data with SVD
    X = data - np.mean(data, axis=1, keepdims=True)
    cov0 = (X @ X.T) / (n_times - 1)
    d, u = np.linalg.eigh(cov0)
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

    # Joint approximate diagonalization (Jacobi rotations)
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
# 2. Artifact Isolation Engines (AATS, ICA, SOBI)
# ─────────────────────────────────────────────────────────────────────────────

def isolate_noise_aats(
    raw_ci: mne.io.Raw,
    ref_channels: List[str] = CI_REFERENCE_CHANNELS,
    n_taps: int = 6,
    reg_lambda: float = 1e-3,
    active_conditions: List[str] = ["A", "AV"],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Isolates pure CI artifact using Stimulus-Locked Adaptive Artifact Template Subtraction.
    Returns: (noise_matrix, clean_matrix) of shape (n_channels, n_times).
    """
    data = raw_ci.get_data()
    ch_names = raw_ci.ch_names
    n_channels, n_times = data.shape

    ref_indices = [ch_names.index(ch) for ch in ref_channels if ch in ch_names]
    valid_refs = [idx for idx in ref_indices if np.std(data[idx]) > 1e-6 and not np.isnan(data[idx]).any()]

    if not valid_refs:
        print("  Warning: No valid CI reference electrodes found for AATS. Returning zero noise.")
        return np.zeros_like(data), data.copy()

    n_ref = len(valid_refs)
    R = np.zeros((n_times, n_ref * n_taps))
    col = 0
    for r_idx in valid_refs:
        sig = data[r_idx]
        for tau in range(n_taps):
            if tau == 0:
                R[:, col] = sig
            else:
                R[tau:, col] = sig[:-tau]
            col += 1

    fit_mask = np.ones(n_times, dtype=bool)
    if raw_ci.annotations and len(raw_ci.annotations) > 0:
        stim_mask = np.zeros(n_times, dtype=bool)
        sfreq = raw_ci.info["sfreq"]
        for annot in raw_ci.annotations:
            if any(cond in annot["description"] for cond in active_conditions):
                s0 = max(0, int(annot["onset"] * sfreq))
                s1 = min(n_times, s0 + int(0.6 * sfreq))
                stim_mask[s0:s1] = True
        if np.any(stim_mask) and np.sum(stim_mask) > (n_ref * n_taps * 2):
            fit_mask = stim_mask

    R_fit = R[fit_mask]
    RTR = R_fit.T @ R_fit
    trace_val = np.trace(RTR) / (n_ref * n_taps) + 1e-12
    reg = max(reg_lambda * trace_val, 1e-4 * trace_val)
    A_reg = RTR + reg * np.eye(n_ref * n_taps)
    inv_RTR = np.linalg.pinv(A_reg, rcond=1e-6)

    noise_data = np.zeros_like(data)
    clean_data = data.copy()

    for ch_idx in range(n_channels):
        x_fit = data[ch_idx, fit_mask]
        w = inv_RTR @ (R_fit.T @ x_fit)
        art_est = R @ w
        orig_std = np.std(data[ch_idx]) + 1e-12
        if np.isnan(art_est).any() or (np.std(art_est) / orig_std > 5.0):
            noise_data[ch_idx] = 0.0
            clean_data[ch_idx] = data[ch_idx]
        else:
            noise_data[ch_idx] = art_est
            clean_data[ch_idx] = data[ch_idx] - art_est

    return noise_data, clean_data


def isolate_noise_ica(
    raw_ci: mne.io.Raw,
    ref_channels: List[str] = CI_REFERENCE_CHANNELS,
    n_components: int = 15,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Isolates pure CI artifact using FastICA by identifying components
    maximizing projection weight on CI mastoid/retroauricular sensors.
    """
    from mne.preprocessing import ICA
    raw_copy = raw_ci.copy()
    n_ch = min(len(raw_copy.ch_names), n_components)
    ica = ICA(n_components=n_ch, method="fastica", random_state=42, max_iter="auto")
    ica.fit(raw_copy, verbose=False)

    comp_matrix = ica.get_components()
    ch_names = raw_copy.ch_names

    ci_comp_indices = []
    for i in range(comp_matrix.shape[1]):
        col = np.abs(comp_matrix[:, i])
        top_idx = np.argsort(col)[::-1]
        top_chs = [ch_names[idx] for idx in top_idx[:3]]
        if any(ref in top_chs for ref in ref_channels):
            ci_comp_indices.append(i)

    if not ci_comp_indices:
        ci_comp_indices = [int(np.argmax(np.mean(comp_matrix[[ch_names.index(c) for c in ref_channels if c in ch_names]], axis=0)))]

    print(f"  ICA identified CI artifact components: {ci_comp_indices}")

    clean_raw = raw_copy.copy()
    ica.apply(clean_raw, exclude=ci_comp_indices, verbose=False)

    data = raw_ci.get_data()
    clean_data = clean_raw.get_data()
    noise_data = data - clean_data
    return noise_data, clean_data


def isolate_noise_sobi(
    raw_ci: mne.io.Raw,
    ref_channels: List[str] = CI_REFERENCE_CHANNELS,
    n_components: int = 15,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Isolates pure CI artifact using Second-Order Blind Identification (SOBI).
    """
    data = raw_ci.get_data()
    ch_names = raw_ci.ch_names
    n_ch = min(len(ch_names), n_components)
    A, W = run_sobi(data, n_components=n_ch)

    sources = W @ data
    ref_indices = [ch_names.index(ch) for ch in ref_channels if ch in ch_names]

    scores = np.zeros(n_ch)
    for i in range(n_ch):
        src = sources[i]
        ref_power = np.mean([np.corrcoef(src, data[r_idx])[0, 1] ** 2 for r_idx in ref_indices if np.std(data[r_idx]) > 1e-6])
        scores[i] = ref_power

    artifact_idx = int(np.argmax(scores))
    print(f"  SOBI identified top CI artifact component: #{artifact_idx} (ref correlation score: {scores[artifact_idx]:.3f})")

    artifact_source = np.zeros_like(sources)
    artifact_source[artifact_idx] = sources[artifact_idx]
    noise_data = A @ artifact_source
    clean_data = data - noise_data
    return noise_data, clean_data


def isolate_noise_aats_sobi(
    raw_ci: mne.io.Raw,
    ref_channels: List[str] = CI_REFERENCE_CHANNELS,
    n_components: int = 15,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Two-stage cascading artifact removal pipeline:
    Stage 1: AATS removes the primary deterministic stimulus-locked RF pulse envelope.
    Stage 2: SOBI decomposes the residual EEG to isolate secondary non-stationary artifacts
             (e.g., post-auricular myogenic reflexes (PAMR), residual mastoid leakage).
    """
    print("  [Stage 1/2]: Running AATS to remove primary stimulus-locked pulse train...")
    noise_aats, clean_aats_data = isolate_noise_aats(raw_ci, ref_channels=ref_channels)

    print("  [Stage 2/2]: Running SOBI on AATS-cleaned residuals...")
    raw_aats_cleaned = mne.io.RawArray(clean_aats_data, raw_ci.info.copy(), verbose=False)
    if raw_ci.annotations:
        raw_aats_cleaned.set_annotations(raw_ci.annotations)

    noise_sobi_res, clean_final = isolate_noise_sobi(
        raw_aats_cleaned,
        ref_channels=ref_channels,
        n_components=n_components,
    )

    raw_data = raw_ci.get_data()
    total_noise = raw_data - clean_final
    return total_noise, clean_final


def isolate_noise_aats_ica(
    raw_ci: mne.io.Raw,
    ref_channels: List[str] = CI_REFERENCE_CHANNELS,
    n_components: int = 15,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Two-stage cascading artifact removal pipeline:
    Stage 1: AATS removes the primary deterministic stimulus-locked RF pulse envelope.
    Stage 2: FastICA decomposes the residual EEG to isolate secondary non-stationary artifacts
             (e.g., post-auricular myogenic reflexes (PAMR), ocular blinks).
    """
    print("  [Stage 1/2]: Running AATS to remove primary stimulus-locked pulse train...")
    noise_aats, clean_aats_data = isolate_noise_aats(raw_ci, ref_channels=ref_channels)

    print("  [Stage 2/2]: Running FastICA on AATS-cleaned residuals...")
    raw_aats_cleaned = mne.io.RawArray(clean_aats_data, raw_ci.info.copy(), verbose=False)
    if raw_ci.annotations:
        raw_aats_cleaned.set_annotations(raw_ci.annotations)

    noise_ica_res, clean_final = isolate_noise_ica(
        raw_aats_cleaned,
        ref_channels=ref_channels,
        n_components=n_components,
    )

    raw_data = raw_ci.get_data()
    total_noise = raw_data - clean_final
    return total_noise, clean_final


# ─────────────────────────────────────────────────────────────────────────────
# 3. Permutation Discovery & Pairing Logic
# ─────────────────────────────────────────────────────────────────────────────

def find_matching_pair(
    data_dir: str,
    target_perm: int = 1,
    ci_pref: Optional[str] = None,
    hearing_pref: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    ci_files = sorted(glob.glob(os.path.join(data_dir, "**", "*09*-raw.fif"), recursive=True))
    hearing_files = sorted(glob.glob(os.path.join(data_dir, "**", "*08*-raw.fif"), recursive=True))

    chosen_ci = None
    chosen_hearing = None

    if ci_pref:
        for f in ci_files:
            if ci_pref in os.path.basename(f):
                chosen_ci = f
                break

    if hearing_pref:
        for f in hearing_files:
            if hearing_pref in os.path.basename(f):
                chosen_hearing = f
                break

    if chosen_ci is None and ci_files:
        chosen_ci = ci_files[0]
    if chosen_hearing is None and hearing_files:
        chosen_hearing = hearing_files[0]

    return chosen_ci, chosen_hearing


# ─────────────────────────────────────────────────────────────────────────────
# 4. Multi-Trial Continuous Block Comparison
# ─────────────────────────────────────────────────────────────────────────────

def extract_and_plot_multitrial_block(
    raw_ci: mne.io.Raw,
    raw_hearing: mne.io.Raw,
    extracted_noise_mats: Dict[str, np.ndarray],
    common_chs: List[str],
    target_ch: str,
    output_dir: str,
    ci_subject: str,
    hearing_subject: str,
    permutation: int,
    start_trial: int = 1,
    num_trials: int = 8,
    trial_duration_s: float = 1.5,
):
    """
    Extracts and stitches a sequence of consecutive trials (e.g. Trials 1 to 8)
    containing alternating Audio (A), AudioVisual (AV), and Visual-only (V) conditions.
    Plots continuous waveforms comparing Raw CI, Clean Hearing, Isolated Noise, and Denoised EEG.
    """
    ci_annots = [a for a in raw_ci.annotations if a["description"] in ["A", "AV", "V"]]
    h_annots = [a for a in raw_hearing.annotations if a["description"] in ["A", "AV", "V"]]

    max_trials = min(len(ci_annots), len(h_annots))
    if max_trials == 0:
        return

    s_idx = max(0, start_trial - 1)
    e_idx = min(max_trials, s_idx + num_trials)
    chosen_ci = ci_annots[s_idx:e_idx]
    chosen_h = h_annots[s_idx:e_idx]
    actual_num_trials = len(chosen_ci)

    if actual_num_trials == 0:
        return

    print(f"\nConstructing Multi-Trial Block Sequence (Trials {s_idx + 1} to {e_idx})...")
    conditions_seq = [a["description"] for a in chosen_ci]
    print(f"  Trial Condition Sequence: {' -> '.join(conditions_seq)}")

    sfreq_ci = raw_ci.info["sfreq"]
    sfreq_h = raw_hearing.info["sfreq"]
    sfreq = min(sfreq_ci, sfreq_h)
    pts_per_trial = int(trial_duration_s * sfreq)

    ch_idx = common_chs.index(target_ch) if target_ch in common_chs else 0

    ci_raw_data = raw_ci.get_data()
    h_raw_data = raw_hearing.get_data()

    ci_chunks = []
    h_chunks = []
    noise_chunks = {m: [] for m in extracted_noise_mats}

    for k in range(actual_num_trials):
        ci_t0 = chosen_ci[k]["onset"]
        ci_samp0 = int(ci_t0 * sfreq_ci)
        ci_chunk = ci_raw_data[ch_idx, ci_samp0 : ci_samp0 + pts_per_trial] * 1e6
        if len(ci_chunk) < pts_per_trial:
            ci_chunk = np.pad(ci_chunk, (0, pts_per_trial - len(ci_chunk)))
        ci_chunks.append(ci_chunk[:pts_per_trial])

        h_t0 = chosen_h[k]["onset"]
        h_samp0 = int(h_t0 * sfreq_h)
        h_chunk = h_raw_data[ch_idx, h_samp0 : h_samp0 + pts_per_trial] * 1e6
        if len(h_chunk) < pts_per_trial:
            h_chunk = np.pad(h_chunk, (0, pts_per_trial - len(h_chunk)))
        h_chunks.append(h_chunk[:pts_per_trial])

        for m, mat in extracted_noise_mats.items():
            n_chunk = mat[ch_idx, ci_samp0 : ci_samp0 + pts_per_trial] * 1e6
            if len(n_chunk) < pts_per_trial:
                n_chunk = np.pad(n_chunk, (0, pts_per_trial - len(n_chunk)))
            noise_chunks[m].append(n_chunk[:pts_per_trial])

    full_ci = np.concatenate(ci_chunks)
    full_h = np.concatenate(h_chunks)
    full_noise = {m: np.concatenate(noise_chunks[m]) for m in extracted_noise_mats}
    full_denoised = {m: full_ci - full_noise[m] for m in extracted_noise_mats}

    total_time_s = len(full_ci) / sfreq
    time_axis = np.linspace(0, total_time_s, len(full_ci))

    # Plot Multi-Trial Block Comparison
    fig, axes = plt.subplots(4, 1, figsize=(15, 11), sharex=True)
    colors = {"aats": "#1f77b4", "ica": "#ff7f0e", "sobi": "#2ca02c", "aats_sobi": "#d62728"}
    labels = {"aats": "AATS", "ica": "FastICA", "sobi": "SOBI", "aats_sobi": "AATS+SOBI"}

    cond_colors = {"V": "#2ca02c", "A": "#d62728", "AV": "#9467bd"}

    # Add background shading for each trial condition
    for k in range(actual_num_trials):
        t0 = k * trial_duration_s
        t1 = (k + 1) * trial_duration_s
        cond = conditions_seq[k]
        bg_c = cond_colors.get(cond, "#cccccc")
        for ax in axes:
            ax.axvspan(t0, t1, color=bg_c, alpha=0.10)
            ax.axvline(t0, color="gray", ls="--", alpha=0.5, lw=0.8)

    # Panel 1: Raw CI Recording with trial banners
    axes[0].plot(time_axis, full_ci, color="#333333", lw=1.1, label=f"Raw CI Recording ({ci_subject})")
    axes[0].set_ylabel(r"Voltage ($\mu$V)")
    axes[0].set_title(f"Panel 1: Continuous Multi-Trial CI Recording at {target_ch} (Permutation {permutation})", fontweight="bold")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    # Annotate trial boxes at the top of Panel 1
    y_min, y_max = axes[0].get_ylim()
    y_pos = y_max - 0.15 * (y_max - y_min)
    for k in range(actual_num_trials):
        t_mid = (k + 0.5) * trial_duration_s
        cond = conditions_seq[k]
        lbl = f"T{s_idx + k + 1}: [{cond}]" + ("\n(SILENCE)" if cond == "V" else "")
        axes[0].text(t_mid, y_pos, lbl, ha="center", va="top", fontsize=8, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor=cond_colors.get(cond, "gray")))

    # Panel 2: Clean Hearing Control Recording across the same sentences
    axes[1].plot(time_axis, full_h, color="#1f77b4", lw=1.3, label=f"Clean Hearing Ground Truth ({hearing_subject})")
    axes[1].set_ylabel(r"Voltage ($\mu$V)")
    axes[1].set_title(f"Panel 2: Continuous Clean Normal-Hearing Control at {target_ch} (Matching Sentences)", fontweight="bold")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Multi-Method Isolated CI Noise
    for m in extracted_noise_mats:
        axes[2].plot(time_axis, full_noise[m], lw=1.2, color=colors.get(m, "gray"), alpha=0.85, label=f"CI Noise: {labels.get(m, m.upper())}")
    axes[2].set_ylabel(r"Voltage ($\mu$V)")
    axes[2].set_title("Panel 3: Isolated CI Stimulation Noise by Method (Note Flatline During Visual [V] Trials)", fontweight="bold")
    axes[2].legend(loc="upper right", ncol=4)
    axes[2].grid(True, alpha=0.3)

    # Panel 4: Denoised CI Signal (Raw CI - Noise)
    axes[3].plot(time_axis, full_h, color="#999999", lw=1.0, ls="--", alpha=0.7, label="Clean Hearing Reference")
    for m in extracted_noise_mats:
        axes[3].plot(time_axis, full_denoised[m], lw=1.2, color=colors.get(m, "gray"), alpha=0.85, label=f"Denoised: {labels.get(m, m.upper())}")
    axes[3].set_xlabel("Continuous Timeline within Block (seconds)", fontweight="bold")
    axes[3].set_ylabel(r"Voltage ($\mu$V)")
    axes[3].set_title("Panel 4: Denoised CI EEG across Transitions between Speech Audio and Visual Silence", fontweight="bold")
    axes[3].legend(loc="upper right", ncol=5)
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"pairing_demo_{ci_subject}_{hearing_subject}_perm{permutation}_multitrial_block.png")
    plt.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved Multi-Trial Block Comparison Visualization → {plot_path}")

    # Export multi-trial data to NPZ and CSV
    npz_path = os.path.join(output_dir, f"multitrial_block_data_perm{permutation}.npz")
    npz_dict = {
        "time_s": time_axis,
        "raw_ci": full_ci,
        "clean_hearing": full_h,
        "conditions": np.array(conditions_seq),
        "target_channel": target_ch,
    }
    for m in extracted_noise_mats:
        npz_dict[f"noise_{m}"] = full_noise[m]
        npz_dict[f"denoised_{m}"] = full_denoised[m]
    np.savez_compressed(npz_path, **npz_dict)
    print(f"Saved Multi-Trial Block Data Matrix → {npz_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main Workflow & Visualization
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Isolate CI Noise and Pair with Clean Hearing Data on Same Permutation")
    parser.add_argument("--data-dir", default="/quobyte/millerlmgrp/annotated_data", help="Directory of annotated FIF files")
    parser.add_argument("--ci-subject", default="0901y2", help="Specific CI subject ID (e.g. 0901y2)")
    parser.add_argument("--hearing-subject", default="0801y2", help="Specific Hearing control subject ID (e.g. 0801y2)")
    parser.add_argument("--permutation", type=int, default=1, help="Permutation index (1, 2, or 3)")
    parser.add_argument("--method", choices=["aats", "ica", "sobi", "aats_sobi", "aats_ica", "all"], default="all", help="Noise isolation method ('all' runs all individual and hybrid methods)")
    parser.add_argument("--alpha", type=float, default=1.0, help="Mixing coefficient (dirty = hearing + alpha * noise)")
    parser.add_argument("--output-dir", default="./benchmark_results/single_pair_demo", help="Output directory")
    parser.add_argument("--num-block-trials", type=int, default=8, help="Number of consecutive trials for multi-trial block comparison (default: 8)")
    parser.add_argument("--start-trial", type=int, default=1, help="Starting trial index for multi-trial block comparison (default: 1)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("  CI NOISE ISOLATION & PERMUTATION-MATCHED PAIRING")
    print("=" * 70)
    print(f"Permutation:     perm{args.permutation}")
    print(f"Noise Extractor: {args.method.upper()}")
    print(f"Mixing Alpha:    {args.alpha}\n")

    ci_path, hearing_path = find_matching_pair(
        args.data_dir,
        target_perm=args.permutation,
        ci_pref=args.ci_subject,
        hearing_pref=args.hearing_subject,
    )

    if not ci_path or not os.path.exists(ci_path):
        print(f"Error: Could not find CI file for subject {args.ci_subject} under {args.data_dir}")
        return
    if not hearing_path or not os.path.exists(hearing_path):
        print(f"Error: Could not find Hearing file for subject {args.hearing_subject} under {args.data_dir}")
        return

    print(f"1. Loading CI Recording:      {os.path.basename(ci_path)}")
    print(f"2. Loading Hearing Recording: {os.path.basename(hearing_path)}")

    raw_ci = mne.io.read_raw_fif(ci_path, preload=True, verbose=False)
    raw_hearing = mne.io.read_raw_fif(hearing_path, preload=True, verbose=False)

    # Align channel sets
    common_chs = [ch for ch in raw_hearing.ch_names if ch in raw_ci.ch_names]
    raw_ci.pick(common_chs)
    raw_hearing.pick(common_chs)
    print(f"Common Channels ({len(common_chs)}): {common_chs[:8]}...\n")

    # Export full trial timeline and visual-only intervals from CI annotations
    trial_records = []
    if raw_ci.annotations:
        for idx, annot in enumerate(raw_ci.annotations):
            desc = annot["description"]
            if desc in ["A", "AV", "V"]:
                onset = float(annot["onset"])
                dur = float(annot["duration"]) if annot["duration"] > 0 else 1.5
                trial_records.append({
                    "trial_idx": len(trial_records) + 1,
                    "condition": desc,
                    "onset_s": round(onset, 3),
                    "end_s": round(onset + dur, 3),
                    "duration_s": round(dur, 3),
                })

    if trial_records:
        import csv
        timeline_csv = os.path.join(args.output_dir, f"trial_timeline_{args.ci_subject}.csv")
        with open(timeline_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["trial_idx", "condition", "onset_s", "end_s", "duration_s"])
            writer.writeheader()
            writer.writerows(trial_records)
        print(f"Exported Complete Trial Timeline ({len(trial_records)} trials) → {timeline_csv}")

        visual_csv = os.path.join(args.output_dir, f"visual_only_intervals_{args.ci_subject}.csv")
        v_records = [r for r in trial_records if r["condition"] == "V"]
        with open(visual_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["trial_idx", "condition", "onset_s", "end_s", "duration_s"])
            writer.writeheader()
            writer.writerows(v_records)
        print(f"Exported Visual-Only Silent Intervals ({len(v_records)} trials) → {visual_csv}\n")

    # Determine methods to run
    methods_to_run = ["aats", "ica", "sobi", "aats_sobi", "aats_ica"] if args.method == "all" else [args.method]

    # Pre-compute Clean Hearing Evoked responses
    conditions = ["A", "AV", "V"]
    evokeds_clean = {}
    for cond in conditions:
        if raw_hearing.annotations and cond in raw_hearing.annotations.description:
            ev_h, _ = mne.events_from_annotations(raw_hearing, event_id={cond: 1}, verbose=False)
            if len(ev_h) > 0:
                ep_h = mne.Epochs(raw_hearing, ev_h, event_id={cond: 1}, tmin=-0.2, tmax=0.8, baseline=(-0.2, 0.0), preload=True, verbose=False)
                evokeds_clean[cond] = ep_h.average()

    eval_cond = "A" if "A" in evokeds_clean else ("AV" if "AV" in evokeds_clean else None)
    if not eval_cond or eval_cond not in evokeds_clean:
        print("Error: Hearing recording has no A or AV events.")
        return

    h_ev = evokeds_clean[eval_cond]
    target_ch = "Cz" if "Cz" in common_chs else common_chs[0]
    ch_idx = common_chs.index(target_ch)

    extracted_noise_mats = {}
    evokeds_noise_dict = {}

    for m in methods_to_run:
        print(f"\n--- Extracting CI noise matrix using [{m.upper()}] ---")
        if m == "aats":
            n_mat, _ = isolate_noise_aats(raw_ci)
        elif m == "ica":
            n_mat, _ = isolate_noise_ica(raw_ci)
        elif m == "sobi":
            n_mat, _ = isolate_noise_sobi(raw_ci)
        elif m == "aats_sobi":
            n_mat, _ = isolate_noise_aats_sobi(raw_ci)
        else:
            n_mat, _ = isolate_noise_aats_ica(raw_ci)

        extracted_noise_mats[m] = n_mat
        raw_noise = mne.io.RawArray(n_mat, raw_ci.info.copy(), verbose=False)
        raw_noise.set_annotations(raw_ci.annotations)

        evokeds_noise_dict[m] = {}
        for cond in conditions:
            if raw_noise.annotations and cond in raw_noise.annotations.description:
                ev_n, _ = mne.events_from_annotations(raw_noise, event_id={cond: 1}, verbose=False)
                if len(ev_n) > 0:
                    ep_n = mne.Epochs(raw_noise, ev_n, event_id={cond: 1}, tmin=-0.2, tmax=0.8, baseline=(-0.2, 0.0), preload=True, verbose=False)
                    evokeds_noise_dict[m][cond] = ep_n.average()

    # If running ALL methods, compute pairwise correlation of extracted noise signals at target channel
    min_t = len(h_ev.times)
    for m in methods_to_run:
        if eval_cond in evokeds_noise_dict[m]:
            min_t = min(min_t, len(evokeds_noise_dict[m][eval_cond].times))

    times = h_ev.times[:min_t] * 1000.0  # ms
    h_sig = h_ev.data[ch_idx, :min_t] * 1e6  # uV

    noise_signals = {}
    dirty_signals = {}
    v_signals = {}

    for m in methods_to_run:
        if eval_cond in evokeds_noise_dict[m]:
            n_sig = evokeds_noise_dict[m][eval_cond].data[ch_idx, :min_t] * 1e6
            noise_signals[m] = n_sig
            dirty_signals[m] = h_sig + args.alpha * n_sig
        else:
            noise_signals[m] = np.zeros_like(h_sig)
            dirty_signals[m] = h_sig

        if "V" in evokeds_noise_dict[m]:
            v_signals[m] = evokeds_noise_dict[m]["V"].data[ch_idx, :min_t] * 1e6
        else:
            v_signals[m] = np.zeros_like(h_sig)

    colors = {
        "aats": "#1f77b4",
        "ica": "#ff7f0e",
        "sobi": "#2ca02c",
        "aats_sobi": "#d62728",
        "aats_ica": "#9467bd",
    }
    labels = {
        "aats": "AATS",
        "ica": "FastICA",
        "sobi": "SOBI",
        "aats_sobi": "AATS+SOBI (Hybrid)",
        "aats_ica": "AATS+ICA (Hybrid)",
    }

    if args.method == "all":
        print("\n" + "=" * 65)
        print("  PAIRWISE CORRELATION AMONG EXTRACTED NOISE SIGNALS (at Cz)")
        print("=" * 65)
        pair_list = [
            ("aats", "ica"),
            ("aats", "sobi"),
            ("ica", "sobi"),
            ("aats", "aats_sobi"),
            ("sobi", "aats_sobi"),
            ("ica", "aats_ica"),
            ("aats_sobi", "aats_ica"),
        ]
        for m1, m2 in pair_list:
            if np.std(noise_signals[m1]) > 1e-6 and np.std(noise_signals[m2]) > 1e-6:
                r_val = float(np.corrcoef(noise_signals[m1], noise_signals[m2])[0, 1])
                print(f"  Pearson r({labels[m1]}, {labels[m2]}): {r_val:+.4f}")
            else:
                print(f"  Pearson r({labels[m1]}, {labels[m2]}): N/A (low variance)")
        print("=" * 65 + "\n")

        # Multi-Method Comparative 4-Panel Visualization
        fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)

        # Panel 1: Extracted Noise Comparison
        for m in methods_to_run:
            axes[0].plot(times, noise_signals[m], lw=1.5, color=colors[m], label=f"CI Noise: {labels[m]}")
        axes[0].axvline(0, color="k", ls="--", alpha=0.6, label="Stimulus Onset")
        axes[0].set_ylabel(r"Voltage ($\mu$V)")
        axes[0].set_title(f"Panel 1: Isolated Speech-Evoked CI Noise Matrix at {target_ch} (Subject {args.ci_subject}, perm{args.permutation})", fontweight="bold")
        axes[0].legend(loc="upper right", ncol=4)
        axes[0].grid(True, alpha=0.3)

        # Panel 2: Clean Hearing Control ERP
        axes[1].plot(times, h_sig, color="#333333", lw=2.0, label=f"Clean Hearing Ground Truth ({args.hearing_subject})")
        axes[1].axvline(0, color="k", ls="--", alpha=0.6)
        axes[1].set_ylabel(r"Voltage ($\mu$V)")
        axes[1].set_title(f"Panel 2: Clean Normal-Hearing Auditory ERP at {target_ch} (Intact N1-P2 Dipole)", fontweight="bold")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)

        # Panel 3: Synthesized Semi-Synthetic Mixtures
        axes[2].plot(times, h_sig, color="#888888", lw=1.2, ls="--", alpha=0.8, label="Clean Ground Truth")
        for m in methods_to_run:
            axes[2].plot(times, dirty_signals[m], lw=1.4, color=colors[m], alpha=0.85, label=f"Hearing + {args.alpha}x {labels[m]}")
        axes[2].axvline(0, color="k", ls="--", alpha=0.6)
        axes[2].set_ylabel(r"Voltage ($\mu$V)")
        axes[2].set_title(f"Panel 3: Synthesized Semi-Synthetic Training Mixtures (perm{args.permutation})", fontweight="bold")
        axes[2].legend(loc="upper right", ncol=4)
        axes[2].grid(True, alpha=0.3)

        # Panel 4: Visual-only vs Active Audio
        axes[3].plot(times, v_signals["aats"], color="#7f7f7f", lw=1.5, ls=":", label="Visual-only [V] (Ambient Silence Baseline)")
        for m in methods_to_run:
            axes[3].plot(times, noise_signals[m], lw=1.2, color=colors[m], alpha=0.7, label=f"Audio [A] ({labels[m]})")
        axes[3].axvline(0, color="k", ls="--", alpha=0.6)
        axes[3].set_xlabel("Time relative to stimulus onset (ms)", fontweight="bold")
        axes[3].set_ylabel(r"Voltage ($\mu$V)")
        axes[3].set_title("Panel 4: Visual-Only 'Silence' Baseline vs. Active Audio Stimulation Artifact", fontweight="bold")
        axes[3].legend(loc="upper right", ncol=4)
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(args.output_dir, f"pairing_demo_{args.ci_subject}_{args.hearing_subject}_perm{args.permutation}_comparison_all.png")
        plt.savefig(plot_path, dpi=200)
        plt.close(fig)
        print(f"Saved Multi-Method Comparative 4-Panel Visualization → {plot_path}")

        # Save Combined .npz
        npz_dict = {
            "clean": h_ev.data[:, :min_t],
            "times": h_ev.times[:min_t],
            "channels": common_chs,
            "permutation": args.permutation,
            "ci_subject": args.ci_subject,
            "hearing_subject": args.hearing_subject,
        }
        for m in methods_to_run:
            npz_dict[f"noise_{m}"] = evokeds_noise_dict[m][eval_cond].data[:, :min_t]
            npz_dict[f"dirty_{m}"] = h_ev.data[:, :min_t] + args.alpha * evokeds_noise_dict[m][eval_cond].data[:, :min_t]

        npz_path = os.path.join(args.output_dir, f"paired_epochs_perm{args.permutation}_all.npz")
        np.savez_compressed(npz_path, **npz_dict)
        print(f"Saved Combined Multi-Method Ground-Truth Matrix → {npz_path}")

    else:
        # Single-method visualization and save
        m = args.method
        n_sig = noise_signals[m]
        dirty_sig = dirty_signals[m]
        v_sig = v_signals[m]
        n_ev = evokeds_noise_dict[m][eval_cond]

        fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)

        axes[0].plot(times, n_sig, color="#d62728", lw=1.5, label=f"Isolated CI Noise ({m.upper()})")
        axes[0].axvline(0, color="k", ls="--", alpha=0.6, label="Stimulus Onset")
        axes[0].set_ylabel(r"Voltage ($\mu$V)")
        axes[0].set_title(f"Panel 1: Isolated Speech-Evoked CI Noise Matrix at {target_ch} (Subject {args.ci_subject})", fontweight="bold")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(times, h_sig, color="#1f77b4", lw=1.8, label=f"Clean Hearing Ground Truth ({args.hearing_subject})")
        axes[1].axvline(0, color="k", ls="--", alpha=0.6)
        axes[1].set_ylabel(r"Voltage ($\mu$V)")
        axes[1].set_title(f"Panel 2: Clean Normal-Hearing Auditory ERP at {target_ch} (Intact N1-P2 Dipole)", fontweight="bold")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(times, dirty_sig, color="#9467bd", lw=1.5, label=f"Synthesized Input (Hearing + {args.alpha}x CI Noise)")
        axes[2].plot(times, h_sig, color="#1f77b4", lw=1.0, ls="--", alpha=0.7, label="Underlying Brainwave")
        axes[2].axvline(0, color="k", ls="--", alpha=0.6)
        axes[2].set_ylabel(r"Voltage ($\mu$V)")
        axes[2].set_title(f"Panel 3: Semi-Synthetic Training Mixture (Permutation Matched: perm{args.permutation})", fontweight="bold")
        axes[2].legend(loc="upper right")
        axes[2].grid(True, alpha=0.3)

        axes[3].plot(times, v_sig, color="#2ca02c", lw=1.4, label="Visual-only [V] (Ambient Silence / Hardware Noise)")
        axes[3].plot(times, n_sig, color="#d62728", lw=1.2, alpha=0.7, label="Audio-only [A] (Speech Stimulation Pulses)")
        axes[3].axvline(0, color="k", ls="--", alpha=0.6)
        axes[3].set_xlabel("Time relative to stimulus onset (ms)", fontweight="bold")
        axes[3].set_ylabel(r"Voltage ($\mu$V)")
        axes[3].set_title("Panel 4: Visual-Only 'Silence' Baseline vs. Active Audio Stimulation Artifact", fontweight="bold")
        axes[3].legend(loc="upper right")
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(args.output_dir, f"pairing_demo_{args.ci_subject}_{args.hearing_subject}_perm{args.permutation}_{m}.png")
        plt.savefig(plot_path, dpi=200)
        plt.close(fig)
        print(f"\nSaved Comparative 4-Panel Visualization → {plot_path}")

        npz_path = os.path.join(args.output_dir, f"paired_epochs_perm{args.permutation}_{m}.npz")
        np.savez_compressed(
            npz_path,
            clean=h_ev.data[:, :min_t],
            noise=n_ev.data[:, :min_t],
            dirty=(h_ev.data[:, :min_t] + args.alpha * n_ev.data[:, :min_t]),
            times=h_ev.times[:min_t],
            channels=common_chs,
            permutation=args.permutation,
            ci_subject=args.ci_subject,
            hearing_subject=args.hearing_subject,
            method=m,
        )
        print(f"Saved Paired Ground-Truth Matrix → {npz_path}")

    # Multi-Trial Continuous Block Comparison (Audio vs Visual Sequences)
    extract_and_plot_multitrial_block(
        raw_ci=raw_ci,
        raw_hearing=raw_hearing,
        extracted_noise_mats=extracted_noise_mats,
        common_chs=common_chs,
        target_ch=target_ch,
        output_dir=args.output_dir,
        ci_subject=args.ci_subject,
        hearing_subject=args.hearing_subject,
        permutation=args.permutation,
        start_trial=args.start_trial,
        num_trials=args.num_block_trials,
    )

    print("\nPairing and Noise Extraction Workflow Finished Successfully!")


if __name__ == "__main__":
    main()
