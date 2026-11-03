"""
compare_sobi_ica.py

Head-to-Head Comparison: SOBI vs. FastICA for Cochlear Implant (CI) EEG Denoising.
Evaluates Subject 0901y2 (CI) vs. 0801y2 (Hearing Control) on Permutation 1.

Metrics Evaluated:
1. Component Localization: Mastoid/Coil artifact isolation vs. diffuse neural preservation.
2. 2D Brain Scalp Potential Topographies at N1 (~100 ms) and P2 (~200 ms).
3. Vertex (Cz) ERP Morphometry: Peak-to-peak amplitude, N1/P2 latencies, baseline noise floor.
4. Biological Ground-Truth Recovery: Pearson correlation (r), RMSE, and SNR vs Normal Hearing.
"""

import argparse
import glob
import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne.preprocessing import ICA


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
    Second-Order Blind Identification (SOBI) via joint diagonalization of
    time-lagged cross-covariance matrices.

    Args:
        data: (n_channels, n_times) zero-mean continuous or epoched EEG.
        n_components: Number of components to extract.
        lags: Sample lags (default: [1, 2, 4, 8, 16, 25, 50]).

    Returns:
        mixing_matrix A: (n_channels, n_components)
        unmixing_matrix W: (n_components, n_channels)
    """
    n_channels, n_times = data.shape
    if n_components is None or n_components > n_channels:
        n_components = n_channels
    if lags is None:
        lags = [1, 2, 4, 8, 16, 25, 50]

    # Pre-whiten with SVD
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
# 2. Helpers & Montage Setup
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
        print(f"Montage setup notice: {e}")
    return raw


def classify_mixing_matrix(
    A: np.ndarray,
    ch_names: List[str],
    ref_channels: List[str] = CI_REFERENCE_CHANNELS,
    central_channels: List[str] = CENTRAL_AUDITORY_CHANNELS,
) -> Dict[int, Dict]:
    """
    Classifies spatial projection components based on mixing matrix A (n_channels, n_components).
    """
    n_comps = A.shape[1]
    results = {}

    ref_idxs = [ch_names.index(c) for c in ref_channels if c in ch_names]
    central_idxs = [ch_names.index(c) for c in central_channels if c in ch_names]

    for k in range(n_comps):
        weights = A[:, k]
        abs_weights = np.abs(weights)
        top_idx = int(np.argmax(abs_weights))
        top_ch = ch_names[top_idx]

        ref_power = np.mean(abs_weights[ref_idxs]) if ref_idxs else 0.0
        central_power = np.mean(abs_weights[central_idxs]) if central_idxs else 0.0
        ref_ratio = float(ref_power / (central_power + 1e-9))

        if any(ch_names[idx] in ref_channels for idx in np.argsort(abs_weights)[::-1][:2]):
            cat = "CI ARTIFACT (Coil/Mastoid)"
            color = "#d62728"
        elif any(ch_names[idx] in ["Fp1", "Fp2", "AF3", "AF4"] for idx in np.argsort(abs_weights)[::-1][:2]):
            cat = "OCULAR (Blink/Eye)"
            color = "#ff7f0e"
        elif central_power > 1.3 * (ref_power + 1e-9) and (top_ch in central_channels):
            cat = "CENTRAL AUDITORY DIPOLE"
            color = "#2ca02c"
        else:
            cat = "DIFFUSE / NEURAL"
            color = "#1f77b4"

        results[k] = {
            "top_channel": top_ch,
            "category": cat,
            "color": color,
            "ref_ratio": ref_ratio,
        }

    return results


def find_matching_pair(
    data_dir: str,
    target_perm: int = 1,
    ci_pref: Optional[str] = "0901y2",
    hearing_pref: Optional[str] = "0801y2",
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
# 3. Epoching and ERP Extraction
# ─────────────────────────────────────────────────────────────────────────────

def get_averaged_erp(
    raw: mne.io.Raw,
    conditions: List[str] = ["A", "AV"],
    tmin: float = -0.2,
    tmax: float = 0.5,
) -> mne.Evoked:
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    target_ids = [event_id[c] for c in conditions if c in event_id]
    epochs = mne.Epochs(
        raw,
        events,
        event_id=target_ids,
        tmin=tmin,
        tmax=tmax,
        baseline=(tmin, 0.0),
        preload=True,
        verbose=False,
    )
    return epochs.average()


def compute_metrics(
    evoked_test: mne.Evoked,
    evoked_ref: mne.Evoked,
    target_channel: str = "Cz",
) -> Dict[str, float]:
    """Computes peak amplitudes, latencies, baseline RMS, Pearson r, and RMSE."""
    times = evoked_test.times
    ch_idx = evoked_test.ch_names.index(target_channel)
    ref_ch_idx = evoked_ref.ch_names.index(target_channel)

    sig_test = evoked_test.data[ch_idx] * 1e6   # µV
    sig_ref = evoked_ref.data[ref_ch_idx] * 1e6 # µV

    # Baseline RMS (-200 to 0 ms)
    base_mask = (times >= -0.2) & (times <= 0.0)
    base_rms = float(np.sqrt(np.mean(sig_test[base_mask] ** 2)))

    # N1 search: 80 to 140 ms (minimum)
    n1_mask = (times >= 0.080) & (times <= 0.140)
    n1_times = times[n1_mask]
    n1_vals = sig_test[n1_mask]
    n1_idx = int(np.argmin(n1_vals))
    n1_amp = float(n1_vals[n1_idx])
    n1_lat = float(n1_times[n1_idx] * 1000.0)

    # P2 search: 160 to 240 ms (maximum)
    p2_mask = (times >= 0.160) & (times <= 0.240)
    p2_times = times[p2_mask]
    p2_vals = sig_test[p2_mask]
    p2_idx = int(np.argmax(p2_vals))
    p2_amp = float(p2_vals[p2_idx])
    p2_lat = float(p2_times[p2_idx] * 1000.0)

    p2p_amp = float(p2_amp - n1_amp)

    # Post-stimulus window (0 to 450 ms) metrics against normal hearing
    post_mask = (times >= 0.0) & (times <= 0.450)
    test_post = sig_test[post_mask]
    ref_post = sig_ref[post_mask]

    corr = float(np.corrcoef(test_post, ref_post)[0, 1])
    rmse = float(np.sqrt(np.mean((test_post - ref_post) ** 2)))

    # ERP SNR: (Peak-to-Peak / (2 * sqrt(2) * baseline_rms))
    erp_snr = float(20.0 * np.log10(max(p2p_amp / (2.0 * np.sqrt(2.0) * (base_rms + 1e-9)), 1e-3)))

    return {
        "baseline_rms_uv": base_rms,
        "n1_amp_uv": n1_amp,
        "n1_lat_ms": n1_lat,
        "p2_amp_uv": p2_amp,
        "p2_lat_ms": p2_lat,
        "p2p_amp_uv": p2p_amp,
        "corr_with_hearing": corr,
        "rmse_with_hearing": rmse,
        "erp_snr_db": erp_snr,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Main Comparison Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SOBI vs. FastICA Head-to-Head Comparison for CI Denoising")
    parser.add_argument("--data-dir", default="/quobyte/millerlmgrp/annotated_data", help="Directory of annotated FIF files")
    parser.add_argument("--ci-subject", default="0901y2", help="CI subject ID (e.g. 0901y2)")
    parser.add_argument("--hearing-subject", default="0801y2", help="Hearing subject ID (e.g. 0801y2)")
    parser.add_argument("--permutation", type=int, default=1, help="Permutation index (1, 2, or 3)")
    parser.add_argument("--n-components", type=int, default=15, help="Number of components to fit")
    parser.add_argument("--output-dir", default="./benchmark_results/single_pair_demo", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("  HEAD-TO-HEAD BSS COMPARISON: SOBI vs. FASTICA")
    print("=" * 70)
    print(f"CI Subject:      {args.ci_subject}")
    print(f"Hearing Control: {args.hearing_subject}")
    print(f"Permutation:     perm{args.permutation}")
    print(f"Components:      {args.n_components}\n")

    ci_path, hearing_path = find_matching_pair(
        args.data_dir,
        target_perm=args.permutation,
        ci_pref=args.ci_subject,
        hearing_pref=args.hearing_subject,
    )

    if not ci_path or not os.path.exists(ci_path):
        print(f"Error: Could not find CI file for subject {args.ci_subject}")
        return
    if not hearing_path or not os.path.exists(hearing_path):
        print(f"Error: Could not find Hearing file for subject {args.hearing_subject}")
        return

    print(f"1. Loading CI Recording:      {os.path.basename(ci_path)}")
    print(f"2. Loading Hearing Recording: {os.path.basename(hearing_path)}")

    raw_ci = mne.io.read_raw_fif(ci_path, preload=True, verbose=False)
    raw_hearing = mne.io.read_raw_fif(hearing_path, preload=True, verbose=False)

    common_chs = [ch for ch in raw_hearing.ch_names if ch in raw_ci.ch_names]
    raw_ci.pick(common_chs)
    raw_hearing.pick(common_chs)

    raw_ci = setup_montage(raw_ci)
    raw_hearing = setup_montage(raw_hearing)

    ci_data = raw_ci.get_data()
    n_chs, n_times = ci_data.shape
    n_comps = min(n_chs, args.n_components)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: Fit FastICA
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[1/4] Fitting FastICA ({n_comps} components)...")
    ica = ICA(n_components=n_comps, method="fastica", random_state=42, max_iter="auto")
    ica.fit(raw_ci, verbose=False)

    ica_comp_matrix = ica.get_components()
    ica_classifications = classify_mixing_matrix(ica_comp_matrix, common_chs)
    ica_art_indices = [k for k, v in ica_classifications.items() if "CI ARTIFACT" in v["category"]]
    if not ica_art_indices:
        ica_art_indices = [0]
    print(f"  FastICA Excluded Artifact Components: {ica_art_indices}")

    raw_ica_clean = raw_ci.copy()
    ica.apply(raw_ica_clean, exclude=ica_art_indices, verbose=False)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2: Fit SOBI
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[2/4] Fitting SOBI ({n_comps} components, multiple time-lags)...")
    A_sobi, W_sobi = run_sobi(ci_data, n_components=n_comps)

    sobi_classifications = classify_mixing_matrix(A_sobi, common_chs)
    sobi_art_indices = [k for k, v in sobi_classifications.items() if "CI ARTIFACT" in v["category"]]
    if not sobi_art_indices:
        sobi_art_indices = [0]
    print(f"  SOBI Excluded Artifact Components: {sobi_art_indices}")

    # Reconstruct cleaned data by zeroing artifact sources
    sobi_sources = W_sobi @ ci_data
    sobi_sources_clean = sobi_sources.copy()
    sobi_sources_clean[sobi_art_indices, :] = 0.0
    sobi_clean_data = A_sobi @ sobi_sources_clean

    raw_sobi_clean = mne.io.RawArray(sobi_clean_data, raw_ci.info.copy(), verbose=False)
    raw_sobi_clean.set_annotations(raw_ci.annotations)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3: Epoching & Metric Calculation
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[3/4] Epoching and computing Auditory ERP metrics at Cz...")
    evoked_raw_ci = get_averaged_erp(raw_ci)
    evoked_ica_clean = get_averaged_erp(raw_ica_clean)
    evoked_sobi_clean = get_averaged_erp(raw_sobi_clean)
    evoked_hearing = get_averaged_erp(raw_hearing)

    metrics_raw = compute_metrics(evoked_raw_ci, evoked_hearing, "Cz")
    metrics_ica = compute_metrics(evoked_ica_clean, evoked_hearing, "Cz")
    metrics_sobi = compute_metrics(evoked_sobi_clean, evoked_hearing, "Cz")
    metrics_hearing = compute_metrics(evoked_hearing, evoked_hearing, "Cz")

    df_metrics = pd.DataFrame([
        {"Method": "Raw CI EEG", **metrics_raw},
        {"Method": "FastICA Denoised", **metrics_ica},
        {"Method": "SOBI Denoised", **metrics_sobi},
        {"Method": "Clean Hearing Control", **metrics_hearing},
    ])

    csv_out = os.path.join(args.output_dir, f"sobi_vs_ica_metrics_{args.ci_subject}.csv")
    df_metrics.to_csv(csv_out, index=False)

    json_out = os.path.join(args.output_dir, f"sobi_vs_ica_metrics_{args.ci_subject}.json")
    with open(json_out, "w") as f:
        json.dump(df_metrics.to_dict(orient="records"), f, indent=2)

    print("\n" + "=" * 80)
    print("                     BSS HEAD-TO-HEAD BENCHMARK METRICS (Cz)")
    print("=" * 80)
    print(df_metrics.to_string(index=False))
    print("=" * 80 + "\n")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4: Comprehensive Comparison Visualizations
    # ─────────────────────────────────────────────────────────────────────────
    print("[4/4] Generating publication-grade comparison figures...")

    # Figure 1: Waveform & Topography Side-by-Side
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.2, 1, 1], hspace=0.35, wspace=0.25)

    times = evoked_raw_ci.times * 1000.0  # ms
    cz_idx = evoked_raw_ci.ch_names.index("Cz")

    # Panel A: ERP Waveforms at Cz
    ax_wave = fig.add_subplot(gs[0, :])
    ax_wave.plot(times, evoked_raw_ci.data[cz_idx] * 1e6, label="Raw CI EEG (Contaminated)", color="#333333", lw=1.2, alpha=0.7)
    ax_wave.plot(times, evoked_hearing.data[cz_idx] * 1e6, label="Clean Hearing Control (Ground Truth)", color="#1f77b4", lw=2.2, ls="--")
    ax_wave.plot(times, evoked_ica_clean.data[cz_idx] * 1e6, label=f"FastICA Denoised (r={metrics_ica['corr_with_hearing']:.2f})", color="#ff7f0e", lw=1.8)
    ax_wave.plot(times, evoked_sobi_clean.data[cz_idx] * 1e6, label=f"SOBI Denoised (r={metrics_sobi['corr_with_hearing']:.2f})", color="#2ca02c", lw=2.0)

    ax_wave.axvspan(-200, 0, color="gray", alpha=0.12, label="Pre-stimulus Baseline")
    ax_wave.axvline(100, color="darkred", ls=":", alpha=0.6, label="Canonical N1 (~100 ms)")
    ax_wave.axvline(200, color="navy", ls=":", alpha=0.6, label="Canonical P2 (~200 ms)")

    ax_wave.set_title(f"A. Auditory Cortical ERP Waveform at Vertex (Cz) - Subject {args.ci_subject} vs. Hearing Control", fontsize=11, fontweight="bold")
    ax_wave.set_xlabel("Time post-stimulus onset (ms)", fontweight="bold")
    ax_wave.set_ylabel(r"Voltage ($\mu$V)", fontweight="bold")
    ax_wave.set_xlim(-200, 450)
    ax_wave.legend(loc="upper right", ncol=3, fontsize=9)
    ax_wave.grid(True, alpha=0.3)

    # 2D Topographies at N1 (~100 ms) and P2 (~200 ms)
    # Rows: Row 1 = N1, Row 2 = P2
    # Cols: Raw CI, FastICA, SOBI, Normal Hearing
    methods_evokeds = [
        ("Raw CI", evoked_raw_ci),
        ("FastICA Cleaned", evoked_ica_clean),
        ("SOBI Cleaned", evoked_sobi_clean),
        ("Hearing Control", evoked_hearing),
    ]

    for col_idx, (m_label, ev_obj) in enumerate(methods_evokeds):
        # Row 1: N1 (100 ms)
        ax_n1 = fig.add_subplot(gs[1, col_idx])
        im1, _ = mne.viz.plot_topomap(
            ev_obj.get_data(tmin=0.090, tmax=0.110).mean(axis=1) * 1e6,
            ev_obj.info,
            axes=ax_n1,
            show=False,
            sphere=0.095,
            cmap="RdBu_r",
        )
        ax_n1.set_title(f"{m_label}\nN1 Peak (~100 ms)", fontsize=9.5, fontweight="bold")
        plt.colorbar(im1, ax=ax_n1, fraction=0.046, pad=0.04, label=r"$\mu$V")

        # Row 2: P2 (200 ms)
        ax_p2 = fig.add_subplot(gs[2, col_idx])
        im2, _ = mne.viz.plot_topomap(
            ev_obj.get_data(tmin=0.190, tmax=0.210).mean(axis=1) * 1e6,
            ev_obj.info,
            axes=ax_p2,
            show=False,
            sphere=0.095,
            cmap="RdBu_r",
        )
        ax_p2.set_title(f"{m_label}\nP2 Peak (~200 ms)", fontsize=9.5, fontweight="bold")
        plt.colorbar(im2, ax=ax_p2, fraction=0.046, pad=0.04, label=r"$\mu$V")

    plt.suptitle(
        f"BSS Denoising Benchmark: SOBI vs. FastICA on Subject {args.ci_subject}\n"
        "SOBI preserves time-lagged temporal covariance for continuous speech denoising",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )

    plot_out = os.path.join(args.output_dir, f"sobi_vs_ica_erp_comparison_{args.ci_subject}.png")
    plt.savefig(plot_out, dpi=200)
    plt.close(fig)
    print(f"Saved ERP & Topomap Comparison Figure → {plot_out}")

    # Figure 2: SOBI All Component Scalp Topographies
    fig_sobi, axes_sobi = plt.subplots(3, 5, figsize=(16, 9.6))
    axes_sobi = axes_sobi.reshape(-1)

    for k in range(n_comps):
        ax = axes_sobi[k]
        weights = A_sobi[:, k]
        im_s, _ = mne.viz.plot_topomap(
            weights,
            raw_ci.info,
            axes=ax,
            show=False,
            sphere=0.095,
            cmap="RdBu_r",
        )
        cls = sobi_classifications.get(k, {})
        cat = cls.get("category", "")
        col = cls.get("color", "black")
        top_ch = cls.get("top_channel", "")

        flag = " [EXCLUDED]" if k in sobi_art_indices else ""
        ax.set_title(f"SOBI #{k:02d} [{top_ch}]\n{cat}{flag}", fontsize=8.5, fontweight="bold", color=col)

    plt.suptitle(
        f"2D Scalp Topography of All {n_comps} SOBI Components (Subject {args.ci_subject})\n"
        "Red = Flagged Coil/Mastoid Artifact | Green = Central Auditory Dipole | Blue = Diffuse Neural",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    sobi_topo_out = os.path.join(args.output_dir, f"sobi_all_components_topomap_{args.ci_subject}.png")
    plt.savefig(sobi_topo_out, dpi=200)
    plt.close(fig_sobi)
    print(f"Saved SOBI Component Topographies → {sobi_topo_out}")

    print("\nSOBI vs. FastICA Head-to-Head Comparison Run Completed Successfully!")


if __name__ == "__main__":
    main()
