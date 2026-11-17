"""
run_sobi_multi_pass_collapse.py

Empirical stress-test of recursive SOBI:
Executes 6 consecutive passes of SOBI on continuous raw CI EEG (0901y2):
- Pass 1: Primary coil artifact
- Pass 2: Secondary return lead
- Passes 3 to 6: Progressive subspace erosion

Evaluates at each pass k in [1..6]:
1. Isolated component scalp topography and top channel
2. Mastoid-to-central power ratio (artifact vs. neural signature)
3. Cortical Auditory Evoked Potential (Cz): N1, P2 amplitude, latency, peak-to-peak
4. Ground-truth fidelity against Normal Hearing control (Pearson r, RMSE in uV)
5. RF hash suppression (20-50 Hz) vs. Neural preservation (1-15 Hz)
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
from scipy import signal


CI_REFERENCE_CHANNELS = ["P7", "T7", "M1", "M2", "P8"]
CENTRAL_AUDITORY_CHANNELS = ["Cz", "Fz", "FC1", "FC2", "C3", "C4"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. SOBI Implementation with Rank-Deficient Safeguards
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
    with rank-deficiency handling.
    """
    n_channels, n_times = data.shape
    if lags is None:
        lags = [1, 2, 4, 8, 16, 25, 50]

    # Pre-whitening via SVD/eigh
    X = data - np.mean(data, axis=1, keepdims=True)
    cov0 = (X @ X.T) / (n_times - 1)
    d, u = np.linalg.eigh(cov0)

    # Sort descending
    sort_idx = np.argsort(d)[::-1]
    d = d[sort_idx]
    u = u[:, sort_idx]

    # Detect numerical rank
    max_d = np.max(d)
    valid_mask = d > 1e-7 * max_d
    rank = int(np.sum(valid_mask))
    if rank == 0:
        rank = 1

    if n_components is None or n_components > rank:
        n_comps = rank
    else:
        n_comps = n_components

    d_sub = np.maximum(d[:n_comps], 1e-12)
    u_sub = u[:, :n_comps]

    whitening = np.diag(1.0 / np.sqrt(d_sub)) @ u_sub.T
    dewhitening = u_sub @ np.diag(np.sqrt(d_sub))
    Z = whitening @ X

    # Lagged cross-covariance matrices
    K = len(lags)
    M = np.zeros((n_comps, n_comps * K))
    for k_idx, lag in enumerate(lags):
        if lag >= n_times:
            continue
        c_lag = (Z[:, :n_times - lag] @ Z[:, lag:].T) / (n_times - lag)
        c_sym = 0.5 * (c_lag + c_lag.T)
        M[:, k_idx * n_comps : (k_idx + 1) * n_comps] = c_sym

    # Jacobi rotations
    V = np.eye(n_comps)
    for _ in range(max_iter):
        delta = 0.0
        for p in range(n_comps - 1):
            for q in range(p + 1, n_comps):
                h_p = M[p, :].reshape(K, n_comps)
                h_q = M[q, :].reshape(K, n_comps)
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
                        col_p = M[:, k * n_comps + p].copy()
                        col_q = M[:, k * n_comps + q].copy()
                        M[:, k * n_comps + p] = c * col_p + s * col_q
                        M[:, k * n_comps + q] = -s * col_p + c * col_q

        if delta < eps:
            break

    W = V.T @ whitening
    A = dewhitening @ V
    return A, W


# ─────────────────────────────────────────────────────────────────────────────
# 2. Setup & Diagnostic Helpers
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
        print(f"Montage notice: {e}", flush=True)
    return raw


def identify_dominant_component(
    A: np.ndarray,
    S: np.ndarray,
    ch_names: List[str],
    ref_channels: List[str] = CI_REFERENCE_CHANNELS,
) -> Tuple[int, Dict]:
    """
    Selects the component that accounts for maximum variance / power.
    Also records spatial power ratios to classify whether it is an artifact or neural dipole.
    """
    n_chs, n_comps = A.shape
    ref_idxs = [ch_names.index(c) for c in ref_channels if c in ch_names]
    central_idxs = [ch_names.index(c) for c in CENTRAL_AUDITORY_CHANNELS if c in ch_names]

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

    # Select dominant component by variance
    dom_idx = int(np.argmax(pvaf))

    info = {
        "dom_idx": dom_idx,
        "pvaf": float(pvaf[dom_idx]),
        "top_channel": top_chs[dom_idx],
        "ref_ratio": float(ratios[dom_idx]),
        "is_mastoid_focal": bool(top_chs[dom_idx] in ref_channels or ratios[dom_idx] > 2.0),
    }
    return dom_idx, info


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

    eval_mask = (times >= 0.0) & (times <= 0.500)
    y_test = data_ch[eval_mask]
    y_true = ref_ch[eval_mask]

    corr = float(np.corrcoef(y_test, y_true)[0, 1]) if (np.std(y_test) > 1e-6 and np.std(y_true) > 1e-6) else 0.0
    rmse = float(np.sqrt(np.mean((y_test - y_true) ** 2)))

    evoked_power = np.mean(y_test ** 2)
    noise_power = np.mean(data_ch[baseline_mask] ** 2) + 1e-12
    erp_snr_db = float(10.0 * np.log10(evoked_power / noise_power))

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
    }


def compute_band_power(sig: np.ndarray, fs: float, fmin: float, fmax: float) -> float:
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
# 3. Multi-Pass Execution
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-Pass SOBI Stress Test (Passes 1 to 6)")
    parser.add_argument("--data-dir", default="/quobyte/millerlmgrp/annotated_data", help="FIF data directory")
    parser.add_argument("--ci-subject", default="0901y2", help="CI subject ID")
    parser.add_argument("--hearing-subject", default="0801y2", help="Hearing subject ID")
    parser.add_argument("--max-passes", type=int, default=6, help="Maximum number of consecutive SOBI passes")
    parser.add_argument("--n-components", type=int, default=15, help="Number of SOBI components to fit")
    parser.add_argument("--output-dir", default="./benchmark_results/single_pair_demo", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80, flush=True)
    print(f"  MULTI-PASS SOBI STRESS TEST (PASSES 1 TO {args.max_passes})", flush=True)
    print("=" * 80, flush=True)

    ci_files = sorted(glob.glob(os.path.join(args.data_dir, "**", f"*{args.ci_subject}*-raw.fif"), recursive=True))
    hearing_files = sorted(glob.glob(os.path.join(args.data_dir, "**", f"*{args.hearing_subject}*-raw.fif"), recursive=True))

    raw_ci = mne.io.read_raw_fif(ci_files[0], preload=True, verbose=False)
    raw_hearing = mne.io.read_raw_fif(hearing_files[0], preload=True, verbose=False)

    common_chs = [ch for ch in raw_hearing.ch_names if ch in raw_ci.ch_names]
    raw_ci.pick(common_chs)
    raw_hearing.pick(common_chs)

    raw_ci = setup_montage(raw_ci)
    raw_hearing = setup_montage(raw_hearing)

    fs = float(raw_ci.info["sfreq"])
    cz_idx = common_chs.index("Cz")
    n_chs = len(common_chs)

    # Reference Ground Truth
    evoked_hearing = get_averaged_erp(raw_hearing)
    evoked_raw_ci = get_averaged_erp(raw_ci)
    m_hearing = compute_metrics(evoked_hearing, evoked_hearing, "Cz")
    m_raw = compute_metrics(evoked_raw_ci, evoked_hearing, "Cz")

    records = [
        {
            "Pass": "Normal Hearing (GT)",
            "Top_Channel": "N/A",
            "PVAF_%": 0.0,
            "Mastoid_Ratio": 0.0,
            "Category": "Ground Truth",
            **m_hearing,
            "neural_power_1_15hz": compute_band_power(raw_hearing.get_data()[cz_idx], fs, 1.0, 15.0),
            "rf_hash_power_20_50hz": compute_band_power(raw_hearing.get_data()[cz_idx], fs, 20.0, 50.0),
        },
        {
            "Pass": "Raw CI (Contaminated)",
            "Top_Channel": "N/A",
            "PVAF_%": 0.0,
            "Mastoid_Ratio": 0.0,
            "Category": "Contaminated Input",
            **m_raw,
            "neural_power_1_15hz": compute_band_power(raw_ci.get_data()[cz_idx], fs, 1.0, 15.0),
            "rf_hash_power_20_50hz": compute_band_power(raw_ci.get_data()[cz_idx], fs, 20.0, 50.0),
        },
    ]

    evoked_list = [
        ("Clean Hearing Control", evoked_hearing, "#000000", 2.2, "--"),
        ("Raw CI EEG", evoked_raw_ci, "#888888", 1.0, "-"),
    ]

    pass_components = []
    current_data = raw_ci.get_data().copy()

    # Pass colors for progressive plotting
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b"]

    print("\nStarting recursive SOBI passes...", flush=True)

    for p in range(1, args.max_passes + 1):
        print(f"\n--- [Pass {p}/{args.max_passes}] Running SOBI on residual (Rank: {n_chs - (p - 1)}) ---", flush=True)
        A, W = run_sobi(current_data, n_components=args.n_components)
        S = W @ current_data

        dom_idx, info = identify_dominant_component(A, S, common_chs)
        top_ch = info["top_channel"]
        pvaf = info["pvaf"]
        ref_ratio = info["ref_ratio"]

        # Classification label
        if ref_ratio > 3.0:
            cat = "CI Hardware Artifact"
        elif ref_ratio > 1.2:
            cat = "Mixed / Transition"
        else:
            cat = "CORTICAL NEURAL / BACKGROUND"

        print(f"  Dominant Component: SOBI {dom_idx}", flush=True)
        print(f"  Top Channel:        {top_ch}", flush=True)
        print(f"  Explained Variance: {pvaf:.2f}%", flush=True)
        print(f"  Mastoid/Central:    {ref_ratio:.2f}x ({cat})", flush=True)

        # Save component topography for plotting
        pass_components.append({
            "pass": p,
            "top_ch": top_ch,
            "pvaf": pvaf,
            "ref_ratio": ref_ratio,
            "weights": A[:, dom_idx],
            "category": cat,
        })

        # Project out the single dominant component
        S_clean = S.copy()
        S_clean[dom_idx, :] = 0.0
        current_data = A @ S_clean

        # Wrap in Raw to epoch
        raw_pass = mne.io.RawArray(current_data, raw_ci.info.copy(), verbose=False)
        raw_pass.set_annotations(raw_ci.annotations)
        evoked_pass = get_averaged_erp(raw_pass)

        m_pass = compute_metrics(evoked_pass, evoked_hearing, "Cz")
        neural_pow = compute_band_power(current_data[cz_idx], fs, 1.0, 15.0)
        rf_pow = compute_band_power(current_data[cz_idx], fs, 20.0, 50.0)

        records.append({
            "Pass": f"SOBI Pass {p}",
            "Top_Channel": top_ch,
            "PVAF_%": pvaf,
            "Mastoid_Ratio": ref_ratio,
            "Category": cat,
            **m_pass,
            "neural_power_1_15hz": neural_pow,
            "rf_hash_power_20_50hz": rf_pow,
        })

        evoked_list.append((f"Pass {p} ({top_ch})", evoked_pass, colors[p - 1], 1.6 if p <= 2 else 1.2, "-"))

    df = pd.DataFrame(records)
    csv_path = os.path.join(args.output_dir, f"sobi_multi_pass_metrics_{args.ci_subject}.csv")
    df.to_csv(csv_path, index=False)
    json_path = os.path.join(args.output_dir, f"sobi_multi_pass_metrics_{args.ci_subject}.json")
    with open(json_path, "w") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)

    print("\n" + "=" * 95, flush=True)
    print("                     MULTI-PASS SOBI BENCHMARK RESULTS", flush=True)
    print("=" * 95, flush=True)
    disp_cols = ["Pass", "Top_Channel", "Category", "corr_with_hearing", "rmse_with_hearing", "p2_amp_uv", "p2_lat_ms", "baseline_rms_uv", "rf_hash_power_20_50hz"]
    print(df[disp_cols].to_string(index=False), flush=True)
    print("=" * 95 + "\n", flush=True)

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Multi-Panel Visualizations
    # ─────────────────────────────────────────────────────────────────────────
    print("Rendering diagnostic figures...", flush=True)

    # 4.1 FIGURE 1: Scalp Topographies of Passes 1 to 6
    fig1, axes1 = plt.subplots(1, args.max_passes, figsize=(18, 4), facecolor="#ffffff")
    fig1.suptitle("Recursive SOBI: Progressive Component Extraction (Passes 1 to 6)", fontsize=13, fontweight="bold", y=1.02)

    for i, cinfo in enumerate(pass_components):
        ax = axes1[i]
        im, _ = mne.viz.plot_topomap(cinfo["weights"], raw_ci.info, axes=ax, show=False, cmap="RdBu_r", sphere=0.095)
        ax.set_title(f"Pass {cinfo['pass']}: {cinfo['top_ch']}\n{cinfo['pvaf']:.1f}% var | {cinfo['ref_ratio']:.1f}x\n({cinfo['category'][:15]})",
                     fontsize=9, fontweight="bold", pad=4)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig1.tight_layout()
    fig1_path = os.path.join(args.output_dir, f"sobi_multi_pass_topomaps_{args.ci_subject}.png")
    fig1.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved Topomap Figure -> {fig1_path}", flush=True)

    # 4.2 FIGURE 2: Cz ERP Waveforms across Passes 1 to 6
    fig2, (ax2_wave, ax2_metric) = plt.subplots(2, 1, figsize=(14, 10), facecolor="#ffffff", gridspec_kw={"height_ratios": [1.5, 1]})
    fig2.suptitle(f"Cortical Auditory ERP at Cz Across Progressive SOBI Passes (Subject {args.ci_subject})", fontsize=13, fontweight="bold")

    times = evoked_hearing.times * 1000.0  # ms
    cz_idx = common_chs.index("Cz")

    for name, evk, col, lw, ls in evoked_list:
        sig = evk.data[cz_idx] * 1e6
        ax2_wave.plot(times, sig, label=name, color=col, lw=lw, ls=ls, alpha=0.9)

    ax2_wave.axvspan(-200, 0, color="gray", alpha=0.08)
    ax2_wave.axvline(100, color="darkred", ls=":", alpha=0.5, label="N1 Target (~100 ms)")
    ax2_wave.axvline(200, color="navy", ls=":", alpha=0.5, label="P2 Target (~200 ms)")
    ax2_wave.set_title("A. Vertex (Cz) Waveform Overlays: Evolution from Pass 1 to Pass 6", fontsize=11, fontweight="bold")
    ax2_wave.set_xlabel("Time post-stimulus onset (ms)", fontweight="bold")
    ax2_wave.set_ylabel("Amplitude (µV)", fontweight="bold")
    ax2_wave.legend(loc="upper right", fontsize=8.5, frameon=True, ncol=2)
    ax2_wave.grid(True, linestyle=":", alpha=0.4)

    # Metric Curves: Correlation r and P2 Amplitude vs. Pass Number
    sobi_passes = [r for r in records if "SOBI Pass" in r["Pass"]]
    pass_nums = [int(r["Pass"].split()[-1]) for r in sobi_passes]
    corrs = [r["corr_with_hearing"] for r in sobi_passes]
    p2_amps = [r["p2_amp_uv"] for r in sobi_passes]
    rf_powers = [r["rf_hash_power_20_50hz"] * 1e11 for r in sobi_passes]

    ax2_metric.plot(pass_nums, corrs, marker="o", color="#1f77b4", lw=2.0, label="Pearson r vs Hearing Control")
    ax2_metric.set_xlabel("SOBI Pass Number", fontweight="bold")
    ax2_metric.set_ylabel("Pearson Correlation (r)", color="#1f77b4", fontweight="bold")
    ax2_metric.tick_params(axis="y", labelcolor="#1f77b4")
    ax2_metric.set_xticks(pass_nums)
    ax2_metric.grid(True, linestyle=":", alpha=0.5)

    ax2_twin = ax2_metric.twinx()
    ax2_twin.plot(pass_nums, p2_amps, marker="s", color="#d62728", lw=2.0, ls="--", label="P2 Peak Amplitude (µV)")
    ax2_twin.set_ylabel("Cz P2 Amplitude (µV)", color="#d62728", fontweight="bold")
    ax2_twin.tick_params(axis="y", labelcolor="#d62728")

    # Add threshold note
    ax2_metric.axvline(2.0, color="green", ls="-.", alpha=0.7, label="Optimal Hardware Cutoff (Pass 2)")
    ax2_metric.set_title("B. Metric Trajectory: Point of Brain Activity Erosion", fontsize=11, fontweight="bold")

    lines_1, labels_1 = ax2_metric.get_legend_handles_labels()
    lines_2, labels_2 = ax2_twin.get_legend_handles_labels()
    ax2_metric.legend(lines_1 + lines_2, labels_1 + labels_2, loc="lower left", fontsize=8.5)

    fig2.tight_layout()
    fig2_path = os.path.join(args.output_dir, f"sobi_multi_pass_erp_collapse_{args.ci_subject}.png")
    fig2.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved ERP Collapse Figure -> {fig2_path}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("  EXPERIMENT COMPLETE! All outputs saved to:", args.output_dir, flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
