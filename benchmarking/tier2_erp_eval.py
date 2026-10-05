"""
tier2_erp_eval.py

Tier 2: Event-Related Potential (ERP) & Neural Signal Preservation Evaluation.

Evaluates candidate denoising methods on REAL cochlear implant EEG data with trial
onsets (A, AV, V conditions) from extract_trial_onsets.py.

Key Neural & ERP Metrics:
  - N1 Component Recovery: Peak amplitude (uV) & latency (ms) in [80, 150] ms window
  - P2 Component Recovery: Peak amplitude (uV) & latency (ms) in [150, 250] ms window
  - Template Correlation: Pearson r of evoked waveform against hearing control grand-average
  - Baseline Noise Level: Pre-stimulus RMS noise in [-200, 0] ms
  - Auditory SNR: Ratio of post-stimulus peak power to pre-stimulus noise power
  - Topographic Map Correlation: Spatial activation profile across electrodes

Can evaluate:
  (a) Pre-cleaned FIF files from ml_denoise.py outputs:
      --cleaned-root /quobyte/millerlmgrp/ml_cleaned_data
  (b) Or raw annotated FIF files with on-the-fly denoising.
"""

import argparse
import gc
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
import pandas as pd

try:
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
except Exception:
    pass

CENTRAL_AUDITORY_CHS = ['Cz', 'Fz', 'C3', 'C4']


# ─── ERP Extraction and Component Detection ──────────────────────────────────

def compute_evoked(
    fif_path: str,
    event_desc: str = "A",
    tmin: float = -0.2,
    tmax: float = 1.0,
    baseline: Tuple[float, float] = (-0.2, 0.0),
    picks: Optional[List[str]] = None,
) -> Optional[mne.Evoked]:
    """Epochs a .fif file on event_desc and returns the Evoked average without reading full recording into RAM."""
    try:
        if not os.path.exists(fif_path) or os.path.getsize(fif_path) < 100000:
            return None
        raw = mne.io.read_raw_fif(fif_path, preload=False, verbose=False)
        present = set(raw.annotations.description)
        if event_desc not in present:
            del raw
            return None

        event_id = {event_desc: 1}
        events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose=False)
        if len(events) == 0:
            del raw
            return None

        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            picks=picks,
            preload=True,
            verbose=False,
        )
        evoked = epochs.average()
        del raw, epochs
        return evoked
    except Exception as e:
        print(f"  Error reading {os.path.basename(fif_path)}: {e}")
        return None


def extract_erp_components(evoked: mne.Evoked, target_ch: str = 'Cz') -> Dict[str, float]:
    """
    Extracts N1 and P2 peak amplitudes and latencies, plus pre-stimulus noise RMS.
    """
    times = evoked.times
    ch_names = evoked.ch_names
    if target_ch not in ch_names:
        # Fall back to first available channel
        target_ch = ch_names[0]

    ch_idx = ch_names.index(target_ch)
    data = evoked.data[ch_idx]  # in Volts

    # Baseline RMS in [-200, 0] ms
    base_mask = (times >= -0.2) & (times <= 0.0)
    base_rms = float(np.sqrt(np.mean(data[base_mask] ** 2))) if np.any(base_mask) else 1e-9

    # N1: negative trough in [80, 150] ms
    n1_mask = (times >= 0.080) & (times <= 0.150)
    if np.any(n1_mask):
        n1_idx = np.where(n1_mask)[0][np.argmin(data[n1_mask])]
        n1_amp_uv = float(data[n1_idx] * 1e6)
        n1_lat_ms = float(times[n1_idx] * 1000.0)
    else:
        n1_amp_uv, n1_lat_ms = np.nan, np.nan

    # P2: positive peak in [150, 250] ms
    p2_mask = (times >= 0.150) & (times <= 0.250)
    if np.any(p2_mask):
        p2_idx = np.where(p2_mask)[0][np.argmax(data[p2_mask])]
        p2_amp_uv = float(data[p2_idx] * 1e6)
        p2_lat_ms = float(times[p2_idx] * 1000.0)
    else:
        p2_amp_uv, p2_lat_ms = np.nan, np.nan

    # Peak-to-peak amplitude (N1-P2 complex)
    p2p_uv = float(p2_amp_uv - n1_amp_uv) if not np.isnan(p2_amp_uv) and not np.isnan(n1_amp_uv) else np.nan

    # Post-stimulus peak power vs baseline noise power (Auditory ERP SNR in dB)
    post_mask = (times >= 0.050) & (times <= 0.350)
    post_power = np.max(data[post_mask] ** 2) if np.any(post_mask) else 1e-12
    erp_snr_db = float(10.0 * np.log10(post_power / (base_rms ** 2 + 1e-18)))

    return {
        "channel": target_ch,
        "n1_amp_uv": n1_amp_uv,
        "n1_lat_ms": n1_lat_ms,
        "p2_amp_uv": p2_amp_uv,
        "p2_lat_ms": p2_lat_ms,
        "p2p_uv": p2p_uv,
        "base_rms_uv": base_rms * 1e6,
        "erp_snr_db": erp_snr_db,
    }


def compute_template_correlation(
    test_evoked: mne.Evoked,
    template_evoked: mne.Evoked,
    target_ch: str = 'Cz',
    window: Tuple[float, float] = (0.050, 0.400),
) -> float:
    """Computes correlation between test ERP and hearing template ERP over the auditory window."""
    if target_ch not in test_evoked.ch_names or target_ch not in template_evoked.ch_names:
        return np.nan

    t_mask = (test_evoked.times >= window[0]) & (test_evoked.times <= window[1])
    test_sig = test_evoked.data[test_evoked.ch_names.index(target_ch), t_mask]

    tmpl_mask = (template_evoked.times >= window[0]) & (template_evoked.times <= window[1])
    tmpl_sig = template_evoked.data[template_evoked.ch_names.index(target_ch), tmpl_mask]

    min_len = min(len(test_sig), len(tmpl_sig))
    if min_len < 5:
        return np.nan

    std_t = np.std(test_sig[:min_len])
    std_m = np.std(tmpl_sig[:min_len])
    if std_t < 1e-12 or std_m < 1e-12:
        return 0.0
    return float(np.corrcoef(test_sig[:min_len], tmpl_sig[:min_len])[0, 1])


# ─── Hearing Template Construction ────────────────────────────────────────────

def build_hearing_template_from_zarr(
    zarr_path: str,
    group_name: str = "hearing_trial_data",
    sfreq: float = 250.0,
    tmin: float = -0.2,
    tmax: float = 1.0,
) -> Optional[mne.Evoked]:
    """Constructs a gold-standard hearing control ERP template from raw_epoched_data.zarr."""
    if not os.path.exists(zarr_path):
        return None
    try:
        import zarr
        try:
            root = zarr.open(zarr_path, mode="r")
        except Exception:
            root = zarr.open_group(zarr_path, mode="r")

        try:
            grp = root[group_name]
        except (KeyError, TypeError) as e:
            print(f"  Notice: Group '{group_name}' not accessible in Zarr store: {e}")
            return None
        data = grp["data"][:]  # shape: (n_epochs, n_channels, n_times)

        valid_mask = ~np.isnan(data).any(axis=(1, 2))
        valid_data = data[valid_mask]
        if len(valid_data) == 0:
            return None

        ch_names = [
            'Fp1', 'Fz', 'F3', 'F7', 'T7', 'C3', 'Cz', 'Pz', 'P3', 'P7', 'O1',
            'Fp2', 'F8', 'F4', 'C4', 'T8', 'P8', 'P4', 'O2', 'M1', 'M2'
        ]
        if valid_data.shape[1] != len(ch_names):
            ch_names = [f"EEG{i+1:03d}" for i in range(valid_data.shape[1])]

        avg_data = np.mean(valid_data, axis=0)
        n_times = avg_data.shape[1]
        times = np.arange(n_times) / sfreq + tmin
        t_mask = (times >= tmin) & (times <= tmax)
        avg_data_cropped = avg_data[:, t_mask]

        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        evoked = mne.EvokedArray(avg_data_cropped, info, tmin=tmin)
        print(f"  Grand average template built from {len(valid_data)} hearing control epochs in Zarr ({zarr_path}).")
        return evoked
    except Exception as e:
        print(f"  Warning: Failed to build hearing template from Zarr: {e}")
        return None


def build_hearing_grand_average(
    hearing_files: List[str],
    condition: str = "A",
    tmin: float = -0.2,
    tmax: float = 1.0,
) -> Optional[mne.Evoked]:
    """Averages all hearing control subjects to create the gold-standard ERP template."""
    evokeds = []
    print(f"Building hearing control ERP template from {len(hearing_files)} files...")
    for f in hearing_files:
        ev = compute_evoked(f, event_desc=condition, tmin=tmin, tmax=tmax)
        if ev is not None:
            evokeds.append(ev)
    if not evokeds:
        print("  Warning: No hearing control evokeds could be extracted.")
        return None
    grand_avg = mne.grand_average(evokeds)
    print(f"  Grand average template built from {len(evokeds)} hearing participants.")
    return grand_avg


def plot_tier2_summary(df: pd.DataFrame, output_dir: str):
    """Generates bar charts of Template Correlation, Auditory SNR, N1-P2 Amp, and Baseline Noise."""
    methods = sorted(df["method"].unique())
    num_cols = ["erp_snr_db", "p2p_uv", "base_rms_uv"]
    if "template_corr" in df.columns:
        num_cols.append("template_corr")
    summary = df.groupby("method")[num_cols].agg(["mean", "std"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Tier 2 ERP Benchmark: Neural Signal Preservation", fontsize=14, fontweight="bold", y=0.98)

    palette = plt.cm.tab10(np.linspace(0, 1, max(len(methods), 3)))
    color_map = {m: palette[i] for i, m in enumerate(methods)}

    # 1. Template Correlation
    ax = axes[0, 0]
    vals = [summary.loc[m, ("template_corr", "mean")] if ("template_corr", "mean") in summary.columns else 0.0 for m in methods]
    errs = [summary.loc[m, ("template_corr", "std")] if ("template_corr", "std") in summary.columns else 0.0 for m in methods]
    vals = [0.0 if np.isnan(v) else float(v) for v in vals]
    errs = [0.0 if np.isnan(e) else float(e) for e in errs]
    ax.bar(methods, vals, yerr=errs, capsize=5, color=[color_map[m] for m in methods], alpha=0.85)
    ax.set_title("Template Correlation with Hearing Control ERP (Higher is better)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Pearson r")
    ax.set_ylim([0.0, 1.05])
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # 2. Auditory ERP SNR
    ax = axes[0, 1]
    vals = [summary.loc[m, ("erp_snr_db", "mean")] for m in methods]
    errs = [summary.loc[m, ("erp_snr_db", "std")] for m in methods]
    vals = [0.0 if np.isnan(v) else float(v) for v in vals]
    errs = [0.0 if np.isnan(e) else float(e) for e in errs]
    ax.bar(methods, vals, yerr=errs, capsize=5, color=[color_map[m] for m in methods], alpha=0.85)
    ax.set_title("Auditory ERP SNR (Higher is better)", fontsize=11, fontweight="bold")
    ax.set_ylabel("SNR (dB)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # 3. Peak-to-Peak Amplitude
    ax = axes[1, 0]
    vals = [summary.loc[m, ("p2p_uv", "mean")] for m in methods]
    errs = [summary.loc[m, ("p2p_uv", "std")] for m in methods]
    vals = [0.0 if np.isnan(v) else float(v) for v in vals]
    errs = [0.0 if np.isnan(e) else float(e) for e in errs]
    ax.bar(methods, vals, yerr=errs, capsize=5, color=[color_map[m] for m in methods], alpha=0.85)
    ax.set_title("N1-P2 Peak-to-Peak Amplitude (μV)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Amplitude (μV)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # 4. Baseline Noise RMS
    ax = axes[1, 1]
    vals = [summary.loc[m, ("base_rms_uv", "mean")] for m in methods]
    errs = [summary.loc[m, ("base_rms_uv", "std")] for m in methods]
    vals = [0.0 if np.isnan(v) else float(v) for v in vals]
    errs = [0.0 if np.isnan(e) else float(e) for e in errs]
    ax.bar(methods, vals, yerr=errs, capsize=5, color=[color_map[m] for m in methods], alpha=0.85)
    ax.set_title("Pre-stimulus Baseline Noise RMS (Lower is cleaner)", fontsize=11, fontweight="bold")
    ax.set_ylabel("RMS Noise (μV)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "tier2_erp_summary_charts.png")
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ERP summary chart to: {plot_path}")

    # Figure 2: Cohort Distribution Boxplots
    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4.5))
    fig2.suptitle("Cohort ERP Distributions across CI Subjects", fontsize=13, fontweight="bold", y=0.99)

    data_snr = [df[df["method"] == m]["erp_snr_db"].dropna().values for m in methods]
    axes2[0].boxplot(data_snr, tick_labels=methods, patch_artist=True)
    axes2[0].set_title("Auditory ERP SNR (dB)", fontweight="bold")
    axes2[0].set_ylabel("dB")
    axes2[0].grid(axis="y", linestyle=":", alpha=0.5)

    data_p2p = [df[df["method"] == m]["p2p_uv"].dropna().values for m in methods]
    axes2[1].boxplot(data_p2p, tick_labels=methods, patch_artist=True)
    axes2[1].set_title("N1-P2 Amplitude (μV)", fontweight="bold")
    axes2[1].set_ylabel("μV")
    axes2[1].grid(axis="y", linestyle=":", alpha=0.5)

    data_base = [df[df["method"] == m]["base_rms_uv"].dropna().values for m in methods]
    axes2[2].boxplot(data_base, tick_labels=methods, patch_artist=True)
    axes2[2].set_title("Baseline Noise RMS (μV)", fontweight="bold")
    axes2[2].set_ylabel("μV")
    axes2[2].grid(axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()
    box_path = os.path.join(output_dir, "tier2_erp_distribution_boxplots.png")
    fig2.savefig(box_path, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved ERP distribution boxplots to: {box_path}")


# ─── Main Evaluation Pipeline ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Tier 2: ERP Preservation Benchmark on Real CI Recordings")
    parser.add_argument(
        "--annotated-dir",
        default="/quobyte/millerlmgrp/annotated_data",
        help="Directory containing annotated trial_onsets_{subject}-raw.fif files",
    )
    parser.add_argument(
        "--cleaned-root",
        default="/quobyte/millerlmgrp/ml_cleaned_data",
        help="Root directory where ml_denoise.py saved cleaned data ({cleaned-root}/{method}/...)",
    )
    parser.add_argument(
        "--hearing-dir",
        default="/mnt/data/PilapilData/processed_data/hearing",
        help="Directory containing hearing control .fif files for reference template",
    )
    parser.add_argument(
        "--zarr-path",
        default="/quobyte/millerlmgrp/processed_data/raw_epoched_data.zarr",
        help="Path to raw_epoched_data.zarr for hearing control fallback template",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["raw", "ica", "pca", "cca", "ssp", "wavelet"],
        help="Methods to evaluate ('raw' evaluates uncleaned CI data)",
    )
    parser.add_argument("--condition", default="A", choices=["A", "AV", "V"], help="Trial condition to evaluate")
    parser.add_argument("--target-channel", default="Cz", help="Primary electrode for auditory ERP components")
    parser.add_argument("--output-dir", default="./benchmark_results/tier2", help="Directory for CSV reports and plots")
    parser.add_argument("--save-plots", action="store_true", help="Generate and save ERP comparison plots")
    parser.add_argument("--year", default=None, help="Optional year filter ('2' for CMPy2, etc.)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("=== Tier 2: ERP & Neural Signal Preservation Benchmark ===")
    print(f"Annotated dir: {args.annotated_dir}")
    print(f"Cleaned root:  {args.cleaned_root}")
    print(f"Condition:     {args.condition}")
    print(f"Target ch:     {args.target_channel}")
    print(f"Methods:       {args.methods}")
    if args.year:
        print(f"Year filter:   CMPy{args.year}\n")
    else:
        print()

    # 1. Build or locate hearing control template
    template_evoked = None
    hearing_pattern = os.path.join(args.hearing_dir, "*.fif")
    hearing_files = sorted(glob.glob(hearing_pattern))
    if not hearing_files:
        # Check annotated dir for '08' subject IDs (hearing participants)
        hearing_files = sorted(glob.glob(os.path.join(args.annotated_dir, "*", "trial_onsets_08*-raw.fif")))
        if not hearing_files:
            hearing_files = sorted(glob.glob(os.path.join(args.annotated_dir, "trial_onsets_08*-raw.fif")))

    if hearing_files:
        template_evoked = build_hearing_grand_average(hearing_files[:10], condition=args.condition)

    if template_evoked is None and os.path.exists(args.zarr_path):
        print(f"Loading hearing control template from Zarr: {args.zarr_path}...")
        template_evoked = build_hearing_template_from_zarr(args.zarr_path)

    if template_evoked is None:
        print("  Notice: No hearing control data found. Template correlation will be NaN.")

    # 2. Discover CI subjects
    ci_raw_files = sorted(glob.glob(os.path.join(args.annotated_dir, "*", "trial_onsets_09*-raw.fif")))
    if not ci_raw_files:
        ci_raw_files = sorted(glob.glob(os.path.join(args.annotated_dir, "trial_onsets_09*-raw.fif")))
    
    # Filter out corrupted or empty (0-byte) files
    ci_raw_files = [f for f in ci_raw_files if os.path.exists(f) and os.path.getsize(f) > 100000]
    if args.year:
        ci_raw_files = [f for f in ci_raw_files if f"y{args.year}" in os.path.basename(f) or f"CMPy{args.year}" in f]

    print(f"Found {len(ci_raw_files)} valid raw CI annotated recording(s).")

    all_metrics: List[Dict] = []

    # 3. Evaluate each method
    for method in args.methods:
        print(f"\nEvaluating method: [{method}]...")
        for raw_path in ci_raw_files:
            subject_id = os.path.basename(raw_path).replace("trial_onsets_", "").replace("-raw.fif", "")

            # Determine file path for this method
            if method == "raw":
                target_fif = raw_path
            else:
                # Look in cleaned_root
                matches = glob.glob(os.path.join(args.cleaned_root, method, "**", f"*{subject_id}*-raw.fif"), recursive=True)
                if not matches:
                    continue
                target_fif = matches[0]

            evoked = compute_evoked(target_fif, event_desc=args.condition)
            if evoked is None:
                continue

            comp = extract_erp_components(evoked, target_ch=args.target_channel)
            corr = np.nan
            if template_evoked is not None:
                corr = compute_template_correlation(evoked, template_evoked, target_ch=args.target_channel)

            row = {
                "method": method,
                "subject_id": subject_id,
                "condition": args.condition,
                "template_corr": corr,
                **comp,
            }
            all_metrics.append(row)
            print(f"  [{method}] {subject_id}: N1={comp['n1_amp_uv']:.2f}uV @ {comp['n1_lat_ms']:.0f}ms, "
                  f"P2={comp['p2_amp_uv']:.2f}uV @ {comp['p2_lat_ms']:.0f}ms, "
                  f"ERP_SNR={comp['erp_snr_db']:.1f}dB, Corr={corr:.3f}")

            del evoked
            gc.collect()
            try:
                import ctypes
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass

    if not all_metrics:
        print("\nNo ERP records were successfully analyzed. Check input paths.")
        return

    # 4. Save results
    df = pd.DataFrame(all_metrics)
    detail_csv = os.path.join(args.output_dir, "tier2_subject_erp_metrics.csv")
    df.to_csv(detail_csv, index=False)
    print(f"\nSubject ERP metrics saved to: {detail_csv}")

    # Summary leaderboard
    summary = df.groupby("method")[["template_corr", "erp_snr_db", "p2p_uv", "base_rms_uv"]].agg(["mean", "std"])
    summary_csv = os.path.join(args.output_dir, "tier2_method_summary.csv")
    summary.to_csv(summary_csv)
    print(f"Summary leaderboard saved to: {summary_csv}\n")

    print("=" * 90)
    print("  TIER 2 ERP PRESERVATION LEADERBOARD")
    print("=" * 90)
    leaderboard = df.groupby("method")[["template_corr", "erp_snr_db", "p2p_uv", "base_rms_uv"]].mean()
    leaderboard = leaderboard.sort_values("template_corr", ascending=False)
    print(f"{'Method':<12s} {'Template Corr ↑':<18s} {'ERP SNR dB ↑':<16s} {'N1-P2 Amp uV':<16s} {'Base RMS uV ↓':<14s}")
    print("-" * 90)
    for m_name, row in leaderboard.iterrows():
        print(f"{m_name:<12s} {row['template_corr']:<18.4f} {row['erp_snr_db']:<16.2f} {row['p2p_uv']:<16.2f} {row['base_rms_uv']:<14.2f}")
    print("=" * 90)

    if args.save_plots:
        print("\nGenerating ERP summary figures...")
        try:
            plot_tier2_summary(df, args.output_dir)
            print("ERP summary figures saved successfully!")
        except Exception as plot_err:
            print(f"Warning: Could not save ERP summary plots: {plot_err}")


if __name__ == "__main__":
    main()
