"""
tier1_screening.py

Tier 1: Fast Screening of CI Denoising Methods on Synthetic Mixtures.

Loads pairs of clean (hearing) and noise (CI) epochs from Zarr storage, mixes them
(dirty = clean + alpha * noise), applies candidate methods, and computes objective
reconstruction and artifact-suppression metrics:
  - RRMSE (Relative Root Mean Squared Error)
  - SNR Improvement (Delta SNR in dB)
  - Artifact Suppression Ratio (dB)
  - Pearson Correlation with clean ground-truth
  - Per-channel distortion & execution latency

Designed for execution on the Hive cluster (low overhead, memory-safe, unbuffered stdout).
"""

import argparse
import gc
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import mne
import numpy as np
import pandas as pd
from scipy import signal
import zarr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure unbuffered output so SLURM .out logs show real-time progress
try:
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
except Exception:
    pass

# Standard channel list used across the project
CH_NAMES_21 = [
    'Fp1', 'Fz', 'F3', 'F7', 'T7', 'C3', 'Cz', 'Pz', 'P3', 'P7', 'O1',
    'Fp2', 'F8', 'F4', 'C4', 'T8', 'P8', 'P4', 'O2', 'M1', 'M2'
]
CI_CHANNELS = ['P7', 'T7', 'M2', 'M1', 'P8']


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(
    clean: np.ndarray,
    noise: np.ndarray,
    dirty: np.ndarray,
    denoised: np.ndarray,
    duration_s: float
) -> Dict[str, float]:
    """
    Compute rigorous objective metrics for an epoch (shape: channels, timepoints).
    """
    eps = 1e-12

    # Residual error
    err = denoised - clean

    # Power computations
    p_clean = np.mean(clean ** 2)
    p_noise = np.mean(noise ** 2)
    p_err = np.mean(err ** 2)

    # 1. RRMSE: relative error to clean signal
    rrmse = np.sqrt(p_err) / (np.sqrt(p_clean) + eps)

    # 2. Input and Output SNR
    snr_in = 10.0 * np.log10((p_clean + eps) / (p_noise + eps))
    snr_out = 10.0 * np.log10((p_clean + eps) / (p_err + eps))
    delta_snr = snr_out - snr_in

    # 3. Artifact Suppression (dB): how much the added artifact was attenuated
    suppression_db = 10.0 * np.log10((p_noise + eps) / (p_err + eps))

    # 4. Pearson correlation across channels and time
    c_flat = clean.ravel()
    d_flat = denoised.ravel()
    std_c = np.std(c_flat)
    std_d = np.std(d_flat)
    if std_c > eps and std_d > eps:
        corr = float(np.corrcoef(c_flat, d_flat)[0, 1])
    else:
        corr = 0.0

    # 5. Channel-specific RRMSE for CI-proximal channels vs scalp channels
    ch_map = {name: i for i, name in enumerate(CH_NAMES_21[:clean.shape[0]])}
    ci_indices = [ch_map[ch] for ch in CI_CHANNELS if ch in ch_map]
    other_indices = [i for i in range(clean.shape[0]) if i not in ci_indices]

    ci_rrmse = 0.0
    if ci_indices:
        p_err_ci = np.mean(err[ci_indices] ** 2)
        p_clean_ci = np.mean(clean[ci_indices] ** 2)
        ci_rrmse = np.sqrt(p_err_ci) / (np.sqrt(p_clean_ci) + eps)

    scalp_rrmse = 0.0
    if other_indices:
        p_err_other = np.mean(err[other_indices] ** 2)
        p_clean_other = np.mean(clean[other_indices] ** 2)
        scalp_rrmse = np.sqrt(p_err_other) / (np.sqrt(p_clean_other) + eps)

    return {
        "rrmse": float(rrmse),
        "snr_in_db": float(snr_in),
        "snr_out_db": float(snr_out),
        "delta_snr_db": float(delta_snr),
        "suppression_db": float(suppression_db),
        "correlation": float(corr),
        "ci_ch_rrmse": float(ci_rrmse),
        "scalp_ch_rrmse": float(scalp_rrmse),
        "time_ms": float(duration_s * 1000.0),
    }


# ─── Method Runners ───────────────────────────────────────────────────────────

def create_mne_raw(data: np.ndarray, sfreq: float) -> mne.io.RawArray:
    """Wraps (n_channels, n_times) in an MNE RawArray with 10-20 montage."""
    n_ch = data.shape[0]
    ch_names = CH_NAMES_21[:n_ch] if n_ch <= len(CH_NAMES_21) else [f"EEG{i+1}" for i in range(n_ch)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg', verbose=False)
    raw = mne.io.RawArray(data, info, verbose=False)
    try:
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, on_missing='ignore', verbose=False)
    except Exception:
        pass
    return raw


def run_classical_method(method_obj, dirty_epoch: np.ndarray, sfreq: float) -> np.ndarray:
    """Runs a classical method instance from processing/ml_denoise.py on a single epoch."""
    raw = create_mne_raw(dirty_epoch, sfreq)
    clean_raw, _ = method_obj.fit_transform(raw)
    out_data = clean_raw.get_data()
    del raw, clean_raw
    return out_data


def load_unet_model(checkpoint_path: str, device: str):
    """Loads UNet from modeling/unet.py."""
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'modeling')))
    try:
        import torch
        from unet import UNet
        model = UNet().to(device)
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        print(f"Warning: Failed to load UNet model from {checkpoint_path}: {e}")
        return None


def run_unet(model, dirty_epoch: np.ndarray, device: str) -> np.ndarray:
    import torch
    inp = torch.tensor(dirty_epoch, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(inp)
        min_t = min(out.shape[-1], dirty_epoch.shape[-1])
        denoised = out[0, :, :min_t].cpu().numpy()
    return denoised


# ─── Visualization Functions ─────────────────────────────────────────────────

def plot_tier1_leaderboard(df: pd.DataFrame, output_dir: str):
    """Plots bar charts of RRMSE, Delta SNR, Artifact Suppression, and Correlation."""
    methods = sorted(df["method"].unique())
    summary = df.groupby("method").agg(["mean", "std"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Tier 1 Screening: Method Comparison Summary", fontsize=14, fontweight="bold", y=0.98)

    palette = plt.cm.tab10(np.linspace(0, 1, max(len(methods), 3)))
    color_map = {m: palette[i] for i, m in enumerate(methods)}

    # 1. RRMSE
    ax = axes[0, 0]
    vals = [summary.loc[m, ("rrmse", "mean")] for m in methods]
    errs = [summary.loc[m, ("rrmse", "std")] for m in methods]
    bars = ax.bar(methods, vals, yerr=errs, capsize=5, color=[color_map[m] for m in methods], alpha=0.85)
    ax.set_title("Relative RMSE (Lower is better)", fontsize=11, fontweight="bold")
    ax.set_ylabel("RRMSE")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # 2. Delta SNR
    ax = axes[0, 1]
    vals = [summary.loc[m, ("delta_snr_db", "mean")] for m in methods]
    errs = [summary.loc[m, ("delta_snr_db", "std")] for m in methods]
    ax.bar(methods, vals, yerr=errs, capsize=5, color=[color_map[m] for m in methods], alpha=0.85)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("SNR Improvement (Higher is better)", fontsize=11, fontweight="bold")
    ax.set_ylabel("ΔSNR (dB)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # 3. Artifact Suppression
    ax = axes[1, 0]
    vals = [summary.loc[m, ("suppression_db", "mean")] for m in methods]
    errs = [summary.loc[m, ("suppression_db", "std")] for m in methods]
    ax.bar(methods, vals, yerr=errs, capsize=5, color=[color_map[m] for m in methods], alpha=0.85)
    ax.set_title("Artifact Suppression (Higher is better)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Suppression (dB)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # 4. Correlation
    ax = axes[1, 1]
    vals = [summary.loc[m, ("correlation", "mean")] for m in methods]
    errs = [summary.loc[m, ("correlation", "std")] for m in methods]
    ax.bar(methods, vals, yerr=errs, capsize=5, color=[color_map[m] for m in methods], alpha=0.85)
    ax.set_title("Pearson Correlation with Clean Ground-Truth", fontsize=11, fontweight="bold")
    ax.set_ylabel("Correlation (r)")
    ax.set_ylim([0.0, 1.05])
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "tier1_metric_leaderboard.png")
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved leaderboard chart to:  {plot_path}")


def plot_waveform_comparison(
    clean: np.ndarray,
    dirty: np.ndarray,
    denoised_dict: Dict[str, np.ndarray],
    sfreq: float,
    output_dir: str,
    duration_s: float = 3.0,
):
    """
    Plots stacked/overlaid multi-channel EEG waveforms comparing Clean, Dirty, and Denoised outputs,
    matching the comparison style in shyam/pipeline.ipynb.
    """
    n_samples = min(int(duration_s * sfreq), clean.shape[-1])
    time_s = np.arange(n_samples) / sfreq

    # Channels to display: Vertex (Cz), Frontal (Fz), CI-proximal (T7, P7)
    display_chs = ['Cz', 'Fz', 'P7', 'T7']
    available_chs = [ch for ch in display_chs if ch in CH_NAMES_21[:clean.shape[0]]]
    if not available_chs:
        available_chs = [CH_NAMES_21[i] for i in range(min(4, clean.shape[0]))]

    n_rows = len(available_chs)
    fig, axes = plt.subplots(n_rows, 1, figsize=(15, 2.8 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    fig.suptitle(f"EEG Waveform Denoising Comparison (Sample Epoch, first {duration_s}s)", fontsize=13, fontweight="bold", y=0.995)

    method_colors = plt.cm.tab10(np.linspace(0, 1, max(len(denoised_dict), 3)))

    for i, ch_name in enumerate(available_chs):
        ch_idx = CH_NAMES_21.index(ch_name)
        ax = axes[i]

        # Convert to uV for standard clinical EEG visualization
        c_uV = clean[ch_idx, :n_samples] * 1e6
        d_uV = dirty[ch_idx, :n_samples] * 1e6

        # Plot Dirty input (light gray)
        ax.plot(time_s, d_uV, color="#b0b0b0", linewidth=1.0, alpha=0.7, label="Dirty Input (Clean + CI Noise)")

        # Plot Clean ground truth (thick green line)
        ax.plot(time_s, c_uV, color="#2ca02c", linewidth=2.0, linestyle="--", label="Clean Ground Truth (Hearing)")

        # Plot each denoised candidate method
        for m_idx, (m_name, m_data) in enumerate(denoised_dict.items()):
            min_t = min(n_samples, m_data.shape[-1])
            m_uV = m_data[ch_idx, :min_t] * 1e6
            ax.plot(time_s[:min_t], m_uV, color=method_colors[m_idx], linewidth=1.2, alpha=0.9, label=f"Denoised: {m_name}")

        ax.set_ylabel(f"{ch_name} (μV)", fontsize=10, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.5)

        if i == 0:
            ax.legend(loc="upper right", ncol=min(3, len(denoised_dict) + 2), fontsize=8, framealpha=0.9)

    axes[-1].set_xlabel("Time (seconds)", fontsize=11)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "tier1_eeg_waveform_comparison.png")
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved waveform plot to:      {plot_path}")


def plot_psd_comparison(
    clean: np.ndarray,
    dirty: np.ndarray,
    denoised_dict: Dict[str, np.ndarray],
    sfreq: float,
    output_dir: str,
):
    """
    Plots Power Spectral Density (PSD) comparing Clean, Dirty, and Denoised spectra.
    Shows artifact attenuation at high frequencies while preserving 1-30 Hz neural bands.
    """
    display_chs = ['Cz', 'P7']
    available_chs = [ch for ch in display_chs if ch in CH_NAMES_21[:clean.shape[0]]]
    if not available_chs:
        available_chs = [CH_NAMES_21[0]]

    fig, axes = plt.subplots(1, len(available_chs), figsize=(7 * len(available_chs), 4.5))
    if len(available_chs) == 1:
        axes = [axes]

    fig.suptitle("Power Spectral Density (PSD) Comparison (Welch)", fontsize=13, fontweight="bold", y=0.99)
    method_colors = plt.cm.tab10(np.linspace(0, 1, max(len(denoised_dict), 3)))

    for i, ch_name in enumerate(available_chs):
        ch_idx = CH_NAMES_21.index(ch_name)
        ax = axes[i]

        nperseg = min(clean.shape[-1], int(sfreq * 2.0))
        freqs, psd_clean = signal.welch(clean[ch_idx], fs=sfreq, nperseg=nperseg)
        _, psd_dirty = signal.welch(dirty[ch_idx], fs=sfreq, nperseg=nperseg)

        # Plot 10 log10 PSD (dB/Hz)
        ax.plot(freqs, 10 * np.log10(psd_dirty + 1e-18), color="#b0b0b0", linewidth=1.2, label="Dirty Input")
        ax.plot(freqs, 10 * np.log10(psd_clean + 1e-18), color="#2ca02c", linewidth=2.0, linestyle="--", label="Clean (Hearing)")

        for m_idx, (m_name, m_data) in enumerate(denoised_dict.items()):
            min_t = min(clean.shape[-1], m_data.shape[-1])
            f_m, psd_m = signal.welch(m_data[ch_idx, :min_t], fs=sfreq, nperseg=nperseg)
            ax.plot(f_m, 10 * np.log10(psd_m + 1e-18), color=method_colors[m_idx], linewidth=1.2, label=f"{m_name}")

        ax.set_title(f"Channel {ch_name}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Frequency (Hz)", fontsize=10)
        ax.set_ylabel("Power Spectral Density (dB / Hz)", fontsize=10)
        ax.set_xlim([1.0, min(100.0, sfreq / 2.0)])
        ax.grid(True, linestyle=":", alpha=0.5)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "tier1_psd_spectrum_comparison.png")
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved PSD comparison plot to: {plot_path}")


# ─── Main Screening Pipeline ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Tier 1: Screening CI Denoising Methods on Synthetic Mixtures")
    parser.add_argument(
        "--zarr-path",
        default="/quobyte/millerlmgrp/processed_data/raw_epoched_data.zarr",
        help="Path to raw_epoched_data.zarr or epoched_data.zarr",
    )
    parser.add_argument("--clean-group", default="hearing_trial_data", help="Clean dataset group name")
    parser.add_argument("--noise-group", default="ci_trial_data", help="Noise dataset group name")
    parser.add_argument("--output-dir", default="./benchmark_results/tier1", help="Output directory for reports")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["ica", "pca", "cca", "ssp", "wavelet"],
        help="Methods to test (from ml_denoise: ica, pca, cca, ssp, wavelet, iva, emd_ica, emd_cca, unet)",
    )
    parser.add_argument("--unet-checkpoint", default=None, help="Path to best.pt for unet testing")
    parser.add_argument("--num-epochs", type=int, default=30, help="Number of benchmark epoch pairs to evaluate")
    parser.add_argument("--alpha", type=float, default=1.0, help="Noise scaling factor (clean + alpha * noise)")
    parser.add_argument("--sfreq", type=float, default=250.0, help="Sampling frequency (Hz)")
    parser.add_argument("--device", default="cpu", help="Compute device ('cpu' or 'cuda')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for epoch pairing")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"=== Tier 1 Screening: Benchmarking Candidate Methods ===")
    print(f"Zarr input:  {args.zarr_path}")
    print(f"Methods:     {args.methods}")
    print(f"Num epochs:  {args.num_epochs}")
    print(f"Alpha:       {args.alpha}")
    print(f"Output dir:  {args.output_dir}\n")

    # Add project root and processing directory to sys.path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    proj_root = os.path.abspath(os.path.join(script_dir, '..'))
    processing_dir = os.path.join(proj_root, 'processing')
    for p in [proj_root, processing_dir, os.getcwd(), os.path.expanduser('~/cidenoise')]:
        if os.path.exists(p) and p not in sys.path:
            sys.path.insert(0, p)

    try:
        from processing.ml_denoise import METHOD_REGISTRY
    except ModuleNotFoundError:
        try:
            from ml_denoise import METHOD_REGISTRY
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(f"Could not import METHOD_REGISTRY. sys.path is: {sys.path}") from e

    # Load Zarr
    if not os.path.exists(args.zarr_path):
        raise FileNotFoundError(f"Zarr store not found at {args.zarr_path}")

    root = zarr.open(args.zarr_path, mode='r')
    clean_arr = root[args.clean_group]['data']
    noise_arr = root[args.noise_group]['data']

    total_clean = clean_arr.shape[0]
    total_noise = noise_arr.shape[0]
    print(f"Available clean epochs: {total_clean}, noise epochs: {total_noise}")

    # Build reproducible pairs
    rng = np.random.default_rng(args.seed)
    n_eval = min(args.num_epochs, total_clean, total_noise)
    clean_indices = rng.choice(total_clean, size=n_eval, replace=False)
    noise_indices = rng.choice(total_noise, size=n_eval, replace=False)

    # Prepare UNet if requested
    unet_model = None
    import torch
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but torch.cuda.is_available() is False. Falling back to CPU.")
        device = "cpu"

    if "unet" in args.methods:
        if args.unet_checkpoint and os.path.exists(args.unet_checkpoint):
            print(f"Loading UNet on {device} from {args.unet_checkpoint}...")
            unet_model = load_unet_model(args.unet_checkpoint, device)
        else:
            print(f"Warning: 'unet' requested but checkpoint not found at: {args.unet_checkpoint}. Skipping UNet.")

    all_results: List[Dict] = []
    sample_clean: Optional[np.ndarray] = None
    sample_dirty: Optional[np.ndarray] = None
    sample_denoised: Dict[str, np.ndarray] = {}

    # Evaluation loop
    for pair_idx in range(n_eval):
        c_idx = int(clean_indices[pair_idx])
        n_idx = int(noise_indices[pair_idx])

        clean = np.array(clean_arr[c_idx], dtype=np.float64)
        noise = np.array(noise_arr[n_idx], dtype=np.float64)

        # Check for NaNs
        if np.isnan(clean).any() or np.isnan(noise).any():
            print(f"  Skipping pair {pair_idx} (contains NaNs)")
            continue

        # Match dimensions if needed
        min_ch = min(clean.shape[0], noise.shape[0])
        min_t = min(clean.shape[1], noise.shape[1])
        clean = clean[:min_ch, :min_t]
        noise = noise[:min_ch, :min_t]
        scaled_noise = args.alpha * noise
        dirty = clean + scaled_noise

        is_sample_pair = (sample_clean is None)
        if is_sample_pair:
            sample_clean = clean.copy()
            sample_dirty = dirty.copy()

        print(f"Processing epoch pair {pair_idx+1}/{n_eval} (clean #{c_idx}, noise #{n_idx})...")

        for method_name in args.methods:
            t0 = time.time()
            try:
                if method_name == "unet":
                    if unet_model is None:
                        continue
                    denoised = run_unet(unet_model, dirty, device)
                else:
                    if method_name not in METHOD_REGISTRY:
                        print(f"  Unknown method '{method_name}', skipping")
                        continue
                    method_cls = METHOD_REGISTRY[method_name]
                    method_inst = method_cls()
                    denoised = run_classical_method(method_inst, dirty, args.sfreq)

                if is_sample_pair and method_name not in sample_denoised:
                    sample_denoised[method_name] = denoised.copy()

                dt = time.time() - t0
                m = compute_metrics(clean, scaled_noise, dirty, denoised, dt)
                m.update({
                    "pair_idx": pair_idx,
                    "clean_idx": c_idx,
                    "noise_idx": n_idx,
                    "method": method_name,
                })
                all_results.append(m)
                print(f"  [{method_name:<8s}] RRMSE={m['rrmse']:.4f}  ΔSNR={m['delta_snr_db']:+.2f}dB  "
                      f"Suppr={m['suppression_db']:.2f}dB  Corr={m['correlation']:.4f}  ({m['time_ms']:.1f}ms)")
            except Exception as e:
                print(f"  [{method_name}] ERROR: {e}")

        gc.collect()

    if not all_results:
        print("No successful results computed.")
        return

    # Save detailed CSV
    df = pd.DataFrame(all_results)
    detail_csv = os.path.join(args.output_dir, "tier1_epoch_metrics.csv")
    df.to_csv(detail_csv, index=False)
    print(f"\nDetailed metrics saved to: {detail_csv}")

    # Generate and print aggregated summary
    summary_cols = ["rrmse", "delta_snr_db", "suppression_db", "correlation", "ci_ch_rrmse", "scalp_ch_rrmse", "time_ms"]
    summary = df.groupby("method")[summary_cols].agg(["mean", "std"])
    summary_csv = os.path.join(args.output_dir, "tier1_summary.csv")
    summary.to_csv(summary_csv)
    print(f"Summary table saved to:    {summary_csv}\n")

    print("=" * 90)
    print("  TIER 1 SCREENING LEADERBOARD (Mean ± Std)")
    print("=" * 90)
    leaderboard = df.groupby("method")[["rrmse", "delta_snr_db", "suppression_db", "correlation", "time_ms"]].mean()
    leaderboard = leaderboard.sort_values("rrmse", ascending=True)
    print(f"{'Method':<12s} {'RRMSE (lower)':<16s} {'ΔSNR dB (higher)':<18s} {'Suppr dB (higher)':<18s} {'Correlation':<14s} {'Time/Epoch':<12s}")
    print("-" * 90)
    for m_name, row in leaderboard.iterrows():
        print(f"{m_name:<12s} {row['rrmse']:<16.4f} {row['delta_snr_db']:<+18.2f} {row['suppression_db']:<18.2f} {row['correlation']:<14.4f} {row['time_ms']:<10.1f}ms")
    print("=" * 90)

    # ── Generate publication-quality comparison figures ────────────────────────
    print("\nGenerating comparison figures (matching pipeline.ipynb style)...")
    try:
        plot_tier1_leaderboard(df, args.output_dir)
        if sample_clean is not None and sample_dirty is not None and sample_denoised:
            plot_waveform_comparison(sample_clean, sample_dirty, sample_denoised, args.sfreq, args.output_dir)
            plot_psd_comparison(sample_clean, sample_dirty, sample_denoised, args.sfreq, args.output_dir)
        print("All comparison figures generated successfully!")
    except Exception as fig_err:
        print(f"Warning: Could not generate some figures: {fig_err}")


if __name__ == "__main__":
    main()
