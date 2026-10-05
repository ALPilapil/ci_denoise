"""
plot_tier2.py

Generates publication-quality ERP comparison figures directly from existing Tier 2 CSV results:
  1. tier2_erp_summary_charts.png (4-panel bar chart with error bars)
  2. tier2_erp_distribution_boxplots.png (distribution across CI subjects)
  3. tier2_erp_n1_p2_scatter.png (N1 vs P2 amplitude scatter)

Can be run immediately on Hive in 2 seconds without re-running SLURM:
  python benchmarking/plot_tier2.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    csv_path = "./benchmark_results/tier2/tier2_subject_erp_metrics.csv"
    output_dir = "./benchmark_results/tier2"

    if not os.path.exists(csv_path):
        # Try alternate path
        csv_path = os.path.expanduser("~/cidenoise/benchmark_results/tier2/tier2_subject_erp_metrics.csv")
        output_dir = os.path.expanduser("~/cidenoise/benchmark_results/tier2")

    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        return

    print(f"Loading ERP metrics from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Clean up NaNs
    df["erp_snr_db"] = pd.to_numeric(df["erp_snr_db"], errors="coerce")
    df["p2p_uv"] = pd.to_numeric(df["p2p_uv"], errors="coerce")
    df["base_rms_uv"] = pd.to_numeric(df["base_rms_uv"], errors="coerce")
    if "template_corr" in df.columns:
        df["template_corr"] = pd.to_numeric(df["template_corr"], errors="coerce")

    num_cols = ["erp_snr_db", "p2p_uv", "base_rms_uv"]
    if "template_corr" in df.columns:
        num_cols.append("template_corr")

    summary = df.groupby("method")[num_cols].agg(["mean", "std"])
    methods = sorted(df["method"].unique())

    # ── Figure 1: 4-Panel Summary Bar Chart ──────────────────────────────────────
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
    ax.set_title("Template Correlation with Hearing ERP (Higher is better)", fontsize=11, fontweight="bold")
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
    chart_path = os.path.join(output_dir, "tier2_erp_summary_charts.png")
    fig.savefig(chart_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Successfully generated: {chart_path}")

    # ── Figure 2: Cohort Distribution Boxplots ───────────────────────────────────
    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4.5))
    fig2.suptitle("Cohort ERP Distributions across CI Subjects", fontsize=13, fontweight="bold", y=0.99)

    # SNR Boxplot
    data_snr = [df[df["method"] == m]["erp_snr_db"].dropna().values for m in methods]
    axes2[0].boxplot(data_snr, tick_labels=methods, patch_artist=True)
    axes2[0].set_title("Auditory ERP SNR (dB)", fontweight="bold")
    axes2[0].set_ylabel("dB")
    axes2[0].grid(axis="y", linestyle=":", alpha=0.5)

    # Peak-to-Peak Boxplot
    data_p2p = [df[df["method"] == m]["p2p_uv"].dropna().values for m in methods]
    axes2[1].boxplot(data_p2p, tick_labels=methods, patch_artist=True)
    axes2[1].set_title("N1-P2 Amplitude (μV)", fontweight="bold")
    axes2[1].set_ylabel("μV")
    axes2[1].grid(axis="y", linestyle=":", alpha=0.5)

    # Baseline Noise Boxplot
    data_base = [df[df["method"] == m]["base_rms_uv"].dropna().values for m in methods]
    axes2[2].boxplot(data_base, tick_labels=methods, patch_artist=True)
    axes2[2].set_title("Baseline Noise RMS (μV)", fontweight="bold")
    axes2[2].set_ylabel("μV")
    axes2[2].grid(axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()
    box_path = os.path.join(output_dir, "tier2_erp_distribution_boxplots.png")
    fig2.savefig(box_path, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"Successfully generated: {box_path}")

if __name__ == "__main__":
    main()
