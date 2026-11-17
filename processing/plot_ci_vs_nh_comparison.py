#!/usr/bin/env python3
"""
plot_ci_vs_nh_comparison.py
Generates publication-quality comparative figure directly contrasting
Cochlear Implant Patients (N = 37) against Normal-Hearing Controls (N = 15).
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_comparison_plot():
    ci_csv = "./benchmark_results/sobi_polarity_cohort/cohort_sobi_polarity_summary.csv"
    nh_csv = "./benchmark_results/sobi_polarity_nh_cohort/cohort_sobi_polarity_summary.csv"
    output_dir = "./benchmark_results/sobi_polarity_cohort"
    
    df_ci = pd.read_csv(ci_csv)
    df_nh = pd.read_csv(nh_csv)
    
    # Filter uncalibrated hardware sensor scaling outlier (0923y2)
    df_ci_clean = df_ci[(df_ci['silent_cz_floor_uv'] < 50.0) & (df_ci['p2_amplitude_uv'].abs() < 50.0)]
    df_nh_clean = df_nh[(df_nh['silent_cz_floor_uv'] < 50.0) & (df_nh['p2_amplitude_uv'].abs() < 50.0)]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    palette = {'NH': '#2b5c8f', 'CI': '#d95f02'}
    
    # -------------------------------------------------------------
    # Panel A: P2 Latency Comparison (Neuroplastic Latency Delay)
    # -------------------------------------------------------------
    ax = axes[0]
    box_data = [df_nh_clean['p2_latency_ms'], df_ci_clean['p2_latency_ms']]
    bp = ax.boxplot(box_data, patch_artist=True, labels=[f"Normal-Hearing\n(N={len(df_nh_clean)})", f"Cochlear Implant\n(N={len(df_ci_clean)})"],
                    medianprops=dict(color='black', linewidth=2),
                    widths=0.4)
    bp['boxes'][0].set_facecolor('#a6bddb')
    bp['boxes'][1].set_facecolor('#fdbb84')
    
    # Jitter scatter overlay
    np.random.seed(42)
    x0 = np.random.normal(1, 0.04, size=len(df_nh_clean))
    x1 = np.random.normal(2, 0.04, size=len(df_ci_clean))
    ax.scatter(x0, df_nh_clean['p2_latency_ms'], color=palette['NH'], alpha=0.7, edgecolors='k', s=45)
    ax.scatter(x1, df_ci_clean['p2_latency_ms'], color=palette['CI'], alpha=0.7, edgecolors='k', s=45)
    
    mean_nh_lat = df_nh_clean['p2_latency_ms'].mean()
    mean_ci_lat = df_ci_clean['p2_latency_ms'].mean()
    delta_lat = mean_ci_lat - mean_nh_lat
    
    ax.axhspan(150, 240, color='green', alpha=0.1, label='Canonical P2 Window (150-240 ms)')
    ax.set_title(f"A. Cortical P2 Latency (Δ = +{delta_lat:.1f} ms)", fontsize=11, fontweight='bold')
    ax.set_ylabel("P2 Peak Latency (ms)", fontsize=10)
    ax.set_ylim(135, 255)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=8)
    
    # -------------------------------------------------------------
    # Panel B: Evoked P2 Amplitude Preservation
    # -------------------------------------------------------------
    ax = axes[1]
    box_data_amp = [df_nh_clean['p2_amplitude_uv'], df_ci_clean['p2_amplitude_uv']]
    bp2 = ax.boxplot(box_data_amp, patch_artist=True, labels=[f"Normal-Hearing\n(N={len(df_nh_clean)})", f"Cochlear Implant\n(N={len(df_ci_clean)})"],
                     medianprops=dict(color='black', linewidth=2),
                     widths=0.4)
    bp2['boxes'][0].set_facecolor('#a6bddb')
    bp2['boxes'][1].set_facecolor('#fdbb84')
    
    ax.scatter(x0, df_nh_clean['p2_amplitude_uv'], color=palette['NH'], alpha=0.7, edgecolors='k', s=45)
    ax.scatter(x1, df_ci_clean['p2_amplitude_uv'], color=palette['CI'], alpha=0.7, edgecolors='k', s=45)
    
    mean_nh_amp = df_nh_clean['p2_amplitude_uv'].mean()
    mean_ci_amp = df_ci_clean['p2_amplitude_uv'].mean()
    retention_pct = (mean_ci_amp / mean_nh_amp) * 100.0
    
    ax.axhline(0, color='gray', linestyle=':', alpha=0.6)
    ax.set_title(f"B. Evoked P2 Amplitude ({retention_pct:.1f}% Preserved)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Cz P2 Amplitude (µV)", fontsize=10)
    ax.set_ylim(-1.5, 4.5)
    ax.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------
    # Panel C: Condition V Silent Baseline Floor
    # -------------------------------------------------------------
    ax = axes[2]
    box_data_sil = [df_nh_clean['silent_cz_floor_uv'], df_ci_clean['silent_cz_floor_uv']]
    bp3 = ax.boxplot(box_data_sil, patch_artist=True, labels=[f"Normal-Hearing\n(N={len(df_nh_clean)})", f"Cochlear Implant\n(N={len(df_ci_clean)})"],
                     medianprops=dict(color='black', linewidth=2),
                     widths=0.4)
    bp3['boxes'][0].set_facecolor('#a6bddb')
    bp3['boxes'][1].set_facecolor('#fdbb84')
    
    ax.scatter(x0, df_nh_clean['silent_cz_floor_uv'], color=palette['NH'], alpha=0.7, edgecolors='k', s=45)
    ax.scatter(x1, df_ci_clean['silent_cz_floor_uv'], color=palette['CI'], alpha=0.7, edgecolors='k', s=45)
    
    ax.axhline(2.0, color='red', linestyle='--', linewidth=1.5, label='Noise Floor Ceiling (< 2.0 µV)')
    mean_nh_sil = df_nh_clean['silent_cz_floor_uv'].mean()
    mean_ci_sil = df_ci_clean['silent_cz_floor_uv'].mean()
    
    ax.set_title(f"C. Silent Cz Floor (NH: {mean_nh_sil:.2f} µV | CI: {mean_ci_sil:.2f} µV)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Silent Cz RMS Noise Floor (µV)", fontsize=10)
    ax.set_ylim(0, 3.8)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "cohort_ci_vs_nh_comparative_summary.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved comparative summary figure to: {save_path}")

if __name__ == "__main__":
    generate_comparison_plot()
