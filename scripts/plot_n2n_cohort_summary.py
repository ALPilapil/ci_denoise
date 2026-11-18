#!/usr/bin/env python3
"""
plot_n2n_cohort_summary.py

Generates publication-quality diagnostic visualizations for the SOBI + Symmetric N2N
pipeline across the entire N=36 clinical CI cohort.
"""

import os
import glob
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Plot N2N Cohort Evaluation Summary")
    parser.add_argument("--cohort-dir", type=str, default="./benchmark_results/sobi_n2n_cohort")
    parser.add_argument("--output-png", type=str, default=None)
    parser.add_argument("--title-suffix", type=str, default="")
    return parser.parse_args()

def main():
    args = parse_args()
    cohort_dir = args.cohort_dir
    output_png = args.output_png or os.path.join(cohort_dir, "sobi_n2n_cohort_validation_figure.png")
    
    files = sorted(glob.glob(os.path.join(cohort_dir, "sobi_n2n_eval_*.json")))
    records = []
    for f in files:
        if "summary" in f:
            continue
        with open(f, 'r') as jf:
            d = json.load(jf)
        if d.get('evaluated'):
            records.append(d)
            
    print(f"Loaded {len(records)} valid CI cohort evaluation records.")
    
    atten = np.array([r['p7_additional_attenuation_db'] for r in records])
    cz_r = np.array([r['cz_template_r'] for r in records])
    v_floors = np.array([r['v_cz_noise_floor_uv'] for r in records])
    p2_lat = np.array([r['p2_latency_ms'] for r in records])
    p2_amp = np.array([r['p2_amplitude_uv'] for r in records])
    
    mean_att = float(np.mean(atten))
    med_att = float(np.median(atten))
    mean_v = float(np.mean(v_floors))
    
    # Setup Figure
    fig = plt.figure(figsize=(15, 11), facecolor="#ffffff")
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)
    
    # -------------------------------------------------------------
    # Panel A: P7 Additional Artifact Power Attenuation (dB)
    # -------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.hist(atten, bins=12, color="#1f77b4", edgecolor="black", alpha=0.7, density=True)
    ax_a.axvline(mean_att, color="red", linestyle="--", lw=2, label=f"Mean: {mean_att:+.2f} dB")
    ax_a.axvline(med_att, color="darkgreen", linestyle=":", lw=2, label=f"Median: {med_att:+.2f} dB")
    ax_a.axvline(0.0, color="black", linestyle="-", lw=1.2, alpha=0.6)
    ax_a.set_title("A. Cohort Residual Coil Attenuation Beyond SOBI (P7)", fontweight="bold", fontsize=12)
    ax_a.set_xlabel("Additional Artifact Attenuation (dB) [Negative = Cleaner]", fontsize=10)
    ax_a.set_ylabel("Probability Density", fontsize=10)
    ax_a.legend(loc="upper left", frameon=True, fontsize=9)
    ax_a.grid(True, alpha=0.25)
    
    # -------------------------------------------------------------
    # Panel B: Midline Biological Fidelity (r at Cz)
    # -------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.scatter(range(len(records)), cz_r, color="#2ca02c", s=45, edgecolors="black", alpha=0.85)
    ax_b.axhline(0.70, color="crimson", linestyle="--", lw=1.5, label="Dr. Vance Fidelity Gate (r >= 0.70)")
    ax_b.axhline(1.0, color="gray", linestyle=":", lw=1.0)
    ax_b.set_ylim(0.60, 1.02)
    ax_b.set_title("B. Vertex Biological Preservation (Cz Pearson r vs SOBI)", fontweight="bold", fontsize=12)
    ax_b.set_xlabel("CI Subject Index (1 to 36)", fontsize=10)
    ax_b.set_ylabel("Pearson Correlation r", fontsize=10)
    ax_b.legend(loc="lower left", frameon=True, fontsize=9)
    ax_b.grid(True, alpha=0.25)
    
    # -------------------------------------------------------------
    # Panel C: Condition V Silent Visual Floor at Cz
    # -------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.bar(range(len(records)), v_floors, color="#9467bd", edgecolor="black", alpha=0.75, width=0.7)
    ax_c.axhline(2.0, color="crimson", linestyle="--", lw=2, label="Dr. Vance Silent Gate (< 2.0 uV)")
    ax_c.axhline(mean_v, color="blue", linestyle=":", lw=2, label=f"Cohort Mean: {mean_v:.2f} uV")
    ax_c.set_title("C. Silent Visual Condition (V) Noise Floor at Cz", fontweight="bold", fontsize=12)
    ax_c.set_xlabel("CI Subject Index (1 to 36)", fontsize=10)
    ax_c.set_ylabel("RMS Amplitude (uV)", fontsize=10)
    ax_c.legend(loc="upper right", frameon=True, fontsize=9)
    ax_c.grid(True, alpha=0.25, axis="y")
    
    # -------------------------------------------------------------
    # Panel D: Auditory P2 Cortical Potential (Latency vs Amplitude)
    # -------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])
    sc = ax_d.scatter(p2_lat, p2_amp, c=atten, cmap="coolwarm", s=55, edgecolors="black", alpha=0.85)
    cbar = plt.colorbar(sc, ax=ax_d)
    cbar.set_label("P7 Attenuation (dB)", fontsize=9)
    ax_d.axvspan(150.0, 240.0, color="green", alpha=0.12, label="Canonical P2 Window [150-240 ms]")
    ax_d.set_title("D. Auditory P2 Latency & Evoked Amplitude", fontweight="bold", fontsize=12)
    ax_d.set_xlabel("P2 Latency (ms)", fontsize=10)
    ax_d.set_ylabel("P2 Peak Amplitude (uV)", fontsize=10)
    ax_d.legend(loc="upper left", frameon=True, fontsize=9)
    ax_d.grid(True, alpha=0.25)
    
    title_str = f"Clinical CI Cohort Validation: Two-Pass SOBI + Subspace-Constrained N2N (N=36){args.title_suffix}"
    plt.suptitle(title_str, fontsize=15, fontweight="bold", y=0.98)
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Diagnostic figure successfully saved to: {output_png}")

if __name__ == "__main__":
    main()
