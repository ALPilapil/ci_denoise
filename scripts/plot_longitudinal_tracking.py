#!/usr/bin/env python3
"""
scripts/plot_longitudinal_tracking.py

Publication-quality longitudinal neuroplastic tracking figure across Year 2, Year 3,
and Year 4 clinical CI cohorts.
"""

import os
import glob
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Y2_DIR = os.path.join(REPO_ROOT, "benchmark_results", "sobi_n2n_foundation_cohort")
Y3_DIR = os.path.join(REPO_ROOT, "benchmark_results", "sobi_n2n_cmpy3_cohort")
Y4_DIR = os.path.join(REPO_ROOT, "benchmark_results", "sobi_n2n_cmpy4_cohort")
OUTPUT_PNG = os.path.join(REPO_ROOT, "benchmark_results", "longitudinal_tracking_figure.png")

def main():
    y2_files = {os.path.basename(f).split('_')[-1][:4]: f for f in glob.glob(os.path.join(Y2_DIR, "sobi_n2n_eval_*.json")) if 'summary' not in f}
    y3_files = {os.path.basename(f).split('_')[-1][:4]: f for f in glob.glob(os.path.join(Y3_DIR, "sobi_n2n_eval_*.json")) if 'summary' not in f}
    y4_files = {os.path.basename(f).split('_')[-1][:4]: f for f in glob.glob(os.path.join(Y4_DIR, "sobi_n2n_eval_*.json")) if 'summary' not in f}

    matched = sorted(set(y2_files.keys()) & set(y3_files.keys()))
    
    y2_lats, y3_lats = [], []
    y2_amps, y3_amps = [], []
    valid_subs = []

    for sub in matched:
        with open(y2_files[sub]) as f2, open(y3_files[sub]) as f3:
            d2 = json.load(f2)
            d3 = json.load(f3)
        if d2.get('evaluated') and d3.get('evaluated'):
            y2_lats.append(d2['p2_latency_ms'])
            y3_lats.append(d3['p2_latency_ms'])
            y2_amps.append(d2['p2_amplitude_uv'])
            y3_amps.append(d3['p2_amplitude_uv'])
            valid_subs.append(sub)

    y2_lats = np.array(y2_lats)
    y3_lats = np.array(y3_lats)
    y2_amps = np.array(y2_amps)
    y3_amps = np.array(y3_amps)

    n_matched = len(valid_subs)
    print(f"Loaded {n_matched} paired Year 2 <-> Year 3 subjects.")

    # Multi-year tracking for 0938 (Y2, Y3, Y4)
    with open(y2_files['0938']) as f2, open(y3_files['0938']) as f3, open(y4_files['0938']) as f4:
        sub0938_y2 = json.load(f2)
        sub0938_y3 = json.load(f3)
        sub0938_y4 = json.load(f4)

    years_0938 = [2, 3, 4]
    amps_0938 = [sub0938_y2['p2_amplitude_uv'], sub0938_y3['p2_amplitude_uv'], sub0938_y4['p2_amplitude_uv']]
    lats_0938 = [sub0938_y2['p2_latency_ms'], sub0938_y3['p2_latency_ms'], sub0938_y4['p2_latency_ms']]

    # Setup 2x2 publication figure
    fig = plt.figure(figsize=(15, 11), facecolor="#ffffff")
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28)

    # -------------------------------------------------------------
    # Panel A: Paired P2 Latency Trajectory (Y2 vs Y3)
    # -------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.axhspan(160, 200, color="#2ca02c", alpha=0.15, label="Canonical Auditory P2 Range (160–200 ms)")
    ax_a.axhline(180, color="#2ca02c", linestyle="--", linewidth=1.2, alpha=0.7, label="Canonical Mean (180 ms)")

    for i in range(n_matched):
        ax_a.plot([2, 3], [y2_lats[i], y3_lats[i]], color="#4a5568", alpha=0.45, linewidth=1.2, marker="o", markersize=4)

    # Highlighting key examples
    sub_0901_idx = valid_subs.index('0901')
    ax_a.plot([2, 3], [y2_lats[sub_0901_idx], y3_lats[sub_0901_idx]], color="#e53e3e", linewidth=2.5, marker="o", markersize=7, label="Sub 0901 (209.9 -> 182.3 ms)")
    sub_0903_idx = valid_subs.index('0903')
    ax_a.plot([2, 3], [y2_lats[sub_0903_idx], y3_lats[sub_0903_idx]], color="#3182ce", linewidth=2.5, marker="s", markersize=7, label="Sub 0903 (229.6 -> 186.2 ms)")

    # Mean trajectory
    ax_a.plot([2, 3], [np.mean(y2_lats), np.mean(y3_lats)], color="#1a202c", linewidth=3.5, marker="D", markersize=9, label=f"Cohort Mean ({np.mean(y2_lats):.1f} -> {np.mean(y3_lats):.1f} ms)")

    ax_a.set_xticks([2, 3])
    ax_a.set_xticklabels(["Year 2 (CMPy2)", "Year 3 (CMPy3)"], fontsize=12, fontweight="bold")
    ax_a.set_ylabel("Cz Cortical P2 Latency (ms)", fontsize=12, fontweight="bold")
    ax_a.set_title("A. Cortical P2 Latency Dynamics (N=23 Paired Recipients)", fontsize=13, fontweight="bold", pad=10)
    ax_a.set_ylim(130, 260)
    ax_a.grid(True, linestyle=":", alpha=0.6)
    ax_a.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    # -------------------------------------------------------------
    # Panel B: Paired P2 Amplitude Trajectory (Y2 vs Y3)
    # -------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    for i in range(n_matched):
        ax_b.plot([2, 3], [y2_amps[i], y3_amps[i]], color="#4a5568", alpha=0.45, linewidth=1.2, marker="o", markersize=4)

    ax_b.plot([2, 3], [y2_amps[sub_0901_idx], y3_amps[sub_0901_idx]], color="#e53e3e", linewidth=2.5, marker="o", markersize=7, label="Sub 0901 (1.29 -> 1.59 uV)")
    ax_b.plot([2, 3], [y2_amps[sub_0903_idx], y3_amps[sub_0903_idx]], color="#3182ce", linewidth=2.5, marker="s", markersize=7, label="Sub 0903 (-1.89 -> +1.31 uV)")
    ax_b.plot([2, 3], [np.mean(y2_amps), np.mean(y3_amps)], color="#1a202c", linewidth=3.5, marker="D", markersize=9, label=f"Cohort Mean ({np.mean(y2_amps):.2f} -> {np.mean(y3_amps):.2f} uV)")

    ax_b.set_xticks([2, 3])
    ax_b.set_xticklabels(["Year 2 (CMPy2)", "Year 3 (CMPy3)"], fontsize=12, fontweight="bold")
    ax_b.set_ylabel("Cz Cortical P2 Amplitude (uV)", fontsize=12, fontweight="bold")
    ax_b.set_title("B. Cortical P2 Amplitude Dynamics (N=23 Paired Recipients)", fontsize=13, fontweight="bold", pad=10)
    ax_b.grid(True, linestyle=":", alpha=0.6)
    ax_b.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    # -------------------------------------------------------------
    # Panel C: Longitudinal Neuroplastic Recovery in Patient 0938 (3-Year Follow-up)
    # -------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.plot(years_0938, amps_0938, color="#805ad5", linewidth=3.0, marker="o", markersize=10, label="P2 Amplitude (uV)")
    for yr, amp in zip(years_0938, amps_0938):
        ax_c.annotate(f"{amp:.3f} uV", (yr, amp), textcoords="offset points", xytext=(0, 10), ha="center", fontweight="bold", color="#805ad5", fontsize=11)

    ax_c.set_xticks([2, 3, 4])
    ax_c.set_xticklabels(["Year 2", "Year 3", "Year 4"], fontsize=12, fontweight="bold")
    ax_c.set_ylabel("Cz Cortical P2 Amplitude (uV)", fontsize=12, fontweight="bold", color="#805ad5")
    ax_c.set_title("C. Multi-Year Neuroplastic Tracking: Patient 0938", fontsize=13, fontweight="bold", pad=10)
    ax_c.set_ylim(0.0, 1.1)
    ax_c.grid(True, linestyle=":", alpha=0.6)

    # Inset / twin axis for latency
    ax_c_lat = ax_c.twinx()
    ax_c_lat.plot(years_0938, lats_0938, color="#dd6b20", linewidth=2.0, linestyle="--", marker="s", markersize=8, label="P2 Latency (ms)")
    for yr, lat in zip(years_0938, lats_0938):
        ax_c_lat.annotate(f"{lat:.1f} ms", (yr, lat), textcoords="offset points", xytext=(0, -18), ha="center", fontweight="bold", color="#dd6b20", fontsize=10)
    ax_c_lat.set_ylabel("Cz Cortical P2 Latency (ms)", fontsize=12, fontweight="bold", color="#dd6b20")
    ax_c_lat.set_ylim(160, 230)

    # Combined legend
    lines_c1, labels_c1 = ax_c.get_legend_handles_labels()
    lines_c2, labels_c2 = ax_c_lat.get_legend_handles_labels()
    ax_c.legend(lines_c1 + lines_c2, labels_c1 + labels_c2, loc="upper left", fontsize=9.5, framealpha=0.9)

    # -------------------------------------------------------------
    # Panel D: Cohort-Wide Latency Discrepancy & Convergence
    # -------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])
    lat_diff = y3_lats - y2_lats
    n_bins = 10
    counts, bins, patches = ax_d.hist(lat_diff, bins=n_bins, color="#4299e1", edgecolor="#2b6cb0", alpha=0.75, linewidth=1.2)
    ax_d.axvline(0.0, color="#e53e3e", linestyle="--", linewidth=2.0, label="Zero Shift (Invariant)")
    ax_d.axvline(np.mean(lat_diff), color="#1a202c", linestyle="-", linewidth=2.5, label=f"Mean Shift = {np.mean(lat_diff):+.2f} ms")

    ax_d.set_xlabel("P2 Latency Shift (Year 3 - Year 2, ms)", fontsize=12, fontweight="bold")
    ax_d.set_ylabel("Patient Count", fontsize=12, fontweight="bold")
    ax_d.set_title("D. Stability Distribution: Latency Delta (Y3 - Y2)", fontsize=13, fontweight="bold", pad=10)
    ax_d.grid(True, linestyle=":", alpha=0.6)
    ax_d.legend(loc="upper right", fontsize=9.5, framealpha=0.9)

    fig.suptitle("Longitudinal Auditory Cortical Neuroplasticity Tracking Across Multi-Year CI Cohorts", fontsize=16, fontweight="bold", y=0.98)
    
    os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Longitudinal tracking diagnostic figure successfully saved to: {OUTPUT_PNG}")

if __name__ == "__main__":
    main()
