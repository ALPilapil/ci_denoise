#!/usr/bin/env python3
"""
aggregate_cohort_results.py
Aggregates all single-subject audit JSONs from benchmark_results/sobi_polarity_cohort/
into a population-level CSV and publication grand-average figure.
"""

import os
import sys
import glob
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def aggregate_cohort(results_dir="./benchmark_results/sobi_polarity_cohort", output_dir="./benchmark_results/sobi_polarity_cohort"):
    json_files = sorted(glob.glob(os.path.join(results_dir, "sobi_polarity_audit_*.json")))
    print(f"Found {len(json_files)} single-subject audit files in {results_dir}")
    if not json_files:
        print("No audit JSON files found. Run the cohort array first.")
        return None
        
    records = []
    for jf in json_files:
        with open(jf, 'r') as f:
            d = json.load(f)
            
        records.append({
            'subject': d.get('subject', 'unknown'),
            'total_runtime_s': d.get('total_runtime_s', np.nan),
            'stage1_rf_attenuation_db': d.get('stage1_sobi', {}).get('artifact_attenuation_db', np.nan),
            'stage2_polarity_attenuation_db': d.get('stage2_polarity_averaging', {}).get('additional_attenuation_db', np.nan),
            'total_artifact_attenuation_db': d.get('total_pipeline', {}).get('total_artifact_attenuation_db', np.nan),
            'power_reduction_pct': d.get('total_pipeline', {}).get('power_reduction_pct', np.nan),
            'n1_latency_ms': d.get('cz_cortical_erp', {}).get('n1_latency_ms', np.nan),
            'n1_amplitude_uv': d.get('cz_cortical_erp', {}).get('n1_amplitude_uv', np.nan),
            'n1_gate_passed': d.get('cz_cortical_erp', {}).get('n1_gate_passed', False),
            'p2_latency_ms': d.get('cz_cortical_erp', {}).get('p2_latency_ms', np.nan),
            'p2_amplitude_uv': d.get('cz_cortical_erp', {}).get('p2_amplitude_uv', np.nan),
            'p2_gate_passed': d.get('cz_cortical_erp', {}).get('p2_gate_passed', False),
            'peak_to_peak_uv': d.get('cz_cortical_erp', {}).get('peak_to_peak_uv', np.nan),
            'template_fidelity_r': d.get('cz_cortical_erp', {}).get('template_fidelity_r', np.nan),
            'silent_cz_floor_uv': d.get('condition_v_silent_baseline', {}).get('cz_noise_floor_uv', np.nan),
            'silent_p7_floor_uv': d.get('condition_v_silent_baseline', {}).get('p7_noise_floor_uv', np.nan),
            'silent_gate_passed': d.get('condition_v_silent_baseline', {}).get('silent_gate_passed', False)
        })
        
    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "cohort_sobi_polarity_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved cohort summary table to {csv_path} ({len(df)} subjects)")
    
    # Population-level statistics
    print("\n" + "=" * 80)
    print(f"POPULATION SUMMARY STATISTICS (N = {len(df)} CI Patients)")
    print("=" * 80)
    print(f"Mean RF Attenuation:          {df['total_artifact_attenuation_db'].mean():.2f} +/- {df['total_artifact_attenuation_db'].std():.2f} dB")
    print(f"Mean Total Power Reduction:   {df['power_reduction_pct'].mean():.2f}% +/- {df['power_reduction_pct'].std():.2f}%")
    print(f"Mean Silent Cz Noise Floor:   {df['silent_cz_floor_uv'].mean():.3f} +/- {df['silent_cz_floor_uv'].std():.3f} uV")
    print(f"Silent Floor Compliance (<2uV): {(df['silent_gate_passed'].sum() / len(df)) * 100:.1f}%")
    print(f"Mean P2 Evoked Latency:       {df['p2_latency_ms'].mean():.1f} +/- {df['p2_latency_ms'].std():.1f} ms")
    print(f"P2 Gate Compliance (150-240ms): {(df['p2_gate_passed'].sum() / len(df)) * 100:.1f}%")
    print(f"Mean Peak-to-Peak Amplitude:  {df['peak_to_peak_uv'].mean():.2f} +/- {df['peak_to_peak_uv'].std():.2f} uV")
    print(f"Mean Runtime per Subject:     {df['total_runtime_s'].mean():.1f} seconds")
    print("=" * 80)
    
    # Population Grand-Average Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Subplot A: P2 Latency vs Amplitude Distribution
    ax = axes[0]
    ax.scatter(df['p2_latency_ms'], df['p2_amplitude_uv'], color='crimson', edgecolors='k', s=60, alpha=0.8)
    ax.axvspan(150, 240, color='green', alpha=0.15, label='Canonical P2 Window (150-240 ms)')
    ax.set_title(f"A. Clinical P2 Distribution (N = {len(df)} CI Patients)", fontsize=11, fontweight='bold')
    ax.set_xlabel("P2 Latency (ms)")
    ax.set_ylabel("P2 Amplitude (µV)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)
    
    # Subplot B: Total Artifact Attenuation Distribution
    ax = axes[1]
    ax.hist(df['total_artifact_attenuation_db'], bins=12, color='royalblue', edgecolor='k', alpha=0.7)
    mean_db = df['total_artifact_attenuation_db'].mean()
    ax.axvline(mean_db, color='red', linestyle='--', linewidth=2, label=f'Mean Rejection: {mean_db:.2f} dB')
    ax.set_title("B. Population Artifact Attenuation Distribution", fontsize=11, fontweight='bold')
    ax.set_xlabel("Total Artifact Rejection (dB)")
    ax.set_ylabel("Subject Count")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=9)
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "cohort_grand_average_summary.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved population grand-average figure to {fig_path}")
    return df

if __name__ == "__main__":
    aggregate_cohort()
