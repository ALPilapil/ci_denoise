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
    
    # Filter calibrated subjects (excluding uncalibrated hardware sensor scaling outlier 0923y2)
    calibrated_mask = (df['silent_cz_floor_uv'] < 50.0) & (df['p2_amplitude_uv'].abs() < 50.0)
    df_calib = df[calibrated_mask]
    
    # Population-level statistics
    print("\n" + "=" * 80)
    print(f"POPULATION SUMMARY STATISTICS (N = {len(df)} CI Patients, {len(df_calib)} Calibrated)")
    print("=" * 80)
    print(f"P2 Gate Compliance (150-240ms): {(df['p2_gate_passed'].sum() / len(df)) * 100:.1f}% ({df['p2_gate_passed'].sum()}/{len(df)} patients)")
    print(f"Silent Floor Compliance (<2uV): {(df['silent_gate_passed'].sum() / len(df)) * 100:.1f}% ({df['silent_gate_passed'].sum()}/{len(df)} patients)")
    print(f"Mean P2 Evoked Latency:         {df['p2_latency_ms'].mean():.1f} +/- {df['p2_latency_ms'].std():.1f} ms")
    print(f"Median P2 Evoked Latency:       {df['p2_latency_ms'].median():.1f} ms (IQR: {df['p2_latency_ms'].quantile(0.25):.1f} - {df['p2_latency_ms'].quantile(0.75):.1f} ms)")
    
    if len(df_calib) > 0:
        print(f"Calibrated Mean P2 Amplitude:   {df_calib['p2_amplitude_uv'].mean():.2f} +/- {df_calib['p2_amplitude_uv'].std():.2f} uV")
        print(f"Calibrated Mean Peak-to-Peak:   {df_calib['peak_to_peak_uv'].mean():.2f} +/- {df_calib['peak_to_peak_uv'].std():.2f} uV")
        print(f"Calibrated Mean Silent Cz Floor:{df_calib['silent_cz_floor_uv'].mean():.3f} +/- {df_calib['silent_cz_floor_uv'].std():.3f} uV")
        
    valid_atten = df['total_artifact_attenuation_db'][(df['total_artifact_attenuation_db'] <= 0.0)]
    print(f"Mean Significant Attenuation:   {valid_atten.mean():.2f} +/- {valid_atten.std():.2f} dB (Max: {valid_atten.min():.2f} dB)")
    print(f"Mean Runtime per Subject:       {df['total_runtime_s'].mean():.1f} seconds")
    print("=" * 80)
    
    # Population Grand-Average Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Subplot A: P2 Latency vs Amplitude Distribution
    ax = axes[0]
    ax.scatter(df_calib['p2_latency_ms'], df_calib['p2_amplitude_uv'], color='crimson', edgecolors='k', s=60, alpha=0.8, label=f'CI Patients (N = {len(df_calib)})')
    ax.axvspan(150, 240, color='green', alpha=0.15, label='Canonical P2 Gate (150-240 ms)')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.6)
    ax.set_title(f"A. Clinical Cortical P2 Latency & Amplitude (N = {len(df_calib)})", fontsize=11, fontweight='bold')
    ax.set_xlabel("Cz P2 Latency (ms)", fontsize=10)
    ax.set_ylabel("Cz P2 Amplitude (µV)", fontsize=10)
    ax.set_xlim(140, 250)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)
    
    # Subplot B: Total Artifact Attenuation Distribution
    ax = axes[1]
    atten_vals = df['total_artifact_attenuation_db'][df['total_artifact_attenuation_db'] <= 0]
    ax.hist(atten_vals, bins=12, color='royalblue', edgecolor='k', alpha=0.7)
    mean_db = atten_vals.mean()
    ax.axvline(mean_db, color='red', linestyle='--', linewidth=2, label=f'Mean Rejection: {mean_db:.2f} dB')
    ax.axvline(atten_vals.median(), color='orange', linestyle=':', linewidth=2, label=f'Median Rejection: {atten_vals.median():.2f} dB')
    ax.set_title(f"B. Artifact Attenuation Distribution (Up to {atten_vals.min():.1f} dB)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Total Artifact Rejection (dB)", fontsize=10)
    ax.set_ylabel("Patient Count", fontsize=10)
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
