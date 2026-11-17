#!/usr/bin/env python3
"""
aggregate_cohort_results.py
Aggregates all single-subject audit JSONs from benchmark_results/sobi_polarity_cohort/
into a population-level CSV and publication grand-average figure.
Implements formal QA/QC exclusion of corrupted hardware recordings and transparent,
un-cherry-picked reporting of attenuation and group delay metrics.
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
            'hardware_failure': d.get('hardware_failure', False),
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
            'cz_group_delay_ms': d.get('cz_cortical_erp', {}).get('cz_group_delay_ms', np.nan),
            'cz_raw_clean_corr_r': d.get('cz_cortical_erp', {}).get('cz_raw_clean_corr_r', np.nan),
            'silent_cz_floor_uv': d.get('condition_v_silent_baseline', {}).get('cz_noise_floor_uv', np.nan),
            'silent_p7_floor_uv': d.get('condition_v_silent_baseline', {}).get('p7_noise_floor_uv', np.nan),
            'silent_gate_passed': d.get('condition_v_silent_baseline', {}).get('silent_gate_passed', False)
        })
        
    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "cohort_sobi_polarity_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved cohort summary table to {csv_path} ({len(df)} subjects)")
    
    # QA/QC Exclusion: Filter corrupted hardware files (e.g. 0923y2 with uncalibrated amplifier overflow)
    hw_corrupt_mask = df['hardware_failure'] | (df['silent_cz_floor_uv'] > 50.0) | (df['p2_amplitude_uv'].abs() > 50.0)
    df_valid = df[~hw_corrupt_mask]
    n_total = len(df)
    n_valid = len(df_valid)
    n_excluded = n_total - n_valid
    
    # Population-level statistics
    print("\n" + "=" * 80)
    print(f"POPULATION SUMMARY STATISTICS (N = {n_total} Enrolled, {n_excluded} Excluded for Hardware Saturation, N = {n_valid} Valid)")
    print("=" * 80)
    print(f"P2 Compliance (150-240ms & Valid Amp): {(df_valid['p2_gate_passed'].sum() / n_valid) * 100:.1f}% ({df_valid['p2_gate_passed'].sum()}/{n_valid} patients)")
    print(f"Silent Floor Compliance (<2uV):         {(df_valid['silent_gate_passed'].sum() / n_valid) * 100:.1f}% ({df_valid['silent_gate_passed'].sum()}/{n_valid} patients)")
    print(f"Mean P2 Evoked Latency:                 {df_valid['p2_latency_ms'].mean():.1f} +/- {df_valid['p2_latency_ms'].std():.1f} ms")
    print(f"Median P2 Evoked Latency:               {df_valid['p2_latency_ms'].median():.1f} ms (IQR: {df_valid['p2_latency_ms'].quantile(0.25):.1f} - {df_valid['p2_latency_ms'].quantile(0.75):.1f} ms)")
    print(f"Mean P2 Amplitude:                      {df_valid['p2_amplitude_uv'].mean():.2f} +/- {df_valid['p2_amplitude_uv'].std():.2f} uV")
    print(f"Mean Peak-to-Peak Amplitude:            {df_valid['peak_to_peak_uv'].mean():.2f} +/- {df_valid['peak_to_peak_uv'].std():.2f} uV")
    print(f"Mean Silent Cz Noise Floor:             {df_valid['silent_cz_floor_uv'].mean():.3f} +/- {df_valid['silent_cz_floor_uv'].std():.3f} uV")
    
    # Honest Attenuation Reporting (all valid subjects without dropping positive dB values)
    all_atten = df_valid['total_artifact_attenuation_db'].dropna()
    print(f"\n[Transparent Attenuation Across All N = {n_valid} Valid Patients]")
    print(f"  Parametric (Mean +/- Std):            {all_atten.mean():.2f} +/- {all_atten.std():.2f} dB")
    print(f"  Non-Parametric (Median [IQR]):        {all_atten.median():.2f} dB [{all_atten.quantile(0.25):.2f} to {all_atten.quantile(0.75):.2f} dB]")
    print(f"  Dynamic Range (Min to Max):           {all_atten.min():.2f} to {all_atten.max():.2f} dB")
    
    # Group Delay and Phase Integrity
    if 'cz_group_delay_ms' in df_valid.columns:
        valid_gd = df_valid['cz_group_delay_ms'].dropna()
        print(f"\n[Phase & Group Delay Integrity on Shielded Cz ERP]")
        print(f"  Median Group Delay:                   {valid_gd.median():.1f} ms (IQR: {valid_gd.quantile(0.25):.1f} - {valid_gd.quantile(0.75):.1f} ms)")
        valid_corr = df_valid['cz_raw_clean_corr_r'].dropna()
        print(f"  Mean Cz Waveform Fidelity r:          {valid_corr.mean():.4f} +/- {valid_corr.std():.4f}")
        
    print(f"\nMean Runtime per Subject:               {df_valid['total_runtime_s'].mean():.1f} seconds")
    print("=" * 80)
    
    # Population Grand-Average Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Subplot A: P2 Latency vs Amplitude Distribution
    ax = axes[0]
    ax.scatter(df_valid['p2_latency_ms'], df_valid['p2_amplitude_uv'], color='crimson', edgecolors='k', s=60, alpha=0.8, label=f'CI Patients (N = {n_valid})')
    ax.axvspan(150, 240, color='green', alpha=0.15, label='Canonical P2 Gate (150-240 ms)')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.6)
    ax.set_title(f"A. Clinical Cortical P2 Latency & Amplitude (N = {n_valid})", fontsize=11, fontweight='bold')
    ax.set_xlabel("Cz P2 Latency (ms)", fontsize=10)
    ax.set_ylabel("Cz P2 Amplitude (µV)", fontsize=10)
    ax.set_xlim(140, 250)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)
    
    # Subplot B: Total Artifact Attenuation Distribution (Unclipped)
    ax = axes[1]
    ax.hist(all_atten, bins=12, color='royalblue', edgecolor='k', alpha=0.7)
    mean_db = all_atten.mean()
    median_db = all_atten.median()
    ax.axvline(mean_db, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_db:.2f} dB')
    ax.axvline(median_db, color='orange', linestyle=':', linewidth=2, label=f'Median: {median_db:.2f} dB')
    ax.set_title(f"B. Artifact Attenuation Distribution (N = {n_valid})", fontsize=11, fontweight='bold')
    ax.set_xlabel("Total Artifact Rejection (dB)", fontsize=10)
    ax.set_ylabel("Patient Count", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=9)
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "cohort_grand_average_summary.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved population grand-average figure to {fig_path}")

if __name__ == "__main__":
    res_dir = sys.argv[1] if len(sys.argv) > 1 else "./benchmark_results/sobi_polarity_cohort"
    aggregate_cohort(res_dir, res_dir)
