#!/usr/bin/env python3
"""
compare_n2n_models.py

Compares Single-Subject N2N (trained on 0901y2) vs Multi-Subject Foundation N2N
(trained on 8 pooled CI subjects) across the entire clinical CI cohort (N=36).
"""

import os
import glob
import json
import numpy as np

SINGLE_DIR = "./benchmark_results/sobi_n2n_cohort"
FOUNDATION_DIR = "./benchmark_results/sobi_n2n_foundation_cohort"

def load_cohort(directory):
    files = sorted(glob.glob(os.path.join(directory, "sobi_n2n_eval_*.json")))
    records = {}
    for f in files:
        if "summary" in f:
            continue
        with open(f, 'r') as jf:
            d = json.load(jf)
        if d.get('evaluated'):
            records[d['subject']] = d
    return records

def main():
    single_records = load_cohort(SINGLE_DIR)
    found_records = load_cohort(FOUNDATION_DIR)
    
    common_subjs = sorted(list(set(single_records.keys()) & set(found_records.keys())))
    print(f"Loaded {len(single_records)} single-subject evals, {len(found_records)} foundation evals.")
    print(f"Common evaluated CI subjects: {len(common_subjs)}")
    
    if len(common_subjs) == 0:
        print("No overlapping subjects yet.")
        return
        
    s_att = np.array([single_records[s]['p7_additional_attenuation_db'] for s in common_subjs])
    f_att = np.array([found_records[s]['p7_additional_attenuation_db'] for s in common_subjs])
    
    s_cz_r = np.array([single_records[s]['cz_template_r'] for s in common_subjs])
    f_cz_r = np.array([found_records[s]['cz_template_r'] for s in common_subjs])
    
    s_v_floor = np.array([single_records[s]['v_cz_noise_floor_uv'] for s in common_subjs])
    f_v_floor = np.array([found_records[s]['v_cz_noise_floor_uv'] for s in common_subjs])
    
    diff_att = f_att - s_att  # negative means foundation got more attenuation
    
    print("\n" + "="*70)
    print("COHORT BENCHMARK COMPARISON: SINGLE-SUBJECT vs FOUNDATION N2N")
    print("="*70)
    print(f"{'Metric':<35} | {'Single-Subj (0901)':<15} | {'Foundation (8-Subj)':<15}")
    print("-"*70)
    print(f"{'Mean P7 Attenuation (dB)':<35} | {np.mean(s_att):>+10.2f} dB   | {np.mean(f_att):>+10.2f} dB")
    print(f"{'Median P7 Attenuation (dB)':<35} | {np.median(s_att):>+10.2f} dB   | {np.median(f_att):>+10.2f} dB")
    print(f"{'Max P7 Attenuation (dB)':<35} | {np.min(s_att):>+10.2f} dB   | {np.min(f_att):>+10.2f} dB")
    print(f"{'Subjects with Clean Atten (< 0 dB)':<35} | {np.sum(s_att < 0):>10d} / {len(common_subjs)} | {np.sum(f_att < 0):>10d} / {len(common_subjs)}")
    print(f"{'Mean Cz Pearson r (SOBI template)':<35} | {np.mean(s_cz_r):>13.6f} | {np.mean(f_cz_r):>13.6f}")
    print(f"{'Min Cz Pearson r':<35} | {np.min(s_cz_r):>13.6f} | {np.min(f_cz_r):>13.6f}")
    print(f"{'Mean Silent Floor at Cz (uV)':<35} | {np.mean(s_v_floor):>10.3f} uV   | {np.mean(f_v_floor):>10.3f} uV")
    print(f"{'Max Silent Floor at Cz (uV)':<35} | {np.max(s_v_floor):>10.3f} uV   | {np.max(f_v_floor):>10.3f} uV")
    print("="*70)
    
    # Check subjects where foundation improved upon single-subject
    improved = np.sum(diff_att < -0.01)
    equivalent = np.sum(np.abs(diff_att) <= 0.01)
    worse = np.sum(diff_att > 0.01)
    print(f"Foundation Model vs Single-Subject: Improved: {improved}, Equivalent: {equivalent}, Lower Atten: {worse}")
    print(f"Mean Difference in Attenuation: {np.mean(diff_att):+.2f} dB")

if __name__ == "__main__":
    main()
