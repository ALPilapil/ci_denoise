#!/usr/bin/env python3
"""
scripts/build_multi_year_master_table.py

Aggregates audited evaluation results across Year 2 (CMPy2), Year 3 (CMPy3),
and Year 4 (CMPy4) into a unified master clinical CSV table.
"""

import os
import glob
import json
import csv

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Y2_DIR = os.path.join(REPO_ROOT, "benchmark_results", "sobi_n2n_foundation_cohort")
Y3_DIR = os.path.join(REPO_ROOT, "benchmark_results", "sobi_n2n_cmpy3_cohort")
Y4_DIR = os.path.join(REPO_ROOT, "benchmark_results", "sobi_n2n_cmpy4_cohort")
OUTPUT_CSV = os.path.join(REPO_ROOT, "benchmark_results", "multi_year_cohort_master_table.csv")

def process_cohort(cohort_dir, cohort_label):
    records = []
    files = sorted(glob.glob(os.path.join(cohort_dir, "sobi_n2n_eval_*.json")))
    for f in files:
        if "summary" in f:
            continue
        with open(f, 'r') as jf:
            d = json.load(jf)
            
        sub = d.get('subject')
        hw_fail = d.get('hardware_failure', False)
        eval_ok = d.get('evaluated', False)
        
        row = {
            "subject": sub,
            "cohort": cohort_label,
            "hardware_failure": hw_fail,
            "evaluated": eval_ok,
            "inference_time_s": d.get('inference_time_s'),
            "cz_template_r": d.get('cz_template_r'),
            "cz_rmse_uv": d.get('cz_rmse_uv'),
            "p2_latency_ms": d.get('p2_latency_ms'),
            "p2_amplitude_uv": d.get('p2_amplitude_uv'),
            "sobi_p7_power_uv2": d.get('sobi_p7_power_uv2'),
            "n2n_p7_power_uv2": d.get('n2n_p7_power_uv2'),
            "p7_additional_attenuation_db": d.get('p7_additional_attenuation_db'),
            "v_cz_noise_floor_uv": d.get('v_cz_noise_floor_uv'),
            "v_p7_noise_floor_uv": d.get('v_p7_noise_floor_uv'),
            "all_gates_passed": all(d.get('gates_passed', {}).values()) if d.get('gates_passed') else False
        }
        records.append(row)
    return records

def main():
    y2_records = process_cohort(Y2_DIR, "Year 2 (CMPy2)")
    y3_records = process_cohort(Y3_DIR, "Year 3 (CMPy3)")
    y4_records = process_cohort(Y4_DIR, "Year 4 (CMPy4)")
    
    all_records = y2_records + y3_records + y4_records
    
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    fieldnames = list(all_records[0].keys())
    
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)
        
    print(f"Master clinical CSV table created successfully: {OUTPUT_CSV}")
    print(f"Total rows: {len(all_records)}")
    y2_count = len(y2_records)
    y3_count = len(y3_records)
    y4_count = len(y4_records)
    print(f"  Year 2: {y2_count}")
    print(f"  Year 3: {y3_count}")
    print(f"  Year 4: {y4_count}")
    
    evaluated = [r for r in all_records if r['evaluated']]
    print(f"Evaluated subjects: {len(evaluated)} / {len(all_records)}")
    mean_atten = sum(r['p7_additional_attenuation_db'] for r in evaluated) / len(evaluated)
    mean_r = sum(r['cz_template_r'] for r in evaluated) / len(evaluated)
    print(f"Mean P7 attenuation across all evaluated: {mean_atten:.2f} dB")
    print(f"Mean Cz template fidelity: {mean_r:.6f}")

if __name__ == "__main__":
    main()
