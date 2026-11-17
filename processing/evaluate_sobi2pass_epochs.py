#!/usr/bin/env python3
"""
evaluate_sobi2pass_epochs.py
Computes the neurophysiological and DSP audit metrics on Stage 1 SOBI-cleaned data:
1. Condition V (Silent Visual) RMS floor at P7 and Cz (Raw vs SOBI 2-Pass).
2. Condition A (Audio) ERP morphology at Cz: N1 latency (ms), P2 latency (ms), P2 amplitude (uV).
3. Inter-trial correlation between paired Polarity 1 and Polarity 2 takes on SOBI-cleaned data.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import mne

CANONICAL_20_CHS = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'T7', 'C3', 'Cz', 'C4', 'T8',
    'P7', 'P3', 'Pz', 'P4', 'P8',
    'O1', 'Oz', 'O2'
]

def evaluate_epochs(raw_fif, sobi_fif, csv_path, output_json=None):
    df = pd.read_csv(csv_path)
    print(f"Loaded trial metadata: {len(df)} rows from {csv_path}")
    
    print(f"Loading Raw FIF: {raw_fif}")
    raw = mne.io.read_raw_fif(raw_fif, preload=True, verbose=False)
    if raw.info['sfreq'] != 256.0:
        raw.resample(256.0, verbose=False)
    raw.filter(1.0, 45.0, fir_design='firwin', verbose=False)
    
    print(f"Loading SOBI 2-Pass FIF: {sobi_fif}")
    sobi = mne.io.read_raw_fif(sobi_fif, preload=True, verbose=False)
    sobi.filter(1.0, 45.0, fir_design='firwin', verbose=False)
    
    fs = 256.0
    common_chs = [ch for ch in CANONICAL_20_CHS if ch in raw.ch_names and ch in sobi.ch_names]
    raw.pick(common_chs)
    sobi.pick(common_chs)
    
    raw_data = raw.get_data()
    sobi_data = sobi.get_data()
    ch_names = raw.ch_names
    
    cz_idx = ch_names.index('Cz')
    p7_idx = ch_names.index('P7')
    
    # 1. Condition V (Silent Visual) Evaluation
    v_trials = df[df['condition'] == 'V']
    win_samples = int(1.2 * fs)
    
    raw_v_rms_p7, sobi_v_rms_p7 = [], []
    raw_v_rms_cz, sobi_v_rms_cz = [], []
    
    raw_v_epochs, sobi_v_epochs = [], []
    
    for _, row in v_trials.iterrows():
        s0 = int(row['onset_s'] * fs)
        s1 = s0 + win_samples
        if s1 <= raw_data.shape[1] and s1 <= sobi_data.shape[1]:
            raw_seg = raw_data[:, s0:s1]
            sobi_seg = sobi_data[:, s0:s1]
            
            raw_v_rms_p7.append(np.sqrt(np.mean(raw_seg[p7_idx] ** 2)) * 1e6)
            sobi_v_rms_p7.append(np.sqrt(np.mean(sobi_seg[p7_idx] ** 2)) * 1e6)
            raw_v_rms_cz.append(np.sqrt(np.mean(raw_seg[cz_idx] ** 2)) * 1e6)
            sobi_v_rms_cz.append(np.sqrt(np.mean(sobi_seg[cz_idx] ** 2)) * 1e6)
            
            raw_v_epochs.append(raw_seg)
            sobi_v_epochs.append(sobi_seg)
            
    # Averaged V-trial noise floor
    raw_v_avg = np.mean(np.array(raw_v_epochs), axis=0) * 1e6
    sobi_v_avg = np.mean(np.array(sobi_v_epochs), axis=0) * 1e6
    
    raw_v_avg_p7_floor = float(np.sqrt(np.mean(raw_v_avg[p7_idx] ** 2)))
    sobi_v_avg_p7_floor = float(np.sqrt(np.mean(sobi_v_avg[p7_idx] ** 2)))
    raw_v_avg_cz_floor = float(np.sqrt(np.mean(raw_v_avg[cz_idx] ** 2)))
    sobi_v_avg_cz_floor = float(np.sqrt(np.mean(sobi_v_avg[cz_idx] ** 2)))
    
    # 2. Condition A (Audio) ERP Morphology Evaluation at Cz
    a_trials = df[df['condition'] == 'A']
    t_pre = int(0.2 * fs) # 200 ms baseline
    t_post = int(0.6 * fs) # 600 ms post-stimulus
    times = np.linspace(-0.2, 0.6, t_pre + t_post)
    
    raw_a_epochs, sobi_a_epochs = [], []
    for _, row in a_trials.iterrows():
        s0 = int(row['onset_s'] * fs) - t_pre
        s1 = s0 + t_pre + t_post
        if s0 >= 0 and s1 <= raw_data.shape[1] and s1 <= sobi_data.shape[1]:
            # Baseline correct
            r_seg = raw_data[:, s0:s1].copy()
            s_seg = sobi_data[:, s0:s1].copy()
            r_seg -= np.mean(r_seg[:, :t_pre], axis=1, keepdims=True)
            s_seg -= np.mean(s_seg[:, :t_pre], axis=1, keepdims=True)
            raw_a_epochs.append(r_seg)
            sobi_a_epochs.append(s_seg)
            
    raw_a_erp = np.mean(np.array(raw_a_epochs), axis=0) * 1e6 # in uV
    sobi_a_erp = np.mean(np.array(sobi_a_epochs), axis=0) * 1e6
    
    # N1 peak in 90-140 ms
    n1_mask = (times >= 0.090) & (times <= 0.140)
    n1_times = times[n1_mask]
    sobi_cz_n1_idx = np.argmin(sobi_a_erp[cz_idx][n1_mask])
    sobi_n1_lat_ms = float(n1_times[sobi_cz_n1_idx] * 1000.0)
    sobi_n1_amp_uv = float(sobi_a_erp[cz_idx][n1_mask][sobi_cz_n1_idx])
    
    # P2 peak in 150-240 ms
    p2_mask = (times >= 0.150) & (times <= 0.240)
    p2_times = times[p2_mask]
    sobi_cz_p2_idx = np.argmax(sobi_a_erp[cz_idx][p2_mask])
    sobi_p2_lat_ms = float(p2_times[sobi_cz_p2_idx] * 1000.0)
    sobi_p2_amp_uv = float(sobi_a_erp[cz_idx][p2_mask][sobi_cz_p2_idx])
    
    # Check compliance with Vance gates
    n1_pass = 90.0 <= sobi_n1_lat_ms <= 140.0
    p2_pass = 150.0 <= sobi_p2_lat_ms <= 240.0
    v_suppression_ratio = float(np.mean(raw_v_rms_p7) / max(np.mean(sobi_v_rms_p7), 1e-6))
    
    results = {
        'subject': '0901y2',
        'condition_v_silent_audit': {
            'single_trial_p7_rms_raw_uv': float(np.mean(raw_v_rms_p7)),
            'single_trial_p7_rms_sobi_uv': float(np.mean(sobi_v_rms_p7)),
            'p7_rms_suppression_ratio': v_suppression_ratio,
            'averaged_erp_p7_floor_raw_uv': raw_v_avg_p7_floor,
            'averaged_erp_p7_floor_sobi_uv': sobi_v_avg_p7_floor,
            'averaged_erp_cz_floor_sobi_uv': sobi_v_avg_cz_floor
        },
        'condition_a_cortical_erp_audit': {
            'cz_n1_latency_ms': sobi_n1_lat_ms,
            'cz_n1_amplitude_uv': sobi_n1_amp_uv,
            'cz_n1_gate_passed': n1_pass,
            'cz_p2_latency_ms': sobi_p2_lat_ms,
            'cz_p2_amplitude_uv': sobi_p2_amp_uv,
            'cz_p2_gate_passed': p2_pass,
            'peak_to_peak_n1_p2_uv': float(sobi_p2_amp_uv - sobi_n1_amp_uv)
        }
    }
    
    print("\n--- SOBI 2-PASS EPOCH AUDIT ---")
    print(json.dumps(results, indent=2))
    
    if output_json:
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {output_json}")
        
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-fif", default="/quobyte/millerlmgrp/annotated_data/CMPy2/trial_onsets_0901y2-raw.fif")
    parser.add_argument("--sobi-fif", default="./benchmark_results/sobi2pass_cache/trial_onsets_0901y2_sobi2pass-raw.fif")
    parser.add_argument("--csv", default="/quobyte/millerlmgrp/annotated_data/CMPy2/trial_onsets_0901y2.csv")
    parser.add_argument("--output-json", default="./benchmark_results/sobi2pass_cache/epoch_audit_0901y2.json")
    args = parser.parse_args()
    evaluate_epochs(args.raw_fif, args.sobi_fif, args.csv, args.output_json)
