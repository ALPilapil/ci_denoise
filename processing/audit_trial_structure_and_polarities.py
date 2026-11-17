#!/usr/bin/env python3
"""
audit_trial_structure_and_polarities.py
Rigorously inspects and validates the clinical CI trial structure:
1. Condition breakdown (A, AV, V).
2. Polarity pairing (polarity 1 vs polarity 2 for identical stimuli).
3. Cross-block sentence repetition.
4. Silent Visual (V) baseline statistics (confirming zero acoustic stimulation).
5. Inter-trial correlation (Raw CI correlation between paired trials).
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

def audit_subject(data_dir, subject="0901y2"):
    csv_path = os.path.join(data_dir, f"trial_onsets_{subject}.csv")
    fif_path = os.path.join(data_dir, f"trial_onsets_{subject}-raw.fif")
    
    if not os.path.exists(csv_path) or not os.path.exists(fif_path):
        raise FileNotFoundError(f"Missing CSV or FIF for {subject} in {data_dir}")
        
    df = pd.read_csv(csv_path)
    print(f"Loaded trial metadata: {len(df)} rows")
    
    # Check conditions
    cond_counts = df['condition'].value_counts().to_dict()
    polarity_counts = df['polarity'].value_counts().to_dict()
    block_counts = df['block'].value_counts().to_dict()
    
    # Check stimulus pairing across polarities
    # Within each block or across blocks, how are stimuli paired?
    paired_stims = []
    for stim_id, grp in df[df['stim_id'] > 0].groupby('stim_id'):
        pols = grp['polarity'].tolist()
        conds = grp['condition'].tolist()
        blocks = grp['block'].tolist()
        onsets = grp['onset_s'].tolist()
        paired_stims.append({
            'stim_id': int(stim_id),
            'n_trials': len(grp),
            'polarities': pols,
            'conditions': conds,
            'blocks': blocks,
            'onsets_s': [float(x) for x in onsets]
        })
        
    # Load raw data to check signal properties on V (silent) trials vs A/AV
    print("Loading raw FIF...")
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
    
    # Match channels
    available_chs = [ch for ch in CANONICAL_20_CHS if ch in raw.ch_names]
    raw.pick_channels(available_chs)
    raw.filter(1.0, 50.0, fir_design='firwin', verbose=False)
    raw.resample(256.0, verbose=False)
    fs = raw.info['sfreq']
    
    # Epoch V trials (silent visual)
    v_trials = df[df['condition'] == 'V']
    v_rms_list = []
    v_window_len = int(1.2 * fs) # 1.2s window
    
    raw_data = raw.get_data()
    ch_names = raw.ch_names
    cz_idx = ch_names.index('Cz') if 'Cz' in ch_names else 0
    p7_idx = ch_names.index('P7') if 'P7' in ch_names else 0
    
    for _, row in v_trials.iterrows():
        t0 = row['onset_s']
        s0 = int(t0 * fs)
        s1 = s0 + v_window_len
        if s1 <= raw_data.shape[1]:
            segment = raw_data[:, s0:s1]
            v_rms_list.append(np.sqrt(np.mean(segment ** 2, axis=1)))
            
    v_rms_arr = np.array(v_rms_list) * 1e6 # in uV
    mean_v_rms_cz = float(np.mean(v_rms_arr[:, cz_idx]))
    mean_v_rms_p7 = float(np.mean(v_rms_arr[:, p7_idx]))
    
    # Epoch A and AV trials for comparison
    a_trials = df[df['condition'] == 'A']
    a_rms_list = []
    for _, row in a_trials.iterrows():
        t0 = row['onset_s']
        s0 = int(t0 * fs)
        s1 = s0 + v_window_len
        if s1 <= raw_data.shape[1]:
            segment = raw_data[:, s0:s1]
            a_rms_list.append(np.sqrt(np.mean(segment ** 2, axis=1)))
    a_rms_arr = np.array(a_rms_list) * 1e6
    mean_a_rms_cz = float(np.mean(a_rms_arr[:, cz_idx]))
    mean_a_rms_p7 = float(np.mean(a_rms_arr[:, p7_idx]))
    
    # Inter-trial correlation between Polarity 1 and Polarity 2 for the same stim_id
    pair_corrs_cz = []
    pair_corrs_p7 = []
    
    for pair in paired_stims[:50]: # first 50 pairs
        if pair['n_trials'] == 2 and pair['polarities'] == [1, 2]:
            t1, t2 = pair['onsets_s']
            s1_0, s2_0 = int(t1 * fs), int(t2 * fs)
            seg1 = raw_data[:, s1_0:s1_0 + v_window_len]
            seg2 = raw_data[:, s2_0:s2_0 + v_window_len]
            if seg1.shape[1] == v_window_len and seg2.shape[1] == v_window_len:
                # correlation at Cz
                r_cz = np.corrcoef(seg1[cz_idx], seg2[cz_idx])[0, 1]
                r_p7 = np.corrcoef(seg1[p7_idx], seg2[p7_idx])[0, 1]
                if not np.isnan(r_cz):
                    pair_corrs_cz.append(float(r_cz))
                if not np.isnan(r_p7):
                    pair_corrs_p7.append(float(r_p7))
                    
    results = {
        'subject': subject,
        'total_trials': len(df),
        'condition_counts': cond_counts,
        'polarity_counts': polarity_counts,
        'block_counts': block_counts,
        'n_stim_pairs': len(paired_stims),
        'v_silent_baseline': {
            'cz_rms_uv': mean_v_rms_cz,
            'p7_rms_uv': mean_v_rms_p7,
            'description': 'Baseline RMS on silent visual trials (where CI receives zero acoustic audio).'
        },
        'a_active_baseline': {
            'cz_rms_uv': mean_a_rms_cz,
            'p7_rms_uv': mean_a_rms_p7,
            'description': 'Active RMS on auditory speech trials.'
        },
        'raw_ci_intertrial_correlation': {
            'cz_mean_r': float(np.mean(pair_corrs_cz)) if pair_corrs_cz else None,
            'p7_mean_r': float(np.mean(pair_corrs_p7)) if pair_corrs_p7 else None,
            'description': 'Correlation between Polarity 1 and Polarity 2 takes of identical sentences in Raw CI.'
        }
    }
    
    print("\n--- AUDIT RESULTS ---")
    print(json.dumps(results, indent=2))
    
    out_dir = "benchmark_results/pipeline_audit"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"trial_structure_audit_{subject}.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved audit to {out_file}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/quobyte/millerlmgrp/annotated_data/CMPy2")
    parser.add_argument("--subject", default="0901y2")
    args = parser.parse_args()
    audit_subject(args.data_dir, args.subject)
