#!/usr/bin/env python3
"""
evaluate_sobi_n2n_cohort.py

Full-cohort downstream evaluation of the Subspace-Constrained Symmetric Noise2Noise model
across all N=37 CI subjects in the CMPy2 dataset.

For each subject:
1. Loads continuous SOBI Stage 1 output (FIF).
2. Extracts subject-specific artifact projection operator P_art (midline Cz/Fz/Pz protected).
3. Applies sliding-window overlap-add inference using the calibrated N2N checkpoint.
4. Evaluates:
   - Midline template fidelity (Pearson r, RMSE at Cz).
   - Auditory P2 latency & amplitude preservation.
   - Condition V silent visual noise floor (< 2.0 uV gate).
   - Mastoid P7 artifact power attenuation (dB).
5. Outputs per-subject JSON and aggregated cohort metrics.
"""

import os
import sys
import glob
import json
import time
import argparse
import numpy as np
import pandas as pd
import mne
import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if os.path.join(REPO_ROOT, "track_a_deep_learning") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "track_a_deep_learning"))

from model import UNet1D
from train_sobi_n2n import ResidualConstrainedUNet1D, compute_sagittal_permutation_matrix

CANONICAL_20_CHS = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'T7', 'C3', 'Cz', 'C4', 'T8',
    'P7', 'P3', 'Pz', 'P4', 'P8',
    'O1', 'Oz', 'O2'
]

DATA_DIR = "/quobyte/millerlmgrp/annotated_data/CMPy2"
SOBI_DIR = "./benchmark_results/sobi_polarity_cohort"

def get_ci_subjects():
    pattern = os.path.join(DATA_DIR, "trial_onsets_09*y2-raw.fif")
    files = sorted(glob.glob(pattern))
    subjects = []
    for f in files:
        sub = os.path.basename(f).replace("trial_onsets_", "").replace("-raw.fif", "")
        csv_file = os.path.join(DATA_DIR, f"trial_onsets_{sub}.csv")
        sobi_fif = os.path.join(SOBI_DIR, f"trial_onsets_{sub}_sobi2pass-raw.fif")
        summary_json = os.path.join(SOBI_DIR, f"sobi2pass_summary_{sub}.json")
        subjects.append({
            'subject_id': sub,
            'raw_fif': f,
            'csv_file': csv_file,
            'sobi_fif': sobi_fif,
            'summary_json': summary_json
        })
    return subjects

def apply_sobi_n2n_sliding_window(model, data, p_art_tensor, device, win_len=512, hop_len=256, scale_uv=100.0):
    model.eval()
    n_chs, n_times = data.shape
    data_scaled = (data * 1e6) / scale_uv
    output_acc = np.zeros_like(data_scaled, dtype=np.float32)
    weight_acc = np.zeros(n_times, dtype=np.float32)
    hann = np.hanning(win_len).astype(np.float32)
    
    with torch.no_grad():
        for start in range(0, n_times - win_len + 1, hop_len):
            end = start + win_len
            chunk = data_scaled[:, start:end]
            x_tensor = torch.tensor(chunk[None, :, :], dtype=torch.float32).to(device)
            out_tensor, _ = model(x_tensor, p_art_tensor)
            out_np = out_tensor.cpu().numpy()[0]
            output_acc[:, start:end] += out_np * hann[None, :]
            weight_acc[start:end] += hann
            
        if n_times % hop_len != 0:
            start = n_times - win_len
            end = n_times
            chunk = data_scaled[:, start:end]
            x_tensor = torch.tensor(chunk[None, :, :], dtype=torch.float32).to(device)
            out_tensor, _ = model(x_tensor, p_art_tensor)
            out_np = out_tensor.cpu().numpy()[0]
            output_acc[:, start:end] += out_np * hann[None, :]
            weight_acc[start:end] += hann
            
    weight_acc = np.maximum(weight_acc, 1e-6)
    output_scaled = output_acc / weight_acc[None, :]
    return (output_scaled * scale_uv) / 1e6

def evaluate_subject(sub_info, checkpoint_path, output_dir, device):
    os.makedirs(output_dir, exist_ok=True)
    sub = sub_info['subject_id']
    out_json = os.path.join(output_dir, f"sobi_n2n_eval_{sub}.json")
    
    # 0. Check Hardware Gate / Exclusion
    if not os.path.exists(sub_info['sobi_fif']) or not os.path.exists(sub_info['summary_json']):
        print(f"[{sub}] SOBI preprocessed files missing. Checking exclusion...")
        audit_file = os.path.join(SOBI_DIR, f"sobi_polarity_audit_{sub}.json")
        if os.path.exists(audit_file):
            with open(audit_file, 'r') as f:
                audit = json.load(f)
            if audit.get('hardware_failure'):
                print(f"[{sub}] Excluded due to hardware failure: {audit.get('hardware_failure_reason')}")
                res = {
                    'subject': sub,
                    'hardware_failure': True,
                    'hardware_failure_reason': audit.get('hardware_failure_reason'),
                    'evaluated': False
                }
                with open(out_json, 'w') as f:
                    json.dump(res, f, indent=2)
                return res
        print(f"[{sub}] Files missing and no audit found. Skipping.")
        return None

    with open(sub_info['summary_json'], 'r') as f:
        meta = json.load(f)
    if meta.get('hardware_failure'):
        print(f"[{sub}] Excluded by hardware failure flag in metadata.")
        res = {
            'subject': sub,
            'hardware_failure': True,
            'hardware_failure_reason': meta.get('hardware_failure_reason'),
            'evaluated': False
        }
        with open(out_json, 'w') as f:
            json.dump(res, f, indent=2)
        return res

    print(f"\n========================================================")
    print(f"Evaluating Subject: {sub}")
    print(f"========================================================")
    t0 = time.time()
    
    # 1. Load SOBI Continuous FIF
    sobi_raw = mne.io.read_raw_fif(sub_info['sobi_fif'], preload=True, verbose=False)
    available_chs = [ch for ch in CANONICAL_20_CHS if ch in sobi_raw.ch_names]
    sobi_raw.pick(available_chs)
    fs = 256.0
    
    df = pd.read_csv(sub_info['csv_file'])
    
    # 2. Extract Artifact Projection Operator P_artifact
    A_art_list = []
    for p in meta.get('pass_history', []):
        weights = p.get('mixing_weights', {})
        vec = np.array([weights.get(ch, 0.0) for ch in available_chs], dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            A_art_list.append(vec / norm)
            
    if A_art_list:
        U = np.column_stack(A_art_list)
        P_artifact = U @ np.linalg.pinv(U.T @ U) @ U.T
        for m_ch in ['Cz', 'Fz', 'Pz']:
            if m_ch in available_chs:
                m_idx = available_chs.index(m_ch)
                P_artifact[m_idx, :] = 0.0
                P_artifact[:, m_idx] = 0.0
    else:
        P_artifact = np.zeros((len(available_chs), len(available_chs)), dtype=np.float32)
        
    p_art_tensor = torch.tensor(P_artifact, dtype=torch.float32).to(device)
    
    # 3. Model setup
    n_channels = len(available_chs)
    P_sag = compute_sagittal_permutation_matrix(available_chs)
    model = ResidualConstrainedUNet1D(in_channels=n_channels, out_channels=n_channels, p_sagittal=P_sag, use_sagittal=False).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    
    # 4. Sliding-Window Inference
    sobi_data = sobi_raw.get_data()
    n2n_data = apply_sobi_n2n_sliding_window(model, sobi_data, p_art_tensor, device)
    infer_time = time.time() - t0
    print(f"[{sub}] Inference complete in {infer_time:.2f}s ({sobi_data.shape[1]/fs:.1f}s EEG)")
    
    # 5. Auditory ERP Metrics (Condition A)
    cz_idx = available_chs.index('Cz') if 'Cz' in available_chs else None
    p7_idx = available_chs.index('P7') if 'P7' in available_chs else None
    
    a_trials = df[df['condition'] == 'A']
    t_pre = int(0.2 * fs)
    t_post = int(0.6 * fs)
    times = np.linspace(-0.2, 0.6, t_pre + t_post)
    post_mask = (times >= 0.0) & (times <= 0.5)
    
    n2n_epochs, sobi_epochs = [], []
    for _, row in a_trials.iterrows():
        s0 = int(row['onset_s'] * fs) - t_pre
        s1 = s0 + t_pre + t_post
        if s0 >= 0 and s1 <= n2n_data.shape[1]:
            n_seg = n2n_data[:, s0:s1].copy()
            s_seg = sobi_data[:, s0:s1].copy()
            n_seg -= np.mean(n_seg[:, :t_pre], axis=1, keepdims=True)
            s_seg -= np.mean(s_seg[:, :t_pre], axis=1, keepdims=True)
            n2n_epochs.append(n_seg)
            sobi_epochs.append(s_seg)
            
    if n2n_epochs and cz_idx is not None:
        n2n_erp = np.mean(np.array(n2n_epochs), axis=0) * 1e6 # uV
        sobi_erp = np.mean(np.array(sobi_epochs), axis=0) * 1e6
        
        # Template fidelity at Cz
        r_cz = float(np.corrcoef(n2n_erp[cz_idx][post_mask], sobi_erp[cz_idx][post_mask])[0, 1])
        rmse_cz = float(np.sqrt(np.mean((n2n_erp[cz_idx][post_mask] - sobi_erp[cz_idx][post_mask]) ** 2)))
        
        # Latencies
        p2_mask = (times >= 0.150) & (times <= 0.240)
        p2_times = times[p2_mask]
        p2_idx = np.argmax(n2n_erp[cz_idx][p2_mask])
        p2_lat_ms = float(p2_times[p2_idx] * 1000.0)
        p2_amp_uv = float(n2n_erp[cz_idx][p2_mask][p2_idx])
    else:
        r_cz, rmse_cz, p2_lat_ms, p2_amp_uv = np.nan, np.nan, np.nan, np.nan

    # 6. Artifact attenuation at P7 (Lindqvist Occam's Razor Criterion)
    if p7_idx is not None and len(sobi_epochs) > 0:
        sobi_p7_power = float(np.mean(sobi_erp[p7_idx][post_mask] ** 2))
        n2n_p7_power = float(np.mean(n2n_erp[p7_idx][post_mask] ** 2))
        p7_atten_db = float(10 * np.log10(n2n_p7_power / (sobi_p7_power + 1e-12)))
    else:
        sobi_p7_power, n2n_p7_power, p7_atten_db = np.nan, np.nan, 0.0

    # 7. Silent Visual (V) Noise Floor Audit
    v_trials = df[df['condition'] == 'V']
    v_n2n_epochs = []
    for _, row in v_trials.iterrows():
        s0 = int(row['onset_s'] * fs) - t_pre
        s1 = s0 + t_pre + t_post
        if s0 >= 0 and s1 <= n2n_data.shape[1]:
            v_seg = n2n_data[:, s0:s1].copy()
            v_seg -= np.mean(v_seg[:, :t_pre], axis=1, keepdims=True)
            v_n2n_epochs.append(v_seg)
            
    if v_n2n_epochs and cz_idx is not None:
        v_erp = np.mean(np.array(v_n2n_epochs), axis=0) * 1e6
        v_cz_floor = float(np.sqrt(np.mean(v_erp[cz_idx] ** 2)))
        v_p7_floor = float(np.sqrt(np.mean(v_erp[p7_idx] ** 2))) if p7_idx is not None else np.nan
    else:
        v_cz_floor, v_p7_floor = np.nan, np.nan

    res = {
        'subject': sub,
        'hardware_failure': False,
        'evaluated': True,
        'inference_time_s': infer_time,
        'cz_template_r': r_cz,
        'cz_rmse_uv': rmse_cz,
        'p2_latency_ms': p2_lat_ms,
        'p2_amplitude_uv': p2_amp_uv,
        'sobi_p7_power_uv2': sobi_p7_power,
        'n2n_p7_power_uv2': n2n_p7_power,
        'p7_additional_attenuation_db': p7_atten_db,
        'v_cz_noise_floor_uv': v_cz_floor,
        'v_p7_noise_floor_uv': v_p7_floor,
        'gates_passed': {
            'cz_template_gate': (not np.isnan(r_cz)) and (r_cz >= 0.70),
            'silence_gate': (not np.isnan(v_cz_floor)) and (v_cz_floor < 2.0),
            'p7_attenuation_gate': p7_atten_db <= 0.0
        }
    }
    
    with open(out_json, 'w') as f:
        json.dump(res, f, indent=2)
        
    print(f"[{sub}] Results: P7 Atten = {p7_atten_db:+.2f} dB | Cz r = {r_cz:.4f} | V floor = {v_cz_floor:.2f} uV")
    return res

def aggregate_cohort(output_dir):
    json_files = sorted(glob.glob(os.path.join(output_dir, "sobi_n2n_eval_*.json")))
    print(f"\n[Aggregation] Found {len(json_files)} subject evaluation records.")
    
    records = []
    for jf in json_files:
        with open(jf, 'r') as f:
            d = json.load(f)
        if d.get('evaluated'):
            records.append(d)
            
    df = pd.DataFrame(records)
    if len(df) == 0:
        print("No evaluated records found.")
        return
        
    summary = {
        'total_evaluated_subjects': len(df),
        'mean_p7_attenuation_db': float(df['p7_additional_attenuation_db'].mean()),
        'median_p7_attenuation_db': float(df['p7_additional_attenuation_db'].median()),
        'std_p7_attenuation_db': float(df['p7_additional_attenuation_db'].std()),
        'min_p7_attenuation_db': float(df['p7_additional_attenuation_db'].min()),
        'max_p7_attenuation_db': float(df['p7_additional_attenuation_db'].max()),
        'mean_cz_template_r': float(df['cz_template_r'].mean()),
        'min_cz_template_r': float(df['cz_template_r'].min()),
        'mean_v_cz_floor_uv': float(df['v_cz_noise_floor_uv'].mean()),
        'max_v_cz_floor_uv': float(df['v_cz_noise_floor_uv'].max()),
        'pct_subjects_passing_all_gates': float(np.mean([
            all(r['gates_passed'].values()) for r in records
        ]) * 100.0)
    }
    
    agg_file = os.path.join(output_dir, "sobi_n2n_cohort_summary.json")
    with open(agg_file, 'w') as f:
        json.dump(summary, f, indent=2)
        
    print("\n" + "=" * 80)
    print("  COHORT-WIDE SOBI + SYMMETRIC N2N SUMMARY (N = %d)" % len(df))
    print("=" * 80)
    print(f"Mean P7 Additional Attenuation: {summary['mean_p7_attenuation_db']:+.2f} dB (Median: {summary['median_p7_attenuation_db']:+.2f} dB)")
    print(f"Mean Cz Fidelity (r vs SOBI)  : {summary['mean_cz_template_r']:.6f} (Min: {summary['min_cz_template_r']:.4f})")
    print(f"Mean Silent V Floor at Cz     : {summary['mean_v_cz_floor_uv']:.3f} uV (Max: {summary['max_v_cz_floor_uv']:.3f} uV)")
    print(f"Subjects Passing All Gates    : {summary['pct_subjects_passing_all_gates']:.1f}%")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SOBI-N2N Across Cohort")
    parser.add_argument("--task-id", type=int, default=None, help="Task ID for Slurm array (0-indexed)")
    parser.add_argument("--subject-id", type=str, default=None, help="Specific subject to evaluate")
    parser.add_argument("--checkpoint", default="./benchmark_results/sobi_n2n_results/best_sobi_n2n_unet.pt", help="N2N checkpoint")
    parser.add_argument("--output-dir", default="./benchmark_results/sobi_n2n_cohort", help="Output directory")
    parser.add_argument("--aggregate", action="store_true", help="Aggregate all evaluation records")
    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    subjects = get_ci_subjects()
    
    if args.aggregate:
        aggregate_cohort(args.output_dir)
    elif args.task_id is not None:
        if args.task_id < 0 or args.task_id >= len(subjects):
            print(f"Task ID {args.task_id} out of bounds (0..{len(subjects)-1})")
            sys.exit(1)
        sub_info = subjects[args.task_id]
        evaluate_subject(sub_info, args.checkpoint, args.output_dir, device)
    elif args.subject_id is not None:
        matched = [s for s in subjects if s['subject_id'] == args.subject_id]
        if not matched:
            print(f"Subject {args.subject_id} not found.")
            sys.exit(1)
        evaluate_subject(matched[0], args.checkpoint, args.output_dir, device)
    else:
        print(f"Evaluating all {len(subjects)} subjects sequentially...")
        for s in subjects:
            evaluate_subject(s, args.checkpoint, args.output_dir, device)
        aggregate_cohort(args.output_dir)
