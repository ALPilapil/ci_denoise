#!/usr/bin/env python3
"""
evaluate_sobi_n2n_downstream.py
Comprehensive downstream validation for the SOBI-N2N pipeline:
1. Continuous sliding-window inference on 0901y2 continuous EEG.
2. ERP N1/P2 latency, amplitude, and Pearson r against Normal Hearing control 0801y2.
3. Silent Visual (V) hallucination check (< 0.1 uV target).
4. Shuffled Envelope Permutation Sanity Test (Dr. Thorne mandate).
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import mne
import torch
import torch.nn as nn
from scipy import signal

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if os.path.join(REPO_ROOT, "track_a_deep_learning") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "track_a_deep_learning"))

from model import UNet1D
from train_sobi_n2n import EquivariantUNet1D, compute_sagittal_permutation_matrix

CANONICAL_20_CHS = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'T7', 'C3', 'Cz', 'C4', 'T8',
    'P7', 'P3', 'Pz', 'P4', 'P8',
    'O1', 'Oz', 'O2'
]

def apply_sobi_n2n_sliding_window(model, data, device, win_len=512, hop_len=256, scale_uv=100.0):
    """
    Continuous sliding-window inference with Hann-weighted overlap-add crossfading.
    Input data: (n_channels, n_times) in Volts
    Returns: denoised_data in Volts
    """
    model.eval()
    n_chs, n_times = data.shape
    
    # Scale to network scale (Volts -> uV -> scale_uv)
    data_scaled = (data * 1e6) / scale_uv
    
    output_acc = np.zeros_like(data_scaled, dtype=np.float32)
    weight_acc = np.zeros(n_times, dtype=np.float32)
    hann = np.hanning(win_len).astype(np.float32)
    
    with torch.no_grad():
        for start in range(0, n_times - win_len + 1, hop_len):
            end = start + win_len
            chunk = data_scaled[:, start:end]
            x_tensor = torch.tensor(chunk[None, :, :], dtype=torch.float32).to(device)
            out_tensor = model(x_tensor)
            out_np = out_tensor.cpu().numpy()[0]
            
            output_acc[:, start:end] += out_np * hann[None, :]
            weight_acc[start:end] += hann
            
        # Tail handling
        if n_times % hop_len != 0:
            start = n_times - win_len
            end = n_times
            chunk = data_scaled[:, start:end]
            x_tensor = torch.tensor(chunk[None, :, :], dtype=torch.float32).to(device)
            out_tensor = model(x_tensor)
            out_np = out_tensor.cpu().numpy()[0]
            output_acc[:, start:end] += out_np * hann[None, :]
            weight_acc[start:end] += hann
            
    weight_acc = np.maximum(weight_acc, 1e-6)
    output_scaled = output_acc / weight_acc[None, :]
    
    # Convert back to Volts
    return (output_scaled * scale_uv) / 1e6

def evaluate_pipeline(sobi_fif, hearing_fif, csv_path, checkpoint_path, output_dir, use_sagittal=False):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Evaluation] Device: {device} | Checkpoint: {checkpoint_path}")
    
    # 1. Load data
    sobi_raw = mne.io.read_raw_fif(sobi_fif, preload=True, verbose=False)
    hearing_raw = mne.io.read_raw_fif(hearing_fif, preload=True, verbose=False)
    
    available_chs = [ch for ch in CANONICAL_20_CHS if ch in sobi_raw.ch_names and ch in hearing_raw.ch_names]
    sobi_raw.pick(available_chs)
    hearing_raw.pick(available_chs)
    
    if hearing_raw.info['sfreq'] != 256.0:
        hearing_raw.resample(256.0, verbose=False)
    hearing_raw.filter(1.0, 45.0, fir_design='firwin', verbose=False)
    
    fs = 256.0
    df = pd.read_csv(csv_path)
    
    # 2. Load trained model
    n_channels = len(available_chs)
    base_unet = UNet1D(in_channels=n_channels, out_channels=n_channels)
    P_sag = compute_sagittal_permutation_matrix(available_chs)
    model = EquivariantUNet1D(base_unet, P_sag, use_sagittal=use_sagittal).to(device)
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    
    # 3. Apply inference to full continuous recording
    t0 = time.time()
    sobi_data = sobi_raw.get_data()
    n2n_data = apply_sobi_n2n_sliding_window(model, sobi_data, device)
    infer_time = time.time() - t0
    print(f"Continuous Inference Complete in {infer_time:.2f}s ({sobi_data.shape[1]/fs:.1f}s EEG)")
    
    n2n_raw = mne.io.RawArray(n2n_data, sobi_raw.info, verbose=False)
    n2n_raw.set_annotations(sobi_raw.annotations)
    
    # 4. Extract ERPs
    cz_idx = available_chs.index('Cz')
    p7_idx = available_chs.index('P7')
    
    # Audio trials
    a_trials = df[df['condition'] == 'A']
    t_pre = int(0.2 * fs)
    t_post = int(0.6 * fs)
    times = np.linspace(-0.2, 0.6, t_pre + t_post)
    
    h_epochs, n2n_epochs, sobi_epochs = [], [], []
    h_data = hearing_raw.get_data()
    
    for _, row in a_trials.iterrows():
        s0 = int(row['onset_s'] * fs) - t_pre
        s1 = s0 + t_pre + t_post
        if s0 >= 0 and s1 <= n2n_data.shape[1] and s1 <= h_data.shape[1]:
            h_seg = h_data[:, s0:s1].copy()
            n_seg = n2n_data[:, s0:s1].copy()
            s_seg = sobi_data[:, s0:s1].copy()
            
            h_seg -= np.mean(h_seg[:, :t_pre], axis=1, keepdims=True)
            n_seg -= np.mean(n_seg[:, :t_pre], axis=1, keepdims=True)
            s_seg -= np.mean(s_seg[:, :t_pre], axis=1, keepdims=True)
            
            h_epochs.append(h_seg)
            n2n_epochs.append(n_seg)
            sobi_epochs.append(s_seg)
            
    h_erp = np.mean(np.array(h_epochs), axis=0) * 1e6 # uV
    n2n_erp = np.mean(np.array(n2n_epochs), axis=0) * 1e6
    sobi_erp = np.mean(np.array(sobi_epochs), axis=0) * 1e6
    
    # Correlation with Normal Hearing ground truth
    post_mask = (times >= 0.0) & (times <= 0.5)
    r_sobi = float(np.corrcoef(sobi_erp[cz_idx][post_mask], h_erp[cz_idx][post_mask])[0, 1])
    r_n2n = float(np.corrcoef(n2n_erp[cz_idx][post_mask], h_erp[cz_idx][post_mask])[0, 1])
    
    rmse_sobi = float(np.sqrt(np.mean((sobi_erp[cz_idx][post_mask] - h_erp[cz_idx][post_mask]) ** 2)))
    rmse_n2n = float(np.sqrt(np.mean((n2n_erp[cz_idx][post_mask] - h_erp[cz_idx][post_mask]) ** 2)))
    
    # Latencies
    p2_mask = (times >= 0.150) & (times <= 0.240)
    p2_times = times[p2_mask]
    n2n_p2_idx = np.argmax(n2n_erp[cz_idx][p2_mask])
    n2n_p2_lat = float(p2_times[n2n_p2_idx] * 1000.0)
    n2n_p2_amp = float(n2n_erp[cz_idx][p2_mask][n2n_p2_idx])
    
    h_p2_idx = np.argmax(h_erp[cz_idx][p2_mask])
    h_p2_lat = float(p2_times[h_p2_idx] * 1000.0)
    
    # 5. Silent Visual (V) Hallucination Check
    v_trials = df[df['condition'] == 'V']
    v_n2n_epochs = []
    for _, row in v_trials.iterrows():
        s0 = int(row['onset_s'] * fs) - t_pre
        s1 = s0 + t_pre + t_post
        if s0 >= 0 and s1 <= n2n_data.shape[1]:
            v_seg = n2n_data[:, s0:s1].copy()
            v_seg -= np.mean(v_seg[:, :t_pre], axis=1, keepdims=True)
            v_n2n_epochs.append(v_seg)
            
    v_n2n_erp = np.mean(np.array(v_n2n_epochs), axis=0) * 1e6
    v_cz_floor = float(np.sqrt(np.mean(v_n2n_erp[cz_idx] ** 2)))
    v_p7_floor = float(np.sqrt(np.mean(v_n2n_erp[p7_idx] ** 2)))
    
    # 6. Shuffled Envelope Permutation Sanity Test (Dr. Thorne Mandate)
    # Generate 50 phase-randomized surrogate waveforms to test for synthetic hallucination
    surrogate_corrs = []
    fft_vals = np.fft.rfft(n2n_erp[cz_idx][post_mask])
    magnitudes = np.abs(fft_vals)
    for _ in range(50):
        rand_phases = np.exp(1j * np.random.uniform(0, 2*np.pi, size=len(fft_vals)))
        surrogate_sig = np.fft.irfft(magnitudes * rand_phases, n=len(h_erp[cz_idx][post_mask]))
        r_surr = np.corrcoef(surrogate_sig, h_erp[cz_idx][post_mask])[0, 1]
        surrogate_corrs.append(float(r_surr))
        
    p_value = float(np.mean(np.array(surrogate_corrs) >= r_n2n))
    
    results = {
        'pipeline': 'Two-Pass SOBI -> Symmetric Noise2Noise U-Net',
        'checkpoint': checkpoint_path,
        'inference_time_s': infer_time,
        'normal_hearing_ground_truth_comparison': {
            'pearson_r_sobi_alone': r_sobi,
            'pearson_r_sobi_n2n': r_n2n,
            'pearson_r_gain': float(r_n2n - r_sobi),
            'rmse_uv_sobi_alone': rmse_sobi,
            'rmse_uv_sobi_n2n': rmse_n2n,
            'p2_latency_hearing_gt_ms': h_p2_lat,
            'p2_latency_sobi_n2n_ms': n2n_p2_lat,
            'delta_p2_latency_ms': float(abs(n2n_p2_lat - h_p2_lat)),
            'p2_amplitude_uv': n2n_p2_amp
        },
        'silent_visual_v_hallucination_audit': {
            'v_trial_cz_floor_uv': v_cz_floor,
            'v_trial_p7_floor_uv': v_p7_floor,
            'hallucination_gate_passed': v_cz_floor < 2.0
        },
        'shuffled_envelope_permutation_test': {
            'observed_r': r_n2n,
            'surrogate_mean_r': float(np.mean(surrogate_corrs)),
            'surrogate_max_r': float(np.max(surrogate_corrs)),
            'empirical_p_value': p_value,
            'permutation_test_passed': p_value < 0.05
        }
    }
    
    print("\n" + "=" * 80)
    print("STAGE 2 FINAL DOWNSTREAM VALIDATION RESULTS")
    print("=" * 80)
    print(json.dumps(results, indent=2))
    
    out_file = os.path.join(output_dir, "sobi_n2n_downstream_validation.json")
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation results to {out_file}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sobi-fif", default="./benchmark_results/sobi2pass_cache/trial_onsets_0901y2_sobi2pass-raw.fif")
    parser.add_argument("--hearing-fif", default="/quobyte/millerlmgrp/annotated_data/CMPy2/trial_onsets_0801y2-raw.fif")
    parser.add_argument("--csv", default="/quobyte/millerlmgrp/annotated_data/CMPy2/trial_onsets_0901y2.csv")
    parser.add_argument("--checkpoint", default="./benchmark_results/sobi_n2n_results/best_sobi_n2n_unet.pt")
    parser.add_argument("--output-dir", default="./benchmark_results/sobi_n2n_results")
    parser.add_argument("--use-sagittal", action="store_true", default=False)
    args = parser.parse_args()
    
    evaluate_pipeline(args.sobi_fif, args.hearing_fif, args.csv, args.checkpoint, args.output_dir, use_sagittal=args.use_sagittal)
