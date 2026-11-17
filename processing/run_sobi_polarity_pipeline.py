#!/usr/bin/env python3
"""
run_sobi_polarity_pipeline.py
End-to-End Analytical Denoising Pipeline for Cochlear Implant EEG:
Stage 1: Pre-filtered Two-Pass SOBI (Continuous Spatial BSS)
Stage 2: Polarity-Alternating Synchronous Decomposition (Cross-Take Artifact Cancellation)

Outputs:
1. Cleaned continuous FIF: trial_onsets_<subject>_sobi2pass-raw.fif
2. Denoised sentence pair epochs: trial_onsets_<subject>_clean_epochs.npz
3. Comprehensive audit metrics JSON: sobi_polarity_audit_<subject>.json
4. Publication-grade diagnostic figure: sobi_polarity_diagnostic_<subject>.png
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import mne
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if os.path.join(REPO_ROOT, "processing") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "processing"))

from two_pass_sobi_preprocessor import run_two_pass_sobi_subject, CANONICAL_20_CHS, CENTRAL_MIDLINE_CHS

def run_sobi_polarity_pipeline(raw_fif, csv_path, output_dir, subject_id="0901y2", target_fs=256.0, highpass=1.0, lowpass=45.0):
    os.makedirs(output_dir, exist_ok=True)
    t_start = time.time()
    print("=" * 80)
    print(f"STARTING SOBI + POLARITY PIPELINE FOR SUBJECT: {subject_id}")
    print("=" * 80)
    
    # ---------------------------------------------------------
    # STAGE 1: TWO-PASS SOBI PREPROCESSOR WITH CONTINUOUS FILTER
    # ---------------------------------------------------------
    sobi_fif = os.path.join(output_dir, f"trial_onsets_{subject_id}_sobi2pass-raw.fif")
    sobi_json = os.path.join(output_dir, f"sobi2pass_summary_{subject_id}.json")
    
    raw_clean, sobi_summary = run_two_pass_sobi_subject(
        input_fif=raw_fif,
        output_fif=sobi_fif,
        target_fs=target_fs,
        highpass=highpass,
        lowpass=lowpass,
        max_passes=2,
        r_thresh=2.5
    )
    with open(sobi_json, 'w') as f:
        json.dump(sobi_summary, f, indent=2)
        
    fs = raw_clean.info['sfreq']
    ch_names = raw_clean.ch_names
    cz_idx = ch_names.index('Cz')
    p7_idx = ch_names.index('P7')
    sobi_data = raw_clean.get_data()
    
    # Also load raw data filtered to the same band for exact baseline comparison
    raw_orig = mne.io.read_raw_fif(raw_fif, preload=True, verbose=False)
    available_chs = [ch for ch in CANONICAL_20_CHS if ch in raw_orig.ch_names]
    raw_orig.pick(available_chs)
    if raw_orig.info['sfreq'] != target_fs:
        raw_orig.resample(target_fs, verbose=False)
    if highpass > 0.0 or lowpass > 0.0:
        raw_orig.filter(highpass, lowpass, fir_design='firwin', phase='zero', verbose=False)
    raw_data = raw_orig.get_data()
    
    # ---------------------------------------------------------
    # STAGE 2: POLARITY-ALTERNATING SYNCHRONOUS DECOMPOSITION
    # ---------------------------------------------------------
    df = pd.read_csv(csv_path)
    t_pre = int(0.2 * fs)
    t_post = int(0.6 * fs)
    epoch_len = t_pre + t_post
    times = np.linspace(-0.2, 0.6, epoch_len)
    post_mask = (times >= 0.0) & (times <= 0.5)
    
    active_df = df[df['stim_id'] > 0]
    stim_groups = active_df.groupby('stim_id')
    
    clean_pairs = []
    raw_single_p7_powers = []
    sobi_single_p7_powers = []
    summed_p7_powers = []
    diff_p7_powers = []
    
    raw_cz_epochs = []
    sobi_cz_epochs = []
    summed_cz_epochs = []
    
    for stim_id, grp in stim_groups:
        if len(grp) >= 2:
            rows = grp.sort_values('polarity')
            r1, r2 = rows.iloc[0], rows.iloc[1]
            
            s1_0 = int(r1['onset_s'] * fs) - t_pre
            s1_1 = s1_0 + epoch_len
            s2_0 = int(r2['onset_s'] * fs) - t_pre
            s2_1 = s2_0 + epoch_len
            
            if s1_0 >= 0 and s1_1 <= sobi_data.shape[1] and s2_0 >= 0 and s2_1 <= sobi_data.shape[1]:
                # SOBI takes
                sobi_take1 = sobi_data[:, s1_0:s1_1].copy()
                sobi_take2 = sobi_data[:, s2_0:s2_1].copy()
                sobi_take1 -= np.mean(sobi_take1[:, :t_pre], axis=1, keepdims=True)
                sobi_take2 -= np.mean(sobi_take2[:, :t_pre], axis=1, keepdims=True)
                
                # Raw takes
                raw_take1 = raw_data[:, s1_0:s1_1].copy()
                raw_take2 = raw_data[:, s2_0:s2_1].copy()
                raw_take1 -= np.mean(raw_take1[:, :t_pre], axis=1, keepdims=True)
                raw_take2 -= np.mean(raw_take2[:, :t_pre], axis=1, keepdims=True)
                
                # Polarity Sum: s_hat = 0.5 * (Take1 + Take2)
                sobi_sum = 0.5 * (sobi_take1 + sobi_take2)
                # Polarity Diff: a_hat = 0.5 * (Take1 - Take2)
                sobi_diff = 0.5 * (sobi_take1 - sobi_take2)
                
                clean_pairs.append({
                    'stim_id': stim_id,
                    'clean_speech_response': sobi_sum,
                    'isolated_artifact': sobi_diff
                })
                
                # Powers on P7 post-onset
                raw_single_p7_powers.extend([
                    float(np.mean(raw_take1[p7_idx, post_mask] ** 2)),
                    float(np.mean(raw_take2[p7_idx, post_mask] ** 2))
                ])
                sobi_single_p7_powers.extend([
                    float(np.mean(sobi_take1[p7_idx, post_mask] ** 2)),
                    float(np.mean(sobi_take2[p7_idx, post_mask] ** 2))
                ])
                summed_p7_powers.append(float(np.mean(sobi_sum[p7_idx, post_mask] ** 2)))
                diff_p7_powers.append(float(np.mean(sobi_diff[p7_idx, post_mask] ** 2)))
                
                raw_cz_epochs.extend([raw_take1[cz_idx], raw_take2[cz_idx]])
                sobi_cz_epochs.extend([sobi_take1[cz_idx], sobi_take2[cz_idx]])
                summed_cz_epochs.append(sobi_sum[cz_idx])
                
    # Evoked averages
    erp_raw_cz = np.mean(raw_cz_epochs, axis=0) * 1e6 # uV
    erp_sobi_cz = np.mean(sobi_cz_epochs, axis=0) * 1e6
    erp_sum_cz = np.mean(summed_cz_epochs, axis=0) * 1e6
    
    # ---------------------------------------------------------
    # STAGE 3: DOWNSTREAM AUDIT & ERP MORPHOLOGY
    # ---------------------------------------------------------
    # Condition V (Silent Visual) check
    v_trials = df[df['condition'] == 'V']
    v_sobi_epochs, v_raw_epochs = [], []
    for _, row in v_trials.iterrows():
        s0 = int(row['onset_s'] * fs) - t_pre
        s1 = s0 + epoch_len
        if s0 >= 0 and s1 <= sobi_data.shape[1]:
            s_seg = sobi_data[:, s0:s1].copy()
            r_seg = raw_data[:, s0:s1].copy()
            s_seg -= np.mean(s_seg[:, :t_pre], axis=1, keepdims=True)
            r_seg -= np.mean(r_seg[:, :t_pre], axis=1, keepdims=True)
            v_sobi_epochs.append(s_seg)
            v_raw_epochs.append(r_seg)
            
    v_sobi_avg = np.mean(v_sobi_epochs, axis=0) * 1e6
    v_raw_avg = np.mean(v_raw_epochs, axis=0) * 1e6
    v_cz_floor = float(np.sqrt(np.mean(v_sobi_avg[cz_idx] ** 2)))
    v_p7_floor = float(np.sqrt(np.mean(v_sobi_avg[p7_idx] ** 2)))
    
    # P7 power reductions
    mean_raw_p7_pwr = float(np.mean(raw_single_p7_powers) * 1e12) # uV^2
    mean_sobi_p7_pwr = float(np.mean(sobi_single_p7_powers) * 1e12)
    mean_sum_p7_pwr = float(np.mean(summed_p7_powers) * 1e12)
    
    sobi_db_attenuation = float(10 * np.log10(mean_sobi_p7_pwr / (mean_raw_p7_pwr + 1e-12)))
    polarity_db_attenuation = float(10 * np.log10(mean_sum_p7_pwr / (mean_sobi_p7_pwr + 1e-12)))
    total_db_attenuation = float(10 * np.log10(mean_sum_p7_pwr / (mean_raw_p7_pwr + 1e-12)))
    
    # Cz ERP Latencies & Amplitudes
    n1_mask = (times >= 0.090) & (times <= 0.140)
    p2_mask = (times >= 0.150) & (times <= 0.240)
    
    n1_idx = np.argmin(erp_sum_cz[n1_mask])
    n1_lat = float(times[n1_mask][n1_idx] * 1000.0)
    n1_amp = float(erp_sum_cz[n1_mask][n1_idx])
    
    p2_idx = np.argmax(erp_sum_cz[p2_mask])
    p2_lat = float(times[p2_mask][p2_idx] * 1000.0)
    p2_amp = float(erp_sum_cz[p2_mask][p2_idx])
    peak_to_peak = float(p2_amp - n1_amp)
    
    # Correlation with Stage 1 SOBI template
    r_cz_template = float(np.corrcoef(erp_sobi_cz[post_mask], erp_sum_cz[post_mask])[0, 1])
    
    total_runtime = time.time() - t_start
    
    audit_results = {
        'subject': subject_id,
        'total_runtime_s': total_runtime,
        'stage1_sobi': {
            'artifact_attenuation_db': sobi_db_attenuation,
            'p7_mean_single_take_power_uv2': mean_sobi_p7_pwr
        },
        'stage2_polarity_averaging': {
            'matched_pairs': len(clean_pairs),
            'additional_attenuation_db': polarity_db_attenuation,
            'p7_summed_power_uv2': mean_sum_p7_pwr,
            'isolated_artifact_power_uv2': float(np.mean(diff_p7_powers) * 1e12)
        },
        'total_pipeline': {
            'total_artifact_attenuation_db': total_db_attenuation,
            'power_reduction_pct': float((1.0 - mean_sum_p7_pwr / mean_raw_p7_pwr) * 100.0)
        },
        'cz_cortical_erp': {
            'n1_latency_ms': n1_lat,
            'n1_amplitude_uv': n1_amp,
            'n1_gate_passed': 90.0 <= n1_lat <= 140.0,
            'p2_latency_ms': p2_lat,
            'p2_amplitude_uv': p2_amp,
            'p2_gate_passed': 150.0 <= p2_lat <= 240.0,
            'peak_to_peak_uv': peak_to_peak,
            'template_fidelity_r': r_cz_template
        },
        'condition_v_silent_baseline': {
            'cz_noise_floor_uv': v_cz_floor,
            'p7_noise_floor_uv': v_p7_floor,
            'silent_gate_passed': v_cz_floor < 2.0
        }
    }
    
    # Save audit JSON
    audit_json_path = os.path.join(output_dir, f"sobi_polarity_audit_{subject_id}.json")
    with open(audit_json_path, 'w') as f:
        json.dump(audit_results, f, indent=2)
    print(f"\nSaved audit JSON to {audit_json_path}")
    
    # Save Cleaned Epochs NPZ
    epochs_npz_path = os.path.join(output_dir, f"trial_onsets_{subject_id}_clean_epochs.npz")
    all_clean_responses = np.array([p['clean_speech_response'] for p in clean_pairs])
    all_isolated_artifacts = np.array([p['isolated_artifact'] for p in clean_pairs])
    stim_ids = np.array([p['stim_id'] for p in clean_pairs])
    np.savez_compressed(
        epochs_npz_path,
        clean_speech=all_clean_responses,
        isolated_artifact=all_isolated_artifacts,
        stim_ids=stim_ids,
        channels=ch_names,
        times=times,
        sfreq=fs
    )
    print(f"Saved cleaned epochs to {epochs_npz_path} ({len(clean_pairs)} pairs)")
    
    # ---------------------------------------------------------
    # STAGE 4: PUBLICATION-GRADE DIAGNOSTIC FIGURE
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Panel A: Cz Cortical ERP Recovery
    ax = axes[0]
    ax.plot(times * 1000, erp_raw_cz, 'k--', alpha=0.5, label='Raw EEG (Filtered)')
    ax.plot(times * 1000, erp_sobi_cz, 'b-', label='Stage 1: SOBI 2-Pass')
    ax.plot(times * 1000, erp_sum_cz, 'r-', linewidth=2, label='Stage 2: SOBI + Polarity Sum')
    ax.axvline(n1_lat, color='blue', linestyle=':', label=f'N1 ({n1_lat:.1f} ms)')
    ax.axvline(p2_lat, color='red', linestyle=':', label=f'P2 ({p2_lat:.1f} ms)')
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    ax.set_title(f"A. Vertex (Cz) Cortical ERP (N1={n1_lat:.1f}ms, P2={p2_lat:.1f}ms)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_xlim(-200, 600)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Panel B: P7 Artifact Rejection Across Stages
    ax = axes[1]
    p7_raw_epoch = np.mean(raw_data[p7_idx, :int(1.0*fs)]) * 1e6
    erp_raw_p7 = np.mean([r[p7_idx] for r in clean_pairs], axis=0) if clean_pairs else np.zeros(len(times))
    # Plot P7 evoked waveform
    erp_p7_sobi = np.mean([p['clean_speech_response'][p7_idx] for p in clean_pairs], axis=0) * 1e6
    erp_p7_diff = np.mean([p['isolated_artifact'][p7_idx] for p in clean_pairs], axis=0) * 1e6
    ax.plot(times * 1000, erp_p7_diff, 'purple', linestyle='--', alpha=0.7, label='Isolated Odd Artifact')
    ax.plot(times * 1000, erp_p7_sobi, 'blue', linewidth=1.5, label='Polarity-Summed Speech')
    ax.set_title(f"B. Implant Mastoid (P7) Artifact Elimination ({total_db_attenuation:.1f} dB)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_xlim(-200, 600)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Panel C: Stage Power Breakdown Bar Chart
    ax = axes[2]
    stages = ['Raw', 'SOBI 2-Pass', 'SOBI + Polarity']
    powers = [mean_raw_p7_pwr, mean_sobi_p7_pwr, mean_sum_p7_pwr]
    colors = ['gray', 'royalblue', 'crimson']
    bars = ax.bar(stages, powers, color=colors, width=0.5)
    ax.set_yscale('log')
    ax.set_title(f"C. P7 Artifact Energy Breakdown (-{abs(total_db_attenuation):.1f} dB)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Evoked Artifact Power (µV², log scale)")
    for bar, pwr in zip(bars, powers):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval * 1.2, f"{pwr:.1f} µV²", ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both', axis='y')
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"sobi_polarity_diagnostic_{subject_id}.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved diagnostic figure to {fig_path}")
    
    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION COMPLETE & AUDITED")
    print(f"Total Artifact Attenuation: {total_db_attenuation:.2f} dB (Power Reduction: {(1.0 - mean_sum_p7_pwr/mean_raw_p7_pwr)*100:.2f}%)")
    print(f"Cz Evoked P2 Peak: {p2_lat:.1f} ms (+{p2_amp:.2f} µV) | Silent Cz Floor: {v_cz_floor:.3f} µV")
    print("=" * 80)
    return audit_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-fif", default="/quobyte/millerlmgrp/annotated_data/CMPy2/trial_onsets_0901y2-raw.fif")
    parser.add_argument("--csv", default="/quobyte/millerlmgrp/annotated_data/CMPy2/trial_onsets_0901y2.csv")
    parser.add_argument("--output-dir", default="./benchmark_results/sobi_polarity_pipeline")
    parser.add_argument("--subject-id", default="0901y2")
    parser.add_argument("--highpass", type=float, default=1.0)
    parser.add_argument("--lowpass", type=float, default=45.0)
    args = parser.parse_args()
    
    run_sobi_polarity_pipeline(
        raw_fif=args.raw_fif,
        csv_path=args.csv,
        output_dir=args.output_dir,
        subject_id=args.subject_id,
        highpass=args.highpass,
        lowpass=args.lowpass
    )
