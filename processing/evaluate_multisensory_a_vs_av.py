#!/usr/bin/env python3
"""
evaluate_multisensory_a_vs_av.py
Evaluates multisensory integration (Audio-only 'A' vs Audiovisual 'AV' vs Visual-only 'V')
across cleaned epochs for both Cochlear Implant and Normal-Hearing cohorts.
Implements the canonical additive model [AV - (A + V)] to rigorously test
whether enhancement reflects true superadditive neural integration or linear superposition of the VEP.
"""

import os
import glob
import numpy as np
import pandas as pd
import mne
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def evaluate_cohort_multisensory(cohort_dir, cohort_type="ci"):
    csv_dir = "/quobyte/millerlmgrp/annotated_data/CMPy2"
    epoch_files = sorted(glob.glob(os.path.join(cohort_dir, "*_clean_epochs.npz")))
    print(f"[{cohort_type.upper()}] Found {len(epoch_files)} clean epoch files in {cohort_dir}")
    
    results = []
    
    for ep_file in epoch_files:
        sub = os.path.basename(ep_file).replace("trial_onsets_", "").replace("_clean_epochs.npz", "")
        csv_file = os.path.join(csv_dir, f"trial_onsets_{sub}.csv")
        if not os.path.exists(csv_file):
            continue
            
        df = pd.read_csv(csv_file)
        npz = np.load(ep_file, allow_pickle=True)
        clean_speech = npz['clean_speech'] # (n_pairs, n_chs, n_times)
        stim_ids = npz['stim_ids']
        channels = list(npz['channels'])
        times = npz['times']
        
        if 'Cz' not in channels:
            continue
        cz_idx = channels.index('Cz')
        
        # Map stim_id to condition (A, AV, V)
        stim_to_cond = {}
        for sid, grp in df.groupby('stim_id'):
            conds = grp['condition'].unique()
            if len(conds) == 1:
                stim_to_cond[sid] = conds[0]
                
        a_indices = [i for i, sid in enumerate(stim_ids) if stim_to_cond.get(sid) == 'A']
        av_indices = [i for i, sid in enumerate(stim_ids) if stim_to_cond.get(sid) == 'AV']
        
        if not a_indices or not av_indices:
            continue
            
        # P2 window: 150 - 240 ms
        p2_mask = (times >= 0.150) & (times <= 0.240)
        
        # Cz ERP for A (convert Volts -> uV)
        erp_a = np.mean(clean_speech[a_indices, cz_idx, :], axis=0) * 1e6
        p2_amp_a = float(np.max(erp_a[p2_mask]))
        p2_lat_a = float(times[p2_mask][np.argmax(erp_a[p2_mask])] * 1000.0)
        
        # Cz ERP for AV (convert Volts -> uV)
        erp_av = np.mean(clean_speech[av_indices, cz_idx, :], axis=0) * 1e6
        p2_amp_av = float(np.max(erp_av[p2_mask]))
        p2_lat_av = float(times[p2_mask][np.argmax(erp_av[p2_mask])] * 1000.0)
        
        # Condition V (Silent Visual speech) extraction
        erp_v = None
        if 'v_clean' in npz and len(npz['v_clean']) > 0:
            erp_v = np.mean(npz['v_clean'][:, cz_idx, :], axis=0) * 1e6
        else:
            clean_fif = os.path.join(cohort_dir, f"trial_onsets_{sub}_sobi2pass-raw.fif")
            if os.path.exists(clean_fif):
                try:
                    raw_c = mne.io.read_raw_fif(clean_fif, preload=True, verbose=False)
                    v_trials = df[df['condition'] == 'V']
                    v_segs = []
                    fs_fif = raw_c.info['sfreq']
                    t_pre_pts = int(0.2 * fs_fif)
                    ep_len = int(0.8 * fs_fif)
                    if 'Cz' in raw_c.ch_names:
                        cz_c_idx = raw_c.ch_names.index('Cz')
                        d_c = raw_c.get_data(picks=[cz_c_idx])
                        for _, row in v_trials.iterrows():
                            s0 = int(row['onset_s'] * fs_fif) - t_pre_pts
                            s1 = s0 + ep_len
                            if s0 >= 0 and s1 <= d_c.shape[1]:
                                seg = d_c[0, s0:s1].copy()
                                seg -= np.mean(seg[:t_pre_pts])
                                v_segs.append(seg)
                        if v_segs:
                            erp_v = np.mean(v_segs, axis=0) * 1e6
                except Exception:
                    pass
                    
        p2_amp_v = float(np.max(erp_v[p2_mask])) if erp_v is not None else np.nan
        p2_lat_v = float(times[p2_mask][np.argmax(erp_v[p2_mask])] * 1000.0) if erp_v is not None else np.nan
        
        # Canonical Additive Superposition Model: AV vs (A + V)
        additive_expected = (p2_amp_a + p2_amp_v) if erp_v is not None else np.nan
        interaction_diff = (p2_amp_av - additive_expected) if erp_v is not None else np.nan
        
        results.append({
            'subject': sub,
            'cohort': cohort_type,
            'n_pairs_a': len(a_indices),
            'n_pairs_av': len(av_indices),
            'p2_amp_a_uv': p2_amp_a,
            'p2_lat_a_ms': p2_lat_a,
            'p2_amp_v_uv': p2_amp_v,
            'p2_lat_v_ms': p2_lat_v,
            'p2_amp_av_uv': p2_amp_av,
            'p2_lat_av_ms': p2_lat_av,
            'multisensory_gain_uv': p2_amp_av - p2_amp_a,
            'multisensory_gain_pct': ((p2_amp_av - p2_amp_a) / max(abs(p2_amp_a), 1e-6)) * 100.0,
            'additive_expected_uv': additive_expected,
            'interaction_diff_uv': interaction_diff
        })
        
    return pd.DataFrame(results)

def main():
    ci_dir = "./benchmark_results/sobi_polarity_cohort"
    nh_dir = "./benchmark_results/sobi_polarity_nh_cohort"
    output_dir = "./benchmark_results/sobi_polarity_cohort"
    
    df_ci = evaluate_cohort_multisensory(ci_dir, cohort_type="ci")
    df_nh = evaluate_cohort_multisensory(nh_dir, cohort_type="nh")
    
    # Filter uncalibrated hardware outlier (0923y2)
    df_ci = df_ci[df_ci['p2_amp_av_uv'].abs() < 50.0]
    
    print("\n" + "=" * 80)
    print("MULTISENSORY INTEGRATION (A vs AV vs V) SUMMARY")
    print("=" * 80)
    print(f"Normal-Hearing (N = {len(df_nh)}):")
    print(f"  P2 Amplitude Audio (A):          {df_nh['p2_amp_a_uv'].mean():.2f} +/- {df_nh['p2_amp_a_uv'].std():.2f} uV")
    if 'p2_amp_v_uv' in df_nh.columns and df_nh['p2_amp_v_uv'].notna().any():
        print(f"  P2 Amplitude Visual (V):         {df_nh['p2_amp_v_uv'].dropna().mean():.2f} +/- {df_nh['p2_amp_v_uv'].dropna().std():.2f} uV")
    print(f"  P2 Amplitude Audiovisual (AV):   {df_nh['p2_amp_av_uv'].mean():.2f} +/- {df_nh['p2_amp_av_uv'].std():.2f} uV")
    print(f"  Direct Gain (AV - A):            +{df_nh['multisensory_gain_uv'].mean():.2f} uV (+{df_nh['multisensory_gain_pct'].mean():.1f}%)")
    if 'interaction_diff_uv' in df_nh.columns and df_nh['interaction_diff_uv'].notna().any():
        valid_int = df_nh['interaction_diff_uv'].dropna()
        print(f"  Additive Interaction [AV-(A+V)]: {valid_int.mean():.2f} +/- {valid_int.std():.2f} uV")
    
    print(f"\nCochlear Implant (N = {len(df_ci)}):")
    print(f"  P2 Amplitude Audio (A):          {df_ci['p2_amp_a_uv'].mean():.2f} +/- {df_ci['p2_amp_a_uv'].std():.2f} uV")
    if 'p2_amp_v_uv' in df_ci.columns and df_ci['p2_amp_v_uv'].notna().any():
        print(f"  P2 Amplitude Visual (V):         {df_ci['p2_amp_v_uv'].dropna().mean():.2f} +/- {df_ci['p2_amp_v_uv'].dropna().std():.2f} uV")
    print(f"  P2 Amplitude Audiovisual (AV):   {df_ci['p2_amp_av_uv'].mean():.2f} +/- {df_ci['p2_amp_av_uv'].std():.2f} uV")
    print(f"  Direct Gain (AV - A):            +{df_ci['multisensory_gain_uv'].mean():.2f} uV (+{df_ci['multisensory_gain_pct'].mean():.1f}%)")
    if 'interaction_diff_uv' in df_ci.columns and df_ci['interaction_diff_uv'].notna().any():
        valid_int_ci = df_ci['interaction_diff_uv'].dropna()
        print(f"  Additive Interaction [AV-(A+V)]: {valid_int_ci.mean():.2f} +/- {valid_int_ci.std():.2f} uV")
    print("=" * 80)
    
    df_all = pd.concat([df_nh, df_ci], ignore_index=True)
    csv_out = os.path.join(output_dir, "multisensory_a_vs_av_summary.csv")
    df_all.to_csv(csv_out, index=False)
    print(f"Saved multisensory table to {csv_out}")
    
    # Plot Multisensory Enhancement
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel A: Normal Hearing
    ax = axes[0]
    nh_a = df_nh['p2_amp_a_uv']
    nh_av = df_nh['p2_amp_av_uv']
    for i in range(len(df_nh)):
        ax.plot([1, 2], [nh_a.iloc[i], nh_av.iloc[i]], color='gray', alpha=0.5, linewidth=1.2)
    ax.scatter([1]*len(df_nh), nh_a, color='#2b5c8f', s=50, label='Audio (A)', zorder=3)
    ax.scatter([2]*len(df_nh), nh_av, color='#41b6c4', s=50, label='Audiovisual (AV)', zorder=3)
    ax.plot([1, 2], [nh_a.mean(), nh_av.mean()], color='black', linewidth=3, marker='o', markersize=8, label='Group Mean', zorder=4)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Audio (A)', 'Audiovisual (AV)'], fontsize=11)
    ax.set_ylabel('Cz P2 Amplitude (µV)', fontsize=11)
    ax.set_title(f'A. Normal-Hearing Controls (N = {len(df_nh)})\nGain: +{df_nh["multisensory_gain_uv"].mean():.2f} µV', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=9)
    
    # Panel B: Cochlear Implant
    ax = axes[1]
    ci_a = df_ci['p2_amp_a_uv']
    ci_av = df_ci['p2_amp_av_uv']
    for i in range(len(df_ci)):
        ax.plot([1, 2], [ci_a.iloc[i], ci_av.iloc[i]], color='gray', alpha=0.5, linewidth=1.2)
    ax.scatter([1]*len(df_ci), ci_a, color='#d95f02', s=50, label='Audio (A)', zorder=3)
    ax.scatter([2]*len(df_ci), ci_av, color='#fdae61', s=50, label='Audiovisual (AV)', zorder=3)
    ax.plot([1, 2], [ci_a.mean(), ci_av.mean()], color='black', linewidth=3, marker='o', markersize=8, label='Group Mean', zorder=4)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Audio (A)', 'Audiovisual (AV)'], fontsize=11)
    ax.set_ylabel('Cz P2 Amplitude (µV)', fontsize=11)
    ax.set_title(f'B. Cochlear Implant Recipients (N = {len(df_ci)})\nGain: +{df_ci["multisensory_gain_uv"].mean():.2f} µV', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=9)
    
    plt.tight_layout()
    plot_out = os.path.join(output_dir, "multisensory_integration_a_vs_av.png")
    plt.savefig(plot_out, dpi=300)
    plt.close()
    print(f"Saved multisensory integration figure to {plot_out}")

if __name__ == "__main__":
    main()
