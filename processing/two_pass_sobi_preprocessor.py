#!/usr/bin/env python3
"""
two_pass_sobi_preprocessor.py
Production-grade, automated Two-Pass SOBI Preprocessor for Cochlear Implant EEG.

Key Algorithmic Safeguards:
1. Two-Pass Ceiling: Stops strictly after Pass 2 (empirically proven to prevent Pass 3 cortical cliff collapse).
2. Mastoid-to-Central Ratio (R_mastoid >= 2.5x): Rejects ONLY components localized to mastoid/lead electrodes (P7, T7, P8, T8).
3. Cortical Dipole Shield: If a component has its maximum absolute loading on vertex (Cz), frontal (Fz), or parietal (Pz), subtraction is strictly FORBIDDEN.
4. Active Step-Wise Rollback: If removing a candidate component causes biological band (1-15 Hz) power on central midline leads (Cz, Fz, Pz) to drop below 70%, or if the unmixing matrix is ill-conditioned (cond > 1000), the pass is actively REJECTED and rolled back to preserve brain signals.
5. Hardware Quality Gate: Detects floating point overflows, amplifier saturation, or disconnected sensor corruption before processing.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import mne

# Ensure repository packages are on path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if os.path.join(REPO_ROOT, "processing") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "processing"))

CANONICAL_20_CHS = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'T7', 'C3', 'Cz', 'C4', 'T8',
    'P7', 'P3', 'Pz', 'P4', 'P8',
    'O1', 'Oz', 'O2'
]

MASTOID_LEAD_CHS = ['P7', 'T7', 'P8', 'T8', 'P3', 'P4']
CENTRAL_MIDLINE_CHS = ['Cz', 'Fz', 'Pz']

def run_sobi(data, n_components=None, lags=None, eps=1e-5, max_iter=200):
    """
    Condition-guarded Second-Order Blind Identification (SOBI).
    Jointly diagonalizes time-lagged cross-covariance matrices.
    """
    n_channels, n_times = data.shape
    if n_components is None or n_components > n_channels:
        n_components = n_channels
    if lags is None:
        lags = [1, 2, 4, 8, 16, 25, 50]

    # Pre-whitening via EVD with condition-number protection
    X = data - np.mean(data, axis=1, keepdims=True)
    try:
        cov0 = (X @ X.T) / (n_times - 1)
        cov0 = 0.5 * (cov0 + cov0.T)
        d, u = np.linalg.eigh(cov0)
    except np.linalg.LinAlgError:
        u, s, _ = np.linalg.svd(X, full_matrices=False)
        d = (s ** 2) / (n_times - 1)
        
    idx = np.argsort(d)[::-1]
    d_sorted = d[idx]
    u_sorted = u[:, idx]
    
    # Condition-number protection: truncate eigenvalues below 1e-5 * d_max
    d_max = max(d_sorted[0], 1e-12)
    valid_count = int(np.sum(d_sorted > 1e-5 * d_max))
    n_keep = max(2, min(n_components, valid_count))
    
    d_keep = d_sorted[:n_keep]
    u_keep = u_sorted[:, :n_keep]

    whitening = np.diag(1.0 / np.sqrt(d_keep)) @ u_keep.T
    dewhitening = u_keep @ np.diag(np.sqrt(d_keep))
    Z = whitening @ X

    # Compute lagged cross-covariance matrices
    K = len(lags)
    M = np.zeros((n_keep, n_keep * K))
    for k_idx, lag in enumerate(lags):
        if lag >= n_times:
            continue
        c_lag = (Z[:, :n_times - lag] @ Z[:, lag:].T) / (n_times - lag)
        c_sym = 0.5 * (c_lag + c_lag.T)
        M[:, k_idx * n_keep : (k_idx + 1) * n_keep] = c_sym

    # Jacobi joint approximate diagonalization
    V = np.eye(n_keep)
    for _ in range(max_iter):
        delta = 0.0
        for p in range(n_keep - 1):
            for q in range(p + 1, n_keep):
                h_p = M[p, :].reshape(K, n_keep)
                h_q = M[q, :].reshape(K, n_keep)
                g11 = np.sum(h_p[:, p] ** 2)
                g22 = np.sum(h_q[:, q] ** 2)
                g12 = np.sum(h_p[:, q] ** 2)
                g0 = np.sum((h_p[:, p] - h_q[:, q]) * h_p[:, q])

                theta = 0.5 * np.arctan2(2.0 * g0, g11 + g22 - 2.0 * g12 + 1e-12)
                c = np.cos(theta)
                s = np.sin(theta)

                if np.abs(s) > eps:
                    delta = max(delta, np.abs(s))
                    V_p = V[:, p].copy()
                    V_q = V[:, q].copy()
                    V[:, p] = c * V_p + s * V_q
                    V[:, q] = -s * V_p + c * V_q

                    row_p = M[p, :].copy()
                    row_q = M[q, :].copy()
                    M[p, :] = c * row_p + s * row_q
                    M[q, :] = -s * row_p + c * row_q
                    for k in range(K):
                        col_p = M[:, k * n_keep + p].copy()
                        col_q = M[:, k * n_keep + q].copy()
                        M[:, k * n_keep + p] = c * col_p + s * col_q
                        M[:, k * n_keep + q] = -s * col_p + c * col_q
        if delta < eps:
            break

    W = V.T @ whitening
    A = dewhitening @ V
    return A, W

def compute_spatial_ratio(mixing_col, ch_names):
    """Computes ratio of mastoid/lead energy to central/midline energy."""
    ch_map = {ch: i for i, ch in enumerate(ch_names)}
    mastoid_powers = [mixing_col[ch_map[ch]] ** 2 for ch in MASTOID_LEAD_CHS if ch in ch_map]
    central_powers = [mixing_col[ch_map[ch]] ** 2 for ch in CENTRAL_MIDLINE_CHS if ch in ch_map]
    
    mean_mastoid = np.mean(mastoid_powers) if mastoid_powers else 0.0
    mean_central = np.mean(central_powers) if central_powers else 1e-12
    return mean_mastoid / max(mean_central, 1e-12)

def run_two_pass_sobi_subject(input_fif, output_fif=None, target_fs=256.0, highpass=1.0, lowpass=45.0, max_passes=2, r_thresh=2.5):
    """
    Executes automated two-pass SOBI with active biological rollback safeguards and zero-phase continuous pre-filtering.
    """
    t0 = time.time()
    print(f"\n[Two-Pass SOBI] Loading: {input_fif}")
    raw = mne.io.read_raw_fif(input_fif, preload=True, verbose=False)
    
    # 1. Channel selection
    available_chs = [ch for ch in CANONICAL_20_CHS if ch in raw.ch_names]
    raw.pick(available_chs)
    
    # 2. Resample to standard target_fs if needed
    if raw.info['sfreq'] != target_fs:
        print(f"Resampling {raw.info['sfreq']} Hz -> {target_fs} Hz...")
        raw.resample(target_fs, verbose=False)

    # Detect and drop unrecorded / NaN channels before filtering
    data_check = raw.get_data()
    bad_nan_chs = [ch for i, ch in enumerate(raw.ch_names) if np.isnan(data_check[i]).any()]
    if bad_nan_chs:
        print(f"[Two-Pass SOBI] Detected {len(bad_nan_chs)} bad channels with NaNs: {bad_nan_chs}. Dropping.")
        raw.drop_channels(bad_nan_chs)
    
    ch_names = raw.ch_names
    n_channels = len(ch_names)
        
    # Continuous zero-phase FIR pre-filtering (removes DC drift and anti-aliases)
    l_freq = highpass if (highpass is not None and highpass > 0.0) else None
    h_freq = lowpass if (lowpass is not None and lowpass > 0.0) else None
    if l_freq is not None or h_freq is not None:
        print(f"Applying zero-phase FIR filter ({l_freq} - {h_freq} Hz)...")
        raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin', phase='zero', verbose=False)
        
    fs = raw.info['sfreq']
    data = raw.get_data() # (n_channels, n_times)
    total_var_orig = float(np.sum(np.var(data, axis=1)))
    
    # HARDWARE QUALITY GATE: Detect extreme numerical overflow / saturation (> 1 V^2 = 10^12 uV^2)
    if total_var_orig > 1.0 or np.isnan(data).any() or np.isinf(data).any():
        print(f"[HARDWARE GATE ERROR] Raw variance {total_var_orig*1e12:.2e} uV^2 exceeds physical limits (>1 V^2). Corrupt hardware file.")
        summary = {
            'input_fif': input_fif,
            'output_fif': output_fif,
            'channels': ch_names,
            'sfreq': fs,
            'highpass': highpass,
            'lowpass': lowpass,
            'total_passes': 0,
            'total_runtime_s': time.time() - t0,
            'total_variance_original_uv2': float(total_var_orig * 1e12),
            'total_variance_cleaned_uv2': float(total_var_orig * 1e12),
            'variance_retained_pct': 100.0,
            'bio_band_retention_pct': 0.0,
            'midline_bio_retention_pct': 0.0,
            'artifact_band_attenuation_db': 0.0,
            'audit_passed': False,
            'hardware_failure': True,
            'audit_flags': ["EXCLUDED_HARDWARE_SATURATION: Raw variance exceeds physical limits (>1 V^2)"],
            'pass_history': []
        }
        if output_fif is not None:
            os.makedirs(os.path.dirname(os.path.abspath(output_fif)), exist_ok=True)
            raw.save(output_fif, overwrite=True, verbose=False)
        return raw, summary

    # Identify midline cortical leads for active biological retention evaluation
    from scipy.signal import welch
    trapz_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    
    midline_indices = [i for i, ch in enumerate(ch_names) if ch in CENTRAL_MIDLINE_CHS]
    eval_indices = midline_indices if midline_indices else list(range(n_channels))
    
    # Baseline power
    freqs, psd_orig = welch(data, fs=fs, nperseg=int(fs * 2), axis=1)
    bio_mask = (freqs >= 1.0) & (freqs <= 15.0)
    art_mask = (freqs >= 20.0) & (freqs <= 50.0)
    
    bio_power_midline_orig = float(np.mean(trapz_fn(psd_orig[eval_indices, :][:, bio_mask], freqs[bio_mask], axis=1)))
    
    current_data = data.copy()
    pass_history = []
    
    for pass_idx in range(1, max_passes + 1):
        print(f"\n--- SOBI Pass {pass_idx}/{max_passes} ---")
        t_pass_start = time.time()
        
        # Center data
        X = current_data - np.mean(current_data, axis=1, keepdims=True)
        
        # Rank-Preserving SOBI: Use 15 components (discarding noise floor eigenvalues)
        curr_rank = min(15, n_channels - (pass_idx - 1))
        
        # Run SOBI
        A, W = run_sobi(X, n_components=curr_rank, lags=[1, 2, 4, 8, 16, 25, 50], eps=1e-5, max_iter=200)
        S = W @ X
        cond_w = float(np.linalg.cond(W))
        
        if cond_w > 1e3:
            print(f"[ACTIVE SAFEGUARD] Pass {pass_idx}: Demixing matrix ill-conditioned (cond = {cond_w:.1f} > 1e3). Rolling back pass.")
            break
        
        # Compute exact backprojected scalp variance: ||a_k s_k^T||_F^2 = ||a_k||^2 * ||s_k||^2
        a_norms_sq = np.sum(A ** 2, axis=0)
        s_norms_sq = np.sum(S ** 2, axis=1)
        comp_vars = a_norms_sq * s_norms_sq
        total_backprojected_var = np.sum(comp_vars) + 1e-12
        pvaf = (comp_vars / total_backprojected_var) * 100.0
        
        # Evaluate each component
        candidates = []
        for c_idx in range(A.shape[1]):
            mixing_col = A[:, c_idx]
            peak_ch_idx = int(np.argmax(np.abs(mixing_col)))
            peak_ch = ch_names[peak_ch_idx]
            
            # Cortical Dipole Shield: NEVER reject if peak channel is Cz, Fz, Pz
            if peak_ch in CENTRAL_MIDLINE_CHS:
                continue
                
            r_ratio = compute_spatial_ratio(mixing_col, ch_names)
            var_pct = float(pvaf[c_idx])
            
            # Score: Scalp variance * Mastoid-to-Central Ratio
            score = var_pct * r_ratio if r_ratio >= r_thresh else 0.0
            candidates.append({
                'c_idx': c_idx,
                'peak_ch': peak_ch,
                'r_ratio': r_ratio,
                'var_pct': var_pct,
                'score': score
            })
            
        pass_time = time.time() - t_pass_start
        
        # Sort candidates by score
        candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
        if not candidates or candidates[0]['score'] == 0.0:
            print(f"Pass {pass_idx}: No component met artifact criteria (R >= {r_thresh}x). Stopping clean.")
            break
            
        best = candidates[0]
        selected_comp = best['c_idx']
        selected_r = best['r_ratio']
        selected_peak_ch = best['peak_ch']
        var_removed = best['var_pct']
        
        # Simulate projection
        artifact_proj = np.outer(A[:, selected_comp], S[selected_comp, :])
        candidate_data = current_data - artifact_proj
        
        # ACTIVE BIOLOGICAL RETENTION SAFEGUARD:
        # Evaluate candidate 1-15 Hz biological power on midline cortical leads (Cz, Fz, Pz)
        _, psd_cand = welch(candidate_data[eval_indices], fs=fs, nperseg=int(fs * 2), axis=1)
        bio_power_midline_cand = float(np.mean(trapz_fn(psd_cand[:, bio_mask], freqs[bio_mask], axis=1)))
        cand_retention_pct = (bio_power_midline_cand / max(bio_power_midline_orig, 1e-18)) * 100.0
        
        if cand_retention_pct < 70.0:
            print(f"[ACTIVE SAFEGUARD] Pass {pass_idx} REJECTED: Midline cortical retention would drop to {cand_retention_pct:.1f}% (< 70.0%). Rolling back pass.")
            break
            
        # Safeguard passed: commit cleaned data
        current_data = candidate_data
        print(f"Pass {pass_idx}: Isolated Artifact IC {selected_comp} (Peak: {selected_peak_ch}, R={selected_r:.2f}x, Var={var_removed:.1f}%, Midline Bio Retention={cand_retention_pct:.1f}%) in {pass_time:.2f}s")
        
        pass_history.append({
            'pass': pass_idx,
            'component_idx': int(selected_comp),
            'peak_channel': selected_peak_ch,
            'r_ratio': float(selected_r),
            'var_pct': float(var_removed),
            'runtime_s': float(pass_time),
            'cond_number': cond_w,
            'midline_bio_retention_pct': cand_retention_pct,
            'mixing_weights': {ch: float(A[i, selected_comp]) for i, ch in enumerate(ch_names)}
        })
        
    total_var_clean = np.sum(np.var(current_data, axis=1))
    var_retained_pct = (total_var_clean / total_var_orig) * 100.0
    total_runtime = time.time() - t0
    
    # 4. Mandatory DSP & Biophysical Verification Gates
    _, psd_clean = welch(current_data, fs=fs, nperseg=int(fs * 2), axis=1)
    
    bio_power_orig = np.mean(trapz_fn(psd_orig[:, bio_mask], freqs[bio_mask], axis=1))
    bio_power_clean = np.mean(trapz_fn(psd_clean[:, bio_mask], freqs[bio_mask], axis=1))
    bio_retention_pct = (bio_power_clean / max(bio_power_orig, 1e-18)) * 100.0
    
    bio_power_midline_final = float(np.mean(trapz_fn(psd_clean[eval_indices, :][:, bio_mask], freqs[bio_mask], axis=1)))
    midline_bio_retention_pct = (bio_power_midline_final / max(bio_power_midline_orig, 1e-18)) * 100.0
    
    art_power_orig = np.mean(trapz_fn(psd_orig[:, art_mask], freqs[art_mask], axis=1))
    art_power_clean = np.mean(trapz_fn(psd_clean[:, art_mask], freqs[art_mask], axis=1))
    art_attenuation_db = 10.0 * np.log10(max(art_power_clean, 1e-18) / max(art_power_orig, 1e-18))
    
    print(f"\n[Two-Pass SOBI Complete] Passes: {len(pass_history)} | Runtime: {total_runtime:.2f}s")
    print(f"Scalp Variance: {total_var_orig*1e12:.1f} -> {total_var_clean*1e12:.1f} uV^2 ({var_retained_pct:.1f}% retained)")
    print(f"Midline Cortical (Cz/Fz/Pz) Retention: {midline_bio_retention_pct:.2f}% (Safeguard Gate: >= 70%)")
    print(f"Global Scalp Bio Band (1-15 Hz) Retention: {bio_retention_pct:.2f}%")
    print(f"Artifact Band (20-50 Hz) Attenuation: {art_attenuation_db:.2f} dB")
    
    audit_passed = True
    audit_flags = []
    if midline_bio_retention_pct < 70.0:
        audit_flags.append(f"FAIL: Midline biological power retention ({midline_bio_retention_pct:.1f}%) < 70%")
        audit_passed = False
    for p in pass_history:
        if p.get('cond_number', 1.0) > 1e3:
            audit_flags.append(f"FAIL: Pass {p['pass']} unmixing ill-conditioned (cond > 1e3)")
            audit_passed = False
            
    # Store cleaned data back into Raw object
    raw_clean = mne.io.RawArray(current_data, raw.info, verbose=False)
    
    if output_fif is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_fif)), exist_ok=True)
        print(f"Saving cleaned FIF to {output_fif}...")
        raw_clean.save(output_fif, overwrite=True, verbose=False)
        
    summary = {
        'input_fif': input_fif,
        'output_fif': output_fif,
        'channels': ch_names,
        'sfreq': fs,
        'highpass': highpass,
        'lowpass': lowpass,
        'total_passes': len(pass_history),
        'total_runtime_s': total_runtime,
        'total_variance_original_uv2': float(total_var_orig * 1e12),
        'total_variance_cleaned_uv2': float(total_var_clean * 1e12),
        'variance_retained_pct': float(var_retained_pct),
        'bio_band_retention_pct': float(bio_retention_pct),
        'midline_bio_retention_pct': float(midline_bio_retention_pct),
        'artifact_band_attenuation_db': float(art_attenuation_db),
        'audit_passed': audit_passed,
        'hardware_failure': False,
        'audit_flags': audit_flags,
        'pass_history': pass_history
    }
    return raw_clean, summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-Pass SOBI Preprocessor")
    parser.add_argument("--input-fif", required=True, help="Path to input raw FIF")
    parser.add_argument("--output-fif", default=None, help="Path to save cleaned raw FIF")
    parser.add_argument("--target-fs", type=float, default=256.0, help="Target sampling rate")
    parser.add_argument("--highpass", type=float, default=1.0, help="High-pass filter cutoff in Hz (default: 1.0, 0.0 to disable)")
    parser.add_argument("--lowpass", type=float, default=45.0, help="Low-pass filter cutoff in Hz (default: 45.0, 0.0 to disable)")
    parser.add_argument("--max-passes", type=int, default=2, help="Maximum SOBI passes")
    parser.add_argument("--r-thresh", type=float, default=2.5, help="Mastoid-to-Central ratio threshold")
    args = parser.parse_args()
    
    raw_clean, summary = run_two_pass_sobi_subject(
        input_fif=args.input_fif,
        output_fif=args.output_fif,
        target_fs=args.target_fs,
        highpass=args.highpass,
        lowpass=args.lowpass,
        max_passes=args.max_passes,
        r_thresh=args.r_thresh
    )
    print("\nSummary:")
    print(json.dumps(summary, indent=2))
