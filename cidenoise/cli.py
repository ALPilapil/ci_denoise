#!/usr/bin/env python3
"""
cidenoise.cli
Unified Command-Line Interface for the cidenoise package.
"""

import sys
import os
import argparse
import json
import mne
import numpy as np

def clean_command(args):
    print(f"[cidenoise clean] Input: {args.input}")
    print(f"[cidenoise clean] Method: {args.method}")
    print(f"[cidenoise clean] Output: {args.output}")
    
    from processing.two_pass_sobi_preprocessor import run_two_pass_sobi_subject
    raw_clean, summary = run_two_pass_sobi_subject(
        input_fif=args.input,
        output_fif=args.output,
        target_fs=args.target_fs,
        highpass=args.highpass,
        lowpass=args.lowpass,
        max_passes=args.max_passes,
        r_thresh=args.r_thresh
    )
    
    if args.method == "sobi-n2n" and args.checkpoint:
        print(f"[cidenoise clean] Applying Stage 2 Subspace-Constrained N2N from {args.checkpoint}...")
        import torch
        from track_a_deep_learning.train_sobi_n2n import ResidualConstrainedUNet1D
        from track_a_deep_learning.evaluate_sobi_n2n_downstream import apply_sobi_n2n_sliding_window, CANONICAL_20_CHS
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        available_chs = [ch for ch in CANONICAL_20_CHS if ch in raw_clean.ch_names]
        
        # Build P_artifact from summary pass history
        A_art_list = []
        for p in summary.get('pass_history', []):
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
        model = ResidualConstrainedUNet1D(in_channels=len(available_chs), out_channels=len(available_chs)).to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        
        sobi_data = raw_clean.get_data(picks=available_chs)
        n2n_data = apply_sobi_n2n_sliding_window(model, sobi_data, p_art_tensor, device)
        
        raw_clean._data[[raw_clean.ch_names.index(ch) for ch in available_chs]] = n2n_data
        raw_clean.save(args.output, overwrite=True)
        print("[cidenoise clean] Continuous sliding-window N2N applied and saved successfully.")
        
    print("[cidenoise clean] Successfully completed!")

def audit_command(args):
    print(f"[cidenoise audit] Auditing file: {args.input}")
    raw = mne.io.read_raw_fif(args.input, preload=True, verbose=False)
    data = raw.get_data()
    n_chs, n_times = data.shape
    fs = raw.info['sfreq']
    
    # 1. Hardware Quality Check
    max_var = float(np.max(np.var(data, axis=1)))
    is_saturated = max_var > 1.0 # > 1 V^2 = 10^12 uV^2
    
    # 2. Midline Biological Power
    midline_chs = [ch for ch in ['Cz', 'Fz', 'Pz'] if ch in raw.ch_names]
    midline_indices = [raw.ch_names.index(ch) for ch in midline_chs]
    midline_power_uv2 = float(np.mean(np.var(data[midline_indices] * 1e6, axis=1))) if midline_indices else 0.0
    
    # 3. High-Pass Status
    hp = raw.info.get('highpass', 0.0)
    lp = raw.info.get('lowpass', 0.0)
    
    report = {
        'file': args.input,
        'n_channels': n_chs,
        'sfreq_hz': fs,
        'duration_s': n_times / fs,
        'hardware_filter': {
            'highpass_hz': hp,
            'lowpass_hz': lp
        },
        'hardware_saturation_detected': is_saturated,
        'max_channel_variance_v2': max_var,
        'midline_cortical_power_uv2': midline_power_uv2,
        'status': 'EXCLUDED_SATURATION' if is_saturated else 'PASSED'
    }
    
    print(json.dumps(report, indent=2))
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(report, f, indent=2)

def main():
    parser = argparse.ArgumentParser(
        prog="cidenoise",
        description="cidenoise: Turnkey Biophysical Disentanglement & Neural Network Denoising for Cochlear Implant EEG"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available sub-commands")
    
    # Clean sub-command
    clean_p = subparsers.add_parser("clean", help="Clean continuous CI EEG recording")
    clean_p.add_argument("--input", "-i", required=True, help="Input raw FIF file")
    clean_p.add_argument("--output", "-o", required=True, help="Output cleaned FIF file")
    clean_p.add_argument("--method", choices=["sobi-2pass", "sobi-n2n"], default="sobi-2pass", help="Denoising pipeline")
    clean_p.add_argument("--checkpoint", default=None, help="Path to trained PyTorch N2N model checkpoint")
    clean_p.add_argument("--target-fs", type=float, default=256.0, help="Target sampling rate (default: 256.0)")
    clean_p.add_argument("--highpass", type=float, default=1.0, help="High-pass filter cutoff in Hz (default: 1.0)")
    clean_p.add_argument("--lowpass", type=float, default=45.0, help="Low-pass filter cutoff in Hz (default: 45.0)")
    clean_p.add_argument("--max-passes", type=int, default=2, help="Max SOBI passes (default: 2)")
    clean_p.add_argument("--r-thresh", type=float, default=2.5, help="Mastoid-to-central ratio threshold")
    
    # Audit sub-command
    audit_p = subparsers.add_parser("audit", help="Audit recording biophysics and hardware QA")
    audit_p.add_argument("--input", "-i", required=True, help="Input raw FIF file")
    audit_p.add_argument("--output-json", default=None, help="Save audit report JSON")
    
    args = parser.parse_args()
    if args.command == "clean":
        clean_command(args)
    elif args.command == "audit":
        audit_command(args)
    else:
        parser.print_help()
        sys.exit(0)

if __name__ == "__main__":
    main()
