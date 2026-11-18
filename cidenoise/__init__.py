"""
cidenoise: Turnkey Biophysical Disentanglement & Neural Network Denoising for Cochlear Implant EEG
"""

__version__ = "1.0.0"

import os
import sys
import numpy as np

from processing.two_pass_sobi_preprocessor import run_two_pass_sobi_subject

def audit(input_fif: str, output_json: str = None) -> dict:
    """
    Audit recording biophysics, floating-point saturation, and midline cortical power.
    """
    import mne
    import json
    
    raw = mne.io.read_raw_fif(input_fif, preload=True, verbose=False)
    data = raw.get_data()
    n_chs, n_times = data.shape
    fs = raw.info['sfreq']
    
    max_var = float(np.max(np.var(data, axis=1)))
    is_saturated = max_var > 1.0
    
    midline_chs = [ch for ch in ['Cz', 'Fz', 'Pz'] if ch in raw.ch_names]
    midline_indices = [raw.ch_names.index(ch) for ch in midline_chs]
    midline_power_uv2 = float(np.mean(np.var(data[midline_indices] * 1e6, axis=1))) if midline_indices else 0.0
    
    hp = raw.info.get('highpass', 0.0)
    lp = raw.info.get('lowpass', 0.0)
    
    report = {
        'file': input_fif,
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
    
    if output_json:
        with open(output_json, 'w') as f:
            json.dump(report, f, indent=2)
            
    return report

def clean(
    input_fif: str,
    output_fif: str,
    method: str = "sobi-2pass",
    checkpoint: str = None,
    target_fs: float = 256.0,
    highpass: float = 1.0,
    lowpass: float = 45.0,
    max_passes: int = 2,
    r_thresh: float = 2.5
):
    """
    Clean continuous CI-EEG recording using Stage 1 Two-Pass SOBI or Stage 2 SOBI+N2N.
    """
    raw_clean, summary = run_two_pass_sobi_subject(
        input_fif=input_fif,
        output_fif=output_fif,
        target_fs=target_fs,
        highpass=highpass,
        lowpass=lowpass,
        max_passes=max_passes,
        r_thresh=r_thresh
    )
    
    if method == "sobi-n2n" and checkpoint:
        import torch
        from track_a_deep_learning.train_sobi_n2n import ResidualConstrainedUNet1D
        from track_a_deep_learning.evaluate_sobi_n2n_downstream import apply_sobi_n2n_sliding_window, CANONICAL_20_CHS
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        available_chs = [ch for ch in CANONICAL_20_CHS if ch in raw_clean.ch_names]
        
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
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        
        sobi_data = raw_clean.get_data(picks=available_chs)
        n2n_data = apply_sobi_n2n_sliding_window(model, sobi_data, p_art_tensor, device)
        raw_clean._data[[raw_clean.ch_names.index(ch) for ch in available_chs]] = n2n_data
        raw_clean.save(output_fif, overwrite=True)
        
    return raw_clean, summary
