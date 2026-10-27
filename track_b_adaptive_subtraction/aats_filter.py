"""
aats_filter.py - Standalone Stimulus-Locked Adaptive Artifact Template Subtraction (AATS).

Theory:
  - CI stimulation artifacts (RF transmission and ground leakage) are concentrated
    on retroauricular/mastoid reference electrodes (P7, T7, M2, M1, P8) with SNR >20 dB.
  - The central auditory evoked potential (N1/P2 dipole at Cz, Fz) has negligible
    representation on mastoid reference electrodes.
  - Using a regularized multi-tap FIR Wiener filter, we model the transfer function from
    reference electrodes to recording electrodes and subtract the estimated artifact.
  - Because brainwaves are uncorrelated with the reference electrical pulses, the true
    neural response passes through unattenuated.
"""

from typing import List, Optional, Tuple
import numpy as np
import mne


CI_CHANNELS = ['P7', 'T7', 'M2', 'M1', 'P8']


class AATSFilter:
    """
    Adaptive Artifact Template Subtraction filter.
    Operates on MNE Raw instances or NumPy arrays.
    """
    def __init__(
        self,
        ci_channels: List[str] = CI_CHANNELS,
        n_taps: int = 6,
        reg_lambda: float = 1e-3,
        condition_events: Optional[List[str]] = None,
    ):
        self.ci_channels = ci_channels
        self.n_taps = n_taps
        self.reg_lambda = reg_lambda
        self.condition_events = condition_events or ["A", "AV", "V"]
        self.used_ref_channels: List[str] = []

    def fit_transform_raw(self, raw: mne.io.Raw) -> Tuple[mne.io.Raw, mne.io.Raw]:
        """
        Denoises an MNE Raw instance.
        Returns: (clean_raw, noise_raw) where clean_raw + noise_raw == raw.
        """
        data = raw.get_data()  # shape: (n_channels, n_times)
        ch_names = raw.ch_names
        n_channels, n_times = data.shape

        # Filter out dead, flat, or corrupted reference channels
        valid_ref_indices = []
        for idx in ref_indices:
            sig = data[idx]
            if np.isnan(sig).any() or np.isinf(sig).any() or np.std(sig) < 1e-6:
                print(f"  Notice: Excluding bad/flat reference channel [{ch_names[idx]}].", flush=True)
            else:
                valid_ref_indices.append(idx)

        ref_indices = valid_ref_indices
        self.used_ref_channels = [ch_names[idx] for idx in ref_indices]

        if not ref_indices:
            print("  Warning: No valid CI reference channels found in recording. Signal left unchanged.", flush=True)
            clean_raw = raw.copy()
            noise_raw = mne.io.RawArray(np.zeros_like(data), raw.info.copy(), verbose=False)
            return clean_raw, noise_raw

        # Construct multi-tap FIR reference matrix: (n_times, n_ref * n_taps)
        n_ref = len(ref_indices)
        M = self.n_taps
        R = np.zeros((n_times, n_ref * M))

        col = 0
        for r_idx in ref_indices:
            ref_sig = data[r_idx]
            for tau in range(M):
                if tau == 0:
                    R[:, col] = ref_sig
                else:
                    R[tau:, col] = ref_sig[:-tau]
                col += 1

        # Identify adaptation mask: fit on stimulus-active windows if annotations exist
        fit_mask = np.ones(n_times, dtype=bool)
        if raw.annotations and len(raw.annotations) > 0:
            stim_mask = np.zeros(n_times, dtype=bool)
            sfreq = raw.info["sfreq"]
            for annot in raw.annotations:
                if any(cond in annot["description"] for cond in self.condition_events):
                    onset_idx = int(annot["onset"] * sfreq)
                    # Cover 0.6s post-stimulus interval where CI pulses occur
                    dur_idx = int(0.6 * sfreq)
                    s0 = max(0, onset_idx)
                    s1 = min(n_times, onset_idx + dur_idx)
                    stim_mask[s0:s1] = True
            if np.any(stim_mask) and np.sum(stim_mask) > (n_ref * M * 2):
                fit_mask = stim_mask

        R_fit = R[fit_mask]

        # Regularized normal equations: (R^T R + reg * I)
        RTR = R_fit.T @ R_fit
        trace_val = np.trace(RTR) / (n_ref * M) + 1e-12
        reg = max(self.reg_lambda * trace_val, 1e-4 * trace_val)
        A_reg = RTR + reg * np.eye(n_ref * M)

        try:
            inv_RTR = np.linalg.pinv(A_reg, rcond=1e-6)
        except Exception:
            inv_RTR = np.linalg.pinv(A_reg)

        clean_data = data.copy()
        noise_data = np.zeros_like(data)

        # Apply to all scalp recording channels with sanity bounding
        for ch_idx in range(n_channels):
            x_fit = data[ch_idx, fit_mask]
            w = inv_RTR @ (R_fit.T @ x_fit)
            artifact_est = R @ w

            # Sanity check: artifact estimate should not blow up beyond signal scale
            orig_std = np.std(data[ch_idx]) + 1e-12
            art_std = np.std(artifact_est)
            if np.isnan(artifact_est).any() or np.isinf(artifact_est).any() or (art_std / orig_std > 5.0):
                # Regression was numerically unstable on this channel; leave unchanged
                clean_data[ch_idx] = data[ch_idx]
                noise_data[ch_idx] = 0.0
            else:
                noise_data[ch_idx] = artifact_est
                clean_data[ch_idx] = data[ch_idx] - artifact_est

        clean_raw = mne.io.RawArray(clean_data, raw.info.copy(), verbose=False)
        clean_raw.set_annotations(raw.annotations)

        noise_raw = mne.io.RawArray(noise_data, raw.info.copy(), verbose=False)
        noise_raw.set_annotations(raw.annotations)

        return clean_raw, noise_raw
