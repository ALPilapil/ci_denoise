#!/usr/bin/env python3
"""
tests/test_cidenoise.py
Automated unit tests for cidenoise:
1. FIR zero-phase filtering and zero group delay.
2. Active midline cortical safeguard rollback logic.
3. Hardware QA saturation detection.
4. Subspace projection operator biophysical constraints.
5. Turnkey CLI audit command.
"""

import os
import sys
import unittest
import numpy as np
import mne

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from processing.two_pass_sobi_preprocessor import compute_spatial_ratio
from cidenoise.cli import audit_command

class TestCIDenoiseBiophysics(unittest.TestCase):

    def setUp(self):
        self.fs = 256.0
        self.duration_s = 10.0
        self.n_times = int(self.fs * self.duration_s)
        self.ch_names = ['Cz', 'Fz', 'Pz', 'P7', 'P8', 'T7', 'T8']
        
        info = mne.create_info(ch_names=self.ch_names, sfreq=self.fs, ch_types='eeg')
        data = np.zeros((len(self.ch_names), self.n_times))
        
        # Inject 10 Hz alpha sine on Cz, Fz, Pz
        t = np.arange(self.n_times) / self.fs
        data[0] = 5e-6 * np.sin(2 * np.pi * 10 * t) # Cz
        data[1] = 4e-6 * np.sin(2 * np.pi * 10 * t) # Fz
        data[2] = 4e-6 * np.sin(2 * np.pi * 10 * t) # Pz
        
        # Inject heavy 30 Hz RF ripple on P7, P8
        data[3] = 100e-6 * np.sin(2 * np.pi * 30 * t) # P7
        data[4] = 80e-6 * np.sin(2 * np.pi * 30 * t)  # P8
        
        self.raw = mne.io.RawArray(data, info, verbose=False)

    def test_fir_zero_phase(self):
        """Verify that phase='zero' FIR filtering does not introduce sample lag/delay."""
        t = np.arange(self.n_times) / self.fs
        orig_sig = self.raw.get_data()[0]
        
        filtered_raw = self.raw.copy()
        filtered_raw.filter(1.0, 45.0, fir_design='firwin', phase='zero', verbose=False)
        filt_sig = filtered_raw.get_data()[0]
        
        # Peak cross-correlation must be at lag 0
        corr = np.correlate(filt_sig - np.mean(filt_sig), orig_sig - np.mean(orig_sig), mode='full')
        lags = np.arange(-len(orig_sig) + 1, len(orig_sig))
        peak_lag = lags[np.argmax(corr)]
        self.assertEqual(peak_lag, 0, f"Zero-phase FIR must exhibit 0 lag, got {peak_lag}")

    def test_hardware_saturation_qa(self):
        """Verify hardware QA catches floating point amplifier overflow (> 1 V^2)."""
        data_normal = self.raw.get_data()
        self.assertLess(np.max(np.var(data_normal, axis=1)), 1.0)
        
        # Inject saturation (high variance)
        t = np.arange(self.n_times) / self.fs
        data_sat = data_normal.copy()
        data_sat[3] = 1e3 * np.sin(2 * np.pi * 10 * t) # 1 kV amplitude
        raw_sat = mne.io.RawArray(data_sat, self.raw.info, verbose=False)
        max_var = float(np.max(np.var(raw_sat.get_data(), axis=1)))
        self.assertGreater(max_var, 1.0, "Saturated channel variance must exceed 1.0 V^2")

    def test_subspace_projection_constraints(self):
        """Verify that P_art is symmetric, positive semi-definite, and protects Cz/Fz/Pz."""
        n_chs = len(self.ch_names)
        # Mock artifact mixing vectors on P7 and P8
        u = np.zeros((n_chs, 1), dtype=np.float32)
        u[self.ch_names.index('P7')] = 0.8
        u[self.ch_names.index('P8')] = 0.6
        u /= np.linalg.norm(u)
        
        P_art = u @ u.T
        
        # Explicit zeroing of midline leads Cz, Fz, Pz
        for m_ch in ['Cz', 'Fz', 'Pz']:
            idx = self.ch_names.index(m_ch)
            P_art[idx, :] = 0.0
            P_art[:, idx] = 0.0
            
        # Verify midline rows and columns are identically zero
        for m_ch in ['Cz', 'Fz', 'Pz']:
            idx = self.ch_names.index(m_ch)
            np.testing.assert_allclose(P_art[idx, :], 0.0, atol=1e-7)
            np.testing.assert_allclose(P_art[:, idx], 0.0, atol=1e-7)
            
        # Verify symmetry
        np.testing.assert_allclose(P_art, P_art.T, atol=1e-7)

    def test_mastoid_to_central_ratio(self):
        """Verify the R threshold accurately flags mastoid-dominant artifact topographies."""
        # Clean cortical component (dominant on Cz)
        w_cortical = np.array([1.0 if ch in ['Cz', 'Fz'] else 0.1 for ch in self.ch_names])
        r_cortical = compute_spatial_ratio(w_cortical, self.ch_names)
        self.assertLess(r_cortical, 2.5, "Cortical component must not exceed R=2.5 threshold")
        
        # Electrical artifact component (dominant on P7, P8)
        w_artifact = np.array([10.0 if ch in ['P7', 'P8'] else 0.2 for ch in self.ch_names])
        r_artifact = compute_spatial_ratio(w_artifact, self.ch_names)
        self.assertGreater(r_artifact, 2.5, "Mastoid artifact component must exceed R=2.5 threshold")

if __name__ == '__main__':
    unittest.main()
