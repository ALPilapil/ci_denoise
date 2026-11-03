"""
hemispheric_normalization.py

Detects Cochlear Implant (CI) implant laterality (Left CI vs. Right CI vs. Bilateral)
and normalizes multi-channel EEG spatial coordinates across the sagittal midline.

Why this is critical for Supervised Machine Learning:
- Patients with a Left CI have primary stimulation artifacts at P7, T7, TP7, M1.
- Patients with a Right CI have primary stimulation artifacts at P8, T8, TP8, M2.
- Without hemispheric normalization, neural networks (1D U-Nets or 2D spatial CNNs)
  must split their model capacity learning two opposing spatial distributions.
- Normalizing all patients to a canonical "Ipsilateral Left" spatial orientation
  aligns artifact channels across the entire cohort.
- The transformation is strictly reversible (orthogonal permutation matrix).
"""

from typing import Dict, List, Optional, Tuple
import mne
import numpy as np


# Standard 10-05 / 10-20 symmetric bilateral electrode pairs
BILATERAL_SWAP_PAIRS = [
    # Mastoid and temporal reference sensors
    ("P7", "P8"),
    ("T7", "T8"),
    ("M1", "M2"),
    ("LMas", "RMas"),
    ("TP7", "TP8"),
    ("TP9", "TP10"),
    ("FT7", "FT8"),
    ("FT9", "FT10"),
    # Frontal and ocular sensors
    ("Fp1", "Fp2"),
    ("AF3", "AF4"),
    ("AF7", "AF8"),
    ("F1", "F2"),
    ("F3", "F4"),
    ("F5", "F6"),
    ("F7", "F8"),
    ("FC1", "FC2"),
    ("FC3", "FC4"),
    ("FC5", "FC6"),
    # Central and motor strip
    ("C1", "C2"),
    ("C3", "C4"),
    ("C5", "C6"),
    ("T3", "T4"),
    # Centro-parietal and parietal
    ("CP1", "CP2"),
    ("CP3", "CP4"),
    ("CP5", "CP6"),
    ("P1", "P2"),
    ("P3", "P4"),
    ("P5", "P6"),
    ("T5", "T6"),
    # Parieto-occipital and occipital
    ("PO3", "PO4"),
    ("PO7", "PO8"),
    ("O1", "O2"),
]


def detect_implant_laterality(
    raw: mne.io.Raw,
    left_refs: List[str] = ["P7", "T7", "TP7", "M1", "LMas"],
    right_refs: List[str] = ["P8", "T8", "TP8", "M2", "RMas"],
    active_duration_s: float = 60.0,
    threshold: float = 0.15,
) -> Tuple[str, float, float, float]:
    """
    Computes Laterality Index (LI) from relative RMS power of CI-proximal channels:
      LI = (RMS_left - RMS_right) / (RMS_left + RMS_right)

    Returns:
      (laterality_label, laterality_index, rms_left, rms_right)
      - LI > threshold   -> "LEFT_CI"
      - LI < -threshold  -> "RIGHT_CI"
      - |LI| <= threshold -> "BILATERAL_OR_BALANCED"
    """
    sfreq = raw.info["sfreq"]
    n_samples = min(int(active_duration_s * sfreq), raw.n_times)
    data, _ = raw[:, :n_samples]
    chs = raw.ch_names

    valid_left = [chs.index(c) for c in left_refs if c in chs]
    valid_right = [chs.index(c) for c in right_refs if c in chs]

    if not valid_left or not valid_right:
        return "UNKNOWN", 0.0, 0.0, 0.0

    l_rms = float(np.mean([np.std(data[i]) for i in valid_left]) * 1e6)
    r_rms = float(np.mean([np.std(data[i]) for i in valid_right]) * 1e6)

    li = float((l_rms - r_rms) / (l_rms + r_rms + 1e-12))

    if li > threshold:
        label = "LEFT_CI"
    elif li < -threshold:
        label = "RIGHT_CI"
    else:
        label = "BILATERAL_OR_BALANCED"

    return label, li, l_rms, r_rms


def normalize_hemispheric_channels(
    raw: mne.io.Raw,
    target_side: str = "LEFT_CI",
) -> Tuple[mne.io.Raw, bool]:
    """
    Flips channel positions symmetrically across the sagittal midline if the
    implant side does not match target_side (default: canonical LEFT_CI).

    Returns:
      (normalized_raw, was_flipped)
    """
    label, li, l_rms, r_rms = detect_implant_laterality(raw)
    was_flipped = False

    if (target_side == "LEFT_CI" and label == "RIGHT_CI") or (target_side == "RIGHT_CI" and label == "LEFT_CI"):
        raw_flipped = raw.copy()
        ch_names = raw.ch_names
        rename_dict = {}

        for left_ch, right_ch in BILATERAL_SWAP_PAIRS:
            if left_ch in ch_names and right_ch in ch_names:
                # Temporary token to avoid collision
                temp_left = f"__TMP_{left_ch}__"
                rename_dict[left_ch] = temp_left
                rename_dict[right_ch] = left_ch

        # First pass: map left to temp, right to left
        raw_flipped.rename_channels({k: v for k, v in rename_dict.items() if not k.startswith("__TMP_")})
        # Second pass: map temp to right
        temp_to_right = {f"__TMP_{left_ch}__": right_ch for left_ch, right_ch in BILATERAL_SWAP_PAIRS if left_ch in ch_names and right_ch in ch_names}
        raw_flipped.rename_channels(temp_to_right)

        was_flipped = True
        return raw_flipped, was_flipped

    return raw.copy(), was_flipped
