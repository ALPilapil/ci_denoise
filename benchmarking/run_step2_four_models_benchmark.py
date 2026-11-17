"""
run_step2_four_models_benchmark.py

End-to-End Two-Step Denoising Solution:
- Step 1: Two-Pass SOBI (Pass 1: Coil dipole at P7 -> Pass 2: Lead dipole at T7)
- Step 2: Neural Network Temporal Denoising evaluated across 4 architectures in parallel:
    1. UNet1D_Baseline         (Direct 1D Temporal U-Net)
    2. ResAttnUNet1D           (Residual Noise-Predicting U-Net + Channel SE Attention)
    3. STFT_MaskNet            (Complex Time-Frequency Spectrogram U-Net + iSTFT)
    4. SpatioTemporalConformer (Multi-Head Self-Attention + Depthwise Conv)

Each model receives the exact same two-pass SOBI Step 1 output as its input.
Evaluates on real continuous clinical CI EEG (trial_onsets_0901y2-raw.fif)
benchmarked against Normal Hearing Ground Truth (trial_onsets_0801y2-raw.fif).
"""

import argparse
import glob
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import signal

# Add paths
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_DIR, "track_a_deep_learning"))
sys.path.insert(0, os.path.join(REPO_DIR, "processing"))

from models_suite import build_model
from cidenoise_turnkey import CANONICAL_20_CHS
from hemispheric_normalization import normalize_hemispheric_channels


CI_REFERENCE_CHANNELS = ["P7", "T7", "M1", "M2", "P8"]
CENTRAL_AUDITORY_CHANNELS = ["Cz", "Fz", "FC1", "FC2", "C3", "C4"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Step 1: SOBI Implementation (Robust Rank-Deficient Safeguards)
# ─────────────────────────────────────────────────────────────────────────────

def run_sobi(
    data: np.ndarray,
    n_components: Optional[int] = None,
    lags: Optional[List[int]] = None,
    eps: float = 1e-6,
    max_iter: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Second-Order Blind Identification (SOBI) via joint approximate diagonalization
    with rank-deficiency handling.
    """
    n_channels, n_times = data.shape
    if lags is None:
        lags = [1, 2, 4, 8, 16, 25, 50]

    X = data - np.mean(data, axis=1, keepdims=True)
    cov0 = (X @ X.T) / (n_times - 1)
    d, u = np.linalg.eigh(cov0)

    sort_idx = np.argsort(d)[::-1]
    d = d[sort_idx]
    u = u[:, sort_idx]

    max_d = np.max(d)
    valid_mask = d > 1e-7 * max_d
    rank = int(np.sum(valid_mask))
    if rank == 0:
        rank = 1

    n_comps = rank if (n_components is None or n_components > rank) else n_components

    d_sub = np.maximum(d[:n_comps], 1e-12)
    u_sub = u[:, :n_comps]

    whitening = np.diag(1.0 / np.sqrt(d_sub)) @ u_sub.T
    dewhitening = u_sub @ np.diag(np.sqrt(d_sub))
    Z = whitening @ X

    K = len(lags)
    M = np.zeros((n_comps, n_comps * K))
    for k_idx, lag in enumerate(lags):
        if lag >= n_times:
            continue
        c_lag = (Z[:, :n_times - lag] @ Z[:, lag:].T) / (n_times - lag)
        c_sym = 0.5 * (c_lag + c_lag.T)
        M[:, k_idx * n_comps : (k_idx + 1) * n_comps] = c_sym

    V = np.eye(n_comps)
    for _ in range(max_iter):
        delta = 0.0
        for p in range(n_comps - 1):
            for q in range(p + 1, n_comps):
                h_p = M[p, :].reshape(K, n_comps)
                h_q = M[q, :].reshape(K, n_comps)
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
                        col_p = M[:, k * n_comps + p].copy()
                        col_q = M[:, k * n_comps + q].copy()
                        M[:, k * n_comps + p] = c * col_p + s * col_q
                        M[:, k * n_comps + q] = -s * col_p + c * col_q

        if delta < eps:
            break

    W = V.T @ whitening
    A = dewhitening @ V
    return A, W


def identify_dominant_artifact(
    A: np.ndarray,
    S: np.ndarray,
    ch_names: List[str],
    ref_channels: List[str] = CI_REFERENCE_CHANNELS,
) -> Tuple[int, Dict]:
    n_chs, n_comps = A.shape
    ref_idxs = [ch_names.index(c) for c in ref_channels if c in ch_names]
    central_idxs = [ch_names.index(c) for c in CENTRAL_AUDITORY_CHANNELS if c in ch_names]

    a_norms_sq = np.sum(A ** 2, axis=0)
    s_norms_sq = np.sum(S ** 2, axis=1)
    comp_vars = a_norms_sq * s_norms_sq
    var_total = float(np.sum(comp_vars))
    pvaf = (comp_vars / (var_total + 1e-12)) * 100.0

    ratios = []
    top_chs = []

    for k in range(n_comps):
        abs_w = np.abs(A[:, k])
        top_ch_idx = int(np.argmax(abs_w))
        top_chs.append(ch_names[top_ch_idx])

        ref_power = np.mean(abs_w[ref_idxs]) if ref_idxs else 0.0
        central_power = np.mean(abs_w[central_idxs]) if central_idxs else 0.0
        ratios.append(float(ref_power / (central_power + 1e-9)))

    # Artifact priority score: variance weighted by mastoid ratio
    scores = []
    for k in range(n_comps):
        score = pvaf[k] * (1.0 + min(ratios[k], 10.0))
        if top_chs[k] in ref_channels:
            score *= 2.0
        scores.append(score)

    dom_idx = int(np.argmax(scores))
    info = {
        "dom_idx": dom_idx,
        "pvaf": float(pvaf[dom_idx]),
        "top_channel": top_chs[dom_idx],
        "ref_ratio": float(ratios[dom_idx]),
    }
    return dom_idx, info


def execute_two_pass_sobi(
    raw: mne.io.Raw,
    target_channels: List[str],
    n_components: int = 15,
) -> Tuple[np.ndarray, Dict]:
    """
    Executes Two-Pass SOBI on continuous raw EEG.
    Returns:
        sobi_p2_data: Cleaned continuous EEG array (n_channels, n_times)
        metadata: Diagnostic details of Pass 1 and Pass 2
    """
    raw_work = raw.copy().pick(target_channels)
    data = raw_work.get_data()
    n_chs, n_times = data.shape

    # Pass 1: Coil artifact
    print("  [Step 1 - SOBI Pass 1/2] Fitting SOBI on raw CI EEG...", flush=True)
    A1, W1 = run_sobi(data, n_components=n_components)
    S1 = W1 @ data
    dom1, info1 = identify_dominant_artifact(A1, S1, target_channels)
    print(f"    -> Pass 1 Identified Artifact: SOBI {dom1} (Top: {info1['top_channel']}, PVAF: {info1['pvaf']:.1f}%, Ratio: {info1['ref_ratio']:.1f}x)", flush=True)

    S1_clean = S1.copy()
    S1_clean[dom1, :] = 0.0
    data_p1 = A1 @ S1_clean

    # Pass 2: Secondary lead artifact
    print("  [Step 1 - SOBI Pass 2/2] Re-fitting SOBI on Pass 1 residual...", flush=True)
    A2, W2 = run_sobi(data_p1, n_components=n_components)
    S2 = W2 @ data_p1
    dom2, info2 = identify_dominant_artifact(A2, S2, target_channels)
    print(f"    -> Pass 2 Identified Artifact: SOBI {dom2} (Top: {info2['top_channel']}, PVAF: {info2['pvaf']:.1f}%, Ratio: {info2['ref_ratio']:.1f}x)", flush=True)

    S2_clean = S2.copy()
    S2_clean[dom2, :] = 0.0
    data_p2 = A2 @ S2_clean

    meta = {
        "pass_1": info1,
        "pass_2": info2,
    }
    return data_p2, meta


# ─────────────────────────────────────────────────────────────────────────────
# 2. Step 2: Neural Network Sliding-Window Temporal Denoising
# ─────────────────────────────────────────────────────────────────────────────

def apply_neural_denoiser(
    model: nn.Module,
    eeg_data: np.ndarray,
    device: torch.device,
    win_len: int = 1024,
    hop_len: int = 512,
    scale_uv: float = 100.0,
) -> Tuple[np.ndarray, float]:
    """
    Applies neural network model with Hann-weighted sliding window crossfade.
    Input eeg_data is in Volts.
    Returns:
        denoised_data: (n_channels, n_times) in Volts
        latency_sec: Inference duration
    """
    model.eval()
    model.to(device)

    n_chs, n_times = eeg_data.shape
    # Scale to microvolts for model inference
    data_uv = eeg_data * 1e6 / scale_uv

    out_accum = np.zeros_like(data_uv, dtype=np.float32)
    weight_accum = np.zeros(n_times, dtype=np.float32)
    window = np.hanning(win_len).astype(np.float32)

    # Padding
    pad_needed = (hop_len - (n_times - win_len) % hop_len) % hop_len
    if pad_needed > 0:
        data_padded = np.pad(data_uv, ((0, 0), (0, pad_needed)), mode="reflect")
    else:
        data_padded = data_uv

    n_steps = (data_padded.shape[1] - win_len) // hop_len + 1

    t0 = time.time()
    with torch.no_grad():
        for i in range(n_steps):
            start = i * hop_len
            end = start + win_len
            chunk = data_padded[:, start:end]
            x_t = torch.tensor(chunk, dtype=torch.float32, device=device).unsqueeze(0)  # (1, C, T)

            y_t = model(x_t).squeeze(0).cpu().numpy()  # (C, T)

            # Accumulate with Hann window
            actual_end = min(end, n_times)
            valid_len = actual_end - start
            if valid_len > 0:
                out_accum[:, start:actual_end] += y_t[:, :valid_len] * window[:valid_len]
                weight_accum[start:actual_end] += window[:valid_len]

    t1 = time.time()
    latency_sec = float(t1 - t0)

    # Normalize by window weights
    weight_accum = np.maximum(weight_accum, 1e-6)
    out_uv = (out_accum / weight_accum) * scale_uv
    denoised_volts = (out_uv * 1e-6).astype(np.float64)

    return denoised_volts, latency_sec


# ─────────────────────────────────────────────────────────────────────────────
# 3. ERP Extraction & Performance Metrics
# ─────────────────────────────────────────────────────────────────────────────

def setup_montage(raw: mne.io.Raw) -> mne.io.Raw:
    try:
        rename_map = {"LMas": "M1", "RMas": "M2"}
        actual_renames = {k: v for k, v in rename_map.items() if k in raw.ch_names and v not in raw.ch_names}
        if actual_renames:
            raw.rename_channels(actual_renames)
        montage = mne.channels.make_standard_montage("standard_1005")
        raw.set_montage(montage, on_missing="ignore", match_case=False, verbose=False)
    except Exception as e:
        print(f"Montage setup notice: {e}", flush=True)
    return raw


def get_averaged_erp(
    raw: mne.io.Raw,
    conditions: List[str] = ["A", "AV"],
    tmin: float = -0.2,
    tmax: float = 0.8,
) -> mne.Evoked:
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    target_ids = [event_id[c] for c in conditions if c in event_id]
    if not target_ids:
        target_ids = list(event_id.values())

    epochs = mne.Epochs(
        raw,
        events,
        event_id=target_ids,
        tmin=tmin,
        tmax=tmax,
        baseline=(tmin, 0.0),
        preload=True,
        reject=None,
        verbose=False,
    )
    return epochs.average(picks="all")


def compute_metrics(
    evoked: mne.Evoked,
    evoked_ref: mne.Evoked,
    channel: str = "Cz",
) -> Dict[str, float]:
    times = evoked.times
    ch_idx = evoked.ch_names.index(channel)
    ref_ch_idx = evoked_ref.ch_names.index(channel)

    data_ch = evoked.data[ch_idx] * 1e6  # uV
    ref_ch = evoked_ref.data[ref_ch_idx] * 1e6  # uV

    baseline_mask = (times >= -0.2) & (times <= 0.0)
    baseline_rms = float(np.sqrt(np.mean(data_ch[baseline_mask] ** 2)))

    # N1 peak (70-140 ms)
    n1_mask = (times >= 0.070) & (times <= 0.140)
    n1_data = data_ch[n1_mask]
    n1_times = times[n1_mask]
    n1_idx = int(np.argmin(n1_data))
    n1_amp = float(n1_data[n1_idx])
    n1_lat = float(n1_times[n1_idx] * 1000.0)

    # P2 peak (140-230 ms)
    p2_mask = (times >= 0.140) & (times <= 0.230)
    p2_data = data_ch[p2_mask]
    p2_times = times[p2_mask]
    p2_idx = int(np.argmax(p2_data))
    p2_amp = float(p2_data[p2_idx])
    p2_lat = float(p2_times[p2_idx] * 1000.0)

    p2p_amp = float(p2_amp - n1_amp)

    eval_mask = (times >= 0.0) & (times <= 0.500)
    y_test = data_ch[eval_mask]
    y_true = ref_ch[eval_mask]

    corr = float(np.corrcoef(y_test, y_true)[0, 1]) if (np.std(y_test) > 1e-6 and np.std(y_true) > 1e-6) else 0.0
    rmse = float(np.sqrt(np.mean((y_test - y_true) ** 2)))

    evoked_power = np.mean(y_test ** 2)
    noise_power = np.mean(data_ch[baseline_mask] ** 2) + 1e-12
    erp_snr_db = float(10.0 * np.log10(evoked_power / noise_power))

    return {
        "baseline_rms_uv": baseline_rms,
        "n1_amp_uv": n1_amp,
        "n1_lat_ms": n1_lat,
        "p2_amp_uv": p2_amp,
        "p2_lat_ms": p2_lat,
        "p2p_amp_uv": p2p_amp,
        "corr_with_hearing": corr,
        "rmse_with_hearing": rmse,
        "erp_snr_db": erp_snr_db,
    }


def compute_band_power(sig: np.ndarray, fs: float, fmin: float, fmax: float) -> float:
    freqs, psd = signal.welch(sig, fs=fs, nperseg=min(len(sig), int(fs * 2)))
    mask = (freqs >= fmin) & (freqs <= fmax)
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(psd[mask], freqs[mask]))
    elif hasattr(np, "trapz"):
        return float(np.trapz(psd[mask], freqs[mask]))
    else:
        from scipy import integrate
        return float(integrate.trapezoid(psd[mask], freqs[mask]))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Main Parallel Benchmark Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Parallel 2-Step Denoising: SOBI Two Passes into 4 Neural Networks")
    parser.add_argument("--data-dir", default="/quobyte/millerlmgrp/annotated_data", help="Directory of FIF files")
    parser.add_argument("--checkpoint-dir", default="/quobyte/millerlmgrp/checkpoints/track_a_arch_bench", help="Directory of trained model checkpoints")
    parser.add_argument("--ci-subject", default="0901y2", help="CI subject ID")
    parser.add_argument("--hearing-subject", default="0801y2", help="Hearing subject ID")
    parser.add_argument("--target-fs", type=float, default=256.0, help="Target sampling rate in Hz")
    parser.add_argument("--output-dir", default="./benchmark_results/single_pair_demo", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("=" * 85, flush=True)
    print("  TWO-STEP SOLUTION BENCHMARK: TWO-PASS SOBI -> 4 NEURAL NETWORKS IN PARALLEL", flush=True)
    print("=" * 85, flush=True)
    print(f"CI Subject:         {args.ci_subject} (Left CI, Permutation 1)", flush=True)
    print(f"Hearing Control:    {args.hearing_subject} (Normal Hearing Ground Truth)", flush=True)
    print(f"Compute Device:     {device}", flush=True)
    print(f"Target Sample Rate: {args.target_fs} Hz\n", flush=True)

    # 1. Load continuous raw EEG
    ci_files = sorted(glob.glob(os.path.join(args.data_dir, "**", f"*{args.ci_subject}*-raw.fif"), recursive=True))
    hearing_files = sorted(glob.glob(os.path.join(args.data_dir, "**", f"*{args.hearing_subject}*-raw.fif"), recursive=True))

    raw_ci = mne.io.read_raw_fif(ci_files[0], preload=True, verbose=False)
    raw_hearing = mne.io.read_raw_fif(hearing_files[0], preload=True, verbose=False)

    raw_ci = setup_montage(raw_ci)
    raw_hearing = setup_montage(raw_hearing)

    # Resample to 256 Hz and pick 19 canonical channels
    print(f"Resampling EEG recordings from {raw_ci.info['sfreq']} Hz to {args.target_fs} Hz...", flush=True)
    raw_ci.resample(args.target_fs, verbose=False)
    raw_hearing.resample(args.target_fs, verbose=False)

    raw_ci.pick(CANONICAL_20_CHS)
    raw_ci.reorder_channels(CANONICAL_20_CHS)
    raw_hearing.pick(CANONICAL_20_CHS)
    raw_hearing.reorder_channels(CANONICAL_20_CHS)

    fs = float(raw_ci.info["sfreq"])
    cz_idx = CANONICAL_20_CHS.index("Cz")
    p7_idx = CANONICAL_20_CHS.index("P7")
    n_chs = len(CANONICAL_20_CHS)
    n_times = raw_ci.n_times
    duration_min = n_times / (fs * 60.0)

    print(f"Data Prepared: {n_chs} Channels, {n_times} Timepoints ({duration_min:.2f} minutes continuous recording)", flush=True)

    # Ground Truth & Raw Baselines
    evoked_hearing = get_averaged_erp(raw_hearing)
    evoked_raw_ci = get_averaged_erp(raw_ci)
    m_hearing = compute_metrics(evoked_hearing, evoked_hearing, "Cz")
    m_raw = compute_metrics(evoked_raw_ci, evoked_hearing, "Cz")

    # =========================================================================
    # STEP 1: EXECUTE TWO-PASS SOBI ONCE (SHARED INPUT FOR ALL 4 MODELS)
    # =========================================================================
    print("\n" + "=" * 80, flush=True)
    print("  STEP 1: EXECUTING TWO-PASS SOBI ON CONTINUOUS CI EEG", flush=True)
    print("=" * 80, flush=True)

    t_sobi_start = time.time()
    sobi_cleaned_data, sobi_meta = execute_two_pass_sobi(raw_ci, CANONICAL_20_CHS, n_components=15)
    t_sobi_end = time.time()
    sobi_duration = t_sobi_end - t_sobi_start

    print(f"  Step 1 SOBI Two-Pass Completed in {sobi_duration:.2f} seconds!", flush=True)

    # Wrap SOBI output in MNE Raw for ERP evaluation
    raw_sobi_p2 = mne.io.RawArray(sobi_cleaned_data, raw_ci.info.copy(), verbose=False)
    raw_sobi_p2.set_annotations(raw_ci.annotations)
    evoked_sobi_p2 = get_averaged_erp(raw_sobi_p2)
    m_sobi_p2 = compute_metrics(evoked_sobi_p2, evoked_hearing, "Cz")

    # =========================================================================
    # STEP 2: APPLY ALL 4 NEURAL NETWORKS IN PARALLEL ON SOBI STEP 1 OUTPUT
    # =========================================================================
    print("\n" + "=" * 80, flush=True)
    print("  STEP 2: TEMPORAL NEURAL DENOISING ACROSS ALL 4 ARCHITECTURES", flush=True)
    print("=" * 80, flush=True)

    candidate_models = [
        ("UNet1D_Baseline", "best_UNet1D_Baseline.pt", "Direct 1D U-Net Baseline"),
        ("ResAttnUNet1D", "best_ResAttnUNet1D.pt", "Residual Noise U-Net + Channel SE Attention"),
        ("STFT_MaskNet", "best_STFT_MaskNet.pt", "Complex Time-Frequency STFT U-Net"),
        ("SpatioTemporalConformer", "best_SpatioTemporalConformer.pt", "Conformer: Self-Attention + Depthwise Conv"),
    ]

    results_table = [
        {
            "Pipeline": "Clean Hearing Control (GT)",
            "Step1_SOBI": "N/A",
            "Step2_Model": "N/A",
            "Inference_Sec": 0.0,
            "Param_Count_M": 0.0,
            **m_hearing,
            "rf_hash_power_20_50hz": compute_band_power(raw_hearing.get_data()[cz_idx], fs, 20.0, 50.0),
            "neural_power_1_15hz": compute_band_power(raw_hearing.get_data()[cz_idx], fs, 1.0, 15.0),
        },
        {
            "Pipeline": "Raw CI (Contaminated)",
            "Step1_SOBI": "None",
            "Step2_Model": "None",
            "Inference_Sec": 0.0,
            "Param_Count_M": 0.0,
            **m_raw,
            "rf_hash_power_20_50hz": compute_band_power(raw_ci.get_data()[cz_idx], fs, 20.0, 50.0),
            "neural_power_1_15hz": compute_band_power(raw_ci.get_data()[cz_idx], fs, 1.0, 15.0),
        },
        {
            "Pipeline": "Two-Pass SOBI Only (Step 1 Baseline)",
            "Step1_SOBI": "Pass 1 (P7) + Pass 2 (T7)",
            "Step2_Model": "None",
            "Inference_Sec": sobi_duration,
            "Param_Count_M": 0.0,
            **m_sobi_p2,
            "rf_hash_power_20_50hz": compute_band_power(sobi_cleaned_data[cz_idx], fs, 20.0, 50.0),
            "neural_power_1_15hz": compute_band_power(sobi_cleaned_data[cz_idx], fs, 1.0, 15.0),
        },
    ]

    model_evokeds = {}
    model_signals = {}
    model_latencies = {}

    for model_name, ckpt_name, desc in candidate_models:
        ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name)
        print(f"\n[Model: {model_name}] Loading checkpoint: {ckpt_name}...", flush=True)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        # Build model
        net = build_model(model_name, in_channels=n_chs)
        state = torch.load(ckpt_path, map_location=device)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        net.load_state_dict(state)

        n_params = sum(p.numel() for p in net.parameters()) / 1e6
        print(f"  Architecture: {desc} ({n_params:.2f}M parameters)", flush=True)

        # Feed the EXACT SAME SOBI Step 1 output
        print(f"  Executing Step 2 denoising on SOBI Pass-2 manifold...", flush=True)
        denoised_data, lat_sec = apply_neural_denoiser(
            net, sobi_cleaned_data, device=device, win_len=1024, hop_len=512, scale_uv=100.0
        )

        model_signals[model_name] = denoised_data
        model_latencies[model_name] = lat_sec

        # Wrap in MNE Raw for ERP evaluation
        raw_m = mne.io.RawArray(denoised_data, raw_ci.info.copy(), verbose=False)
        raw_m.set_annotations(raw_ci.annotations)
        evk_m = get_averaged_erp(raw_m)
        model_evokeds[model_name] = evk_m

        metrics_m = compute_metrics(evk_m, evoked_hearing, "Cz")
        rf_pow = compute_band_power(denoised_data[cz_idx], fs, 20.0, 50.0)
        neural_pow = compute_band_power(denoised_data[cz_idx], fs, 1.0, 15.0)

        results_table.append({
            "Pipeline": f"Step 1 SOBI + Step 2 {model_name}",
            "Step1_SOBI": "Pass 1 (P7) + Pass 2 (T7)",
            "Step2_Model": model_name,
            "Inference_Sec": lat_sec,
            "Param_Count_M": n_params,
            **metrics_m,
            "rf_hash_power_20_50hz": rf_pow,
            "neural_power_1_15hz": neural_pow,
        })
        print(f"  -> Completed! Latency: {lat_sec:.2f}s | Cz Pearson r: {metrics_m['corr_with_hearing']:.4f} | RMSE: {metrics_m['rmse_with_hearing']:.2f} µV | P2 Amp: {metrics_m['p2_amp_uv']:.2f} µV", flush=True)

    df_results = pd.DataFrame(results_table)

    # Calculate Hash Attenuation % relative to Raw and Step 1 SOBI
    raw_hash = df_results.loc[1, "rf_hash_power_20_50hz"]
    sobi_hash = df_results.loc[2, "rf_hash_power_20_50hz"]
    df_results["hash_suppression_vs_raw_%"] = (1.0 - (df_results["rf_hash_power_20_50hz"] / raw_hash)) * 100.0
    df_results["hash_suppression_vs_sobi_%"] = (1.0 - (df_results["rf_hash_power_20_50hz"] / sobi_hash)) * 100.0

    # Save metrics to disk
    csv_out = os.path.join(args.output_dir, f"two_step_4models_comparison_{args.ci_subject}.csv")
    df_results.to_csv(csv_out, index=False)
    json_out = os.path.join(args.output_dir, f"two_step_4models_comparison_{args.ci_subject}.json")
    with open(json_out, "w") as f:
        json.dump(df_results.to_dict(orient="records"), f, indent=2)

    print("\n" + "=" * 100, flush=True)
    print("                 TWO-STEP 4-MODEL PARALLEL BENCHMARK LEADERBOARD (Cz)", flush=True)
    print("=" * 100, flush=True)
    disp_cols = [
        "Pipeline", "corr_with_hearing", "rmse_with_hearing", "p2_lat_ms",
        "p2_amp_uv", "baseline_rms_uv", "hash_suppression_vs_sobi_%", "Inference_Sec"
    ]
    print(df_results[disp_cols].to_string(index=False), flush=True)
    print("=" * 100 + "\n", flush=True)

    # =========================================================================
    # STEP 4: COMPREHENSIVE PUBLICATION VISUALIZATIONS
    # =========================================================================
    print("Generating comprehensive multi-panel comparison figures...", flush=True)

    # 4.1 FIGURE 1: 5-Panel Waveform Comparison at Vertex (Cz)
    fig1, axes1 = plt.subplots(5, 1, figsize=(15, 15), sharex=True, facecolor="#ffffff")
    fig1.suptitle(f"Two-Step Denoising Solution: SOBI Two Passes into 4 Neural Architectures (Subject {args.ci_subject})", fontsize=13, fontweight="bold", y=0.99)

    times = evoked_hearing.times * 1000.0  # ms
    cz_h = evoked_hearing.data[cz_idx] * 1e6
    cz_raw = evoked_raw_ci.data[cz_idx] * 1e6
    cz_sobi = evoked_sobi_p2.data[cz_idx] * 1e6

    model_colors = {
        "UNet1D_Baseline": "#ff7f0e",
        "ResAttnUNet1D": "#d62728",
        "STFT_MaskNet": "#9467bd",
        "SpatioTemporalConformer": "#2ca02c",
    }

    # Panels A-D: Each model individually against Raw, SOBI, and Ground Truth
    for idx, (m_name, _, m_desc) in enumerate(candidate_models):
        ax = axes1[idx]
        evk_m = model_evokeds[m_name]
        sig_m = evk_m.data[cz_idx] * 1e6
        r_val = df_results.loc[df_results["Step2_Model"] == m_name, "corr_with_hearing"].values[0]
        rmse_val = df_results.loc[df_results["Step2_Model"] == m_name, "rmse_with_hearing"].values[0]

        ax.plot(times, cz_raw, label="Raw CI EEG", color="#aaaaaa", lw=1.0, alpha=0.5)
        ax.plot(times, cz_h, label="Clean Hearing Ground Truth", color="#000000", lw=2.0, ls="--")
        ax.plot(times, cz_sobi, label="Step 1: Two-Pass SOBI Only", color="#1f77b4", lw=1.5, alpha=0.7)
        ax.plot(times, sig_m, label=f"Step 2: {m_name} (r={r_val:.3f}, RMSE={rmse_val:.2f} µV)", color=model_colors[m_name], lw=2.0)

        ax.axvspan(-200, 0, color="gray", alpha=0.08)
        ax.axvline(100, color="darkred", ls=":", alpha=0.5, label="N1 Target (~100 ms)")
        ax.axvline(200, color="navy", ls=":", alpha=0.5, label="P2 Target (~200 ms)")
        ax.set_title(f"({chr(65 + idx)}) {m_name}: {m_desc}", fontsize=10.5, fontweight="bold")
        ax.set_ylabel("Amplitude (µV)", fontweight="bold")
        ax.legend(loc="upper right", fontsize=8.0, frameon=True, ncol=2)
        ax.grid(True, linestyle=":", alpha=0.4)

    # Panel E: Head-to-Head Direct Overlay of all 4 Models
    ax_e = axes1[4]
    ax_e.plot(times, cz_h, label="Clean Hearing Ground Truth", color="#000000", lw=2.4, ls="--")
    ax_e.plot(times, cz_sobi, label="Step 1: Two-Pass SOBI", color="#1f77b4", lw=1.5, alpha=0.5)
    for m_name, _, _ in candidate_models:
        sig_m = model_evokeds[m_name].data[cz_idx] * 1e6
        r_val = df_results.loc[df_results["Step2_Model"] == m_name, "corr_with_hearing"].values[0]
        ax_e.plot(times, sig_m, label=f"{m_name} (r={r_val:.3f})", color=model_colors[m_name], lw=1.8)

    ax_e.axvspan(-200, 0, color="gray", alpha=0.08)
    ax_e.axvline(100, color="darkred", ls=":", alpha=0.5)
    ax_e.axvline(200, color="navy", ls=":", alpha=0.5)
    ax_e.set_title("(E) Head-to-Head Comparison: All 4 Architectures Overlay at Vertex (Cz)", fontsize=10.5, fontweight="bold")
    ax_e.set_xlabel("Time post-stimulus onset (ms)", fontweight="bold")
    ax_e.set_ylabel("Amplitude (µV)", fontweight="bold")
    ax_e.legend(loc="upper right", fontsize=8.5, frameon=True, ncol=3)
    ax_e.grid(True, linestyle=":", alpha=0.4)

    fig1.tight_layout()
    fig1_path = os.path.join(args.output_dir, f"two_step_4models_erp_comparison_{args.ci_subject}.png")
    fig1.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved ERP Comparison Figure -> {fig1_path}", flush=True)

    # 4.2 FIGURE 2: Power Spectral Density (PSD) Overlays across 4 Models
    fig2, (ax2_spec, ax2_bar) = plt.subplots(1, 2, figsize=(15, 6), facecolor="#ffffff", gridspec_kw={"width_ratios": [1.5, 1]})
    fig2.suptitle(f"High-Frequency RF Hash (20-50 Hz) Attenuation Across 4 Architectures (Subject {args.ci_subject})", fontsize=13, fontweight="bold")

    # Spectral Curves
    f_h, p_h = signal.welch(raw_hearing.get_data()[cz_idx] * 1e6, fs=fs, nperseg=int(fs * 2))
    f_raw, p_raw = signal.welch(raw_ci.get_data()[cz_idx] * 1e6, fs=fs, nperseg=int(fs * 2))
    f_sobi, p_sobi = signal.welch(sobi_cleaned_data[cz_idx] * 1e6, fs=fs, nperseg=int(fs * 2))

    ax2_spec.semilogy(f_h, p_h, label="Clean Hearing Control", color="#000000", lw=2.2, ls="--")
    ax2_spec.semilogy(f_raw, p_raw, label="Raw CI EEG", color="#aaaaaa", lw=1.2, alpha=0.6)
    ax2_spec.semilogy(f_sobi, p_sobi, label="Step 1: Two-Pass SOBI", color="#1f77b4", lw=1.8)

    for m_name, _, _ in candidate_models:
        sig_uv = model_signals[m_name][cz_idx] * 1e6
        f_m, p_m = signal.welch(sig_uv, fs=fs, nperseg=int(fs * 2))
        ax2_spec.semilogy(f_m, p_m, label=f"Step 2: {m_name}", color=model_colors[m_name], lw=1.8)

    ax2_spec.axvspan(20, 50, color="crimson", alpha=0.08, label="RF Hash Band (20-50 Hz)")
    ax2_spec.set_xlim(1, 60)
    ax2_spec.set_title("A. Power Spectral Density at Cz (1 to 60 Hz)", fontsize=11, fontweight="bold")
    ax2_spec.set_xlabel("Frequency (Hz)", fontweight="bold")
    ax2_spec.set_ylabel("Power Spectral Density (µV²/Hz)", fontweight="bold")
    ax2_spec.grid(True, linestyle=":", alpha=0.5)
    ax2_spec.legend(loc="upper right", fontsize=8.0)

    # Bar chart of Hash Suppression % vs SOBI
    m_rows = df_results[df_results["Step2_Model"].isin([m[0] for m in candidate_models])]
    bar_names = m_rows["Step2_Model"].tolist()
    supp_vals = m_rows["hash_suppression_vs_sobi_%"].tolist()
    lat_vals = m_rows["Inference_Sec"].tolist()

    x_pos = np.arange(len(bar_names))
    ax2_bar.bar(x_pos, supp_vals, color=[model_colors[m] for m in bar_names], width=0.55, edgecolor="black", alpha=0.85)
    for i, v in enumerate(supp_vals):
        ax2_bar.text(i, v + 1.0, f"{v:.1f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    ax2_bar.set_xticks(x_pos)
    ax2_bar.set_xticklabels(["UNet1D", "ResAttnUNet", "STFT_MaskNet", "Conformer"], rotation=20, ha="right", fontweight="bold")
    ax2_bar.set_ylabel("RF Hash Suppression vs SOBI Step 1 (%)", fontweight="bold")
    ax2_bar.set_title("B. Hash Suppression Relative to SOBI", fontsize=11, fontweight="bold")
    ax2_bar.set_ylim(0, 105)
    ax2_bar.grid(True, axis="y", linestyle=":", alpha=0.5)

    fig2.tight_layout()
    fig2_path = os.path.join(args.output_dir, f"two_step_4models_psd_attenuation_{args.ci_subject}.png")
    fig2.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved PSD Figure -> {fig2_path}", flush=True)

    print("\n" + "=" * 85, flush=True)
    print("  TWO-STEP 4-MODEL BENCHMARK COMPLETE! All outputs saved to:", args.output_dir, flush=True)
    print("=" * 85, flush=True)


if __name__ == "__main__":
    main()
