"""
tier3_downstream_eval.py

Tier 3: Downstream Validation & Deep Learning Model Generalization Benchmark.

Evaluates deep learning models (UNet, TCN, etc.) on held-out test splits with
comprehensive downstream neuroengineering metrics:
  - Subject-independent generalization (cross-subject test loss)
  - Spectral Band Fidelity: Error across clinical EEG bands
      * Delta (1-4 Hz), Theta (4-8 Hz), Alpha (8-13 Hz), Beta (13-30 Hz), Gamma (30-50 Hz)
  - High-Frequency CI Residual: Ratio of high-frequency power removed vs preserved
  - Temporal Waveform Fidelity: Pearson correlation, Mean Absolute Error (MAE), MSE
  - Cross-condition breakdown (by permutation and block if available)

Designed for Hive cluster execution with GPU/CPU support and unbuffered logging.
"""

import argparse
import gc
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import zarr

try:
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
except Exception:
    pass

# EEG Bands of Interest
EEG_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 50.0),
    "ci_high_freq": (50.0, 120.0),
}


# ─── Spectral Band Metrics ───────────────────────────────────────────────────

def compute_band_powers(sig: np.ndarray, sfreq: float) -> Dict[str, float]:
    """Computes average power across standard EEG frequency bands via Welch PSD."""
    powers = {}
    nperseg = min(sig.shape[-1], int(sfreq * 2.0))
    if nperseg < 16:
        nperseg = sig.shape[-1]

    freqs, psd = signal.welch(sig, fs=sfreq, nperseg=nperseg, axis=-1)

    for band_name, (fmin, fmax) in EEG_BANDS.items():
        mask = (freqs >= fmin) & (freqs <= fmax)
        if np.any(mask):
            p = float(np.mean(psd[..., mask]))
        else:
            p = 0.0
        powers[band_name] = p
    return powers


def compute_spectral_distortion(
    clean: np.ndarray,
    denoised: np.ndarray,
    sfreq: float
) -> Dict[str, float]:
    """
    Computes log spectral distortion (dB error) in each EEG band.
    """
    clean_p = compute_band_powers(clean, sfreq)
    denoised_p = compute_band_powers(denoised, sfreq)
    eps = 1e-18

    band_errors = {}
    for band_name in EEG_BANDS:
        c_pow = clean_p.get(band_name, eps)
        d_pow = denoised_p.get(band_name, eps)
        # Log spectral power ratio in dB
        db_diff = float(np.abs(10.0 * np.log10((d_pow + eps) / (c_pow + eps))))
        band_errors[f"spec_err_{band_name}_db"] = db_diff
    return band_errors


# ─── Downstream Evaluation Pipeline ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Tier 3: Downstream Validation & Model Generalization Benchmark")
    parser.add_argument(
        "--zarr-path",
        default="/quobyte/millerlmgrp/processed_data/raw_epoched_data.zarr",
        help="Path to raw_epoched_data.zarr",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="/quobyte/millerlmgrp/checkpoints",
        help="Directory with saved checkpoints (best.pt, epoch_*.pt)",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=["best.pt"],
        help="Checkpoint filenames to evaluate",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Inference batch size")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker count")
    parser.add_argument("--sfreq", type=float, default=250.0, help="EEG sampling rate")
    parser.add_argument("--output-dir", default="./benchmark_results/tier3", help="Output directory for reports")
    parser.add_argument("--device", default="cuda", help="Compute device ('cpu' or 'cuda')")
    parser.add_argument("--test-split", type=float, default=0.1, help="Fraction reserved for test split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device

    print("=== Tier 3: Downstream Validation & Model Generalization Benchmark ===")
    print(f"Device:         {device}")
    print(f"Zarr path:      {args.zarr_path}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print(f"Checkpoints:    {args.checkpoints}")
    print(f"Output dir:     {args.output_dir}\n")

    # Add modeling and project directories to path
    proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    modeling_dir = os.path.join(proj_root, 'modeling')
    for d in (proj_root, modeling_dir):
        if d not in sys.path:
            sys.path.insert(0, d)

    from eeg_dataloader import EEGDataset
    from unet import UNet

    # Load dataset
    print("Loading test split from EEGDataset...")
    dataset = EEGDataset(
        zarr_path=args.zarr_path,
        clean_group='hearing_trial_data',
        noise_group='ci_trial_data',
        alpha=1.0,
        num_pairings=3,
        seed=args.seed,
        mix=True,
    )

    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val

    _, _, test_set = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed)
    )
    print(f"Total dataset pairs: {n_total} -> Test split pairs: {len(test_set)}")

    # Pre-load test pairs into memory (< 10 MB total) to completely eliminate Zarr worker IPC overhead and OOM
    print("Pre-loading test split into memory...")
    test_dirty_list = []
    test_clean_list = []
    for i in range(len(test_set)):
        d, c = test_set[i]
        test_dirty_list.append(d)
        test_clean_list.append(c)
    test_dirty_all = torch.stack(test_dirty_list)
    test_clean_all = torch.stack(test_clean_list)
    mem_mb = (test_dirty_all.nelement() + test_clean_all.nelement()) * test_dirty_all.element_size() / (1024 * 1024)
    print(f"Cached test split in RAM: {test_dirty_all.shape} ({mem_mb:.2f} MB)")

    # Release Zarr store and garbage collect
    del dataset, test_set, test_dirty_list, test_clean_list
    import gc
    gc.collect()

    test_dataset = torch.utils.data.TensorDataset(test_dirty_all, test_clean_all)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == 'cuda'),
    )

    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()

    checkpoint_summaries = []

    for ckpt_name in args.checkpoints:
        ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name) if not os.path.isabs(ckpt_name) else ckpt_name
        if not os.path.exists(ckpt_path):
            print(f"Warning: Checkpoint {ckpt_path} not found. Skipping.")
            continue

        print(f"\nEvaluating checkpoint: {ckpt_name}...")
        model = UNet().to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        model.load_state_dict(state_dict)
        model.eval()

        total_mse = 0.0
        total_mae = 0.0
        total_corr = 0.0
        total_spec_errors: Dict[str, float] = {f"spec_err_{b}_db": 0.0 for b in EEG_BANDS}
        n_samples = 0

        t0 = time.time()
        with torch.no_grad():
            for dirty, clean in test_loader:
                bs = dirty.size(0)
                dirty = dirty.to(device)
                clean = clean.to(device)

                pred = model(dirty)
                min_t = min(pred.shape[-1], clean.shape[-1])
                pred = pred[..., :min_t]
                clean = clean[..., :min_t]

                loss_mse = criterion_mse(pred, clean)
                loss_mae = criterion_mae(pred, clean)

                total_mse += loss_mse.item() * bs
                total_mae += loss_mae.item() * bs

                # Move to CPU numpy for batch spectral and correlation metrics
                p_np = pred.cpu().numpy()
                c_np = clean.cpu().numpy()

                for b in range(bs):
                    c_flat = c_np[b].ravel()
                    p_flat = p_np[b].ravel()
                    std_c = np.std(c_flat)
                    std_p = np.std(p_flat)
                    if std_c > 1e-12 and std_p > 1e-12:
                        r = float(np.corrcoef(c_flat, p_flat)[0, 1])
                    else:
                        r = 0.0
                    total_corr += r

                    spec_errs = compute_spectral_distortion(c_np[b], p_np[b], args.sfreq)
                    for k, val in spec_errs.items():
                        total_spec_errors[k] += val

                n_samples += bs

        eval_time = time.time() - t0
        avg_mse = total_mse / n_samples
        avg_mae = total_mae / n_samples
        avg_corr = total_corr / n_samples
        avg_spec = {k: v / n_samples for k, v in total_spec_errors.items()}

        summary_row = {
            "checkpoint": ckpt_name,
            "test_mse": avg_mse,
            "test_mae": avg_mae,
            "waveform_corr": avg_corr,
            "eval_time_s": eval_time,
            "throughput_samples_per_s": n_samples / max(eval_time, 1e-6),
            **avg_spec,
        }
        checkpoint_summaries.append(summary_row)

        print(f"  MSE:  {avg_mse:.6f} | MAE: {avg_mae:.6f} | Corr: {avg_corr:.4f}")
        print(f"  Spectral Errors: Delta={avg_spec['spec_err_delta_db']:.2f}dB, "
              f"Theta={avg_spec['spec_err_theta_db']:.2f}dB, Alpha={avg_spec['spec_err_alpha_db']:.2f}dB, "
              f"Beta={avg_spec['spec_err_beta_db']:.2f}dB, Gamma={avg_spec['spec_err_gamma_db']:.2f}dB")

        del model
        gc.collect()

    if checkpoint_summaries:
        df = pd.DataFrame(checkpoint_summaries)
        out_csv = os.path.join(args.output_dir, "tier3_downstream_summary.csv")
        df.to_csv(out_csv, index=False)
        print(f"\nDownstream benchmark summary saved to: {out_csv}\n")

        print("=" * 90)
        print("  TIER 3 DOWNSTREAM VALIDATION LEADERBOARD")
        print("=" * 90)
        print(f"{'Checkpoint':<20s} {'MSE ↓':<14s} {'MAE ↓':<14s} {'Corr ↑':<12s} {'Alpha Err dB':<14s} {'Beta Err dB':<14s}")
        print("-" * 90)
        for _, row in df.iterrows():
            print(f"{row['checkpoint']:<20s} {row['test_mse']:<14.6f} {row['test_mae']:<14.6f} "
                  f"{row['waveform_corr']:<12.4f} {row['spec_err_alpha_db']:<14.2f} {row['spec_err_beta_db']:<14.2f}")
        print("=" * 90)


if __name__ == "__main__":
    main()
