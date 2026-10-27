"""
evaluate_downstream.py - Standalone Downstream & Spectral Validation for Track A.

Computes:
  - Subject-independent test MSE & MAE
  - Temporal waveform correlation
  - Spectral band distortion errors across clinical EEG bands (Delta, Theta, Alpha, Beta, Gamma)
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

try:
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
except Exception:
    pass

import numpy as np
import pandas as pd
from scipy import signal
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataloader import TrackADataset, split_dataset
from model import UNet1D

EEG_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 50.0),
}


def compute_band_powers(sig: np.ndarray, sfreq: float) -> Dict[str, float]:
    powers = {}
    nperseg = min(sig.shape[-1], int(sfreq * 2.0))
    if nperseg < 16:
        nperseg = sig.shape[-1]
    freqs, psd = signal.welch(sig, fs=sfreq, nperseg=nperseg, axis=-1)
    for band_name, (fmin, fmax) in EEG_BANDS.items():
        mask = (freqs >= fmin) & (freqs <= fmax)
        powers[band_name] = float(np.mean(psd[..., mask])) if np.any(mask) else 0.0
    return powers


def compute_spectral_distortion(clean: np.ndarray, denoised: np.ndarray, sfreq: float) -> Dict[str, float]:
    clean_p = compute_band_powers(clean, sfreq)
    denoised_p = compute_band_powers(denoised, sfreq)
    eps = 1e-18
    return {
        f"spec_err_{b}_db": float(np.abs(10.0 * np.log10((denoised_p.get(b, eps) + eps) / (clean_p.get(b, eps) + eps))))
        for b in EEG_BANDS
    }


def main():
    parser = argparse.ArgumentParser(description="Track A: Evaluate Trained Checkpoint Downstream")
    parser.add_argument(
        "--zarr-path",
        default="/quobyte/millerlmgrp/processed_data/raw_epoched_data.zarr",
        help="Path to raw_epoched_data.zarr",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="/quobyte/millerlmgrp/checkpoints/track_a",
        help="Directory containing checkpoints",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=["best.pt"],
        help="Checkpoint names to evaluate",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size")
    parser.add_argument("--sfreq", type=float, default=250.0, help="EEG sampling rate")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    parser.add_argument("--seed", type=int, default=42, help="Seed matching training split")
    parser.add_argument("--output-dir", default="./benchmark_results/track_a", help="Output directory for reports")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 70)
    print("  TRACK A: DOWNSTREAM EVALUATION & SPECTRAL FIDELITY BENCHMARK")
    print("=" * 70)
    print(f"Device:         {device}")
    print(f"Zarr path:      {args.zarr_path}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print(f"Checkpoints:    {args.checkpoints}\n")

    # Load dataset & isolate test split
    dataset = TrackADataset(zarr_path=args.zarr_path, seed=args.seed)
    _, _, test_set = split_dataset(dataset, train_frac=0.8, val_frac=0.1, seed=args.seed, strategy="participant")
    print(f"Test split pairs: {len(test_set)}")

    # Pre-cache test set in memory to eliminate DataLoader IPC overhead
    test_dirty_list, test_clean_list = [], []
    for i in range(len(test_set)):
        d, c = test_set[i]
        test_dirty_list.append(d)
        test_clean_list.append(c)

    test_dirty_all = torch.stack(test_dirty_list)
    test_clean_all = torch.stack(test_clean_list)
    del dataset, test_set, test_dirty_list, test_clean_list

    test_dataset = torch.utils.data.TensorDataset(test_dirty_all, test_clean_all)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()
    results = []

    for ckpt_name in args.checkpoints:
        ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name) if not os.path.isabs(ckpt_name) else ckpt_name
        if not os.path.exists(ckpt_path):
            print(f"Warning: Checkpoint {ckpt_path} not found. Skipping.")
            continue

        print(f"\nEvaluating: {ckpt_name}...")
        model = UNet1D().to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        model.load_state_dict(state_dict)
        model.eval()

        total_mse, total_mae, total_corr = 0.0, 0.0, 0.0
        total_spec = {f"spec_err_{b}_db": 0.0 for b in EEG_BANDS}
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

                total_mse += criterion_mse(pred, clean).item() * bs
                total_mae += criterion_mae(pred, clean).item() * bs

                p_np = pred.cpu().numpy()
                c_np = clean.cpu().numpy()

                for b in range(bs):
                    c_flat = c_np[b].ravel()
                    p_flat = p_np[b].ravel()
                    std_c = np.std(c_flat)
                    std_p = np.std(p_flat)
                    r = float(np.corrcoef(c_flat, p_flat)[0, 1]) if (std_c > 1e-12 and std_p > 1e-12) else 0.0
                    total_corr += r

                    errs = compute_spectral_distortion(c_np[b], p_np[b], args.sfreq)
                    for k, v in errs.items():
                        total_spec[k] += v

                n_samples += bs

        eval_time = time.time() - t0
        avg_mse = total_mse / n_samples
        avg_mae = total_mae / n_samples
        avg_corr = total_corr / n_samples
        avg_spec = {k: v / n_samples for k, v in total_spec.items()}

        summary = {
            "checkpoint": ckpt_name,
            "test_mse": avg_mse,
            "test_mae": avg_mae,
            "waveform_corr": avg_corr,
            "eval_time_s": eval_time,
            **avg_spec,
        }
        results.append(summary)

        print(f"  MSE:  {avg_mse:.6f} | MAE: {avg_mae:.6f} | Corr: {avg_corr:.4f}")
        print(f"  Spectral Errors: Delta={avg_spec['spec_err_delta_db']:.2f}dB, "
              f"Theta={avg_spec['spec_err_theta_db']:.2f}dB, Alpha={avg_spec['spec_err_alpha_db']:.2f}dB, "
              f"Beta={avg_spec['spec_err_beta_db']:.2f}dB, Gamma={avg_spec['spec_err_gamma_db']:.2f}dB")

    if results:
        df = pd.DataFrame(results)
        out_csv = os.path.join(args.output_dir, "track_a_downstream_summary.csv")
        df.to_csv(out_csv, index=False)
        print(f"\nDownstream results saved to: {out_csv}")


if __name__ == "__main__":
    main()
