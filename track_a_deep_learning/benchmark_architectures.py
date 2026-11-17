"""
benchmark_architectures.py

Head-to-Head GPU Benchmark of 4 Neural Architectures on Augmented SOBI-Isolated Synthetic Pairs:
1. UNet1D_Baseline         (Direct 1D U-Net baseline)
2. ResAttnUNet1D           (Residual noise-predicting U-Net + GroupNorm + Spatial SE Attention)
3. STFT_MaskNet            (Complex 2D STFT Time-Frequency U-Net + iSTFT residual subtraction)
4. SpatioTemporalConformer (Learnable spatial unmixing + Multi-Head Self-Attention + Depthwise Conv)

Evaluates all 4 models on:
- Held-Out Synthetic Test Set (unseen Hearing & CI subjects)
- Held-Out Real Clinical CI EEG (unseen CI patients)
- Zero-Shot Real CI ERP Recovery (unseen 0901y2 vs. 0801y2)
"""

import argparse
import glob
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import signal
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "processing"))

from losses import MultiResolutionSTFTLoss, PearsonCorrelationLoss
from train_sobi_unet import AugmentedSOBISyntheticDataset, CANONICAL_20_CHS
from models_suite import build_model
from compare_sobi_ica import setup_montage, get_averaged_erp, compute_metrics
from hemispheric_normalization import normalize_hemispheric_channels


def subject_split_pinned(subjects: np.ndarray, pinned_test: str, seed: int = 42):
    uniq = sorted(list(set(subjects)))
    if pinned_test in uniq:
        uniq.remove(pinned_test)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    n_tr = max(1, int(0.75 * (n + 1)))
    n_va = max(1, int(0.125 * (n + 1)))
    tr_s = set(uniq[:n_tr])
    va_s = set(uniq[n_tr:n_tr + n_va])
    te_s = set(uniq[n_tr + n_va:])
    if pinned_test in set(subjects):
        te_s.add(pinned_test)
    tr_m = np.array([s in tr_s for s in subjects])
    va_m = np.array([s in va_s for s in subjects])
    te_m = np.array([s in te_s for s in subjects])
    return tr_m, va_m, te_m, tr_s, va_s, te_s


def eval_sliding_erp(
    model: nn.Module,
    raw_ci_path: str,
    raw_h_path: str,
    device: torch.device,
    scale_uv: float = 100.0,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Runs zero-shot sliding-window denoising on unseen CI subject 0901y2 and computes Cz ERP vs 0801y2."""
    raw_ci = mne.io.read_raw_fif(raw_ci_path, preload=True, verbose=False)
    raw_ci = setup_montage(raw_ci)
    raw_ci.pick(CANONICAL_20_CHS)
    raw_ci.reorder_channels(CANONICAL_20_CHS)
    raw_ci.resample(256.0, verbose=False)
    raw_ci.filter(1.0, 50.0, verbose=False)

    raw_ci_norm, flipped = normalize_hemispheric_channels(raw_ci, target_side="LEFT_CI")
    if flipped:
        raw_ci_norm.reorder_channels(CANONICAL_20_CHS)

    raw_h = mne.io.read_raw_fif(raw_h_path, preload=True, verbose=False)
    raw_h = setup_montage(raw_h)
    raw_h.pick(CANONICAL_20_CHS)
    raw_h.reorder_channels(CANONICAL_20_CHS)
    raw_h.resample(256.0, verbose=False)
    raw_h.filter(1.0, 50.0, verbose=False)

    data_uv = raw_ci_norm.get_data() * 1e6  # (19, T)
    C, T = data_uv.shape
    win_len = 1024
    stride = 512

    out_acc = np.zeros_like(data_uv)
    w_acc = np.zeros((1, T), dtype=np.float64)
    hann = np.hanning(win_len).reshape(1, -1)

    model.eval()
    with torch.no_grad():
        for s in range(0, T - win_len + 1, stride):
            seg = (data_uv[:, s:s + win_len] / scale_uv).astype(np.float32)
            inp = torch.from_numpy(seg).unsqueeze(0).to(device)
            pred = model(inp).squeeze(0).cpu().numpy() * scale_uv
            out_acc[:, s:s + win_len] += pred * hann
            w_acc[:, s:s + win_len] += hann

    mask = w_acc > 1e-6
    denoised_uv = np.where(mask, out_acc / np.maximum(w_acc, 1e-6), data_uv)

    raw_den = mne.io.RawArray(denoised_uv * 1e-6, raw_ci_norm.info.copy(), verbose=False)
    raw_den.set_annotations(raw_ci_norm.annotations)

    evk_den = get_averaged_erp(raw_den)
    evk_h = get_averaged_erp(raw_h)
    erp_metrics = compute_metrics(evk_den, evk_h, "Cz")

    cz_idx = evk_den.ch_names.index("Cz")
    return erp_metrics, evk_den.times * 1000.0, evk_den.data[cz_idx] * 1e6, evk_h.data[cz_idx] * 1e6


def main():
    parser = argparse.ArgumentParser(description="Benchmark 4 Neural Architectures on SOBI Synthetic Pairs")
    parser.add_argument("--dataset-path", default="/quobyte/millerlmgrp/processed_data/sobi_synthetic_4s_256hz.npz")
    parser.add_argument("--data-dir", default="/quobyte/millerlmgrp/annotated_data")
    parser.add_argument("--checkpoint-dir", default="/quobyte/millerlmgrp/checkpoints/track_a_arch_bench")
    parser.add_argument("--output-dir", default="./benchmark_results/architecture_benchmark")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--scale-uv", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 85)
    print(f"  MULTI-ARCHITECTURE BENCHMARK ON SOBI SYNTHETIC PAIRS | Device: {device}")
    print("=" * 85)

    data = np.load(args.dataset_path, allow_pickle=False)
    clean_w, clean_s = data["clean_windows"], data["clean_subjects"]
    noise_w, noise_s = data["noise_windows"], data["noise_subjects"]
    real_ci_w, real_ci_s = data["real_ci_windows"], data["real_ci_subjects"]

    c_tr_m, c_va_m, c_te_m, c_tr_s, c_va_s, c_te_s = subject_split_pinned(clean_s, "0801y2", args.seed)
    n_tr_m, n_va_m, n_te_m, n_tr_s, n_va_s, n_te_s = subject_split_pinned(noise_s, "0901y2", args.seed + 1)

    print(f"Pinned Zero-Leakage Test Subjects: Hearing='0801y2' (+{len(c_te_s)-1}), CI='0901y2' (+{len(n_te_s)-1})")

    train_ds = AugmentedSOBISyntheticDataset(clean_w[c_tr_m], noise_w[n_tr_m], 8000, augment=True, scale_uv=args.scale_uv, seed=args.seed)
    val_ds = AugmentedSOBISyntheticDataset(clean_w[c_va_m], noise_w[n_va_m], 1200, augment=False, scale_uv=args.scale_uv, seed=args.seed + 10)
    test_ds = AugmentedSOBISyntheticDataset(clean_w[c_te_m], noise_w[n_te_m], 1200, augment=False, scale_uv=args.scale_uv, seed=args.seed + 20)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    real_te_mask = np.array([s in n_te_s for s in real_ci_s])
    real_te_all = real_ci_w[real_te_mask]
    # Filter extreme dead-electrode spikes for clean clinical summary
    real_te = real_te_all[np.max(np.abs(real_te_all), axis=(1, 2)) < 25000.0][:300]

    ci_0901_path = sorted(glob.glob(os.path.join(args.data_dir, "**", "*0901y2*-raw.fif"), recursive=True))[0]
    h_0801_path = sorted(glob.glob(os.path.join(args.data_dir, "**", "*0801y2*-raw.fif"), recursive=True))[0]

    arch_names = ["UNet1D_Baseline", "ResAttnUNet1D", "STFT_MaskNet", "SpatioTemporalConformer"]
    l1_loss = nn.L1Loss()
    stft_loss = MultiResolutionSTFTLoss(fft_sizes=(64, 128, 256)).to(device)
    corr_loss = PearsonCorrelationLoss().to(device)

    leaderboard = []
    loss_histories = {}
    erp_traces = {}
    erp_times_ms = None
    hearing_erp_uv = None

    p7_idx, cz_idx = CANONICAL_20_CHS.index("P7"), CANONICAL_20_CHS.index("Cz")

    for name in arch_names:
        print("\n" + "-" * 80)
        print(f"  TRAINING ARCHITECTURE: {name}")
        print("-" * 80)
        torch.manual_seed(args.seed)
        model = build_model(name, in_channels=19).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

        best_val = float("inf")
        ckpt_path = os.path.join(args.checkpoint_dir, f"best_{name}.pt")
        hist = []

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            model.train()
            tr_l = 0.0
            for dirty, clean in train_loader:
                dirty, clean = dirty.to(device), clean.to(device)
                pred = model(dirty)
                loss = l1_loss(pred, clean) + 0.5 * stft_loss(pred, clean) + 0.3 * corr_loss(pred, clean)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                tr_l += loss.item() * dirty.size(0)
            tr_l /= len(train_ds)
            scheduler.step()

            model.eval()
            va_l = 0.0
            with torch.no_grad():
                for dirty, clean in val_loader:
                    dirty, clean = dirty.to(device), clean.to(device)
                    pred = model(dirty)
                    loss = l1_loss(pred, clean) + 0.5 * stft_loss(pred, clean) + 0.3 * corr_loss(pred, clean)
                    va_l += loss.item() * dirty.size(0)
            va_l /= len(val_ds)
            hist.append(va_l)

            mark = ""
            if va_l < best_val:
                best_val = va_l
                torch.save(model.state_dict(), ckpt_path)
                mark = " [BEST]"
            if epoch in (1, 10, 20, 30) or mark:
                print(f"  Epoch {epoch:02d}/{args.epochs} | Train: {tr_l:.4f} | Val: {va_l:.4f} | {time.time()-t0:.1f}s{mark}")

        loss_histories[name] = hist
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        # Evaluate Synthetic Test Set
        dirty_rmses, pred_rmses, dirty_snrs, pred_snrs, pred_corrs, lf_ret = [], [], [], [], [], []
        with torch.no_grad():
            for dirty, clean in test_loader:
                dirty, clean = dirty.to(device), clean.to(device)
                pred = model(dirty)
                d_uv = (dirty.cpu().numpy() * args.scale_uv).astype(np.float64)
                c_uv = (clean.cpu().numpy() * args.scale_uv).astype(np.float64)
                p_uv = (pred.cpu().numpy() * args.scale_uv).astype(np.float64)
                for i in range(d_uv.shape[0]):
                    dirty_rmses.append(float(np.sqrt(np.mean((d_uv[i] - c_uv[i]) ** 2))))
                    pred_rmses.append(float(np.sqrt(np.mean((p_uv[i] - c_uv[i]) ** 2))))
                    p_sig = np.mean(c_uv[i] ** 2) + 1e-8
                    dirty_snrs.append(float(10.0 * np.log10(p_sig / (np.mean((d_uv[i] - c_uv[i]) ** 2) + 1e-8))))
                    pred_snrs.append(float(10.0 * np.log10(p_sig / (np.mean((p_uv[i] - c_uv[i]) ** 2) + 1e-8))))
                    pc = np.corrcoef(p_uv[i, cz_idx], c_uv[i, cz_idx])[0, 1]
                    if not np.isnan(pc):
                        pred_corrs.append(float(pc))
                    freqs, psd_c = signal.welch(c_uv[i, cz_idx], fs=256.0, nperseg=256)
                    _, psd_p = signal.welch(p_uv[i, cz_idx], fs=256.0, nperseg=256)
                    lf = (freqs >= 1.0) & (freqs <= 15.0)
                    lf_ret.append(float(100.0 * np.sum(psd_p[lf]) / (np.sum(psd_c[lf]) + 1e-12)))

        # Evaluate Real CI Holdout
        p7_raw, p7_den, rf_atten = [], [], []
        with torch.no_grad():
            for i in range(0, len(real_te), args.batch_size):
                batch = torch.from_numpy((real_te[i:i + args.batch_size] / args.scale_uv).astype(np.float32)).to(device)
                pred = (model(batch).cpu().numpy() * args.scale_uv).astype(np.float64)
                raw_b = real_te[i:i + args.batch_size].astype(np.float64)
                for j in range(len(raw_b)):
                    p7_raw.append(float(np.sqrt(np.mean(raw_b[j, p7_idx] ** 2))))
                    p7_den.append(float(np.sqrt(np.mean(pred[j, p7_idx] ** 2))))
                    freqs, psd_r = signal.welch(raw_b[j, cz_idx], fs=256.0, nperseg=256)
                    _, psd_d = signal.welch(pred[j, cz_idx], fs=256.0, nperseg=256)
                    hf = (freqs >= 20.0) & (freqs <= 50.0)
                    rf_atten.append(float(100.0 * (1.0 - np.sum(psd_d[hf]) / (np.sum(psd_r[hf]) + 1e-12))))

        # Evaluate Zero-Shot Real CI ERP on unseen 0901y2
        erp_m, t_ms, cz_den_erp, cz_h_erp = eval_sliding_erp(model, ci_0901_path, h_0801_path, device, args.scale_uv)
        erp_times_ms = t_ms
        hearing_erp_uv = cz_h_erp
        erp_traces[name] = cz_den_erp

        row = {
            "Architecture": name,
            "Params": n_params,
            "Best_Val_Loss": round(best_val, 4),
            "Synth_RMSE_uV": round(float(np.mean(pred_rmses)), 2),
            "Synth_SNR_Gain_dB": round(float(np.mean(pred_snrs) - np.mean(dirty_snrs)), 2),
            "Synth_Cz_Corr": round(float(np.mean(pred_corrs)), 4),
            "Synth_1_15Hz_Retained_Pct": round(float(np.mean(lf_ret)), 1),
            "Real_CI_P7_RMS_Red_Pct": round(100.0 * (1.0 - np.mean(p7_den) / (np.mean(p7_raw) + 1e-12)), 1),
            "Real_CI_20_50Hz_RF_Removed_Pct": round(float(np.mean(rf_atten)), 1),
            "Real_0901y2_ERP_P2_Lat_ms": round(erp_m["p2_lat_ms"], 1),
            "Real_0901y2_ERP_RMSE_uV": round(erp_m["rmse_with_hearing"], 2),
            "Real_0901y2_ERP_Corr": round(erp_m["corr_with_hearing"], 4),
        }
        leaderboard.append(row)

    df = pd.DataFrame(leaderboard)
    df.to_csv(os.path.join(args.output_dir, "architecture_comparison_leaderboard.csv"), index=False)
    with open(os.path.join(args.output_dir, "architecture_comparison_leaderboard.json"), "w") as f:
        json.dump(leaderboard, f, indent=2)

    print("\n" + "=" * 110)
    print("                      MULTI-ARCHITECTURE CI-EEG DENOISING LEADERBOARD")
    print("=" * 110)
    print(df.to_string(index=False))
    print("=" * 110 + "\n")

    # Plot Summary Figure
    fig = plt.figure(figsize=(16, 11), facecolor="#ffffff")
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)
    colors = {"UNet1D_Baseline": "#7f7f7f", "ResAttnUNet1D": "#1f77b4", "STFT_MaskNet": "#2ca02c", "SpatioTemporalConformer": "#d62728"}

    ax1 = fig.add_subplot(gs[0, 0])
    for name in arch_names:
        ax1.plot(range(1, args.epochs + 1), loss_histories[name], label=name, color=colors[name], lw=2.0)
    ax1.set_title("A. Validation Composite Loss on Unseen Subjects", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation Loss")
    ax1.legend()
    ax1.grid(True, ls=":", alpha=0.5)

    ax2 = fig.add_subplot(gs[0, 1])
    x_pos = np.arange(len(arch_names))
    rmses = [r["Synth_RMSE_uV"] for r in leaderboard]
    snrs = [r["Synth_SNR_Gain_dB"] for r in leaderboard]
    bars = ax2.bar(x_pos, snrs, color=[colors[n] for n in arch_names], width=0.55)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(arch_names, rotation=15, ha="right", fontsize=9)
    ax2.set_title("B. Synthetic Test SNR Gain (+dB) Across Unseen Subjects", fontweight="bold")
    ax2.set_ylabel("SNR Improvement (dB)")
    for b, val in zip(bars, snrs):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.15, f"+{val:.2f} dB", ha="center", fontweight="bold", fontsize=9)
    ax2.grid(True, axis="y", ls=":", alpha=0.5)

    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(erp_times_ms, hearing_erp_uv, label="Clean Hearing Control 0801y2 (Ground Truth)", color="#111111", lw=2.2, ls="--")
    for name in arch_names:
        r_val = [r["Real_0901y2_ERP_Corr"] for r in leaderboard if r["Architecture"] == name][0]
        p2_l = [r["Real_0901y2_ERP_P2_Lat_ms"] for r in leaderboard if r["Architecture"] == name][0]
        ax3.plot(erp_times_ms, erp_traces[name], label=f"{name} (r={r_val:.2f}, P2={p2_l:.0f}ms)", color=colors[name], lw=1.9)
    ax3.axvline(100, color="darkred", ls=":", alpha=0.5)
    ax3.axvline(200, color="navy", ls=":", alpha=0.5)
    ax3.set_title("C. Zero-Shot Real Clinical CI ERP Recovery at Cz (Unseen Subject 0901y2 vs. 0801y2)", fontweight="bold")
    ax3.set_xlabel("Time post-stimulus (ms)")
    ax3.set_ylabel("Amplitude (µV)")
    ax3.legend(loc="upper right", ncol=2, fontsize=9)
    ax3.grid(True, ls=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "architecture_comparison_summary.png"), dpi=300)
    plt.close(fig)
    print(f"Saved summary figure -> {os.path.join(args.output_dir, 'architecture_comparison_summary.png')}")


if __name__ == "__main__":
    main()
