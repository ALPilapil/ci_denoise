"""
evaluate_erp.py - Standalone Real Auditory ERP Preservation Benchmark for Track B.

Evaluates Raw CI EEG vs. AATS Cleaned EEG (and optionally CCA/PCA baselines):
  - N1 peak amplitude & latency in [80, 150] ms
  - P2 peak amplitude & latency in [150, 250] ms
  - N1-P2 Peak-to-Peak amplitude (uV)
  - Pre-stimulus baseline noise RMS in [-200, 0] ms
  - Auditory ERP SNR (dB)
  - Template correlation with hearing control grand-average
"""

import argparse
import glob
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd


def compute_evoked_from_fif(
    fif_path: str,
    event_desc: str = "A",
    tmin: float = -0.2,
    tmax: float = 1.0,
    baseline: Tuple[float, float] = (-0.2, 0.0),
    picks: Optional[List[str]] = None,
) -> Optional[mne.Evoked]:
    import gc
    raw = None
    epochs = None
    try:
        if not os.path.exists(fif_path) or os.path.getsize(fif_path) < 100000:
            return None
        raw = mne.io.read_raw_fif(fif_path, preload=False, verbose=False)
        if raw.annotations is None or event_desc not in raw.annotations.description:
            return None
        events, _ = mne.events_from_annotations(raw, event_id={event_desc: 1}, verbose=False)
        if len(events) == 0:
            return None

        # Filter picks to channels present in raw
        valid_picks = [ch for ch in (picks or []) if ch in raw.ch_names] if picks else None

        epochs = mne.Epochs(raw, events, event_id={event_desc: 1}, tmin=tmin, tmax=tmax,
                            baseline=baseline, picks=valid_picks, preload=True, verbose=False)
        evoked = epochs.average()
        return evoked
    except Exception as e:
        print(f"  Error reading {os.path.basename(fif_path)}: {e}", flush=True)
        return None
    finally:
        del raw, epochs
        gc.collect()


def extract_erp_metrics(evoked: mne.Evoked, target_ch: str = "Cz") -> Dict[str, float]:
    times = evoked.times
    ch_names = evoked.ch_names
    ch_idx = ch_names.index(target_ch) if target_ch in ch_names else 0
    data = evoked.data[ch_idx]

    # Baseline RMS in [-200, 0] ms
    b_mask = (times >= -0.2) & (times <= 0.0)
    base_rms_uv = float(np.sqrt(np.mean(data[b_mask] ** 2))) * 1e6 if np.any(b_mask) else 1e-3

    # N1 in [80, 150] ms
    n1_mask = (times >= 0.080) & (times <= 0.150)
    if np.any(n1_mask):
        n1_idx = np.where(n1_mask)[0][np.argmin(data[n1_mask])]
        n1_amp_uv = float(data[n1_idx] * 1e6)
        n1_lat_ms = float(times[n1_idx] * 1000.0)
    else:
        n1_amp_uv, n1_lat_ms = np.nan, np.nan

    # P2 in [150, 250] ms
    p2_mask = (times >= 0.150) & (times <= 0.250)
    if np.any(p2_mask):
        p2_idx = np.where(p2_mask)[0][np.argmax(data[p2_mask])]
        p2_amp_uv = float(data[p2_idx] * 1e6)
        p2_lat_ms = float(times[p2_idx] * 1000.0)
    else:
        p2_amp_uv, p2_lat_ms = np.nan, np.nan

    p2p_uv = float(p2_amp_uv - n1_amp_uv) if not np.isnan(p2_amp_uv) and not np.isnan(n1_amp_uv) else np.nan

    post_mask = (times >= 0.050) & (times <= 0.350)
    post_power = np.max(data[post_mask] ** 2) if np.any(post_mask) else 1e-12
    erp_snr_db = float(10.0 * np.log10(post_power / ((base_rms_uv * 1e-6) ** 2 + 1e-18)))

    return {
        "channel": target_ch,
        "n1_amp_uv": n1_amp_uv,
        "n1_lat_ms": n1_lat_ms,
        "p2_amp_uv": p2_amp_uv,
        "p2_lat_ms": p2_lat_ms,
        "p2p_uv": p2p_uv,
        "base_rms_uv": base_rms_uv,
        "erp_snr_db": erp_snr_db,
    }


def compute_template_corr(test_evoked: mne.Evoked, tmpl_evoked: mne.Evoked, ch: str = "Cz") -> float:
    if ch not in test_evoked.ch_names or ch not in tmpl_evoked.ch_names:
        return np.nan
    mask_t = (test_evoked.times >= 0.050) & (test_evoked.times <= 0.400)
    mask_m = (tmpl_evoked.times >= 0.050) & (tmpl_evoked.times <= 0.400)
    s_t = test_evoked.data[test_evoked.ch_names.index(ch), mask_t]
    s_m = tmpl_evoked.data[tmpl_evoked.ch_names.index(ch), mask_m]
    min_l = min(len(s_t), len(s_m))
    if min_l < 5 or np.std(s_t[:min_l]) < 1e-12 or np.std(s_m[:min_l]) < 1e-12:
        return 0.0
    return float(np.corrcoef(s_t[:min_l], s_m[:min_l])[0, 1])


def load_hearing_template(zarr_path: str, max_trials: int = 200) -> Optional[mne.Evoked]:
    """
    Computes hearing control grand-average template using incremental batching
    to avoid loading full multi-gigabyte Zarr arrays into RAM.
    """
    if not os.path.exists(zarr_path):
        return None
    try:
        import zarr
        try:
            root = zarr.open(zarr_path, mode='r')
        except Exception:
            root = zarr.open_group(zarr_path, mode='r')

        if 'hearing_trial_data' not in root or 'data' not in root['hearing_trial_data']:
            return None

        data_arr = root['hearing_trial_data']['data']
        total_trials = data_arr.shape[0] if hasattr(data_arr, 'shape') else len(data_arr)
        n_trials = min(total_trials, max_trials)
        batch_size = 50
        sum_arr = None
        count = 0

        for start_idx in range(0, n_trials, batch_size):
            end_idx = min(start_idx + batch_size, n_trials)
            chunk = data_arr[start_idx:end_idx]
            valid_mask = ~np.isnan(chunk).any(axis=(1, 2))
            valid = chunk[valid_mask]
            if len(valid) > 0:
                if sum_arr is None:
                    sum_arr = np.sum(valid, axis=0)
                else:
                    sum_arr += np.sum(valid, axis=0)
                count += len(valid)

        if count == 0 or sum_arr is None:
            return None

        avg = sum_arr / count
        ch_names = ['Fp1', 'Fz', 'F3', 'F7', 'T7', 'C3', 'Cz', 'Pz', 'P3', 'P7', 'O1',
                    'Fp2', 'F8', 'F4', 'C4', 'T8', 'P8', 'P4', 'O2', 'M1', 'M2']
        info = mne.create_info(ch_names=ch_names[:avg.shape[0]], sfreq=250.0, ch_types='eeg')
        print(f"Loaded hearing template averaged across {count} clean trials (< 5 MB RAM).", flush=True)
        return mne.EvokedArray(avg, info, tmin=-0.2)
    except Exception as e:
        print(f"Notice: Could not load hearing template from Zarr ({e}). Skipping correlation.", flush=True)
        return None


def main():
    parser = argparse.ArgumentParser(description="Track B: Evaluate Real ERP Preservation")
    parser.add_argument("--annotated-dir", default="/quobyte/millerlmgrp/annotated_data", help="Raw annotated FIF dir")
    parser.add_argument("--aats-dir", default="/quobyte/millerlmgrp/aats_cleaned_data", help="AATS cleaned FIF dir")
    parser.add_argument("--classical-root", default="/quobyte/millerlmgrp/ml_cleaned_data", help="Preexisting cleaned root (optional)")
    parser.add_argument("--zarr-path", default="/quobyte/millerlmgrp/processed_data/raw_epoched_data.zarr", help="Zarr for hearing template")
    parser.add_argument("--output-dir", default="./benchmark_results/track_b", help="Output directory")
    parser.add_argument("--condition", default="A", help="Trial condition (A, AV, V)")
    parser.add_argument("--target-channel", default="Cz", help="Evaluation channel")
    parser.add_argument("--year", default=None, help="Year filter")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    template_evoked = load_hearing_template(args.zarr_path)

    # Discover raw files
    raw_files = sorted(glob.glob(os.path.join(args.annotated_dir, "*", "trial_onsets_09*-raw.fif")))
    if not raw_files:
        raw_files = sorted(glob.glob(os.path.join(args.annotated_dir, "trial_onsets_09*-raw.fif")))
    raw_files = [f for f in raw_files if os.path.exists(f) and os.path.getsize(f) > 100000]
    if args.year:
        raw_files = [f for f in raw_files if f"y{args.year}" in os.path.basename(f) or f"CMPy{args.year}" in f]

    print("=" * 70)
    print("  TRACK B: REAL AUDITORY ERP PRESERVATION BENCHMARK")
    print("=" * 70)
    print(f"Condition:      {args.condition}")
    print(f"Channel:        {args.target_channel}")
    print(f"Valid subjects: {len(raw_files)}\n")

    methods_to_test = ["raw", "aats"]
    # Check if preexisting classical methods exist for comparison
    for m in ["cca", "pca"]:
        if os.path.exists(os.path.join(args.classical_root, m)):
            methods_to_test.append(m)

    all_rows = []

    for m in methods_to_test:
        print(f"Evaluating method: [{m}]...")
        for r_path in raw_files:
            sub = os.path.basename(r_path).replace("trial_onsets_", "").replace("-raw.fif", "")

            if m == "raw":
                target_fif = r_path
            elif m == "aats":
                matches = glob.glob(os.path.join(args.aats_dir, "**", f"*{sub}*-raw.fif"), recursive=True)
                if not matches:
                    continue
                target_fif = matches[0]
            else:
                matches = glob.glob(os.path.join(args.classical_root, m, "**", f"*{sub}*-raw.fif"), recursive=True)
                if not matches:
                    continue
                target_fif = matches[0]

            evoked = compute_evoked_from_fif(target_fif, event_desc=args.condition, picks=[args.target_channel])
            if evoked is None:
                continue

            metrics = extract_erp_metrics(evoked, target_ch=args.target_channel)
            corr = compute_template_corr(evoked, template_evoked, ch=args.target_channel) if template_evoked else np.nan

            row = {
                "method": m,
                "subject_id": sub,
                "condition": args.condition,
                "template_corr": corr,
                **metrics,
            }
            all_rows.append(row)
            print(f"  [{m}] {sub}: N1={metrics['n1_amp_uv']:.2f}uV @ {metrics['n1_lat_ms']:.0f}ms | P2P={metrics['p2p_uv']:.2f}uV | BaseRMS={metrics['base_rms_uv']:.2f}uV | Corr={corr:+.3f}", flush=True)

            del evoked
            import gc
            gc.collect()

    if all_rows:
        df = pd.DataFrame(all_rows)
        csv_path = os.path.join(args.output_dir, "track_b_subject_erp_metrics.csv")
        df.to_csv(csv_path, index=False)

        metric_cols = ["n1_amp_uv", "n1_lat_ms", "p2_amp_uv", "p2_lat_ms", "p2p_uv", "base_rms_uv", "erp_snr_db"]
        if "template_corr" in df.columns and df["template_corr"].notna().any():
            metric_cols.insert(0, "template_corr")

        # Robust aggregation: filter unphysiological divergences (|P2P| > 500 uV) from summary
        outliers = df[df["p2p_uv"].abs() > 500]
        if not outliers.empty:
            print(f"\nNotice: Filtered {len(outliers)} anomalous outlier(s) (|P2P| > 500uV) from summary leaderboard: {list(outliers['subject_id'])}", flush=True)

        clean_df = df[df["p2p_uv"].abs() <= 500]
        summary = clean_df.groupby("method")[metric_cols].agg(["mean", "std"])
        summary_csv = os.path.join(args.output_dir, "track_b_method_summary.csv")
        summary.to_csv(summary_csv)
        print(f"\nLeaderboard saved to: {summary_csv}", flush=True)
        print("\n=== SUMMARY LEADERBOARD ===\n", flush=True)
        print(summary.to_string(), flush=True)


if __name__ == "__main__":
    main()
