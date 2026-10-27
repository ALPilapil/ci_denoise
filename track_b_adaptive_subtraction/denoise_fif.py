"""
denoise_fif.py - Batch Processor for Track B AATS Denoising.

Reads annotated raw .fif files, applies 2 Hz high-pass filter, runs AATS filter,
and saves cleaned .fif files with metadata companion CSVs.
"""

import argparse
import glob
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import mne
import numpy as np
import pandas as pd

from aats_filter import AATSFilter


def find_fif_files(data_dir: str, year: Optional[str] = None) -> List[str]:
    pattern = f"CMPy{year}" if year else "CMPy*"
    files = sorted(glob.glob(os.path.join(data_dir, pattern, "*-raw.fif")))
    if not files:
        files = sorted(glob.glob(os.path.join(data_dir, "*-raw.fif")))
    return [f for f in files if os.path.exists(f) and os.path.getsize(f) > 100000]


def parse_subject_year(fif_path: str) -> Tuple[str, Optional[str]]:
    parts = Path(fif_path).parts
    cmp_dir = next((p for p in parts if p.startswith("CMPy")), None)
    year = re.search(r"CMPy(\d+)", cmp_dir).group(1) if cmp_dir else None
    m = re.search(r"trial_onsets_(.+?)-raw\.fif", Path(fif_path).name)
    subject_id = m.group(1) if m else Path(fif_path).stem
    return subject_id, year


def main():
    parser = argparse.ArgumentParser(description="Track B: Run AATS Denoising on Annotated FIF Files")
    parser.add_argument("--data-dir", default="/quobyte/millerlmgrp/annotated_data", help="Root dir of annotated FIF files")
    parser.add_argument("--clean-dir", default="/quobyte/millerlmgrp/aats_cleaned_data", help="Output dir for cleaned FIF files")
    parser.add_argument("--noise-dir", default="/quobyte/millerlmgrp/aats_isolated_noise", help="Output dir for isolated noise FIF files")
    parser.add_argument("--year", default=None, help="Year filter ('2' for CMPy2, etc.)")
    parser.add_argument("--n-taps", type=int, default=6, help="Number of FIR delay taps for AATS")
    parser.add_argument("--reg-lambda", type=float, default=1e-3, help="Ridge regularization parameter")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite already cleaned files")
    args = parser.parse_args()

    files = find_fif_files(args.data_dir, year=args.year)
    if not files:
        print(f"No valid .fif files found under {args.data_dir} (year={args.year})", flush=True)
        return

    print("=" * 70)
    print("  TRACK B: STIMULUS-LOCKED ADAPTIVE ARTIFACT TEMPLATE SUBTRACTION")
    print("=" * 70)
    print(f"Data dir:    {args.data_dir}")
    print(f"Clean dir:   {args.clean_dir}")
    print(f"Files found: {len(files)}")
    print(f"FIR taps:    {args.n_taps}")
    print(f"Lambda:      {args.reg_lambda}")
    print(f"Overwrite:   {args.overwrite}\n", flush=True)

    aats = AATSFilter(n_taps=args.n_taps, reg_lambda=args.reg_lambda)

    import gc

    for fif_path in files:
        subject_id, year = parse_subject_year(fif_path)
        year_label = f"CMPy{year}" if year else "CMPyUnknown"

        sub_clean_dir = os.path.join(args.clean_dir, year_label)
        sub_noise_dir = os.path.join(args.noise_dir, year_label)
        os.makedirs(sub_clean_dir, exist_ok=True)
        os.makedirs(sub_noise_dir, exist_ok=True)

        clean_fif = os.path.join(sub_clean_dir, f"trial_onsets_{subject_id}-raw.fif")
        noise_fif = os.path.join(sub_noise_dir, f"trial_onsets_{subject_id}-raw.fif")

        if not args.overwrite and os.path.exists(clean_fif) and os.path.getsize(clean_fif) > 100000:
            print(f"[{subject_id}] Already cleaned. Skipping → {clean_fif}", flush=True)
            continue

        print(f"[{subject_id}] Loading {os.path.basename(fif_path)}...", flush=True)
        t0 = time.time()
        raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
        raw.filter(l_freq=2.0, h_freq=None, verbose=False)

        clean_raw, noise_raw = aats.fit_transform_raw(raw)
        elapsed = time.time() - t0

        clean_raw.save(clean_fif, overwrite=True, verbose=False)
        noise_raw.save(noise_fif, overwrite=True, verbose=False)

        meta = {
            "participant_id": subject_id,
            "year": year,
            "method": "aats",
            "n_taps": args.n_taps,
            "reg_lambda": args.reg_lambda,
            "reference_channels": ",".join(aats.used_ref_channels),
            "clean_output_path": clean_fif,
            "duration_s": elapsed,
        }
        pd.DataFrame([meta]).to_csv(os.path.join(sub_clean_dir, f"trial_onsets_{subject_id}.csv"), index=False)
        print(f"  [{subject_id}] Cleaned in {elapsed:.2f}s → {clean_fif}", flush=True)

        del raw, clean_raw, noise_raw
        gc.collect()

    print("\nTrack B Batch Denoising Complete!", flush=True)


if __name__ == "__main__":
    main()
