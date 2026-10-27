"""
dump_visual_intervals.py

Quick utility to extract all Visual-Only (V) silent trial intervals
from the annotated FIF file of subject 0901y2 (or any specified subject).
Outputs both to console and as a CSV.
"""

import argparse
import csv
import glob
import os
import mne


def main():
    parser = argparse.ArgumentParser(description="Dump Visual-Only trial intervals from FIF annotations")
    parser.add_argument("--data-dir", default="/quobyte/millerlmgrp/annotated_data", help="Root annotated data directory")
    parser.add_argument("--subject", default="0901y2", help="Subject ID (e.g. 0901y2)")
    parser.add_argument("--output-csv", default="./benchmark_results/single_pair_demo/visual_only_intervals_0901y2.csv", help="Output CSV path")
    args = parser.parse_args()

    # Find the FIF file in any subfolder (e.g. CMPy2/)
    matches = glob.glob(os.path.join(args.data_dir, "**", f"*{args.subject}*-raw.fif"), recursive=True)
    if not matches:
        print(f"Error: Could not find raw FIF file for subject {args.subject} in {args.data_dir}")
        return

    fif_path = sorted(matches)[0]
    print(f"Loading annotations from: {fif_path}")
    raw = mne.io.read_raw_fif(fif_path, preload=False, verbose=False)

    if not raw.annotations:
        print("Error: No annotations found in file.")
        return

    v_intervals = []
    trial_count = 0
    for annot in raw.annotations:
        desc = annot["description"]
        if desc in ["A", "AV", "V"]:
            trial_count += 1
            if desc == "V":
                onset = float(annot["onset"])
                dur = float(annot["duration"]) if annot["duration"] > 0 else 1.5
                v_intervals.append({
                    "trial_idx": trial_count,
                    "condition": "V",
                    "onset_s": round(onset, 3),
                    "end_s": round(onset + dur, 3),
                    "duration_s": round(dur, 3),
                })

    print(f"\nFound {len(v_intervals)} Visual-Only (silent) trials out of {trial_count} total trials.")
    print("-" * 65)
    print(f"{'Trial #':<10} {'Condition':<12} {'Onset (s)':<15} {'End (s)':<15} {'Duration':<10}")
    print("-" * 65)
    for row in v_intervals[:15]:
        print(f"{row['trial_idx']:<10} {row['condition']:<12} {row['onset_s']:<15.3f} {row['end_s']:<15.3f} {row['duration_s']:<10.3f}")
    if len(v_intervals) > 15:
        print(f"... and {len(v_intervals) - 15} more trials.")
    print("-" * 65)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["trial_idx", "condition", "onset_s", "end_s", "duration_s"])
        writer.writeheader()
        writer.writerows(v_intervals)
    print(f"\nSaved all intervals to: {args.output_csv}\n")


if __name__ == "__main__":
    main()
