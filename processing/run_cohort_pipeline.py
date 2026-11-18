#!/usr/bin/env python3
"""
run_cohort_pipeline.py
Slurm Job Array Worker for CMPy2 Cohort Processing.
Executes the validated Two-Pass SOBI + Polarity-Alternating pipeline on subject K.
"""

import os
import sys
import glob
import argparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if os.path.join(REPO_ROOT, "processing") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "processing"))

from run_sobi_polarity_pipeline import run_sobi_polarity_pipeline

DATA_DIR_DEFAULT = "/quobyte/millerlmgrp/annotated_data/CMPy2"

def get_valid_subjects(data_dir=DATA_DIR_DEFAULT, cohort_type="ci", year_tag="y2"):
    prefix = f"09*{year_tag}" if cohort_type == "ci" else f"08*{year_tag}"
    fif_files = sorted(glob.glob(os.path.join(data_dir, f"trial_onsets_{prefix}-raw.fif")))
    valid = []
    for f in fif_files:
        sub = os.path.basename(f).replace("trial_onsets_", "").replace("-raw.fif", "")
        csv_file = f.replace("-raw.fif", ".csv")
        if os.path.exists(csv_file):
            valid.append((sub, f, csv_file))
    return valid

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-type", choices=["ci", "nh"], default="ci", help="Process CI or NH subjects")
    parser.add_argument("--data-dir", default=DATA_DIR_DEFAULT, help="Input raw dataset directory")
    parser.add_argument("--year-tag", default="y2", help="Year tag string, e.g. y2, y3, y4")
    parser.add_argument("--task-id", type=int, default=0, help="Slurm array task ID (0-indexed)")
    parser.add_argument("--output-dir", default="./benchmark_results/sobi_polarity_cohort", help="Output directory")
    parser.add_argument("--list-only", action="store_true", help="Print discovered subjects and exit")
    args = parser.parse_args()
    
    subjects = get_valid_subjects(args.data_dir, args.cohort_type, args.year_tag)
    if args.list_only:
        print(f"Found {len(subjects)} valid {args.cohort_type.upper()} subjects with year tag '{args.year_tag}' in {args.data_dir}:")
        for i, (s, f, c) in enumerate(subjects):
            print(f"  [{i:02d}] {s}")
        sys.exit(0)
    if args.task_id < 0 or args.task_id >= len(subjects):
        print(f"Error: task-id {args.task_id} out of bounds (total subjects: {len(subjects)})")
        sys.exit(1)
        
    sub_id, raw_fif, csv_file = subjects[args.task_id]
    print(f"[Cohort Worker] Task ID: {args.task_id}/{len(subjects)-1} | Subject: {sub_id}")
    print(f"  FIF: {raw_fif}")
    print(f"  CSV: {csv_file}")
    
    run_sobi_polarity_pipeline(
        raw_fif=raw_fif,
        csv_path=csv_file,
        output_dir=args.output_dir,
        subject_id=sub_id,
        highpass=1.0,
        lowpass=45.0
    )
