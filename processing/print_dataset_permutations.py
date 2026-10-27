"""
print_dataset_permutations.py

Utility script to list all subjects and their assigned permutations (1, 2, or 3):
Can check:
1. The Zarr store (/quobyte/millerlmgrp/processed_data/raw_epoched_data.zarr)
2. Companion CSV metadata files (trial_onsets_*.csv)
3. Raw study log files (*_v1.log, *_v2.log, *_v3.log)
"""

import argparse
import glob
import os
import re
import sys
from collections import defaultdict


def check_logs(logs_dir: str):
    print(f"\nScanning Logs directory: {logs_dir}")
    log_files = sorted(glob.glob(os.path.join(logs_dir, "*.log")))
    if not log_files:
        print("  No .log files found.")
        return

    perm_map = defaultdict(lambda: defaultdict(list))
    for p in log_files:
        fname = os.path.basename(p)
        sub = fname[:6]  # e.g. 0901y2 or 0801y2
        group = "CI" if sub.startswith("09") else ("Hearing" if sub.startswith("08") else "Other")
        m = re.search(r"_v(\d)", fname)
        if m:
            version = int(m.group(1))
            perm_map[version][group].append(sub)

    for perm in sorted(perm_map.keys()):
        print(f"\n--- PERMUTATION {perm} ---")
        for grp in ["CI", "Hearing"]:
            subs = sorted(list(set(perm_map[perm][grp])))
            print(f"  {grp} Subjects (N={len(subs)}): {', '.join(subs[:10])}{' ...' if len(subs) > 10 else ''}")


def check_zarr(zarr_path: str):
    print(f"\nScanning Zarr dataset: {zarr_path}")
    if not os.path.exists(zarr_path):
        print(f"  Zarr path not found: {zarr_path}")
        return
    try:
        import zarr
        try:
            root = zarr.open(zarr_path, mode="r")
        except Exception:
            root = zarr.open_group(zarr_path, mode="r")

        for grp_name in ["hearing_trial_data", "ci_trial_data"]:
            if grp_name not in root:
                continue
            grp = root[grp_name]
            perms = grp["perm"][:] if "perm" in grp else []
            participants = grp["participant"][:] if "participant" in grp else []

            p_to_perm = defaultdict(set)
            for p, perm in zip(participants, perms):
                p_to_perm[int(perm)].add(int(p))

            print(f"\nGroup: [{grp_name}]")
            for perm in sorted(p_to_perm.keys()):
                p_list = sorted(list(p_to_perm[perm]))
                print(f"  Permutation {perm}: {len(p_list)} participants -> IDs: {p_list}")
    except Exception as e:
        print(f"  Error reading Zarr: {e}")


def main():
    parser = argparse.ArgumentParser(description="Inspect Subject Permutation Mappings")
    parser.add_argument("--zarr-path", default="/quobyte/millerlmgrp/processed_data/raw_epoched_data.zarr", help="Path to Zarr dataset")
    parser.add_argument("--logs-dir", default="/quobyte/millerlmgrp/annotated_data", help="Path to logs or annotated dir")
    args = parser.parse_args()

    check_zarr(args.zarr_path)
    check_logs(args.logs_dir)


if __name__ == "__main__":
    main()
