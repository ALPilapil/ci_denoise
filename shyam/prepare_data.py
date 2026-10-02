import argparse
import gc
import json
import os
import sys

import mne
import numpy as np
import zarr

from process_util import list_file_paths, permutation_divider


# ─── Default paths (edit for your system) ───────────────────────────────────
DATA_ROOT = '/mnt/data/PilapilData'
YEARS = [2, 3, 4]
DEFAULT_OUTPUT = '/mnt/data/PilapilData/processed_data/eeg_raw.zarr'

# The original preprocessing in isolate_noise.py applied these:
FILTER_LFREQ = 2.0      # Hz high-pass — removes slow drift, keeps neural + CI artifact
FILTER_HFREQ = None     # no low-pass — CRITICAL: preserve CI artifact content above
                        # typical EEG bands so the denoising pipeline has full spectrum

CROP_ALIGN = 65536      # zarr chunk size along time axis (aligns with 4s @ 16384 Hz)
# ────────────────────────────────────────────────────────────────────────────


def read_and_preprocess_raw(set_path):
    """
    Read one raw .set file with minimal preprocessing (matches isolate_noise.py
    minus the ICA step).

    Returns:
        raw: mne.io.Raw with preprocessing applied
        interpolated_channels: list[str] of channels that were bad and got interpolated
    """
    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)

    # Rename mastoids to match the convention used downstream
    rename_map = {}
    if 'LMas' in raw.ch_names:
        rename_map['LMas'] = 'M1'
    if 'RMas' in raw.ch_names:
        rename_map['RMas'] = 'M2'
    if rename_map:
        raw.rename_channels(rename_map)

    # Standard 10-20 montage (for topomap-based visualization later)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='warn', verbose=False)

    # Band-pass filter — KEEP high frequencies where CI artifact lives
    raw.filter(l_freq=FILTER_LFREQ, h_freq=FILTER_HFREQ, verbose=False)

    # Detect NaN/Inf/flat channels
    data = raw.get_data()
    bads = []
    for i, ch_name in enumerate(raw.ch_names):
        ch = data[i, :]
        if np.any(np.isnan(ch)) or np.any(np.isinf(ch)):
            bads.append(ch_name)
        elif np.std(ch) < 1e-10:
            bads.append(ch_name)

    if bads:
        raw.info['bads'] = bads
        # Interpolate so zarr shape is uniform across recordings.
        # Mark them in attrs so downstream code can choose to exclude these channels.
        raw.interpolate_bads(reset_bads=True, verbose=False)

    return raw, bads


def prepare_recording(set_path, zarr_group, participant_id, perm):
    """Read one raw .set file and store continuous data + metadata in zarr_group."""
    raw, interpolated = read_and_preprocess_raw(set_path)
    data = raw.get_data()  # (n_channels, n_samples) float64

    # Per-channel normalization stats
    norm_mean = data.mean(axis=-1, keepdims=True)
    norm_std = data.std(axis=-1, keepdims=True).clip(min=1e-6)

    # EEG data — chunked along time axis for efficient random-crop access
    zarr_group.create_array(
        'data',
        shape=data.shape,
        chunks=(data.shape[0], CROP_ALIGN),
        dtype='float64',
    )
    zarr_group['data'][:] = data

    # Annotations (preserve every event code verbatim)
    annot = raw.annotations
    onset_arr = np.array(annot.onset, dtype='float64')
    duration_arr = np.array(annot.duration, dtype='float64')

    zarr_group.create_array(
        'annotations_onset', shape=onset_arr.shape, dtype='float64',
    )
    if onset_arr.size > 0:
        zarr_group['annotations_onset'][:] = onset_arr

    zarr_group.create_array(
        'annotations_duration', shape=duration_arr.shape, dtype='float64',
    )
    if duration_arr.size > 0:
        zarr_group['annotations_duration'][:] = duration_arr

    # Normalization stats
    zarr_group.create_array('norm_mean', shape=norm_mean.shape, dtype='float64')
    zarr_group['norm_mean'][:] = norm_mean
    zarr_group.create_array('norm_std', shape=norm_std.shape, dtype='float64')
    zarr_group['norm_std'][:] = norm_std

    # Metadata attrs
    zarr_group.attrs['sfreq'] = float(raw.info['sfreq'])
    zarr_group.attrs['ch_names'] = list(raw.info['ch_names'])
    zarr_group.attrs['ch_types'] = [
        mne.channel_type(raw.info, i)
        for i in range(len(raw.info['ch_names']))
    ]
    zarr_group.attrs['participant_id'] = int(participant_id)
    zarr_group.attrs['perm'] = int(perm)
    zarr_group.attrs['source_file'] = str(set_path)
    zarr_group.attrs['annotations_description'] = json.dumps(
        list(annot.description)
    )
    zarr_group.attrs['interpolated_channels'] = list(interpolated)
    zarr_group.attrs['filter_lfreq'] = float(FILTER_LFREQ) if FILTER_LFREQ is not None else None
    zarr_group.attrs['filter_hfreq'] = float(FILTER_HFREQ) if FILTER_HFREQ is not None else None

    # Mark the group as fully written. Used by main() to detect partial writes
    # from a crashed prior run — groups without this attr are re-processed.
    zarr_group.attrs['complete'] = True

    n_annot = len(annot.onset)
    n_ch, n_samples = data.shape
    duration_s = n_samples / raw.info['sfreq']
    interp_str = f" interp={interpolated}" if interpolated else ""
    # Sanity flag for downstream zarr consumers
    ch_warn = " <-- UNUSUAL CHANNEL COUNT" if n_ch != 21 else ""
    print(f"  {os.path.basename(str(set_path))}: {n_ch} ch{ch_warn}, "
          f"{n_samples} samples ({duration_s:.1f}s), "
          f"{n_annot} annotations{interp_str}", flush=True)

    # Aggressively release memory between files. A 745s 21-ch recording is
    # ~2 GB as float64; without this, MNE internals can accumulate and OOM
    # on batch-processing systems with limited per-job memory.
    del data, raw
    gc.collect()


def print_annotation_summary(zarr_path):
    """Print a summary of all unique event descriptions across recordings."""
    store = zarr.open(str(zarr_path), mode='r')
    all_descriptions = {}

    for domain in ['ci', 'hearing']:
        if domain not in store:
            continue
        for rec_name in store[domain]:
            group = store[domain][rec_name]
            descriptions = json.loads(group.attrs['annotations_description'])
            for desc in descriptions:
                if desc not in all_descriptions:
                    all_descriptions[desc] = {'ci': 0, 'hearing': 0}
                all_descriptions[desc][domain] += 1

    print("\n" + "=" * 60)
    print("  ANNOTATION SUMMARY")
    print("=" * 60)
    print(f"  {'Description':<30s} {'CI count':>10s} {'Hearing count':>14s}")
    print("  " + "-" * 56)
    for desc in sorted(all_descriptions.keys()):
        counts = all_descriptions[desc]
        print(f"  {desc:<30s} {counts['ci']:>10d} {counts['hearing']:>14d}")
    print("=" * 60)


def collect_perm_sets(data_root, years):
    """
    Replicates isolate_noise.py's path-gathering logic, using process_util's
    list_file_paths + permutation_divider.

    - list_file_paths reads files at MarkerFixed/ flat (subject IDs are at the
      filename level), and at Logs/ flat (log files contain _v1/_v2/_v3 tags).
    - '/08' path prefix -> hearing controls; '/09' -> CI patients.
    - permutation_divider matches the first 4 digits of each .set filename to
      the corresponding log filename to determine its permutation.

    Returns (ci_paths, hearing_paths), each a list of 3 lists (one per perm).
    """
    ci_paths = [[], [], []]
    hearing_paths = [[], [], []]

    for year in years:
        raw_directory = f'{data_root}/CMPy{year}/MarkerFixed/'
        log_paths     = f'{data_root}/CMPy{year}/Logs/'
        raw_files = list_file_paths(raw_directory)
        log_files = list_file_paths(log_paths)

        hearing_year = [p for p in raw_files if '/08' in p and '.set' in p]
        ci_year      = [p for p in raw_files if '/09' in p and '.set' in p]
        print(f"Year {year}: {len(hearing_year)} hearing, {len(ci_year)} CI .set files found",
              flush=True)

        permed_hearing = permutation_divider(set_paths=hearing_year, log_paths=log_files)
        permed_ci      = permutation_divider(set_paths=ci_year,      log_paths=log_files)

        # Visibility: permutation_divider silently drops files whose subject ID
        # has no matching log entry. Surface that here so we know.
        n_hr_routed = sum(len(p) for p in permed_hearing)
        n_ci_routed = sum(len(p) for p in permed_ci)
        if n_hr_routed != len(hearing_year):
            print(f"  WARN: {len(hearing_year) - n_hr_routed} hearing files dropped "
                  f"(no matching log) -> routed {n_hr_routed}/{len(hearing_year)}",
                  flush=True)
        if n_ci_routed != len(ci_year):
            print(f"  WARN: {len(ci_year) - n_ci_routed} CI files dropped "
                  f"(no matching log) -> routed {n_ci_routed}/{len(ci_year)}",
                  flush=True)

        for i in range(3):
            ci_paths[i].extend(permed_ci[i])
            hearing_paths[i].extend(permed_hearing[i])

    return ci_paths, hearing_paths


def _process_domain(domain_group, perm_path_lists, name_prefix):
    """
    Process a list-of-3-perm-lists for either CI or hearing.

    Resume-safe: groups that already exist with attrs['complete'] == True are
    skipped. Groups that exist but are incomplete (e.g. from a crashed prior
    run) are deleted and re-processed. New groups are created fresh.

    Stdout is flushed after each file so sbatch/SLURM logs reflect progress
    in real time even if the job is SIGKILLed (e.g. OOM) — critical for
    diagnosing where a crash happened.
    """
    n_skipped = n_processed = n_failed = 0
    for perm_idx, perm_list in enumerate(perm_path_lists):
        perm_label = perm_idx + 1
        for idx, set_path in enumerate(perm_list):
            group_name = f'{name_prefix}{idx}_perm{perm_label}'

            # Resume logic — skip if already complete
            if group_name in domain_group:
                existing = domain_group[group_name]
                if existing.attrs.get('complete', False):
                    n_skipped += 1
                    continue
                # Partial from a crash — delete so we can recreate cleanly
                print(f"  {group_name} exists but incomplete (no 'complete' attr) "
                      f"— deleting and re-processing", flush=True)
                del domain_group[group_name]

            rec_group = domain_group.create_group(group_name)
            try:
                prepare_recording(
                    set_path=set_path,
                    zarr_group=rec_group,
                    participant_id=idx,
                    perm=perm_label,
                )
                n_processed += 1
            except Exception as e:
                print(f"  ERROR on {set_path}: {e}", flush=True)
                try:
                    del domain_group[group_name]
                except Exception:
                    pass
                n_failed += 1
                gc.collect()
                continue
    return n_processed, n_skipped, n_failed


def main():
    parser = argparse.ArgumentParser(
        description='Prepare raw (pre-ICA) EEG data for CI artifact removal pipeline.'
    )
    parser.add_argument('--output', default=DEFAULT_OUTPUT,
                        help=f'Output zarr path (default: {DEFAULT_OUTPUT})')
    parser.add_argument('--data-root', default=DATA_ROOT,
                        help=f'Root dir containing CMPy* folders (default: {DATA_ROOT})')
    parser.add_argument('--years', nargs='+', type=int, default=YEARS,
                        help=f'Years to include (default: {YEARS})')
    parser.add_argument('--overwrite', action='store_true',
                        help='Delete any existing zarr at --output and start fresh '
                             '(default: append / resume)')
    args = parser.parse_args()

    # Force line-buffered stdout so sbatch logs see every print even if the
    # job is later SIGKILLed. Needed because stdout redirected to a file is
    # fully-buffered by default, which is why the user's previous sbatch log
    # stopped after the prolog.
    try:
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
    except Exception:
        pass

    ci_paths, hearing_paths = collect_perm_sets(args.data_root, args.years)
    n_ci = sum(len(p) for p in ci_paths)
    n_hr = sum(len(p) for p in hearing_paths)
    print(f"\nTotal: {n_ci} CI + {n_hr} hearing .set files to process", flush=True)

    if n_ci == 0 and n_hr == 0:
        print("No .set files found. Check --data-root and --years paths.", flush=True)
        return

    # Open zarr: overwrite if requested, otherwise append/resume.
    if args.overwrite and os.path.exists(args.output):
        import shutil
        print(f"Overwriting existing {args.output}", flush=True)
        shutil.rmtree(args.output)
    mode = 'a'   # create if missing, modify if exists — safe for resume
    root = zarr.open_group(args.output, mode=mode)
    ci_group = root.require_group('ci')
    hearing_group = root.require_group('hearing')

    print(f"\n=== Processing CI recordings ===", flush=True)
    n_ci_proc, n_ci_skip, n_ci_fail = _process_domain(
        ci_group, ci_paths, name_prefix='noise',
    )
    print(f"CI summary: {n_ci_proc} processed, {n_ci_skip} skipped (already complete), "
          f"{n_ci_fail} failed", flush=True)

    print(f"\n=== Processing hearing recordings ===", flush=True)
    n_hr_proc, n_hr_skip, n_hr_fail = _process_domain(
        hearing_group, hearing_paths, name_prefix='hearing',
    )
    print(f"Hearing summary: {n_hr_proc} processed, {n_hr_skip} skipped (already complete), "
          f"{n_hr_fail} failed", flush=True)

    print(f"\nZarr store saved to: {args.output}", flush=True)
    print_annotation_summary(args.output)


if __name__ == '__main__':
    main()
