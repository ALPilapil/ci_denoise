"""
extract_trial_onsets.py

Determines the onset time and condition (AV / A-only / V-only) for every trial
in the CMPy2 SSVEP/speech dataset and saves the results as MNE Epochs .fif files.

DATA STRUCTURE
--------------
Each subject's Marker Fixed .set file contains three behaviorally meaningful
trigger codes:

  Code 25  – SSVEP ring/flicker onset, fires once per AudioVisual (AV) trial,
              every 3 s, 240 events per recording (40 per block × 6 blocks).
  Code 10  – Audio block onset, positive polarity; fires 6 times (once per block).
  Code 11  – Audio block onset, negative polarity; fires 6 times (once per block,
              marks the second polarity half within each block).

There is NO individual EEG marker for Audio-only (A) or Visual-only (V) trials.
Their onsets are inferred from the stimulus matrix and the trial interleaving:

  AV onset     = time of code-25 event
  Non-AV onset = code-25 time − TRIAL_DURATION_S  (non-AV precedes its paired AV)

where TRIAL_DURATION_S = 1.5 s (= half the 3 s inter-code-25 interval).

Within each block the stim-matrix ordering is:
  [non-AV, AV, non-AV, AV, ...]
so the non-AV trial immediately precedes each AV trial.

STIM MATRIX
-----------
CMPv2_Stim_Matrix_SentenceSet{1,2,3}.mat  →  filled_stim_matrix (480 × 5)

  col 0 : trial index within block (1–80)
  col 1 : block number (1–6)
  col 2 : condition  1 = AV  |  2 = Audio-only  |  3 = Visual-only
  col 3 : stimulus ID (0 for V-only trials)
  col 4 : polarity half (1 = first half, 2 = second half; 0 for V-only)

The correct SentenceSet is chosen from the subject's log filename:
  *_v1.log → SentenceSet1,  *_v2.log → SentenceSet2,  *_v3.log → SentenceSet3

OUTPUT
------
Two files per subject written to the output directory:

  trial_onsets_{subject}-raw.fif
    Raw MNE file with original annotations preserved plus three new labels:
      "A"  – Audio-only trial onset (inferred)
      "AV" – AudioVisual trial onset (from code 25)
      "V"  – Visual-only trial onset (inferred)
    raw.info["description"] : dataset / subject string
    raw.info["subject_info"]["his_id"] : subject ID
    raw.info["proj_name"] : "CMPy2"

  trial_onsets_{subject}.csv
    One row per trial with columns:
      subject, block, trial_idx, condition, stim_id, polarity,
      onset_s, year, permutation

USAGE
-----
  # Single subject (legacy NAS paths):
  python extract_trial_onsets.py 0801y2

  # All subjects (legacy NAS paths):
  python extract_trial_onsets.py

  # Custom output directory:
  python extract_trial_onsets.py 0801y2 --output-dir /path/to/output

  # Single subject from a local study directory:
  python extract_trial_onsets.py 0801y2 --study-dir CMPy2

  # All subjects in a local study directory:
  python extract_trial_onsets.py --study-dir CMPy3

  # All subjects in a study directory with a custom output directory:
  python extract_trial_onsets.py --study-dir CMPy4 --output-dir tagged_files/CMPy4

  # Process each study year into its own output subfolder:
  python extract_trial_onsets.py --study-dir CMPy2 --output-dir tagged_files/CMPy2
  python extract_trial_onsets.py --study-dir CMPy3 --output-dir tagged_files/CMPy3
  python extract_trial_onsets.py --study-dir CMPy4 --output-dir tagged_files/CMPy4

Output files are named trial_onsets_{subject}-raw.fif and contain the original
EEG with "A", "AV", and "V" annotations appended to mark inferred trial onsets.
"""

import os
import re
import glob
import argparse
import warnings
import numpy as np
import pandas as pd
import scipy.io as sio
import mne

# ── paths ─────────────────────────────────────────────────────────────────────

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "tagged_files")

# Resolved at runtime from --study-dir (or legacy NAS defaults)
_NAS_BASE = "/mnt/miller-nas-general/Backup Data/CMP/CMPy2"
MARKER_FIXED_DIR = os.path.join(_NAS_BASE, "Raw Data", "Marker Fixed")
LOGS_DIR = os.path.join(_NAS_BASE, "Logs")
STIM_MATRIX_TEMPLATE = os.path.join(_NAS_BASE, "CMPv2_Stim_Matrix_SentenceSet{n}.mat")

# ── constants ─────────────────────────────────────────────────────────────────

TRIAL_DURATION_S = 1.5      # non-AV trial occurs this many seconds BEFORE its paired code 25
BLOCK_GAP_THRESH_S = 3.5    # gaps > this in the code-25 sequence mark a block boundary
EXPECTED_C25_PER_BLOCK = 40

COND_LABELS = {1: "AV", 2: "A", 3: "V"}

# ── helpers ───────────────────────────────────────────────────────────────────

def load_eeg_events(set_path):
    """Return (events, srate) from an EEGLAB .set file."""
    data = sio.loadmat(set_path, squeeze_me=True, struct_as_record=False)
    eeg = data["EEG"]
    return eeg.event, float(eeg.srate)


def get_event_times(events, srate, code):
    """Sorted array of onset times (s) for events with trigger code == code."""
    times = [float(e.latency) / srate for e in events if str(e.type) == str(code)]
    return np.sort(times)


def detect_block_boundaries(c25_times, gap_thresh=BLOCK_GAP_THRESH_S):
    """
    Split the code-25 sequence into recording blocks by detecting inter-event
    gaps larger than gap_thresh seconds (~4 s gaps separate the 6 blocks).

    Returns a list of 1-D arrays, one per block.
    """
    if len(c25_times) == 0:
        return []
    diffs = np.diff(c25_times)
    boundary_idx = np.where(diffs > gap_thresh)[0] + 1
    return np.split(c25_times, boundary_idx)


def assign_polarity_halves(block_c25, c10_times, c11_times):
    """
    Split one block's code-25 events into two polarity halves.

    The second marker (code 10 or 11) that falls inside the block window
    marks the start of the second polarity half.

    Returns (half1_times, half2_times).
    """
    t_start = block_c25[0] - 10.0
    t_end   = block_c25[-1] + 10.0

    markers = sorted(
        [(t, 10) for t in c10_times if t_start <= t <= t_end] +
        [(t, 11) for t in c11_times if t_start <= t <= t_end],
        key=lambda x: x[0],
    )

    # Always split at the midpoint (20 events per half).
    # Code-25 fires so close to the code-10/11 boundary that a time-based split
    # assigns one event to the wrong half; a count-based split is exact.
    mid = len(block_c25) // 2
    half1, half2 = block_c25[:mid], block_c25[mid:]

    if len(markers) < 1:
        return half1, half2

    # Determine polarity assignment: the half whose start is preceded by code-10
    # is polarity 1; the half preceded by code-11 is polarity 2.
    # If the first marker is code-11, the halves are reversed.
    first_code = markers[0][1]
    if first_code == 11:
        # Block begins with a negative-polarity audio segment
        half1, half2 = half2, half1

    return half1, half2


def get_sentence_set_version(log_prefix, logs_dir=LOGS_DIR, subject_id=None):
    """
    Infer SentenceSet version (1/2/3) from the subject's log filename.
    log_prefix is the numeric part of the subject ID (e.g. "0801").
    subject_id is the full ID (e.g. "0801y4") used as a fallback prefix.
    """
    # Try full subject ID prefix first (handles "0801y4-*.log"), then numeric only
    prefixes = [subject_id, log_prefix] if subject_id and subject_id != log_prefix else [log_prefix]
    matches = []
    for pfx in prefixes:
        matches = glob.glob(os.path.join(logs_dir, f"{pfx}-*.log"))
        if matches:
            break
    if not matches:
        raise FileNotFoundError(
            f"No log file for subject '{subject_id or log_prefix}' in {logs_dir}"
        )
    log_name = os.path.basename(sorted(matches)[0])
    m = re.search(r"_v(\d)", log_name)
    if not m:
        raise ValueError(f"Cannot extract version tag from log filename: {log_name}")
    return int(m.group(1))


def load_stim_matrix(version, template=STIM_MATRIX_TEMPLATE):
    """Return the stimulus matrix as a pandas DataFrame."""
    path = template.format(n=version)
    raw = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    mat = raw["filled_stim_matrix"].astype(int)
    df = pd.DataFrame(mat, columns=["trial_idx", "block", "condition", "stim_id", "polarity"])
    return df


# ── main extraction ───────────────────────────────────────────────────────────

def extract_trial_onsets(subject_id, output_dir=None, study_dir=None):
    """
    Add A/AV/V event markers to raw EEG and save as a .fif + companion CSV.

    Parameters
    ----------
    subject_id : str
        Filename stem of the .set file (e.g. "0801y2").
    output_dir : str, optional
        Where to write output files. Defaults to tagged_files/ next to this script.
    study_dir : str, optional
        Directory containing .set, .log, and stim-matrix .mat files.
        When provided, overrides the default NAS paths.

    Returns
    -------
    mne.io.Raw
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    # Resolve source directories
    set_dir   = study_dir if study_dir else MARKER_FIXED_DIR
    logs_dir  = study_dir if study_dir else LOGS_DIR
    mat_tmpl  = (
        os.path.join(study_dir, "CMPv2_Stim_Matrix_SentenceSet{n}.mat")
        if study_dir else STIM_MATRIX_TEMPLATE
    )

    # Numeric prefix for log lookup (e.g. "0801y2" → "0801")
    log_prefix = re.match(r"(\d+)", subject_id).group(1)

    # ── 1. EEG events ──────────────────────────────────────────────────────────
    set_path = os.path.join(set_dir, f"{subject_id}.set")
    if not os.path.isfile(set_path):
        raise FileNotFoundError(f".set file not found: {set_path}")

    events, srate = load_eeg_events(set_path)
    c25 = get_event_times(events, srate, 25)
    c10 = get_event_times(events, srate, 10)
    c11 = get_event_times(events, srate, 11)

    print(f"[{subject_id}]  code-25: {len(c25)}  code-10: {len(c10)}  code-11: {len(c11)}")

    # ── 2. Block detection ─────────────────────────────────────────────────────
    blocks_c25 = detect_block_boundaries(c25)
    n_blocks = len(blocks_c25)
    print(f"[{subject_id}]  {n_blocks} block(s) detected")
    if n_blocks != 6:
        warnings.warn(f"Expected 6 blocks; found {n_blocks}.")

    for i, blk in enumerate(blocks_c25):
        if len(blk) != EXPECTED_C25_PER_BLOCK:
            warnings.warn(
                f"  Block {i+1}: expected {EXPECTED_C25_PER_BLOCK} code-25 events, "
                f"got {len(blk)}."
            )

    # ── 3. Stimulus matrix ─────────────────────────────────────────────────────
    version = get_sentence_set_version(log_prefix, logs_dir=logs_dir, subject_id=subject_id)
    stim_df = load_stim_matrix(version, template=mat_tmpl)
    print(f"[{subject_id}]  SentenceSet{version} ({len(stim_df)} rows)")

    # ── 4. Build trial table ───────────────────────────────────────────────────
    rows = []

    for block_i, block_c25 in enumerate(blocks_c25):
        block_num = block_i + 1      # 1-indexed

        half1_c25, half2_c25 = assign_polarity_halves(block_c25, c10, c11)

        # Stim-matrix rows for this block, sorted by trial_idx.
        # V-only rows have polarity==0, so we cannot use the polarity column to
        # split them into halves. Instead, use trial_idx position: the first
        # half of rows (trial_idx 1..N/2) = polarity half 1.
        block_stim = (
            stim_df[stim_df["block"] == block_num]
            .sort_values("trial_idx")
            .reset_index(drop=True)
        )
        n_stim = len(block_stim)
        half_stim_size = n_stim // 2   # = 40 per block

        stim_half1 = block_stim.iloc[:half_stim_size].reset_index(drop=True)
        stim_half2 = block_stim.iloc[half_stim_size:].reset_index(drop=True)

        # Pair EEG halves with stim halves (assignment already handled in
        # assign_polarity_halves which swaps if first code is 11)
        halves = [(half1_c25, stim_half1, 1), (half2_c25, stim_half2, 2)]

        for half_c25, stim_half, polarity in halves:
            if len(half_c25) == 0:
                continue

            av_rows = (
                stim_half[stim_half["condition"] == 1]
                .sort_values("trial_idx")
                .reset_index(drop=True)
            )
            non_av_rows = (
                stim_half[stim_half["condition"] != 1]
                .sort_values("trial_idx")
                .reset_index(drop=True)
            )

            n_pairs = min(len(av_rows), len(half_c25))
            if len(av_rows) != len(half_c25):
                warnings.warn(
                    f"  Block {block_num} polarity {polarity}: "
                    f"{len(half_c25)} code-25 events vs {len(av_rows)} AV rows."
                )

            for pair_i in range(n_pairs):
                t_av = half_c25[pair_i]
                av_row = av_rows.iloc[pair_i]

                # AV trial (directly from code 25)
                rows.append({
                    "subject":   subject_id,
                    "block":     block_num,
                    "trial_idx": int(av_row["trial_idx"]),
                    "condition": "AV",
                    "stim_id":   int(av_row["stim_id"]),
                    "polarity":  polarity,
                    "onset_s":   round(t_av, 6),
                })

                # Non-AV trial that precedes this AV trial in the sequence
                if pair_i < len(non_av_rows):
                    non_av_row = non_av_rows.iloc[pair_i]
                    rows.append({
                        "subject":   subject_id,
                        "block":     block_num,
                        "trial_idx": int(non_av_row["trial_idx"]),
                        "condition": COND_LABELS[int(non_av_row["condition"])],
                        "stim_id":   int(non_av_row["stim_id"]),
                        "polarity":  polarity,
                        "onset_s":   round(t_av - TRIAL_DURATION_S, 6),
                    })

    result_df = (
        pd.DataFrame(rows)
        .sort_values("onset_s")
        .reset_index(drop=True)
    )

    # ── 5. Year from subject ID (e.g. "0801y2" → 2) ───────────────────────────
    year_match = re.search(r"y(\d+)", subject_id)
    year = int(year_match.group(1)) if year_match else None

    # ── 6. Load raw EEG via MNE ────────────────────────────────────────────────
    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)

    raw.info["subject_info"] = {"his_id": subject_id}
    raw.info["description"] = (
        f"CMPy2 SSVEP/speech dataset | subject {subject_id} | "
        f"trial codes: A=191 AV=192 V=193"
    )
    raw.info["proj_name"] = "CMPy2"
    raw.info["experimenter"] = "Miller Lab"

    # ── 7. Add A / AV / V annotations to the raw ─────────────────────────────
    new_annotations = mne.Annotations(
        onset=result_df["onset_s"].values,
        duration=np.zeros(len(result_df)),
        description=result_df["condition"].values,
        orig_time=raw.annotations.orig_time,
    )
    raw.set_annotations(raw.annotations + new_annotations)

    # ── 8. Save raw with new markers ──────────────────────────────────────────
    out_path = os.path.join(output_dir, f"trial_onsets_{subject_id}-raw.fif")
    raw.save(out_path, overwrite=True)

    # ── 9. Save companion CSV with full trial metadata ─────────────────────────
    csv_df = result_df.copy()
    csv_df["year"] = year
    csv_df["permutation"] = version
    csv_path = os.path.join(output_dir, f"trial_onsets_{subject_id}.csv")
    csv_df.to_csv(csv_path, index=False)

    print(f"[{subject_id}]  {len(result_df)} markers added → {out_path}")
    print(f"[{subject_id}]  metadata saved → {csv_path}")

    return raw


# ── batch mode ────────────────────────────────────────────────────────────────

def run_all_subjects(output_dir=None, study_dir=None):
    """Process every subject .set file in study_dir (or the default Marker Fixed dir)."""
    search_dir = study_dir if study_dir else MARKER_FIXED_DIR
    set_files = sorted(glob.glob(os.path.join(search_dir, "*.set")))
    for set_file in set_files:
        sid = os.path.splitext(os.path.basename(set_file))[0]
        if "_Andrew" in sid:
            continue
        try:
            extract_trial_onsets(sid, output_dir=output_dir, study_dir=study_dir)
        except Exception as exc:
            warnings.warn(f"Skipping {sid}: {exc}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract CMPy2 trial onsets (AV / A / V) from Marker Fixed .set files."
    )
    parser.add_argument(
        "subject", nargs="?",
        help="Subject ID (e.g. 0801y2). Omit to process all subjects."
    )
    parser.add_argument(
        "--study-dir", default=None,
        help=(
            "Directory containing .set, .log, and stim-matrix .mat files "
            "(e.g. CMPy2/, CMPy3/, CMPy4/). "
            "Overrides the default NAS paths."
        ),
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory for output .fif files (default: tagged_files/ next to this script)."
    )
    args = parser.parse_args()

    if args.subject:
        extract_trial_onsets(args.subject, output_dir=args.output_dir, study_dir=args.study_dir)
    else:
        run_all_subjects(output_dir=args.output_dir, study_dir=args.study_dir)
