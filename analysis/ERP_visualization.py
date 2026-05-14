import argparse
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne


def parse_subject_year(fif_path: str) -> tuple[str, str | None]:
    m = re.search(r"CMPy(\d+)[/\\].*?([^/_\\]+)-raw\.fif", fif_path)
    if m:
        return m.group(2), f"CMPy{m.group(1)}"
    stem = re.sub(r"-raw\.fif$", "", os.path.basename(fif_path))
    return stem, None


def collect_fif_paths(
    files: list[str] | None,
    dir_path: str | None,
) -> list[str]:
    paths = list(files) if files else []
    if dir_path:
        paths.extend(sorted(glob.glob(os.path.join(dir_path, "*-raw.fif"))))
    seen: set[str] = set()
    deduped: list[str] = []
    for p in paths:
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            deduped.append(p)
    if not deduped:
        sys.exit("No .fif files found. Provide --files or --dir.")
    return deduped


def epoch_and_average(
    fif_path: str,
    event_codes: list[int],
    tmin: float,
    tmax: float,
    baseline: tuple[float, float],
    picks: list[str] | None,
) -> dict[int, mne.Evoked]:
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
    present = set(raw.annotations.description)
    event_id = {
        str(c): i + 1
        for i, c in enumerate(sorted(event_codes))
        if str(c) in present
    }

    if not event_id:
        print(f"  WARNING: none of {event_codes} found as annotations in {fif_path}")
        del raw
        return {}

    events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose=False)
    epochs = mne.Epochs(
        raw, events,
        event_id=event_id,
        tmin=tmin, tmax=tmax,
        baseline=baseline,
        picks=picks,
        preload=True,
        verbose=False,
    )

    evokeds: dict[int, mne.Evoked] = {}
    for code_str in event_id:
        subset = epochs[code_str]
        if len(subset) == 0:
            print(f"    code {code_str}: 0 epochs, skipping")
            continue
        evokeds[int(code_str)] = subset.average()
        print(f"    code {code_str}: {len(subset)} epochs averaged")

    del raw, epochs
    return evokeds


def save_evoked_plot(evoked: mne.Evoked, title: str, save_path: str) -> None:
    fig = evoked.plot(show=False)
    fig.suptitle(title, fontsize=11, y=1.02)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {save_path}")


def process_file(
    fif_path: str,
    event_codes: list[int],
    tmin: float,
    tmax: float,
    baseline: tuple[float, float],
    picks: list[str] | None,
    output_dir: str,
) -> None:
    subject_id, year_label = parse_subject_year(fif_path)
    year_label = year_label or "CMPyUnknown"
    print(f"\n[{subject_id} | {year_label}]  {fif_path}")

    os.makedirs(output_dir, exist_ok=True)
    evokeds = epoch_and_average(fif_path, event_codes, tmin, tmax, baseline, picks)

    for code, evoked in evokeds.items():
        title = f"ERP: code {code} | Subject: {subject_id} | Year: {year_label}"
        filename = f"erp_{subject_id}_{year_label}_code{code}.png"
        save_evoked_plot(evoked, title, os.path.join(output_dir, filename))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot per-subject ERPs from annotated CI EEG .fif files."
    )
    parser.add_argument(
        "--files", nargs="+", default=None,
        help="Explicit .fif file paths.",
    )
    parser.add_argument(
        "--dir", default=None,
        help="Directory; all *-raw.fif files inside are processed.",
    )
    parser.add_argument(
        "--event-codes", nargs="+", type=int, required=True,
        help="Integer trigger codes to epoch around (e.g. 25 10 11).",
    )
    parser.add_argument(
        "--tmin", type=float, default=-0.2,
        help="Epoch start in seconds relative to onset (default: -0.2).",
    )
    parser.add_argument(
        "--tmax", type=float, default=1.5,
        help="Epoch end in seconds relative to onset (default: 1.5).",
    )
    parser.add_argument(
        "--baseline", nargs=2, type=float, default=[-0.2, 0],
        metavar=("BMIN", "BMAX"),
        help="Baseline window in seconds (default: -0.2 0).",
    )
    parser.add_argument(
        "--output-dir", default="./erp_plots",
        help="Directory for PNG outputs (default: ./erp_plots).",
    )
    parser.add_argument(
        "--picks", nargs="+", default=None,
        help="Channel names to include (default: all channels in file).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fif_paths = collect_fif_paths(args.files, args.dir)
    baseline = tuple(args.baseline)

    print(f"Processing {len(fif_paths)} file(s), event codes: {args.event_codes}")
    print(f"Epoch window: [{args.tmin}, {args.tmax}] s, baseline: {baseline}")
    print(f"Output directory: {args.output_dir}\n")

    for fif_path in fif_paths:
        process_file(
            fif_path,
            args.event_codes,
            args.tmin,
            args.tmax,
            baseline,
            args.picks,
            args.output_dir,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
