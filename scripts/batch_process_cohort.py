#!/usr/bin/env python3
"""
scripts/batch_process_cohort.py
Turnkey Batch Ingestion and Denoising Pipeline for New CI-EEG Cohorts.

Scans an input directory for continuous .fif recordings, executes automated
hardware saturation auditing, runs Stage 1 Two-Pass SOBI (and optionally Stage 2
Foundation N2N), and aggregates all metrics into a unified summary CSV.
"""

import os
import sys
import glob
import argparse
import csv
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import cidenoise

def batch_process(
    input_dir: str,
    output_dir: str,
    pattern: str = "*.fif",
    method: str = "sobi-2pass",
    checkpoint: str = None,
    highpass: float = 1.0,
    lowpass: float = 45.0,
    target_fs: float = 256.0,
    summary_csv: str = None
):
    os.makedirs(output_dir, exist_ok=True)
    fif_files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not fif_files:
        print(f"[batch_process] No files matching '{pattern}' found in {input_dir}")
        return

    print(f"[batch_process] Discovered {len(fif_files)} candidate recordings in {input_dir}")
    if summary_csv is None:
        summary_csv = os.path.join(output_dir, "batch_processing_summary.csv")

    fieldnames = [
        "recording", "status", "duration_s", "n_channels", "sfreq_hz",
        "hardware_highpass_hz", "hardware_lowpass_hz", "max_channel_variance_v2",
        "midline_power_uv2", "sobi_passes", "mastoid_attenuation_db",
        "midline_retention_pct", "output_file", "runtime_s"
    ]

    records = []
    for idx, fpath in enumerate(fif_files, 1):
        rec_name = os.path.basename(fpath).replace(".fif", "")
        print(f"\n[{idx}/{len(fif_files)}] Processing: {rec_name}")
        t0 = time.time()
        
        # 1. Audit
        try:
            audit_report = cidenoise.audit(fpath)
        except Exception as e:
            print(f"  [ERROR] Audit failed: {e}")
            records.append({
                "recording": rec_name,
                "status": f"FAILED_AUDIT: {e}",
                "runtime_s": round(time.time() - t0, 2)
            })
            continue

        if audit_report["status"] == "EXCLUDED_SATURATION":
            print(f"  [QA INTERCEPT] Hardware saturation detected: {audit_report['max_channel_variance_v2']:.2e} V^2. Excluding.")
            records.append({
                "recording": rec_name,
                "status": "EXCLUDED_SATURATION",
                "duration_s": round(audit_report["duration_s"], 2),
                "n_channels": audit_report["n_channels"],
                "sfreq_hz": audit_report["sfreq_hz"],
                "hardware_highpass_hz": audit_report["hardware_filter"]["highpass_hz"],
                "hardware_lowpass_hz": audit_report["hardware_filter"]["lowpass_hz"],
                "max_channel_variance_v2": audit_report["max_channel_variance_v2"],
                "midline_power_uv2": round(audit_report["midline_cortical_power_uv2"], 2),
                "runtime_s": round(time.time() - t0, 2)
            })
            continue

        # 2. Denoise
        out_fpath = os.path.join(output_dir, f"{rec_name}_clean.fif")
        try:
            raw_clean, summary = cidenoise.clean(
                input_fif=fpath,
                output_fif=out_fpath,
                method=method,
                checkpoint=checkpoint,
                target_fs=target_fs,
                highpass=highpass,
                lowpass=lowpass
            )
            runtime = round(time.time() - t0, 2)
            attenuation = summary.get("artifact_band_attenuation_db", 0.0)
            retention = summary.get("midline_cortical_power_retention_pct", 0.0)
            passes = summary.get("total_passes_performed", 0)

            print(f"  [SUCCESS] Denoised in {runtime}s | Attenuation: {attenuation:.2f} dB | Midline Retention: {retention:.1f}%")
            records.append({
                "recording": rec_name,
                "status": "COMPLETED",
                "duration_s": round(audit_report["duration_s"], 2),
                "n_channels": audit_report["n_channels"],
                "sfreq_hz": audit_report["sfreq_hz"],
                "hardware_highpass_hz": audit_report["hardware_filter"]["highpass_hz"],
                "hardware_lowpass_hz": audit_report["hardware_filter"]["lowpass_hz"],
                "max_channel_variance_v2": audit_report["max_channel_variance_v2"],
                "midline_power_uv2": round(audit_report["midline_cortical_power_uv2"], 2),
                "sobi_passes": passes,
                "mastoid_attenuation_db": round(attenuation, 2),
                "midline_retention_pct": round(retention, 2),
                "output_file": out_fpath,
                "runtime_s": runtime
            })
        except Exception as e:
            print(f"  [ERROR] Denoising failed: {e}")
            records.append({
                "recording": rec_name,
                "status": f"FAILED_CLEAN: {e}",
                "runtime_s": round(time.time() - t0, 2)
            })

    # Save CSV
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    print(f"\n[batch_process] Finished batch run! Summary written to: {summary_csv}")

def main():
    parser = argparse.ArgumentParser(description="Turnkey Batch Ingestion for Continuous CI-EEG Cohorts")
    parser.add_argument("--input-dir", "-i", required=True, help="Input directory containing .fif files")
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory for cleaned files")
    parser.add_argument("--pattern", default="*.fif", help="Glob pattern for recordings (default: *.fif)")
    parser.add_argument("--method", choices=["sobi-2pass", "sobi-n2n"], default="sobi-2pass", help="Cleaning method")
    parser.add_argument("--checkpoint", default=None, help="Path to PyTorch model checkpoint for N2N")
    parser.add_argument("--highpass", type=float, default=1.0, help="High-pass filter cutoff (Hz)")
    parser.add_argument("--lowpass", type=float, default=45.0, help="Low-pass filter cutoff (Hz)")
    parser.add_argument("--target-fs", type=float, default=256.0, help="Target sampling rate (Hz)")
    parser.add_argument("--summary-csv", default=None, help="Custom output CSV path")
    args = parser.parse_args()

    batch_process(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        pattern=args.pattern,
        method=args.method,
        checkpoint=args.checkpoint,
        highpass=args.highpass,
        lowpass=args.lowpass,
        target_fs=args.target_fs,
        summary_csv=args.summary_csv
    )

if __name__ == "__main__":
    main()
