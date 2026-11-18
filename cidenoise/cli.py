#!/usr/bin/env python3
"""
cidenoise.cli
Unified Command-Line Interface for the cidenoise package.
"""

import sys
import os
import argparse
import json

from cidenoise import clean, audit

def clean_command(args):
    print(f"[cidenoise clean] Input: {args.input}")
    print(f"[cidenoise clean] Method: {args.method}")
    print(f"[cidenoise clean] Output: {args.output}")
    
    clean(
        input_fif=args.input,
        output_fif=args.output,
        method=args.method,
        checkpoint=args.checkpoint,
        target_fs=args.target_fs,
        highpass=args.highpass,
        lowpass=args.lowpass,
        max_passes=args.max_passes,
        r_thresh=args.r_thresh
    )
    print("[cidenoise clean] Successfully completed!")

def audit_command(args):
    print(f"[cidenoise audit] Auditing file: {args.input}")
    report = audit(input_fif=args.input, output_json=args.output_json)
    print(json.dumps(report, indent=2))

def main():
    parser = argparse.ArgumentParser(
        prog="cidenoise",
        description="cidenoise: Turnkey Biophysical Disentanglement & Neural Network Denoising for Cochlear Implant EEG"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available sub-commands")
    
    # Clean sub-command
    clean_p = subparsers.add_parser("clean", help="Clean continuous CI EEG recording")
    clean_p.add_argument("--input", "-i", required=True, help="Input raw FIF file")
    clean_p.add_argument("--output", "-o", required=True, help="Output cleaned FIF file")
    clean_p.add_argument("--method", choices=["sobi-2pass", "sobi-n2n"], default="sobi-2pass", help="Denoising pipeline")
    clean_p.add_argument("--checkpoint", default=None, help="Path to trained PyTorch N2N model checkpoint")
    clean_p.add_argument("--target-fs", type=float, default=256.0, help="Target sampling rate (default: 256.0)")
    clean_p.add_argument("--highpass", type=float, default=1.0, help="High-pass filter cutoff in Hz (default: 1.0)")
    clean_p.add_argument("--lowpass", type=float, default=45.0, help="Low-pass filter cutoff in Hz (default: 45.0)")
    clean_p.add_argument("--max-passes", type=int, default=2, help="Max SOBI passes (default: 2)")
    clean_p.add_argument("--r-thresh", type=float, default=2.5, help="Mastoid-to-central ratio threshold")
    
    # Audit sub-command
    audit_p = subparsers.add_parser("audit", help="Audit recording biophysics and hardware QA")
    audit_p.add_argument("--input", "-i", required=True, help="Input raw FIF file")
    audit_p.add_argument("--output-json", default=None, help="Save audit report JSON")
    
    args = parser.parse_args()
    if args.command == "clean":
        clean_command(args)
    elif args.command == "audit":
        audit_command(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
