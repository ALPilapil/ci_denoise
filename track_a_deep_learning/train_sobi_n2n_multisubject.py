#!/usr/bin/env python3
"""
train_sobi_n2n_multisubject.py

Multi-Subject Foundation Training for Stage 2 Subspace-Constrained Symmetric Noise2Noise.
Aggregates stimulus-locked polarity pairs across multiple CI patients to learn
a unified, implant-invariant neural network representation.
"""

import os
import sys
import glob
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if os.path.join(REPO_ROOT, "track_a_deep_learning") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "track_a_deep_learning"))

from n2n_dataloader import build_n2n_splits, CANONICAL_20_CHS
from train_sobi_n2n import ResidualConstrainedUNet1D, compute_sagittal_permutation_matrix

DATA_DIR = "/quobyte/millerlmgrp/annotated_data/CMPy2"
SOBI_DIR = "./benchmark_results/sobi_polarity_cohort"

def get_valid_ci_subjects(max_subjects=8):
    fif_files = sorted(glob.glob(os.path.join(SOBI_DIR, "trial_onsets_09*y2_sobi2pass-raw.fif")))
    valid = []
    for f in fif_files:
        sub = os.path.basename(f).replace("trial_onsets_", "").replace("_sobi2pass-raw.fif", "")
        csv_file = os.path.join(DATA_DIR, f"trial_onsets_{sub}.csv")
        summary_file = os.path.join(SOBI_DIR, f"sobi2pass_summary_{sub}.json")
        
        # Check exclusion
        if os.path.exists(summary_file) and os.path.exists(csv_file):
            with open(summary_file, 'r') as jf:
                meta = json.load(jf)
            if not meta.get('hardware_failure'):
                valid.append({
                    'subject': sub,
                    'sobi_fif': f,
                    'csv_file': csv_file,
                    'summary_json': summary_file
                })
        if len(valid) >= max_subjects:
            break
    return valid

def main():
    parser = argparse.ArgumentParser(description="Multi-Subject SOBI-N2N Foundation Training")
    parser.add_argument("--num-subjects", type=int, default=8, help="Number of CI subjects to pool for foundation pre-training")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output-dir", default="./benchmark_results/sobi_n2n_foundation", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Foundation N2N] Device: {device} | Pooling {args.num_subjects} subjects")

    subjects = get_valid_ci_subjects(args.num_subjects)
    print(f"Found {len(subjects)} valid CI subjects for foundation pooling: {[s['subject'] for s in subjects]}")

    train_datasets = []
    val_datasets = []
    test_datasets = []

    for s in subjects:
        try:
            tr, va, te, p_art = build_n2n_splits(
                sobi_fif_path=s['sobi_fif'],
                csv_path=s['csv_file'],
                summary_json_path=s['summary_json']
            )
            train_datasets.append(tr)
            val_datasets.append(va)
            test_datasets.append(te)
            print(f"  + Added {s['subject']}: {len(tr)} train pairs, {len(va)} val pairs")
        except Exception as e:
            print(f"  ! Error loading {s['subject']}: {e}")

    full_train = ConcatDataset(train_datasets)
    full_val = ConcatDataset(val_datasets)
    full_test = ConcatDataset(test_datasets)

    print(f"\n[Dataset Summary] Pooled Cohort: {len(full_train)} Train Pairs | {len(full_val)} Val Pairs | {len(full_test)} Test Pairs")

    train_loader = DataLoader(full_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(full_val, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(full_test, batch_size=args.batch_size, shuffle=False)

    n_channels = 19
    P_sag = compute_sagittal_permutation_matrix(CANONICAL_20_CHS[:n_channels])
    model = ResidualConstrainedUNet1D(in_channels=n_channels, out_channels=n_channels, p_sagittal=P_sag, use_sagittal=False).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion_n2n = nn.L1Loss()
    criterion_reg = nn.MSELoss()

    best_val_loss = float('inf')
    ckpt_path = os.path.join(args.output_dir, "best_sobi_n2n_foundation_unet.pt")
    history = []

    print("\nStarting Multi-Subject Foundation Training...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0

        for y1, y2, y_v, p_art in train_loader:
            y1, y2, y_v, p_art = y1.to(device), y2.to(device), y_v.to(device), p_art.to(device)

            out_y1, _ = model(y1, p_art)
            out_y2, _ = model(y2, p_art)
            loss_n2n = 0.5 * criterion_n2n(out_y1, y2) + 0.5 * criterion_n2n(out_y2, y1)

            out_v, _ = model(y_v, p_art)
            loss_silent = criterion_reg(out_v, torch.zeros_like(out_v))

            loss = loss_n2n + 0.1 * loss_silent
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for y1, y2, y_v, p_art in val_loader:
                y1, y2, y_v, p_art = y1.to(device), y2.to(device), y_v.to(device), p_art.to(device)
                out_y1, _ = model(y1, p_art)
                out_y2, _ = model(y2, p_art)
                loss_n2n = 0.5 * criterion_n2n(out_y1, y2) + 0.5 * criterion_n2n(out_y2, y1)
                val_loss += loss_n2n.item()
        val_loss /= len(val_loader)
        ep_time = time.time() - t0

        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {ep_time:.1f}s", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"  --> Saved new best foundation checkpoint: {ckpt_path}", flush=True)

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'time_s': ep_time
        })

    summary = {
        'num_subjects_pooled': len(subjects),
        'subjects': [s['subject'] for s in subjects],
        'total_train_pairs': len(full_train),
        'best_val_loss': best_val_loss,
        'history': history
    }
    with open(os.path.join(args.output_dir, "foundation_training_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nFoundation Pre-Training Complete! Best Val Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
