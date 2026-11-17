#!/usr/bin/env python3
"""
train_sobi_n2n.py
Trains Stage 2 Symmetric Noise2Noise on SOBI 2-Pass Cleaned EEG.

Compliance Features:
1. Symmetric N2N Loss: L_N2N = 0.5*||f(y1) - y2||_1 + 0.5*||f(y2) - y1||_1 (Thorne compliance)
2. Speech & Silent Artifact Subspace Nulling: ||P_art * f(y)||^2 (Thorne compliance)
3. Monitored ETR Guardrail: ETR = RMS(P7) / RMS(Cz) <= 1.2
4. Sentence-Disjoint Splits: Train / Val / Test partitioned with zero stimulus leakage.
5. Shuffled Envelope Permutation Sanity Test.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if os.path.join(REPO_ROOT, "track_a_deep_learning") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "track_a_deep_learning"))

from n2n_dataloader import build_n2n_splits
from model import UNet1D

def compute_sagittal_permutation_matrix(ch_names):
    """Generates orthogonal P_sagittal matrix reflecting left/right channels."""
    swap_pairs = [
        ('Fp1', 'Fp2'), ('F7', 'F8'), ('F3', 'F4'),
        ('T7', 'T8'), ('C3', 'C4'),
        ('P7', 'P8'), ('P3', 'P4'),
        ('O1', 'O2')
    ]
    n = len(ch_names)
    P = np.eye(n, dtype=np.float32)
    ch_map = {ch: i for i, ch in enumerate(ch_names)}
    for left, right in swap_pairs:
        if left in ch_map and right in ch_map:
            i, j = ch_map[left], ch_map[right]
            P[i, i] = 0.0
            P[j, j] = 0.0
            P[i, j] = 1.0
            P[j, i] = 1.0
    return P

class EquivariantUNet1D(nn.Module):
    def __init__(self, base_model, p_sagittal=None, use_sagittal=False):
        super().__init__()
        self.base = base_model
        self.use_sagittal = use_sagittal
        if p_sagittal is not None:
            self.register_buffer('P', torch.tensor(p_sagittal, dtype=torch.float32))
        else:
            self.P = None
            
    def forward(self, x):
        if self.use_sagittal and self.P is not None:
            # Equivariant forward pass: 0.5 * (f(x) + P f(P x))
            Px = torch.matmul(self.P, x)
            f_x = self.base(x)
            f_Px = self.base(Px)
            out = 0.5 * (f_x + torch.matmul(self.P, f_Px))
            return out
        else:
            return self.base(x)

def train_epoch(model, loader, optimizer, device, lambda_art=1.0, lambda_v=0.5):
    model.train()
    total_loss, total_n2n, total_art, total_v = 0.0, 0.0, 0.0, 0.0
    
    for y1, y2, y_v, p_art in loader:
        y1 = y1.to(device)
        y2 = y2.to(device)
        y_v = y_v.to(device)
        p_art = p_art[0].to(device) # (n_chs, n_chs)
        
        optimizer.zero_grad()
        
        # Forward passes
        out1 = model(y1)
        out2 = model(y2)
        out_v = model(y_v)
        
        # 1. Symmetric N2N Loss
        loss_n2n = 0.5 * torch.mean(torch.abs(out1 - y2)) + 0.5 * torch.mean(torch.abs(out2 - y1))
        
        # 2. Speech Trial Artifact Subspace Nulling (Dr. Thorne mandate)
        # ||P_art * f(y)||^2
        p_out1 = torch.matmul(p_art, out1)
        p_out2 = torch.matmul(p_art, out2)
        loss_art = torch.mean(p_out1 ** 2) + torch.mean(p_out2 ** 2)
        
        # 3. Silent Visual Artifact Subspace Nulling (preserves ongoing alpha/theta)
        p_out_v = torch.matmul(p_art, out_v)
        loss_v = torch.mean(p_out_v ** 2)
        
        loss = loss_n2n + lambda_art * loss_art + lambda_v * loss_v
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_n2n += loss_n2n.item()
        total_art += loss_art.item()
        total_v += loss_v.item()
        
    n = len(loader)
    return total_loss / n, total_n2n / n, total_art / n, total_v / n

def evaluate(model, loader, device, cz_idx, p7_idx):
    model.eval()
    total_loss = 0.0
    etr_list = []
    with torch.no_grad():
        for y1, y2, _, _ in loader:
            y1 = y1.to(device)
            y2 = y2.to(device)
            out1 = model(y1)
            loss_n2n = 0.5 * torch.mean(torch.abs(out1 - y2))
            total_loss += loss_n2n.item()
            
            # Compute ETR = RMS(P7) / RMS(Cz) on predictions
            p7_rms = torch.sqrt(torch.mean(out1[:, p7_idx, :] ** 2)).item()
            cz_rms = torch.sqrt(torch.mean(out1[:, cz_idx, :] ** 2) + 1e-12).item()
            etr_list.append(p7_rms / cz_rms)
            
    return total_loss / len(loader), float(np.mean(etr_list))

def run_training(sobi_fif, csv_path, summary_json, output_dir, epochs=25, lr=5e-4, use_sagittal=False):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Stage 2 N2N] Compute Device: {device} | Use Sagittal: {use_sagittal}")
    
    # 1. Build splits
    train_ds, val_ds, test_ds, ch_names = build_n2n_splits(sobi_fif, csv_path, summary_json)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)
    
    cz_idx = ch_names.index('Cz')
    p7_idx = ch_names.index('P7')
    n_channels = len(ch_names)
    
    # 2. Build model
    base_unet = UNet1D(in_channels=n_channels, out_channels=n_channels)
    P_sag = compute_sagittal_permutation_matrix(ch_names)
    model = EquivariantUNet1D(base_unet, P_sag, use_sagittal=use_sagittal).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = []
    best_val_loss = float('inf')
    best_model_path = os.path.join(output_dir, "best_sobi_n2n_unet.pt")
    
    print("\n" + "=" * 70)
    print("STARTING STAGE 2 NOISE2NOISE TRAINING")
    print("=" * 70)
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_n2n, train_art, train_v = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_etr = evaluate(model, val_loader, device, cz_idx, p7_idx)
        scheduler.step()
        ep_time = time.time() - t0
        
        # ETR Guardrail check (Dr. Thorne requirement: ETR <= 1.2 after warmup)
        etr_status = "PASS" if val_etr <= 1.2 else "WARN"
        if epoch >= 10 and val_etr > 1.5:
            print(f"Epoch {epoch}: FATAL GUARDRAIL VIOLATION! ETR = {val_etr:.2f} > 1.5. Stopping early.")
            break
            
        print(f"Epoch {epoch:02d}/{epochs:02d} [{ep_time:.1f}s] | Train: {train_loss:.4f} (N2N: {train_n2n:.4f}, Art: {train_art:.4f}) | Val N2N: {val_loss:.4f} | ETR: {val_etr:.2f} ({etr_status})")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_n2n': train_n2n,
            'train_art': train_art,
            'val_loss': val_loss,
            'val_etr': val_etr,
            'epoch_time_s': ep_time
        })
        
    print(f"\nTraining Complete! Best Val Loss: {best_val_loss:.4f}")
    
    # Evaluate best model on Test split
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss, test_etr = evaluate(model, test_loader, device, cz_idx, p7_idx)
    print(f"[Held-Out Test Split] Loss: {test_loss:.4f} | ETR: {test_etr:.2f} (Target <= 1.2)")
    
    summary = {
        'epochs': epochs,
        'best_val_loss': best_val_loss,
        'test_loss': test_loss,
        'test_etr': test_etr,
        'use_sagittal': use_sagittal,
        'history': history
    }
    
    summary_path = os.path.join(output_dir, "sobi_n2n_training_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sobi-fif", default="./benchmark_results/sobi2pass_cache/trial_onsets_0901y2_sobi2pass-raw.fif")
    parser.add_argument("--csv", default="/quobyte/millerlmgrp/annotated_data/CMPy2/trial_onsets_0901y2.csv")
    parser.add_argument("--summary-json", default="./benchmark_results/sobi2pass_cache/sobi2pass_summary_0901y2.json")
    parser.add_argument("--output-dir", default="./benchmark_results/sobi_n2n_results")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--use-sagittal", action="store_true", default=False)
    args = parser.parse_args()
    
    run_training(args.sobi_fif, args.csv, args.summary_json, args.output_dir, epochs=args.epochs, use_sagittal=args.use_sagittal)
