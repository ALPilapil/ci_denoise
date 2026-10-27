"""
train.py - Standalone Training Pipeline for Track A Deep Learning Denoising.

Features:
  - Multi-objective composite neuroengineering loss (STFT spectral + Pearson correlation + ERP window)
  - Subject-isolated cross-validation splitting (zero subject leakage)
  - Checkpoint resume capability and periodic saving
  - Real-time unbuffered stdout streaming for SLURM
"""

import argparse
import os
import sys
import time
from typing import Dict, Tuple

try:
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
except Exception:
    pass

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from dataloader import TrackADataset, split_dataset
from losses import CompositeNeuroLoss
from model import UNet1D


def parse_args():
    parser = argparse.ArgumentParser(description="Track A: Train 1D U-Net with Composite Neuro Loss")
    parser.add_argument(
        "--zarr-path",
        default="/quobyte/millerlmgrp/processed_data/raw_epoched_data.zarr",
        help="Path to raw_epoched_data.zarr",
    )
    parser.add_argument("--clean-group", default="hearing_trial_data", help="Zarr group for clean hearing EEG")
    parser.add_argument("--noise-group", default="ci_trial_data", help="Zarr group for CI artifact noise")
    parser.add_argument(
        "--checkpoint-dir",
        default="/quobyte/millerlmgrp/checkpoints/track_a",
        help="Directory to save model checkpoints",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--alpha", type=float, default=1.0, help="Artifact mixing coefficient")
    parser.add_argument("--num-pairings", type=int, default=3, help="Number of noise pairings per clean epoch")
    parser.add_argument("--train-split", type=float, default=0.8, help="Fraction for training split")
    parser.add_argument("--val-split", type=float, default=0.1, help="Fraction for validation split")
    parser.add_argument(
        "--split-strategy",
        choices=["participant", "random"],
        default="participant",
        help="Splitting strategy: 'participant' strictly isolates subjects, 'random' splits pairs",
    )
    parser.add_argument(
        "--loss-type",
        choices=["composite", "l1", "mse"],
        default="composite",
        help="Loss objective: 'composite' (multi-objective STFT+Corr+ERP), 'l1', or 'mse'",
    )
    parser.add_argument("--lambda-spec", type=float, default=0.5, help="Weight for STFT spectral loss")
    parser.add_argument("--lambda-corr", type=float, default=0.2, help="Weight for Pearson correlation loss")
    parser.add_argument("--lambda-erp", type=float, default=0.3, help="Weight for auditory ERP window loss")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device ('cuda' or 'cpu')",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest/best checkpoint in checkpoint-dir if available",
    )
    return parser.parse_args()


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train: bool = True,
) -> Tuple[float, Dict[str, float]]:
    model.train() if train else model.eval()
    total_loss = 0.0
    accum_sub = {"time": 0.0, "spec": 0.0, "corr": 0.0, "erp": 0.0}
    is_composite = isinstance(criterion, CompositeNeuroLoss)

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch_idx, (dirty, clean) in enumerate(loader, start=1):
            dirty = dirty.to(device)
            clean = clean.to(device)

            pred = model(dirty)
            min_t = min(pred.shape[-1], clean.shape[-1])
            pred = pred[..., :min_t]
            clean = clean[..., :min_t]

            if is_composite:
                loss, metrics = criterion(pred, clean)
                for k in accum_sub:
                    accum_sub[k] += metrics.get(k, 0.0) * dirty.size(0)
            else:
                loss = criterion(pred, clean)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if batch_idx % 50 == 0 or batch_idx == len(loader):
                    print(f"    [Batch {batch_idx:03d}/{len(loader)}] loss={loss.item():.6f}", flush=True)

            total_loss += loss.item() * dirty.size(0)

            # Explicitly release references to graph tensors
            del pred, dirty, clean, loss
            if train and device.type == 'cuda' and (batch_idx % 100 == 0):
                torch.cuda.empty_cache()

    n_samples = max(1, len(loader.dataset))
    avg_loss = total_loss / n_samples
    avg_sub = {k: v / n_samples for k, v in accum_sub.items()}
    return avg_loss, avg_sub


def save_checkpoint(model, optimizer, epoch, val_loss, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch':     epoch,
        'model':     model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'val_loss':  val_loss,
    }, path)
    print(f"  Checkpoint saved → {path}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cpu" and "SLURM_CPUS_PER_TASK" in os.environ:
        torch.set_num_threads(int(os.environ["SLURM_CPUS_PER_TASK"]))
        print(f"PyTorch CPU Threads: {torch.get_num_threads()}")

    print("=" * 70)
    print("  TRACK A: 1D U-NET TRAINING (MULTI-OBJECTIVE NEUROENGINEERING LOSS)")
    print("=" * 70)
    print(f"Device:         {device}")
    print(f"Zarr path:      {args.zarr_path}")
    print(f"Clean group:    {args.clean_group}")
    print(f"Noise group:    {args.noise_group}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print(f"Epochs:         {args.epochs}")
    print(f"Batch size:     {args.batch_size}")
    print(f"Learning rate:  {args.lr}")
    print(f"Loss objective: {args.loss_type}")
    if args.loss_type == "composite":
        print(f"  Weights: λ_spec={args.lambda_spec}, λ_corr={args.lambda_corr}, λ_erp={args.lambda_erp}")
    print(f"Split strategy: {args.split_strategy}\n")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 1. Dataset loading
    print("Loading dataset...")
    dataset = TrackADataset(
        zarr_path=args.zarr_path,
        clean_group=args.clean_group,
        noise_group=args.noise_group,
        alpha=args.alpha,
        num_pairings=args.num_pairings,
        seed=args.seed,
        mix=True,
    )
    print(f"Total pairs: {len(dataset)}")

    train_set, val_set, test_set = split_dataset(
        dataset, args.train_split, args.val_split, args.seed, strategy=args.split_strategy
    )
    print(f"Train: {len(train_set)}  Val: {len(val_set)}  Test: {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))

    # 2. Model & Loss setup
    model = UNet1D(in_channels=21, out_channels=21).to(device)

    if args.loss_type == "composite":
        criterion = CompositeNeuroLoss(
            time_loss="l1",
            lambda_spec=args.lambda_spec,
            lambda_corr=args.lambda_corr,
            lambda_erp=args.lambda_erp,
            sfreq=250.0,
        ).to(device)
    elif args.loss_type == "l1":
        criterion = torch.nn.L1Loss()
    else:
        criterion = torch.nn.MSELoss()

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # 3. Resume logic
    start_epoch = 1
    best_val_loss = float('inf')
    best_ckpt = os.path.join(args.checkpoint_dir, 'best.pt')
    latest_ckpt = os.path.join(args.checkpoint_dir, 'latest.pt')

    if args.resume:
        resume_target = latest_ckpt if os.path.exists(latest_ckpt) else (best_ckpt if os.path.exists(best_ckpt) else None)
        if resume_target:
            print(f"Resuming training from {resume_target}...")
            ckpt = torch.load(resume_target, map_location=device)
            model.load_state_dict(ckpt['model'])
            if 'optimizer' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer'])
            start_epoch = ckpt.get('epoch', 0) + 1
            best_val_loss = ckpt.get('val_loss', float('inf'))
            print(f"  Resumed at epoch {start_epoch}, previous best val_loss: {best_val_loss:.6f}\n")

    # 4. Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_loss, train_sub = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_sub     = run_epoch(model, val_loader,   criterion, optimizer, device, train=False)

        scheduler.step(val_loss)
        elapsed = time.time() - t0

        if args.loss_type == "composite":
            print(f"Epoch {epoch:03d}/{args.epochs}  "
                  f"train={train_loss:.6f} [time={train_sub['time']:.4f}, spec={train_sub['spec']:.4f}, corr={train_sub['corr']:.4f}]  "
                  f"val={val_loss:.6f} [spec={val_sub['spec']:.4f}, corr={val_sub['corr']:.4f}]  "
                  f"({elapsed:.1f}s)")
        else:
            print(f"Epoch {epoch:03d}/{args.epochs}  "
                  f"train={train_loss:.6f}  val={val_loss:.6f}  "
                  f"({elapsed:.1f}s)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, best_ckpt)

        save_checkpoint(model, optimizer, epoch, val_loss, latest_ckpt)

        if epoch % 10 == 0:
            save_checkpoint(model, optimizer, epoch, val_loss,
                            os.path.join(args.checkpoint_dir, f'epoch_{epoch:03d}.pt'))

    # 5. Final test evaluation
    print("\nLoading best checkpoint for held-out test split evaluation...")
    if os.path.exists(best_ckpt):
        checkpoint = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(checkpoint['model'])

        test_loss, test_sub = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
        print(f"Test loss: {test_loss:.6f}")
        if args.loss_type == "composite":
            print(f"  Breakdown: time={test_sub['time']:.6f}, spec={test_sub['spec']:.4f}, corr={test_sub['corr']:.4f}, erp={test_sub['erp']:.4f}")
    else:
        print(f"Warning: {best_ckpt} not found.")


if __name__ == "__main__":
    main()
