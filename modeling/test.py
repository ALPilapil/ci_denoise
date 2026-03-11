import torch
from torch.utils.data import DataLoader
import zarr
import numpy as np
import os

from eeg_dataloader import EEGDataset
from unet import UNet


# ─── Config ───────────────────────────────────────────────────────────────────
ZARR_PATH       = '/quobyte/millerlmgrp/processed_data/processed_raw_epoched_data.zarr'
CLEAN_GROUP     = 'hearing_trial_data'
NOISE_GROUP     = 'ci_trial_data'
CHECKPOINT_PATH = '/quobyte/millerlmgrp/checkpoints/best.pt'
RESULTS_PATH    = '/quobyte/millerlmgrp/processed_data/results.zarr'

ALPHA           = 1.0
NUM_PAIRINGS    = 3
SEED            = 42
MIX             = True

BATCH_SIZE      = 32
NUM_WORKERS     = 4
DEVICE          = 'cuda' if torch.cuda.is_available() else 'cpu'
# ──────────────────────────────────────────────────────────────────────────────


def run_inference(model, loader, device):
    """
    Run the model on all batches in loader.
    Returns three arrays: noisy inputs, model predictions, clean targets.
    """
    model.eval()

    all_noisy  = []
    all_preds  = []
    all_clean  = []

    with torch.no_grad():
        for dirty, clean in loader:
            dirty = dirty.to(device)
            pred  = model(dirty)

            # crop time dim in case of pooling/upsampling rounding
            min_t = min(pred.shape[-1], clean.shape[-1])
            pred  = pred[..., :min_t]
            dirty = dirty[..., :min_t]
            clean = clean[..., :min_t]

            all_noisy.append(dirty.cpu().numpy())
            all_preds.append(pred.cpu().numpy())
            all_clean.append(clean.numpy())

    return (
        np.concatenate(all_noisy, axis=0),
        np.concatenate(all_preds, axis=0),
        np.concatenate(all_clean, axis=0),
    )


def save_results(noisy, preds, clean, path):
    """
    Save noisy inputs, model predictions, and clean targets to a zarr store.
    Shape of each array: (n_epochs, n_channels, n_times)
    """
    root = zarr.open_group(path, mode='w')

    n, c, t = preds.shape
    chunks  = (10, c, t)

    root.create_dataset('noisy',    data=noisy,        chunks=chunks, dtype='float32')
    root.create_dataset('denoised', data=preds,        chunks=chunks, dtype='float32')
    root.create_dataset('clean',    data=clean,        chunks=chunks, dtype='float32')

    # also store residual (what the model removed)
    root.create_dataset('residual', data=noisy - preds, chunks=chunks, dtype='float32')

    print(f"Saved {n} epochs to {path}")
    print(f"  noisy:    {root['noisy'].shape}")
    print(f"  denoised: {root['denoised'].shape}")
    print(f"  clean:    {root['clean'].shape}")
    print(f"  residual: {root['residual'].shape}")


def main():
    print(f"Using device: {DEVICE}")

    # ── data ──────────────────────────────────────────────────────────────────
    print("Loading dataset...")
    dataset = EEGDataset(
        zarr_path=ZARR_PATH,
        clean_group=CLEAN_GROUP,
        noise_group=NOISE_GROUP,
        alpha=ALPHA,
        num_pairings=NUM_PAIRINGS,
        seed=SEED,
        mix=MIX,
    )
    print(f"Total pairs: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

    # ── model ─────────────────────────────────────────────────────────────────
    print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
    model = UNet().to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model'])
    print(f"  (trained to epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.6f})")

    # ── inference ─────────────────────────────────────────────────────────────
    print("Running inference...")
    noisy, preds, clean = run_inference(model, loader, DEVICE)
    print(f"Output shape: {preds.shape}")

    # ── save ──────────────────────────────────────────────────────────────────
    save_results(noisy, preds, clean, RESULTS_PATH)


if __name__ == "__main__":
    main()