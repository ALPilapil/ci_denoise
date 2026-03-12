import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
import time

from eeg_dataloader import EEGDataset
from unet import UNet


# ─── Config ───────────────────────────────────────────────────────────────────
ZARR_PATH       = '/mnt/data/PilapilData/processed_data/processed_raw_epoched_data.zarr'
CLEAN_GROUP     = 'hearing_trial_data'
NOISE_GROUP     = 'ci_trial_data'
CHECKPOINT_DIR  = '/mnt/data/PilapilData/checkpoints/'

ALPHA           = 1.0
NUM_PAIRINGS    = 3
SEED            = 42
MIX             = True

BATCH_SIZE      = 32
NUM_EPOCHS      = 50
LR              = 1e-3
TRAIN_SPLIT     = 0.8
VAL_SPLIT       = 0.1
# remaining is test

NUM_WORKERS     = 4
DEVICE          = 'cuda' if torch.cuda.is_available() else 'cpu'
# ──────────────────────────────────────────────────────────────────────────────


def split_dataset(dataset, train_frac, val_frac, seed):
    n       = len(dataset)
    n_train = int(train_frac * n)
    n_val   = int(val_frac * n)
    n_test  = n - n_train - n_val
    return random_split(dataset, [n_train, n_val, n_test],
                        generator=torch.Generator().manual_seed(seed))


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train() if train else model.eval()
    total_loss = 0.0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for dirty, clean in loader:
            dirty = dirty.to(device)
            clean = clean.to(device)

            pred = model(dirty)

            # match time dimension in case of rounding from pooling/upsampling
            min_t = min(pred.shape[-1], clean.shape[-1])
            pred  = pred[..., :min_t]
            clean = clean[..., :min_t]

            loss = criterion(pred, clean)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * dirty.size(0)

    return total_loss / len(loader.dataset)


def save_checkpoint(model, optimizer, epoch, val_loss, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch':      epoch,
        'model':      model.state_dict(),
        'optimizer':  optimizer.state_dict(),
        'val_loss':   val_loss,
    }, path)
    print(f"  checkpoint saved → {path}")


def main():
    torch.manual_seed(SEED)
    print(f"Using device: {DEVICE}")

    # ── data ──────────────────────────────────────────────────────────────────
    print("Loading dataset...")
    dataset = EEGDataset(
        zarr_path=ZARR_PATH,
        clean_group=CLEAN_GROUP,
        noise_group=NOISE_GROUP,
        alpha=NUM_PAIRINGS,
        num_pairings=NUM_PAIRINGS,
        seed=SEED,
        mix=MIX,
    )
    print(f"Total pairs: {len(dataset)}")

    train_set, val_set, test_set = split_dataset(dataset, TRAIN_SPLIT, VAL_SPLIT, SEED)
    print(f"Train: {len(train_set)}  Val: {len(val_set)}  Test: {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # ── model ─────────────────────────────────────────────────────────────────
    model     = UNet().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

    # ── training loop ─────────────────────────────────────────────────────────
    best_val_loss = float('inf')

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        train_loss = run_epoch(model, train_loader, criterion, optimizer, DEVICE, train=True)
        val_loss   = run_epoch(model, val_loader,   criterion, optimizer, DEVICE, train=False)

        scheduler.step(val_loss)
        elapsed = time.time() - t0

        print(f"Epoch {epoch:03d}/{NUM_EPOCHS}  "
              f"train={train_loss:.6f}  val={val_loss:.6f}  "
              f"({elapsed:.1f}s)")

        # save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss,
                            os.path.join(CHECKPOINT_DIR, 'best.pt'))

        # save periodic checkpoint every 10 epochs
        if epoch % 10 == 0:
            save_checkpoint(model, optimizer, epoch, val_loss,
                            os.path.join(CHECKPOINT_DIR, f'epoch_{epoch:03d}.pt'))

    # ── test ──────────────────────────────────────────────────────────────────
    print("\nLoading best checkpoint for test evaluation...")
    checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, 'best.pt'), map_location=DEVICE)
    model.load_state_dict(checkpoint['model'])

    test_loss = run_epoch(model, test_loader, criterion, optimizer, DEVICE, train=False)
    print(f"Test loss: {test_loss:.6f}")


if __name__ == "__main__":
    main()