from torch.utils.data import Dataset, DataLoader, random_split
import zarr
import torch
import numpy as np
from collections import defaultdict

class EEGDataset(Dataset):
    def __init__(self, zarr_path, clean_group, noise_group, alpha=1.0, num_pairings=3, seed=42, mix=True):
        '''
        Constructs (dirty, clean) pairs where:
            dirty = clean + alpha * noise
        Pairing is done at init time (many-to-one: num_pairings noise epochs per clean epoch).
        Matching condition: same (perm, block) — since stimulus order is fixed within each perm,
        block indices are comparable across hearing and CI participants.
        '''
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)
        self.mix = mix

        # open up the main root 
        root = zarr.open_group(zarr_path, mode='r')
        # divide into the separate groups
        self.clean_data = root[clean_group]
        self.noise_data = root[noise_group]

        # load metadata into memory (small arrays, fine to keep in RAM)
        self.clean_perm  = self.clean_data['perm'][:]
        self.clean_block = self.clean_data['block'][:]
        self.noise_perm  = self.noise_data['perm'][:]
        self.noise_block = self.noise_data['block'][:]

        # build (perm, block) → noise indices lookup for fast sampling
        self.noise_by_cond = defaultdict(list)
        for idx in range(len(self.noise_perm)):
            key = (self.noise_perm[idx], self.noise_block[idx])
            self.noise_by_cond[key].append(idx)

        self.pairs = self._make_pairs(num_pairings)  # list of (clean_idx, noise_idx)

    def _make_pairs(self, num_pairings):
        '''
        For each clean epoch, sample num_pairings noise epochs with matching (perm, block).
        Returns a flat list of (clean_idx, noise_idx) tuples.
        '''
        pairs = []
        for clean_idx in range(len(self.clean_perm)):
            key = (self.clean_perm[clean_idx], self.clean_block[clean_idx])
            candidates = self.noise_by_cond.get(key, [])
            if not candidates:
                print(f"Warning: no noise epochs for perm={key[0]} block={key[1]}, skipping clean idx {clean_idx}")
                continue
            # sample with replacement only if fewer candidates than num_pairings
            chosen = self.rng.choice(candidates, size=num_pairings, replace=len(candidates) < num_pairings)
            for noise_idx in chosen:
                pairs.append((clean_idx, int(noise_idx)))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        '''
        Returns (dirty, clean) as float32 tensors of shape (n_channels, n_times).
        dirty = clean + alpha * noise  (constructed on the fly)
        '''
        clean_idx, noise_idx = self.pairs[idx]

        clean = torch.tensor(self.clean_data['data'][clean_idx], dtype=torch.float32)
        noise = torch.tensor(self.noise_data['data'][noise_idx], dtype=torch.float32)

        # handle channel mismatch: trim to the smaller of the two if needed
        min_ch = min(clean.shape[0], noise.shape[0])
        clean  = clean[:min_ch, :]
        noise  = noise[:min_ch, :]

        # handle mix and match
        if self.mix:
            dirty = clean + self.alpha * noise
            return dirty, clean
        else:
            return dirty, clean


def main():
    root = '/quobyte/millerlmgrp/processed_data/processed_raw_epoched_data.zarr'

    eeg_data = EEGDataset(
        zarr_path=root,
        clean_group='hearing_trial_data',
        noise_group='ci_trial_data',
        alpha=1.0,
        num_pairings=3,
        seed=42,
        mix=True 
        # if true then the pairs are constructed from adding noise to clean
    )

    print(f"Total pairs: {len(eeg_data)}")
    dirty, clean = eeg_data[0]
    print(f"dirty shape: {dirty.shape}, clean shape: {clean.shape}")

    n = len(eeg_data)
    n_train = int(0.8 * n)
    n_val   = int(0.1 * n)
    n_test  = n - n_train - n_val

    train_set, val_set, test_set = random_split(eeg_data, [n_train, n_val, n_test])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_set,   batch_size=32, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_set,  batch_size=32, shuffle=False, num_workers=4)


if __name__ == "__main__":
    main()