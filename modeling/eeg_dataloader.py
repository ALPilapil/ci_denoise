from torch.utils.data import Dataset, DataLoader, random_split
import zarr
import torch
import numpy as np
from collections import defaultdict

class EEGDataset(Dataset):
    def __init__(self, zarr_path, clean_group, noise_group, alpha=1.0, num_pairings=3, seed=42, mix=True):
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)
        self.mix = mix

        root = zarr.open_group(zarr_path, mode='r')
        self.clean_data = root[clean_group]
        self.noise_data = root[noise_group]

        self.clean_perm  = self.clean_data['perm'][:]
        self.clean_block = self.clean_data['block'][:]
        self.noise_perm  = self.noise_data['perm'][:]
        self.noise_block = self.noise_data['block'][:]

        # precompute bad indices
        clean_arr = self.clean_data['data'][:]
        self.bad_clean = set(int(i) for i in np.where(np.isnan(clean_arr).any(axis=(1,2)))[0])
        print(f"Skipping {len(self.bad_clean)} clean epoch(s) with NaNs")

        noise_arr = self.noise_data['data'][:]
        self.bad_noise = set(int(i) for i in np.where(np.isnan(noise_arr).any(axis=(1,2)))[0])
        print(f"Skipping {len(self.bad_noise)} noise epoch(s) with NaNs")

        # build (perm, block) → noise indices lookup
        self.noise_by_cond = defaultdict(list)
        for idx in range(len(self.noise_perm)):
            if idx not in self.bad_noise:
                key = (self.noise_perm[idx], self.noise_block[idx])
                self.noise_by_cond[key].append(idx)

        self.pairs = self._make_pairs(num_pairings)

    def _make_pairs(self, num_pairings):
        pairs = []
        for clean_idx in range(len(self.clean_perm)):
            if clean_idx in self.bad_clean:
                continue
            key = (self.clean_perm[clean_idx], self.clean_block[clean_idx])
            candidates = self.noise_by_cond.get(key, [])
            if not candidates:
                print(f"Warning: no noise epochs for perm={key[0]} block={key[1]}, skipping clean idx {clean_idx}")
                continue
            chosen = self.rng.choice(candidates, size=num_pairings, replace=len(candidates) < num_pairings)
            for noise_idx in chosen:
                pairs.append((clean_idx, int(noise_idx)))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        clean_idx, noise_idx = self.pairs[idx]

        clean = torch.tensor(self.clean_data['data'][clean_idx], dtype=torch.float32)
        noise = torch.tensor(self.noise_data['data'][noise_idx], dtype=torch.float32)

        min_ch = min(clean.shape[0], noise.shape[0])
        clean  = clean[:min_ch, :]
        noise  = noise[:min_ch, :]

        if self.mix:
            dirty = clean + self.alpha * noise
            return dirty, clean
        else:
            return noise, clean


def main():
    root = '/mnt/data/PilapilData/processed_data/raw_epoched_data.zarr'

    eeg_data = EEGDataset(
        zarr_path=root,
        clean_group='hearing_trial_data',
        noise_group='ci_trial_data',
        alpha=1.0,
        num_pairings=3,
        seed=42,
        mix=True,
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