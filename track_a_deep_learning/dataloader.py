"""
dataloader.py - Dataloader for Track A Deep Learning Denoising.

Features:
  - Chunked NaN index precomputation to prevent reading 19+ GB into RAM
  - Participant-grouped cross-validation splitting to prevent subject data leakage
  - Reproducible (perm, block) pairing logic matching project conventions
"""

from collections import defaultdict
from typing import List, Optional, Tuple, Set
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, random_split
import zarr


class TrackADataset(Dataset):
    """
    Synthetic mixture dataset for CI EEG denoising:
      dirty = clean + alpha * noise
    """
    def __init__(
        self,
        zarr_path: str,
        clean_group: str = "hearing_trial_data",
        noise_group: str = "ci_trial_data",
        alpha: float = 1.0,
        num_pairings: int = 3,
        seed: int = 42,
        mix: bool = True,
    ):
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)
        self.mix = mix

        try:
            root = zarr.open(zarr_path, mode='r')
        except Exception:
            root = zarr.open_group(zarr_path, mode='r')
        self.clean_data = root[clean_group]
        self.noise_data = root[noise_group]

        self.clean_perm  = self.clean_data['perm'][:]
        self.clean_block = self.clean_data['block'][:]
        self.noise_perm  = self.noise_data['perm'][:]
        self.noise_block = self.noise_data['block'][:]

        self.clean_participant = self.clean_data['participant'][:] if 'participant' in self.clean_data else None
        self.noise_participant = self.noise_data['participant'][:] if 'participant' in self.noise_data else None

        # Chunked scan for NaN epochs to avoid reading 19+ GB into RAM
        chunk_size = 50
        self.bad_clean: Set[int] = set()
        n_clean = len(self.clean_perm)
        for s in range(0, n_clean, chunk_size):
            e = min(s + chunk_size, n_clean)
            c = self.clean_data['data'][s:e]
            for ni in np.where(np.isnan(c).any(axis=(1, 2)))[0]:
                self.bad_clean.add(s + int(ni))
        if self.bad_clean:
            print(f"Skipping {len(self.bad_clean)} clean epoch(s) containing NaNs")

        self.bad_noise: Set[int] = set()
        n_noise = len(self.noise_perm)
        for s in range(0, n_noise, chunk_size):
            e = min(s + chunk_size, n_noise)
            c = self.noise_data['data'][s:e]
            for ni in np.where(np.isnan(c).any(axis=(1, 2)))[0]:
                self.bad_noise.add(s + int(ni))
        if self.bad_noise:
            print(f"Skipping {len(self.bad_noise)} noise epoch(s) containing NaNs")

        # Map (perm, block) -> noise indices
        self.noise_by_cond = defaultdict(list)
        for idx in range(len(self.noise_perm)):
            if idx not in self.bad_noise:
                key = (self.noise_perm[idx], self.noise_block[idx])
                self.noise_by_cond[key].append(idx)

        self.pairs = self._make_pairs(num_pairings)

    def _make_pairs(self, num_pairings: int) -> List[Tuple[int, int]]:
        pairs = []
        for clean_idx in range(len(self.clean_perm)):
            if clean_idx in self.bad_clean:
                continue
            key = (self.clean_perm[clean_idx], self.clean_block[clean_idx])
            candidates = self.noise_by_cond.get(key, [])
            if not candidates:
                continue
            chosen = self.rng.choice(candidates, size=num_pairings, replace=len(candidates) < num_pairings)
            for noise_idx in chosen:
                pairs.append((clean_idx, int(noise_idx)))
        return pairs

    def get_pair_participant_ids(self) -> Optional[List[int]]:
        if self.clean_participant is None:
            return None
        return [int(self.clean_participant[c_idx]) for c_idx, _ in self.pairs]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        clean_idx, noise_idx = self.pairs[idx]

        clean = torch.tensor(self.clean_data['data'][clean_idx], dtype=torch.float32)
        noise = torch.tensor(self.noise_data['data'][noise_idx], dtype=torch.float32)

        min_ch = min(clean.shape[0], noise.shape[0])
        clean = clean[:min_ch, :]
        noise = noise[:min_ch, :]

        if self.mix:
            dirty = clean + self.alpha * noise
            return dirty, clean
        else:
            return noise, clean


def split_dataset(
    dataset: TrackADataset,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
    strategy: str = "participant",
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Splits dataset into train, val, and test splits.
    If strategy is 'participant' and participant IDs exist in the dataset,
    ensures complete subject isolation across splits (zero data leakage).
    """
    p_ids = dataset.get_pair_participant_ids()

    if strategy == "participant" and p_ids is not None:
        unique_participants = np.array(sorted(list(set(p_ids))))
        rng = np.random.default_rng(seed)
        rng.shuffle(unique_participants)

        n_p = len(unique_participants)
        n_tr = max(1, int(train_frac * n_p))
        n_v = max(1, int(val_frac * n_p))
        if n_tr + n_v >= n_p:
            n_tr = max(1, n_p - 2)
            n_v = 1

        tr_p = set(unique_participants[:n_tr])
        v_p = set(unique_participants[n_tr:n_tr + n_v])
        te_p = set(unique_participants[n_tr + n_v:])

        tr_indices = [i for i, p in enumerate(p_ids) if p in tr_p]
        v_indices = [i for i, p in enumerate(p_ids) if p in v_p]
        te_indices = [i for i, p in enumerate(p_ids) if p in te_p]

        print(f"Participant-grouped split: {len(tr_p)} train subjects ({len(tr_indices)} pairs), "
              f"{len(v_p)} val subjects ({len(v_indices)} pairs), "
              f"{len(te_p)} test subjects ({len(te_indices)} pairs)")

        return Subset(dataset, tr_indices), Subset(dataset, v_indices), Subset(dataset, te_indices)

    # Fallback to random split
    n = len(dataset)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    n_test = n - n_train - n_val
    return random_split(dataset, [n_train, n_val, n_test],
                        generator=torch.Generator().manual_seed(seed))
