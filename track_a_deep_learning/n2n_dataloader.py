#!/usr/bin/env python3
"""
n2n_dataloader.py
Sentence-Disjoint Noise2Noise Dataset and DataLoader for CI EEG.

Key Methodological Guarantees:
1. Sentence-Disjoint Partitioning: All takes of sentence K are grouped exclusively into Train, Val, or Test (zero stimulus leakage).
2. Symmetric Pairing: Pairs Take 1 (Polarity 1) and Take 2 (Polarity 2) of identical sentences.
3. Silent-Visual (V) Integration: Provides matched silent visual trials for artifact-subspace nulling.
4. Artifact Subspace Operator (P_artifact): Computes the projection matrix of the Stage 1 isolated hardware subspace.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import mne

CANONICAL_20_CHS = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'T7', 'C3', 'Cz', 'C4', 'T8',
    'P7', 'P3', 'Pz', 'P4', 'P8',
    'O1', 'Oz', 'O2'
]

class CINoise2NoiseDataset(Dataset):
    """
    Yields (y1, y2, y_silent, p_artifact) tuples for self-supervised training.
    """
    def __init__(self, pairs, silent_trials, p_artifact, window_len=512, scale_uv=100.0):
        self.pairs = pairs # list of (y1_tensor, y2_tensor, stim_id)
        self.silent_trials = silent_trials # list of y_v_tensors
        self.p_artifact = torch.tensor(p_artifact, dtype=torch.float32)
        self.window_len = window_len
        self.scale_uv = scale_uv
        
    def __len__(self):
        return len(self.pairs)
        
    def __getitem__(self, idx):
        y1, y2, stim_id = self.pairs[idx]
        # Random crop to window_len if longer
        T = y1.shape[-1]
        if T > self.window_len:
            start = np.random.randint(0, T - self.window_len + 1)
            y1_crop = y1[:, start:start + self.window_len]
            y2_crop = y2[:, start:start + self.window_len]
        else:
            y1_crop = y1
            y2_crop = y2
            
        # Randomly select a silent visual segment
        v_idx = np.random.randint(0, len(self.silent_trials))
        y_v = self.silent_trials[v_idx]
        T_v = y_v.shape[-1]
        if T_v > self.window_len:
            v_start = np.random.randint(0, T_v - self.window_len + 1)
            y_v_crop = y_v[:, v_start:v_start + self.window_len]
        else:
            y_v_crop = y_v
            
        # Scale to standard neural scale (~1.0 for 100 uV)
        y1_scaled = (y1_crop * 1e6) / self.scale_uv
        y2_scaled = (y2_crop * 1e6) / self.scale_uv
        y_v_scaled = (y_v_crop * 1e6) / self.scale_uv
        
        return (
            torch.tensor(y1_scaled, dtype=torch.float32),
            torch.tensor(y2_scaled, dtype=torch.float32),
            torch.tensor(y_v_scaled, dtype=torch.float32),
            self.p_artifact
        )

def build_n2n_splits(sobi_fif_path, csv_path, summary_json_path, split_ratios=(0.7, 0.15, 0.15), seed=42):
    """
    Constructs sentence-disjoint Train, Val, and Test datasets.
    """
    np.random.seed(seed)
    
    # 1. Load trial metadata
    df = pd.read_csv(csv_path)
    
    # 2. Load continuous SOBI cleaned data
    raw = mne.io.read_raw_fif(sobi_fif_path, preload=True, verbose=False)
    available_chs = [ch for ch in CANONICAL_20_CHS if ch in raw.ch_names]
    raw.pick(available_chs)
    fs = raw.info['sfreq']
    raw_data = raw.get_data() # (n_channels, n_times)
    n_channels = len(available_chs)
    
    # 3. Load SOBI mixing weights to construct P_artifact
    with open(summary_json_path, 'r') as f:
        sobi_meta = json.load(f)
        
    # Extract artifact mixing vectors
    A_art_list = []
    for p in sobi_meta.get('pass_history', []):
        weights = p.get('mixing_weights', {})
        vec = np.array([weights.get(ch, 0.0) for ch in available_chs], dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            A_art_list.append(vec / norm)
            
    if A_art_list:
        U = np.column_stack(A_art_list) # (n_channels, n_art_comps)
        # Orthogonal projection onto artifact subspace: P = U (U^T U)^-1 U^T
        P_artifact = U @ np.linalg.pinv(U.T @ U) @ U.T
    else:
        P_artifact = np.zeros((n_channels, n_channels), dtype=np.float32)
        
    # 4. Extract Silent Visual (V) segments
    v_trials = df[df['condition'] == 'V']
    v_win_len = int(2.0 * fs) # 2.0 seconds
    silent_segments = []
    for _, row in v_trials.iterrows():
        s0 = int(row['onset_s'] * fs)
        s1 = s0 + v_win_len
        if s1 <= raw_data.shape[1]:
            seg = raw_data[:, s0:s1]
            # Baseline correct
            seg -= np.mean(seg[:, :int(0.2*fs)], axis=1, keepdims=True)
            silent_segments.append(seg)
            
    # 5. Extract stimulus pairs (Take 1 and Take 2 of identical sentences)
    # Filter for active audio/audiovisual stimuli
    active_df = df[df['stim_id'] > 0]
    stim_groups = active_df.groupby('stim_id')
    
    sentence_pairs = {}
    for stim_id, grp in stim_groups:
        if len(grp) >= 2:
            rows = grp.sort_values('polarity') # pair polarity 1 with polarity 2
            r1 = rows.iloc[0]
            r2 = rows.iloc[1]
            
            s1_0 = int(r1['onset_s'] * fs) - int(0.2 * fs) # include 200ms baseline
            s1_1 = s1_0 + int(2.0 * fs)
            s2_0 = int(r2['onset_s'] * fs) - int(0.2 * fs)
            s2_1 = s2_0 + int(2.0 * fs)
            
            if s1_0 >= 0 and s1_1 <= raw_data.shape[1] and s2_0 >= 0 and s2_1 <= raw_data.shape[1]:
                seg1 = raw_data[:, s1_0:s1_1].copy()
                seg2 = raw_data[:, s2_0:s2_1].copy()
                # Baseline correct
                seg1 -= np.mean(seg1[:, :int(0.2*fs)], axis=1, keepdims=True)
                seg2 -= np.mean(seg2[:, :int(0.2*fs)], axis=1, keepdims=True)
                sentence_pairs[int(stim_id)] = (seg1, seg2, int(stim_id))
                
    unique_stims = sorted(list(sentence_pairs.keys()))
    np.random.shuffle(unique_stims)
    
    n_total = len(unique_stims)
    n_train = int(n_total * split_ratios[0])
    n_val = int(n_total * split_ratios[1])
    
    train_stims = set(unique_stims[:n_train])
    val_stims = set(unique_stims[n_train:n_train + n_val])
    test_stims = set(unique_stims[n_train + n_val:])
    
    train_pairs = [sentence_pairs[s] for s in train_stims]
    val_pairs = [sentence_pairs[s] for s in val_stims]
    test_pairs = [sentence_pairs[s] for s in test_stims]
    
    print(f"[N2N DataLoader] Total Stimulus Pairs: {n_total}")
    print(f"  -> Train: {len(train_pairs)} pairs (Sentence-Disjoint)")
    print(f"  -> Val:   {len(val_pairs)} pairs (Sentence-Disjoint)")
    print(f"  -> Test:  {len(test_pairs)} pairs (Sentence-Disjoint)")
    print(f"  -> Silent Visual Trials: {len(silent_segments)}")
    print(f"  -> P_artifact rank: {np.linalg.matrix_rank(P_artifact)}")
    
    train_ds = CINoise2NoiseDataset(train_pairs, silent_segments, P_artifact)
    val_ds = CINoise2NoiseDataset(val_pairs, silent_segments, P_artifact)
    test_ds = CINoise2NoiseDataset(test_pairs, silent_segments, P_artifact)
    
    return train_ds, val_ds, test_ds, available_chs
