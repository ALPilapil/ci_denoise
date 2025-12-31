import zarr
import numpy as np
from torch.utils.data import Dataset, DataLoader

class EEGDataset(Dataset):
    def __init__(self, zarr_path):
        self.root = zarr.open_group(zarr_path, mode='r')
        self.clean = self.root['clean']
        self.dirty = self.root['dirty']

    def __len__(self):
        length = min(self.clean.shape[0], self.dirty.shape[0])
        return length

    def __getitem__(self, idx):
        clean_epoch = self.clean[idx]
        dirty_epoch = self.dirty[idx]

        sample = {
            'dirty': dirty_epoch,
            'clean': clean_epoch
        }

        return sample

def main():
    #------------ Paths ------------#
    epoched_pair_path = '/quobyte/millerlmgrp/processed_data/epoched_pairs.zarr'

    #------------ Params ------------#
    pair_truncation = 1000 # load in only this amount of the data, set to None to load all
    batch_size = 64
    shuffle=True
    num_workers=0

    #------------ Data ------------#
    pairs_dataset = EEGDataset(zarr_path=epoched_pair_path) 
    dataloader = DataLoader(pairs_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    #------------ Model ------------#
    


if __name__ == "__main__":
    main()