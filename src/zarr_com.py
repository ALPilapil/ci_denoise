import zarr
import numpy as np
import rich
from pathlib import Path
import sys

epoch_data_storage = '/quobyte/millerlmgrp/processed_data/epoched_data.zarr'
epoch_pairs_storage = '/quobyte/millerlmgrp/processed_data/epoched_pairs.zarr'
n_channels = 21
n_times = 8193
n_chunks = 100
# Shape will be: (8708, 21, 8193)

def initialize_zarr():
    compressors = zarr.codecs.BloscCodec(cname='zstd', clevel=3, shuffle=zarr.codecs.BloscShuffle.bitshuffle)
    # define chunk size
    chunks_shape = (n_chunks, n_channels, n_times)
    # initialize root
    root = zarr.group(epoch_data_storage)
    # initialize groups
    ci_trial_data = root.create_group(name="ci_trial_data")
    hearing_trial_data = root.create_group(name="hearing_trial_data")
    # initialize storage in both groups
    # ci group
    ci_data = ci_trial_data.create_array(name="data", shape=(0, n_channels, n_times), dtype="float32", compressors=compressors, chunks=chunks_shape)
    ci_labels = ci_trial_data.create_array(name="labels", shape=(0,), dtype="int32", chunks=(n_chunks,))
    # hearing group
    hearing_data = hearing_trial_data.create_array(name="data", shape=(0, n_channels, n_times), dtype="float32", compressors=compressors, chunks=chunks_shape)
    hearing_labels = hearing_trial_data.create_array(name="labels", shape=(0,), dtype="int32", chunks=(n_chunks,))

def initialize_pair_zarr():
    compressors = zarr.codecs.BloscCodec(cname='zstd', clevel=3, shuffle=zarr.codecs.BloscShuffle.bitshuffle)
    # define chunk size
    chunks_shape = (n_chunks, n_channels, n_times)
    # initialize root
    root = zarr.open_group(epoch_pairs_storage, mode='w')
    # initialize array
    clean = root.create_array(name="clean", shape=(0, n_channels, n_times), 
                              dtype="float32", compressors=compressors, chunks=chunks_shape)
    dirty = root.create_array(name="dirty", shape=(0, n_channels, n_times), 
                              dtype="float32", compressors=compressors, chunks=chunks_shape)

## APPEND
def appand_zarr():
    root = zarr.open_group(epoch_data_storage)
    ci_trial_data = root['ci_trial_data']
    ci_data = ci_trial_data['data']
    additional_eeg_data = np.random.rand(10, 20, 100)
    ci_data.append(additional_eeg_data)
    print(ci_data.shape)
    ci_data.append(eeg_data)
    print(ci_data.shape)
    print(ci_data.info)

def load_zarr():
    root = zarr.open_group(epoch_data_storage, mode='r')
    ci_data = root['ci_trial_data']['data']
    all_data = ci_data[:]
    print(all_data.shape)

    partial_data = ci_data[:5]
    print(partial_data.shape)

## DELETE
def delete_zarr():
    root = zarr.open_group(epoch_data_storage, mode="a")
    
    # Clear both groups
    for group_name in ["hearing_trial_data", "ci_trial_data"]:
        grp = root[group_name]
        
        # clear all arrays in this group by resizing the first axis to 0
        for name, arr in grp.arrays():
            if arr.ndim >= 1:
                # Build new shape with first dimension = 0, keep other dimensions
                new_shape = (0,) + arr.shape[1:]
                arr.resize(new_shape)
                print(f"Cleared {grp.name}/{name} -> new shape {arr.shape}")

def del_pairs():
    root = zarr.open_group(epoch_pairs_storage, mode="a")
    
    # Clear all arrays in root
    for name, arr in root.arrays():
        new_shape = (0,) + arr.shape[1:]
        arr.resize(new_shape)
        print(f"Cleared {name} -> new shape {arr.shape}")

def main():
    # take in command line args
    arg = sys.argv[1]
    if arg == 'delete':
        delete_zarr()
    elif arg == 'append':
        appand_zarr()
    elif arg == 'init_epoch':
        initialize_zarr()
    elif arg == 'init_pair':
        initialize_pair_zarr()
    elif arg == 'del_pairs':
        del_pairs()
    elif arg == 'load':
        load_zarr()
    else:
        raise ValueError("Argument must be delete, append, init_epoch, init_pair")


if __name__ == "__main__":
    main()
