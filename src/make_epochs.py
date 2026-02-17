import zarr
from pathlib import Path
from process_util import PreprocessConfig, list_file_paths 
import mne
import argparse
import gc

def list_raws(read_path, truncation=None, preload=False):
    '''
    input: path to read from
    output: list of raws
    '''
    file_paths = list_file_paths(read_path)

    if truncation is not None and not isinstance(truncation, int):
        raise TypeError("Function read_raws only accepts None or int as an argument")

    if truncation:
        file_paths = file_paths[:truncation]
    
    return file_paths


def get_dims(raw, config):
    '''
    input: a path to raw data
    output: the dimensions of the epoching of the data
    '''
    # process once
    n_channels = len(raw.ch_names)
    n_times = int((config.tmax - config.tmin) * raw.info['sfreq']) + 1

    return (n_channels, n_times)


def make_epoch_util(raw, config, zarr_group, preload):
    '''
    given a list of mne.raw objects create and save just the epochs that are relevant to us
    these relevant epochs are identified by their event name and come into this function via a list
    input: set_paths, save_path, wanted_events
    output: none, saves to zarr file
    '''   
    events, event_dict = mne.events_from_annotations(raw)
    epochs = mne.Epochs(raw, events, event_id=config.wanted_epochs, tmin=config.tmin, tmax=config.tmax, 
                        baseline=config.baseline, preload=preload, reject_by_annotation=True)

    # get the epoch data for this participant
    epoch_data = epochs.get_data() # (n_epochs, n_channels, n_times) appears in chronological time

    # store the data in an X, data array and Y, label array
    # epoch data is (n_events, n_channels, n_times) and event is ids is (n_events). ith event id = ith epoch data event id
    # just storing 2 numpy arrays in total 
    event_ids = epochs.events[:, 2] # NOTE: from here the events have been translated according to event_dict. so 100 -> 2, 104 -> 4 etc. this is fine 

    # append to the group
    zarr_group['data'].append(epoch_data)
    zarr_group['labels'].append(event_ids)

def make_epoch_data(file_list, config, zarr_group, preload):
    '''
    just take on a filepath and run it through the util function
    '''
    # get dimensions
    raw = mne.io.read_raw_fif(file_list[0], preload=preload)
    n_channels, n_times = get_dims(raw=raw, config=config)

    # create datasets
    zarr_group.create_dataset('data', shape=(0, n_channels, n_times), chunks=(10, n_channels, n_times), dtype='float64')
    zarr_group.create_dataset('labels', shape=(0,), chunks=(10,), dtype='int64')
    
    for filename in file_list:
        # load in the raw data
        raw = mne.io.read_raw_fif(filename, preload=preload)

        # process it into epochs
        make_epoch_util(raw=raw, config=config, zarr_group=zarr_group, preload=preload)

        
def main():
    '''
    Isolates the CI noise from each CI kid via ICA and saves it to [path here]. Does this through running ICA on each datapoint
    and saving the components that primarily come from points of interest: ['P7', 'T7', 'M2', 'M1', 'P8'] since these are the 
    closest to where CIs are placed
    '''
    #----------- Parameters and Paths -----------#
    # where to store noise once isolated
    noise_folder_path = '/quobyte/millerlmgrp/processed_data/noise/'
    noisy_epochs_folder_path = '/quobyte/millerlmgrp/processed_data/noisy_epochs/'
    epoch_data_storage = '/quobyte/millerlmgrp/processed_data/epoched_data.zarr'
    hearing_folder_path = '/quobyte/millerlmgrp/processed_data/hearing/'
    epoch_pair_path = '/quobyte/millerlmgrp/processed_data/epoched_pairs.zarr'

    # preprocessing parameters
    preload = False
    truncation = 1
    CI_chs = ['P7', 'T7', 'M2', 'M1', 'P8'] # points where you would expect lots of CI noise from
    n_components = 0.99999 # tells ICA to use however many components explain 99.9999% of the data
    l_freq = 2 # low frequency band 
    years = [2, 3, 4]
    wanted_epochs = [10, 11] # needs to be a subset of event_dict
    tmin = 0.0
    tmax = 60.0
    baseline = (0,0)
    alpha = 1 # amount to scale noise by, keep it [.1, 2.0], SNR
    channel_scaling = {
        'M1': 2.0, 'M2': 2.0,
        'T7': 1.5, 'P7': 1.5,
        'P8': 1.5,
        # others will default to 1.0
    }
    gaussian_noise = False # adding more noise on top of the CI noise

    config = PreprocessConfig(channels=CI_chs, # expect noise from here
                            n_components=n_components, 
                            l_freq=l_freq, 
                            years=years, 
                            wanted_epochs=wanted_epochs, 
                            tmin=tmin, 
                            tmax=tmax, 
                            baseline=baseline, 
                            alpha=alpha, 
                            channel_scaling=channel_scaling, 
                            gaussian_noise=gaussian_noise)


    # make lists of the data
    ci_raw_list = list_raws(read_path=noise_folder_path, truncation=truncation)
    hearing_raw_list = list_raws(read_path=hearing_folder_path, truncation=truncation)

    # open up the storage
    # open storage
    root = zarr.open_group(epoch_data_storage, mode='w')
    ci_group = root.create_group('ci_trial_data')
    hearing_group = root.create_group('hearing_trial_data')

    # process each file
    make_epoch_data(file_list=ci_raw_list, config=config, zarr_group=ci_group, preload=preload)
    gc.collect()
    make_epoch_data(file_list=hearing_raw_list, config=config, zarr_group=hearing_group, preload=preload)
    gc.collect()
    
if __name__ == "__main__":
    main()
