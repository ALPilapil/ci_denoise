import zarr
from pathlib import Path
from process_util import PreprocessConfig, list_file_paths 
import mne
import argparse


def read_raws(read_path, truncation=None, preload=False):
    '''
    input: path to read from
    output: list of raws
    '''
    file_paths = list_file_paths(read_path)
    # print("list raws: ", file_paths)
    list_raws = []

    if not isinstance(truncation, (int, None)):
        raise TypeError("Function read_raws only accepts None or int as an argument")

    if truncation:
        file_paths = file_paths[:truncation]

    for filename in file_paths:
        # print("filename: ", filename)
        raw_loaded = mne.io.read_raw_fif(filename, preload=preload)
        list_raws.append(raw_loaded)

    return list_raws

def make_epoch_data(eeg_data, config, zarr_group):
    '''
    given a list of mne.raw objects create and save just the epochs that are relevant to us
    these relevant epochs are identified by their event name and come into this function via a list
    input: set_paths, save_path, wanted_events
    output: none, saves to zarr file
    '''   
    for raw in eeg_data[:1]:
        events, event_dict = mne.events_from_annotations(raw)
        epochs = mne.Epochs(raw, events, event_id=config.wanted_epochs, tmin=config.tmin, tmax=config.tmax, 
                            baseline=config.baseline, preload=True, reject_by_annotation=True)

        n_epochs = len(epochs)
        n_channels = len(epochs.ch_names)
        n_times = len(epochs.times)
        print(f"Shape will be: ({n_epochs}, {n_channels}, {n_times})")

        # get the epoch data for this participant
        epoch_data = epochs.get_data() # (n_epochs, n_channels, n_times) appears in chronological time

        # store the data in an X, data array and Y, label array
        # epoch data is (n_events, n_channels, n_times) and event is ids is (n_events). ith event id = ith epoch data event id
        # just storing 2 numpy arrays in total 
        event_ids = epochs.events[:, 2] # NOTE: from here the events have been translated according to event_dict. so 100 -> 2, 104 -> 4 etc. this is fine 

        # append to the group
        zarr_group['data'].append(epoch_data)
        zarr_group['labels'].append(event_ids)

def make_pairs(zarr_root_read, zarr_root_save, config, batch_size=100, truncation=None):
    '''
    read in the epoch data. create pairs out of it by adding noise to the hearing epochs
    '''
    read_root = zarr.open_group(zarr_root_read, mode='r')
    ci_data = read_root['ci_trial_data']['data']
    hearing_data = read_root['hearing_trial_data']['data']
    save_root = zarr.open_group(zarr_root_save, mode='w')

    # load in the data for each group
    if truncation is not None:
        ci_data = ci_data[:truncation]
        hearing_data = hearing_data[:truncation]
    
    clean_batch = []
    dirty_batch = []
    
    # iterate through both groups to make pairs and save them
    for clean_epoch in hearing_data:  # (21, 8193)
        print("appending in process")
        for noisy_epoch in ci_data:    # (21, 8193)
            clean_batch.append(clean_epoch)
            # multiply noise by some alpha
            dirty_epoch = clean_epoch + config.alpha * noisy_epoch
            # if config.channel_scaling:
                # multiply the values in certain bands by a certain amount

                
            if config.gaussian_noise:
                # add some gausian noise in addition to the CI noise to the data
                gausian = np.random.normal(0, sigma, size=noisy_epoch.shape)
                dirty_epoch += gausian
            

            dirty_batch.append(dirty_epoch)
            
            # When batch is full, save and clear
            if len(clean_batch) >= batch_size:
                # Convert to arrays and append
                clean_arr = np.array(clean_batch)  # (batch_size, 21, 8193)
                dirty_arr = np.array(dirty_batch)
                
                save_root['clean'].append(clean_arr)
                save_root['dirty'].append(dirty_arr)
                                
                # Clear batch
                clean_batch = []
                dirty_batch = []
    
    # Save any remaining data
    if len(clean_batch) > 0:
        clean_arr = np.array(clean_batch)
        dirty_arr = np.array(dirty_batch)
        
        save_root['clean'].append(clean_arr)
        save_root['dirty'].append(dirty_arr)
        
        print(f"Saved final batch. Total: {save_root['clean'].shape[0]}")
        
def main():
    '''
    Isolates the CI noise from each CI kid via ICA and saves it to [path here]. Does this through running ICA on each datapoint
    and saving the components that primarily come from points of interest: ['P7', 'T7', 'M2', 'M1', 'P8'] since these are the 
    closest to where CIs are placed
    '''
    # command line args
    parser = argparse.ArgumentParser(description='processing script')
    parser.add_argument("process", help="define what to run. Options: 'make_epoch' or 'make_pairs' or 'both'")
    args = parser.parse_args()

    if args.process == 'make_epoch' or args.process == 'make_pairs' or args.process == 'both':
        print("Processing: ", args.process)
    else:
        raise ValueError("Action must be either 'make_epoch' or 'make_pairs' or 'both'") 

    #----------- Parameters and Paths -----------#
    # where to store noise once isolated
    noise_folder_path = '/quobyte/millerlmgrp/processed_data/noise/'
    noisy_epochs_folder_path = '/quobyte/millerlmgrp/processed_data/noisy_epochs/'
    epoch_data_storage = '/quobyte/millerlmgrp/processed_data/epoched_data.zarr'
    hearing_folder_path = '/quobyte/millerlmgrp/processed_data/hearing/'
    epoch_pair_path = '/quobyte/millerlmgrp/processed_data/epoched_pairs.zarr'

    # preprocessing parameters
    CI_chs = ['P7', 'T7', 'M2', 'M1', 'P8'] # points where you would expect lots of CI noise from
    n_components = 0.99999 # tells ICA to use however many components explain 99.9999% of the data
    l_freq = 2 # low frequency band 
    years = [2, 3, 4]
    wanted_epochs = [10, 11] # needs to be a subset of event_dict
    tmin = 0.0
    tmax = 1.5
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


    # get raw data
    ci_raws = read_raws(read_path=noise_folder_path, truncation=3)
    hearing_raws = read_raws(read_path=hearing_folder_path, truncation=3)

    # save data of just the relevant epochs of interest for the data of both kinds of particpants
    if parser.action == 'make_epoch' or parser.action == 'both': 
        root = zarr.open_group(epoch_data_storage, mode='w')
        ci_group = root['ci_trial_data']
        hearing_group = root['hearing_trial_data']
        # make epoch data for the raw CIs
        make_epoch_data(eeg_data=ci_raws, zarr_group=ci_group, config=config)
        # make epoch data for the hearing participants
        make_epoch_data(eeg_data=hearing_raws, zarr_group=hearing_group, config=config)
    
    # create clean dirty pairs for the data via noise injection
    # save this data as the final result of this script
    if parser.action == 'make_pairs' or parser.action == 'both':
        make_pairs(zarr_root_read=epoch_data_storage, zarr_root_save=epoch_pair_path, config=config, truncation=100)

if __name__ == "__main__":
    main()
