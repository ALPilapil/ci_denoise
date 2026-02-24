from process_util import list_file_paths, permutation_divider, PreprocessConfig
from make_epochs import get_dims
import gc
import zarr
import numpy as np
import mne

def make_epoch_util(raw, config, zarr_group, preload, perm_label):
    events, event_dict = mne.events_from_annotations(raw)
    epochs = mne.Epochs(raw, events, event_id=config.wanted_epochs, tmin=config.tmin, tmax=config.tmax, 
                        baseline=config.baseline, preload=preload, reject_by_annotation=True)

    epoch_data = epochs.get_data()
    event_ids = epochs.events[:, 2]
    
    n = epoch_data.shape[0]
    perm_array = np.full(n, perm_label, dtype='int8')

    zarr_group['data'].append(epoch_data)
    zarr_group['labels'].append(event_ids)
    zarr_group['perm'].append(perm_array)  # new

    print("file processed")

def make_epoch_data(file_list, config, zarr_group, preload, perm_label):
    for filename in file_list:
        try:
            raw = mne.io.read_raw_eeglab(filename, preload=preload)
            make_epoch_util(raw=raw, config=config, zarr_group=zarr_group, preload=preload, perm_label=perm_label)
        except Exception as e:
            print(f"Skipping {filename}: {e}")
            continue

    


def main():
    noise_folder_path = '/quobyte/millerlmgrp/processed_data/noise/'
    hearing_folder_path = '/quobyte/millerlmgrp/processed_data/hearing/'
    epoch_data_storage = '/quobyte/millerlmgrp/processed_data/new_raw_epoched_data.zarr'
    years = [2, 3, 4]

    ci_paths = [[], [], []]
    hearing_paths = [[], [], []]

    for year in years:
        raw_directory = f'/quobyte/millerlmgrp/CMPy{year}/MarkerFixed/'
        raw_data_file_paths = list_file_paths(raw_directory)
        log_paths = f'/quobyte/millerlmgrp/CMPy{year}/Logs/'
        log_files = list_file_paths(log_paths)

        hearing_data_paths = [path for path in raw_data_file_paths if ('/08' in path and '.set' in path)]
        ci_data_paths = [path for path in raw_data_file_paths if ('/09' in path and '.set' in path)]

        permed_hearing_paths = permutation_divider(set_paths=hearing_data_paths, log_paths=log_files)
        permed_ci_paths = permutation_divider(set_paths=ci_data_paths, log_paths=log_files)

        for i in range(3):
            ci_paths[i].extend(permed_ci_paths[i])
            hearing_paths[i].extend(permed_hearing_paths[i])

    # config
    # preprocessing parameters
    preload = False
    truncation = None
    CI_chs = ['P7', 'T7', 'M2', 'M1', 'P8'] # points where you would expect lots of CI noise from
    n_components = 0.99999 # tells ICA to use however many components explain 99.9999% of the data
    l_freq = 2 # low frequency band 
    years = [2, 3, 4]
    wanted_epochs = [10] # needs to be a subset of event_dict
    tmin = 0.0
    tmax = 60.0
    baseline = (0,0)
    alpha = 1 # amount to scale noise by, keep it [.1, 2.0], SNR
    channel_scaling = {}
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

    # open storage
    root = zarr.open_group(epoch_data_storage, mode='w')
    ci_group = root.create_group('ci_trial_data')
    hearing_group = root.create_group('hearing_trial_data')

    # get dims once to initialize datasets
    sample_raw = mne.io.read_raw_eeglab(ci_paths[0][0], preload=preload)
    n_channels, n_times = get_dims(raw=sample_raw, config=config)

    ci_group.create_dataset('data', shape=(0, n_channels, n_times), chunks=(10, n_channels, n_times), dtype='float64')
    ci_group.create_dataset('labels', shape=(0,), chunks=(10,), dtype='int64')
    ci_group.create_dataset('perm', shape=(0,), chunks=(10,), dtype='int8')

    hearing_group.create_dataset('data', shape=(0, n_channels, n_times), chunks=(10, n_channels, n_times), dtype='float64')
    hearing_group.create_dataset('labels', shape=(0,), chunks=(10,), dtype='int64')
    hearing_group.create_dataset('perm', shape=(0,), chunks=(10,), dtype='int8')

    # process each permutation — perm_label is 1, 2, or 3
    for perm_idx in range(3):
        perm_label = perm_idx + 1
        make_epoch_data(file_list=ci_paths[perm_idx], config=config, zarr_group=ci_group, preload=preload, perm_label=perm_label)
        gc.collect()
        make_epoch_data(file_list=hearing_paths[perm_idx], config=config, zarr_group=hearing_group, preload=preload, perm_label=perm_label)
        gc.collect()

if __name__ == "__main__":
    main()