import mne
import numpy as np
import os
from mne.preprocessing import ICA
import sklearn
import matplotlib.pyplot as plt

def list_file_paths(directory_path):
    '''
    given a directory list all the files in it in list form
    '''
    file_names = []
    # Iterate through all entries in the directory
    for entry in os.listdir(directory_path):
        full_path = os.path.join(directory_path, entry)
        # Check if the entry is a file
        if os.path.isfile(full_path):
            file_names.append(directory_path + entry)
    return file_names

def make_epochs_util(set_path, save_path, tmin, tmax):
    raw = mne.io.read_raw_eeglab(set_path, preload=True)
    # NOTE: at this point raw can be now used with any standard mne function
    # data = raw.get_data()

    # data info
    # print("available info", raw.info)
    '''
    access these via json type ex: raw.info['nchan']
    <Info | 8 non-empty values
    bads: []
    ch_names: A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, ...
    chs: 36 EEG
    custom_ref_applied: False
    dig: 36 items (36 EEG)
    highpass: 0.0 Hz
    lowpass: 8192.0 Hz
    meas_date: unspecified
    nchan: 36
    projs: []
    sfreq: 16384.0 Hz
    >
    '''
    events, event_id = mne.events_from_annotations(raw) # shape (n_events, 3)
    # print("raw annotations: ", raw.annotations)
    # values for indivudual events
    # print(f"event onset: {events[0][0].shape}") 
    # print(f"precedding sample signal value: {events[0][1].shape}") # typically just always 0
    # print(f"event onset: {events[0][2].shape}") 
    # print(event_id) # dict that shows how each event type was converted
    # print("list of event ids: ", list_event_ids)
    # print(f"Total events: {events.shape}")  # (8960, 3)
    # print(f"Event samples: {events[:, 0]}")  # All sample times
    # print(f"Event IDs: {events[:, 2]}")      # All event types

    # we only care about events: 98, 99, and the 100s
    filtered_event_id = {
        key: value for key, value in event_id.items() 
        if key in ['98', '99'] or (key.startswith('1') and len(key) == 3)
    }

    # create epochs out of the events
    epochs = mne.Epochs(raw, 
                        events=events, 
                        event_id=filtered_event_id,
                        tmin=tmin,
                        tmax=tmax,
                        preload=True)
    print("----- Epochs made successfully -----")
    
    # save the epochs
    filename = set_path.split('/')[-1].split('.')[0] + '-epo.fif' # need this suffix 
    epochs.save(save_path + filename, overwrite=True)

    # averaging across a single kind of event, using just 10 events
    # evoked_100 = epochs['5'][:10].average()
    # print(evoked_100)

def make_epochs(data_paths, save_path, tmin, tmax):
    '''
    just a loop, above function does real work
    returns the amount of files used to make epochs
    '''
    good_data_counter = 0
    for path in data_paths:
        try: 
            make_epochs_util(path, save_path, tmin=tmin, tmax=tmax)
            good_data_counter += 1
        except ValueError as e:
            print(f"ValueError for {path}: {e}")
            print("Moving on to next file\n")

    return good_data_counter

def plot_ica(n_plot_components, ica, save_dir, idx):
    '''
    given an ICA plot it and save it to the appropriate location
    '''
    component_idxs = list(range(n_plot_components))
    fig = ica.plot_components(picks=component_idxs, ch_type='eeg', show=False)

    # Handle both single figure and list of figures
    if isinstance(fig, list):
        for i, f in enumerate(fig):
            filename = os.path.join(save_dir, f'ica_components_file_{idx}_page_{i}.png')
            f.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(f)
            print(f"Saved plot to {filename}")
    else:
        filename = os.path.join(save_dir, f'ica_components_file_{idx}.png')
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved plot to {filename}")

def explain_variance(ica, raw, n_components):
    '''
    given ICA, explain how much of the data it explains along with how much each
    component explains the data
    '''
    explained_var_ratio = ica.get_explained_variance_ratio(raw)
    print("Fraction of eeg variance explained by all componenets: ", explained_var_ratio)

    # see how much of each component explains the variance
    for i in range(n_components):
        explained_var_ratio = ica.get_explained_variance_ratio(raw, components=[i], ch_type="eeg")
        print(f"Ratio for componenent {i}: {explained_var_ratio}")


def isolate_noise(set_paths, n_components, do_explain_variance, n_plot_components, l_freq, h_freq=None, save_dir='ica_plots'):
    '''
    given a list of .set paths, isolate the noise from each via ICA
    input: list of dirty .set paths
    output: the averaged out noise from each
    '''
    # NOTE: it would be good to see how much the noise from each datapoint varies
    noise = []
    montage = mne.channels.make_standard_montage('standard_1020')
    os.makedirs(save_dir, exist_ok=True)

    # enfore n_plot components < n_compoenents
    if ((n_plot_components is not None) and (n_plot_components > n_components)) :
        raise ValueError("n_plot_components cannot be greater than n_components")

    # loop through each file and get the noise
    for idx, path in enumerate(set_paths[:5]): 
        raw = mne.io.read_raw_eeglab(path, preload=True)

        # rename mastoids
        raw.rename_channels({
            'LMas': 'M1',
            'RMas': 'M2',
        })

        # get m1 and m2 indexes
        m1_idx = raw.ch_names.index('M1')
        m2_idx = raw.ch_names.index('M2')
        
        # set montage
        raw.set_montage(montage)

        # apply filter
        raw.filter(l_freq=l_freq, h_freq=h_freq)

        # run ICA on it
        ica = ICA(n_components=n_components, max_iter="auto", random_state=97)
        ica.fit(raw)

        # explain each component
        if do_explain_variance:
            explain_variance(ica=ica, raw=raw, n_compoenents=n_components)

        # plot where components are coming from
        if n_plot_components is not None:
            plot_ica(n_plot_components=n_plot_components, ica=ica, save_dir=save_dir, idx=idx)

        # save the ICAs of components centered around the mastoids
        componenent_matrix = ica.get_components() # (n_channels, n_components)
        # compare values across the index corresponding to the left and right mastoids
        # get the corresponding rows
        m1_weights = np.abs(componenent_matrix[m1_idx, :]) # (1, 5)
        m2_weights = np.abs(componenent_matrix[m2_idx, :])
        mastoid_weights = (m1_weights + m2_weights) / 2.0
        top_components = np.argsort(mastoid_weights) # indicies of the ordered highest components
        
        print(ica.get_components())

def permutation_divider(log_paths, set_paths):
    '''
    divide a list of set files into their appropriate permutation
    input: list of set files, corresponding logs file
    output: a tuple of 3 lists of set files, ith element of tuple = list of ith permutation
    '''
    # initialize lists
    perm1 = []
    perm2 = []
    perm3 = []

    # Create a mapping from subject ID to permutation version
    subject_to_perm = {}
    
    for log_path in log_paths:
        # Extract filename from path
        log_filename = os.path.basename(log_path)
        
        # Extract subject ID (first 4 digits)
        subject_id = log_filename[:4]
        
        # Extract version (v1, v2, or v3)
        if '_v1' in log_filename:
            subject_to_perm[subject_id] = 1
        elif '_v2' in log_filename:
            subject_to_perm[subject_id] = 2
        elif '_v3' in log_filename:
            subject_to_perm[subject_id] = 3
    
    # Assign set files to appropriate permutation list
    for set_path in set_paths:
        # Extract filename from path
        set_filename = os.path.basename(set_path)
        
        # Extract subject ID (first 4 digits)
        subject_id = set_filename[:4]
        
        # Assign to correct permutation
        if subject_id in subject_to_perm:
            perm_num = subject_to_perm[subject_id]
            if perm_num == 1:
                perm1.append(set_path)
            elif perm_num == 2:
                perm2.append(set_path)
            elif perm_num == 3:
                perm3.append(set_path)
    
    return (perm1, perm2, perm3)

        
def main(do_make_epochs=False, do_make_ERPs=False):
    # parameters and paths 
    ci_epochs_path = '/quobyte/millerlmgrp/CMPy2/ci_epochs/'
    hearing_epochs_path = '/quobyte/millerlmgrp/CMPy2/hearing_epochs/'
    tmin, tmax = -0.2, 0.5 # for epochs

    ci_ERPs_path = '/quobyte/millerlmgrp/CMPy2/ci_ERPs'
    hearing_ERPs_path = '/quobyte/millerlmgrp/CMPy2/hearing_ERPs'

    # get the paths to all the data in this year
    data_path_cmpy2 = '/quobyte/millerlmgrp/CMPy2/MarkerFixed/' # note it's an absolute path
    data_files_paths_cmpy2 = list_file_paths(data_path_cmpy2)
    log_paths_cmpy2 = '/quobyte/millerlmgrp/CMPy2/Logs/'
    log_files_cmpy2 = list_file_paths(log_paths_cmpy2)

    # divide data into hearing and CI, accounting for each permutation
    hearing_cmpy2_data_paths = [path for path in data_files_paths_cmpy2 if ('/08' in path and '.set' in path)]
    ci_cmpy2_data_paths = [path for path in data_files_paths_cmpy2 if ('/09' in path and '.set' in path)]
    print(f"Amount hearing data files: {len(hearing_cmpy2_data_paths)}")
    print(f"Amount ci data files: {len(ci_cmpy2_data_paths)}")

    # divide into appropriate permuation paths
    hearing_1, hearing_2, hearing_3 = permutation_divider(set_paths=hearing_cmpy2_data_paths, log_paths=log_files_cmpy2)
    ci_1, ci_2, ci_3 = permutation_divider(set_paths=ci_cmpy2_data_paths, log_paths=log_files_cmpy2)

    if do_make_epochs:
        # create epochs and store them
        good_ci_data_counter = make_epochs(ci_cmpy2_data_paths, save_path=ci_epochs_path, 
                                        tmin=tmin, tmax=tmax)
        good_ci_cmpy2_data_counter = make_epochs(hearing_cmpy2_data_paths, save_path=hearing_epochs_path, 
                                                tmin=tmin, tmax=tmax)

        print(f"made epochs for {good_ci_data_counter} files")
        print(f"made epochs for {good_ci_cmpy2_data_counter} files")

    # isolate the noise from the data via ICA, high pass it above 2 Hz with Butterwork, zero-phase
    # only need to run this on the CI data
    # n_componenets = 10 is the max because of how much component 0 explains the variance
    isolate_noise(set_paths=ci_cmpy2_data_paths, do_explain_variance=False, n_plot_components=None, n_components=5, l_freq=2)

    # create clean dirty pairs for the data via noise injection

    # output this data as the final result of this script

    
if __name__ == "__main__":
    main(do_make_epochs=False)
    