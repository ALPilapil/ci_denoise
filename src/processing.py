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


def isolate_noise(set_paths, n_components, do_explain_variance, n_plot_components, CI_chs, l_freq, h_freq=None, save_dir='ica_plots'):
    '''
    given a list of .set paths, isolate the noise from each via ICA
    input: list of dirty .set paths
    output: a list of raw instances of the isolated noise
    '''
    noise = []
    montage = mne.channels.make_standard_montage('standard_1020')
    os.makedirs(save_dir, exist_ok=True)

    # enfore n_plot components < n_compoenents
    if ((n_plot_components is not None) and (n_plot_components > n_components)) :
        raise ValueError("n_plot_components cannot be greater than n_components")

    # loop through each file and get the noise
    for idx, path in enumerate(set_paths): 
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

        # detect and drop bad channels
        data = raw.get_data()
        bad_channels = []
        
        for i, ch_name in enumerate(raw.ch_names):
            ch_data = data[i, :]
            
            # check for NaN/Inf
            if np.any(np.isnan(ch_data)) or np.any(np.isinf(ch_data)):
                bad_channels.append(ch_name)
                print(f"  File {idx}: {ch_name} has NaN/Inf - marking as bad")
            # check for flat channels (std very close to 0)
            elif np.std(ch_data) < 1e-10:
                bad_channels.append(ch_name)
                print(f"  File {idx}: {ch_name} is flat - marking as bad")
        
        # drop bad channels
        if bad_channels:
            print(f"File {idx}: Dropping {len(bad_channels)} bad channels: {bad_channels}")
            raw.drop_channels(bad_channels)
        
        # check if we have enough channels left for ICA
        if len(raw.ch_names) < n_components:
            print(f"File {idx}: Not enough channels ({len(raw.ch_names)}) for {n_components} components, skip")
            continue
        
        # Get updated channel indices after dropping
        m1_idx = raw.ch_names.index('M1') if 'M1' in raw.ch_names else None
        m2_idx = raw.ch_names.index('M2') if 'M2' in raw.ch_names else None

        # run ICA on it and store the data itself
        ica = ICA(n_components=n_components, max_iter="auto", random_state=97)
        ica.fit(raw)
        sources = ica.get_sources(raw)

        # explain each component
        if do_explain_variance:
            explain_variance(ica=ica, raw=raw, n_components=n_components)

        # plot where components are coming from
        if n_plot_components is not None:
            plot_ica(n_plot_components=n_plot_components, ica=ica, save_dir=save_dir, idx=idx)
        
        # save the ICAs that have some threshold percentage of their weight coming from around the point of
        # where the cochlear implant is placed
        component_matrix = ica.get_components()  # (n_channels, n_components)

        # get the top channels for each component
        top_components = []
        for i, column in enumerate(component_matrix.T):
            ranked = np.argsort(np.abs(column))[::-1]
            # convert into channel names
            ranked_ch = [raw.ch_names[index] for index in ranked]
            # print(f"{i}th component: {ranked_ch}")
            # if the top component is one that corresponds to being around the implant, include it as noise
            if (ranked_ch[0] in CI_chs) or (ranked_ch[1] in CI_chs):
                # store this component index as one to add later
                top_components.append(i)
        
        # account for no good components found
        if not top_components:
            print(f"no good components found for file {idx}, skip")
            continue

        # add the raw components to noise
        raw_noise = sources.copy().pick(top_components)
        noise.append(raw_noise)

    return noise

def save_raws(list_raws, save_path):
    '''
    input: list of mne.raw, some directory
    output: none
    '''
    for i, raw_obj in enumerate(list_raws):
        filename = os.path.join(save_path, f"raw_noise{i}.fif")
        raw_obj.save(filename, overwrite=True) # overwrite = True replaces folder of the same name

def read_raws(read_path, preload=False):
    '''
    input: path to read from
    output: list of raws
    '''
    list_raws = list_file_paths(read_path)
    for filename in list_raws:
        raw_loaded = mne.io.read_raw_fif(filename, preload=preload)
        list_raws.append(raw_loaded)

    return list_raws

def make_ERP(list_raws):
    '''
    average out instances of mne.raw into a single ERP 
    input: list of mne.raw objs
    output: ERP of them
    '''
    for raw in list_raws:
        print(raw.info)



def main():
    '''
    Isolates the CI noise from each CI kid via ICA and saves it to [path here]. Does this through running ICA on each datapoint
    and saving the components that primarily come from points of interest: ['P7', 'T7', 'M2', 'M1', 'P8'] since these are the 
    closest to where CIs are placed
    '''
    #----------- Parameters and Paths -----------#
    # where to store noise once isolated
    noise_folder_path = '/quobyte/millerlmgrp/processed_data/noise/'
    run_isolation = True
    # preprocessing parameters:
    CI_chs = ['P7', 'T7', 'M2', 'M1', 'P8'] # points where you would expect lots of CI noise from
    n_components = 5 # how many components to run ICA with, 10 is the max because of how much component 0 explains the variance
    l_freq = 2 # low frequency band 

    # get the paths to all the data in all years
    # year 2: 
    data_path_cmpy2 = '/quobyte/millerlmgrp/CMPy2/MarkerFixed/'
    data_files_paths_cmpy2 = list_file_paths(data_path_cmpy2)
    log_paths_cmpy2 = '/quobyte/millerlmgrp/CMPy2/Logs/'
    log_files_cmpy2 = list_file_paths(log_paths_cmpy2)
    # year 3: 
    # data_path_cmpy3 = '/quobyte/millerlmgrp/CMPy3/MarkerFixed/'
    # data_files_paths_cmpy3 = list_file_paths(data_path_cmpy3)
    # log_paths_cmpy3 = '/quobyte/millerlmgrp/CMPy3/Logs/'
    # log_files_cmpy3 = list_file_paths(log_paths_cmpy3)
    # year 4: 
    # data_path_cmpy4 = '/quobyte/millerlmgrp/CMPy4/MarkerFixed/'
    # data_files_paths_cmpy4 = list_file_paths(data_path_cmpy4)
    # log_paths_cmpy4 = '/quobyte/millerlmgrp/CMPy4/Logs/'
    # log_files_cmpy4 = list_file_paths(log_paths_cmpy4)
    # combine them all

    #----------- More Paths -----------#
    # divide data into hearing and CI, accounting for each permutation
    hearing_cmpy2_data_paths = [path for path in data_files_paths_cmpy2 if ('/08' in path and '.set' in path)]
    ci_cmpy2_data_paths = [path for path in data_files_paths_cmpy2 if ('/09' in path and '.set' in path)]
    print(f"Amount hearing data files: {len(hearing_cmpy2_data_paths)}")
    print(f"Amount ci data files: {len(ci_cmpy2_data_paths)}")

    # divide into appropriate permuation paths
    hearing_1, hearing_2, hearing_3 = permutation_divider(set_paths=hearing_cmpy2_data_paths, log_paths=log_files_cmpy2)
    ci_1, ci_2, ci_3 = permutation_divider(set_paths=ci_cmpy2_data_paths, log_paths=log_files_cmpy2)

    #----------- Noise Isolation -----------#
    # isolate the noise from the data via ICA, high pass it above 2 Hz with Butterwork, zero-phase
    if run_isolation:
        ci_1_noise = isolate_noise(set_paths=ci_1, CI_chs=CI_chs, do_explain_variance=False, n_plot_components=None, n_components=n_components, l_freq=l_freq)
        save_raws(list_raws=ci_1_noise, save_path=noise_folder_path)
    else:
        ci_1_noise = read_raws(read_path=noise_folder_path)

    # create an ERP of the dirty signal
    # NOTE: ch_names are now the ICA components, this is what SHOULD happen
    print(ci_1_noise)
    make_ERP(list_raws=ci_1_noise)

    # create clean dirty pairs for the data via noise injection

    # output this data as the final result of this script

    
if __name__ == "__main__":
    main()
    