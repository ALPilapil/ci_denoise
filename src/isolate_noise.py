import mne
import numpy as np
import os
from mne.preprocessing import ICA
from process_util import list_file_paths 


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

def isolate_noise(set_paths, n_components, do_explain_variance, n_plot_components, CI_chs, l_freq, save_path, save_dir=None, h_freq=None):
    '''
    given a list of .set paths, isolate the noise from each via ICA
    input: list of dirty .set paths
    output: nothing
    '''
    montage = mne.channels.make_standard_montage('standard_1020')

    # tracked skipped files
    bad_ch_list = []
    bad_comp_list = []

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
            raw.info['bads'] = bad_channels
            raw.interpolate_bads(reset_bads=True)
        
        # check if we have enough channels left for ICA
        if len(raw.ch_names) < n_components:
            print(f"File {idx}: Not enough channels ({len(raw.ch_names)}) for {n_components} components, skip")
            bad_ch_list.append(f"File {path}, bad ch")
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
            bad_comp_list.append(f"File {path}, bad comp")
            continue

        # reconstruct the data back into its original form
        raw_clean = raw.copy() # copy original data
        ica.apply(raw_clean, exclude=top_components) # exclude the noisy components from it

        raw_noise_sensor = raw.copy() # copy the original data
        raw_noise_sensor._data = raw.get_data() - raw_clean.get_data() # subtract the clean components

        # save the data
        filename = os.path.join(save_path, f"noise{idx}_raw.fif")
        raw_noise_sensor.save(filename, overwrite=True)


    print(f"skipped {len(bad_comp_list) + len(bad_ch_list)} files out of {len(set_paths)}: ")
    print(f"{len(bad_ch_list)} for bad channels", bad_ch_list)
    print(f"{len(bad_comp_list)} for bad components", bad_comp_list)

def process_hearing(set_paths, l_freq, save_path, h_freq=None):
    '''
    Process hearing participant data: rename channels, filter, and save
    input: list of .set paths, filter parameters, save directory
    output: nothing
    '''
    # Create directory if it doesn't exist
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    montage = mne.channels.make_standard_montage('standard_1020')
    
    # Process each hearing file
    for idx, path in enumerate(set_paths):
        raw = mne.io.read_raw_eeglab(path, preload=True)
        
        # Rename mastoids to match CI data
        raw.rename_channels({'LMas': 'M1', 'RMas': 'M2'})
        
        # Set montage
        raw.set_montage(montage)
        
        # Apply same filter as CI data for consistency
        raw.filter(l_freq=l_freq, h_freq=h_freq)
        
        # Save
        filename = os.path.join(save_path, f"hearing{idx}_raw.fif")
        raw.save(filename, overwrite=True)
        print(f"Saved hearing file {idx}")
    
    print(f"Processed and saved {len(set_paths)} hearing files")


def main():
    '''
    isolate noise from data
    '''
    # params and paths
    noise_folder_path = '/quobyte/millerlmgrp/processed_data/noise/'
    hearing_folder_path = '/quobyte/millerlmgrp/processed_data/hearing/'
    CI_chs = ['P7', 'T7', 'M2', 'M1', 'P8'] # points where you would expect lots of CI noise from
    n_components = 0.99999
    years = [2, 3, 4]
    l_freq = 2


    # initialize empty lists to add to later
    ci_paths = []
    hearing_paths = []

    # this will all be absolute paths from quobyte
    for year in years:
        # set paths to raw EEG data and their log paths
        raw_directory = f'/quobyte/millerlmgrp/CMPy{year}/MarkerFixed/'
        raw_data_file_paths = list_file_paths(raw_directory)
        log_paths = f'/quobyte/millerlmgrp/CMPy{year}/Logs/'
        log_files = list_file_paths(log_paths)

        # specify to set paths for hearing vs CI kids
        hearing_data_paths = [path for path in raw_data_file_paths if ('/08' in path and '.set' in path)]
        ci_data_paths = [path for path in raw_data_file_paths if ('/09' in path and '.set' in path)]
        print(f"Amount hearing data files: {len(hearing_data_paths)}")
        print(f"Amount ci data files: {len(ci_data_paths)}")

        # divide into appropriate permuation paths, each of these is a 3 long list
        permed_hearing_paths = permutation_divider(set_paths=hearing_data_paths, log_paths=log_files)
        permed_ci_paths = permutation_divider(set_paths=ci_data_paths, log_paths=log_files)

        # extend to appropriate places
        for i in range(3):
            ci_paths.extend(permed_ci_paths[i])
            hearing_paths.extend(permed_hearing_paths[i])

    # run isolation
    isolate_noise(set_paths=ci_paths, CI_chs=CI_chs, do_explain_variance=False, n_plot_components=None, 
                                n_components=n_components, l_freq=l_freq, save_path=noise_folder_path)
    process_hearing(set_paths=hearing_paths, l_freq=l_freq, save_path=hearing_folder_path)

if __name__ == "__main__":
    main()