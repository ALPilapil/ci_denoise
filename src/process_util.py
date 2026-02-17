import os

class PreprocessConfig():
    def __init__(self, channels, n_components, l_freq, years, wanted_epochs, 
                tmin, tmax, baseline, alpha=1, channel_scaling=None, gaussian_noise=False):
        '''
        configuration class for what parameters to run the data with 
        '''
        self.channels = channels
        self.n_components = n_components
        self.l_freq = l_freq
        self.years = years
        self.wanted_epochs = wanted_epochs
        self.baseline = baseline
        self.alpha = alpha
        self.gaussian_noise = gaussian_noise
        self.tmin = tmin
        self.tmax = tmax
        default_dict = {'Fp1':1, 'Fz':1, 'F3':1, 'F7':1, 'T7':1, 'C3':1, 
                             'Cz':1, 'Pz':1, 'P3':1, 'P7':1, 'O1':1, 'Fp2':1, 
                             'F8':1, 'F4':1, 'C4':1, 'T8':1, 'P8':1, 'P4':1, 
                             'O2':1, 'M1':1, 'M2':1}
        if channel_scaling:
            self.channel_scaling = {**default_dict, **channel_scaling}
        else:
            self.channel_scaling = channel_scaling

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