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