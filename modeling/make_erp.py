import mne
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend, needed when there's no display
import matplotlib.pyplot as plt

def main():
    '''
    make an ERP with an option to plot it
    assume you will be taking in numpy arrays as input
    '''
    # grab info from one reference file
    ref = mne.io.read_raw_fif('noise0_raw_perm1.fif', preload=False)
    info = ref.info

    # load in data into an averaged numpy array


    # now use it to wrap any array with matching channels
    data = np.load('your_array.npy')  # must be (n_channels, n_times) with same ch order
    raw = mne.io.RawArray(data, info)
    fig = raw.plot(show=False)
    fig.savefig('/your/save/dir/plot.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    main()

