import mne
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def make_erp(
    data,           # (n_trials, n_channels, n_times)
    ref_fif_path,
    tmin=0.0,       # time (s) at the start of each trial
    baseline=(None, 0),
    save_path='erp.png',
    picks=None,
):
    ref = mne.io.read_raw_fif(ref_fif_path, preload=False)
    info = ref.info

    epochs = mne.EpochsArray(
        data, info,
        tmin=tmin,
        baseline=baseline,
        verbose=False,
    )

    evoked = epochs.average()

    fig = evoked.plot(show=False)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved ERP to {save_path}")
    return evoked


def main():
    ref_fif = '/mnt/data/PilapilData/processed_data/hearing/hearing0_raw_perm1.fif'

    # data = np.load('model_output.npy')  # (n_batch, 21, 983041)
    rng = np.random.default_rng(42)
    dummy = rng.standard_normal((10, 21, 983041)) * 1e-6  # scale to EEG-like amplitude (volts)

    make_erp(dummy, ref_fif, save_path='erp.png', baseline=(0,0))


if __name__ == "__main__":
    main()