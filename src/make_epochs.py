from process_util import list_file_paths, permutation_divider, PreprocessConfig
from make_epochs import get_dims
import gc
import zarr
import numpy as np
import mne
import re
import os


# ─── Switch between paradigms here ───────────────────────────────────────────
# use fif for ica processed data and set for non
DATA_MODE = 'fif'   # 'set'  →  .set/.fdt files organized by year
                    # 'fif'  →  .fif files with perm encoded in filename
storage_directory = '/mnt/data/PilapilData/processed_data/epoched_data.zarr'
# ─────────────────────────────────────────────────────────────────────────────


def read_raw_file(path, preload):
    """Read a raw file using the appropriate MNE reader for DATA_MODE."""
    if DATA_MODE == 'set':
        return mne.io.read_raw_eeglab(path, preload=preload)
    elif DATA_MODE == 'fif':
        return mne.io.read_raw_fif(path, preload=preload, verbose=False)
    else:
        raise ValueError(f"Unknown DATA_MODE: '{DATA_MODE}'. Choose 'set' or 'fif'.")


def build_fif_path_lists(directory, prefix):
    """
    Scan a directory for .fif files matching <prefix><id>_raw_perm<n>.fif
    and group them into 3 lists by permutation index (0-indexed).

    e.g. noise0_raw_perm1.fif  →  paths[0], participant_id=0
         hearing3_raw_perm2.fif →  paths[1], participant_id=3

    Returns:
        paths          : list of 3 lists of absolute file paths
        participant_ids: list of 3 lists of int participant ids (parallel to paths)
    """
    paths = [[], [], []]
    participant_ids = [[], [], []]
    pattern = re.compile(rf'^{re.escape(prefix)}(\d+)_raw_perm(\d+)\.fif$')

    for fname in sorted(os.listdir(directory)):
        m = pattern.match(fname)
        if m:
            p_id   = int(m.group(1))
            perm   = int(m.group(2)) - 1   # convert 1-based label to 0-based index
            if 0 <= perm < 3:
                paths[perm].append(os.path.join(directory, fname))
                participant_ids[perm].append(p_id)

    return paths, participant_ids


def make_epoch_util(raw, config, zarr_group, preload, perm_label, participant_id):
    """
    Epoch a raw file and append data + metadata to zarr_group.

    For 'set' mode: uses event annotations already present in the file.
    For 'fif' mode: tries annotations first; falls back to fixed-length epochs
                    when no matching events are found (e.g. pure noise recordings).
    """
    epoch_data = None

    if DATA_MODE == 'set':
        events, _ = mne.events_from_annotations(raw)
        epochs = mne.Epochs(raw, events, event_id=config.wanted_epochs,
                            tmin=config.tmin, tmax=config.tmax,
                            baseline=config.baseline, preload=preload,
                            reject_by_annotation=True)
        epoch_data = epochs.get_data()
        event_ids  = epochs.events[:, 2]

    elif DATA_MODE == 'fif':
        try:
            events, event_dict = mne.events_from_annotations(raw)
            # filter to only wanted epochs if annotations exist
            matching = {k: v for k, v in event_dict.items()
                        if v in config.wanted_epochs}
            if matching:
                epochs = mne.Epochs(raw, events, event_id=matching,
                                    tmin=config.tmin, tmax=config.tmax,
                                    baseline=config.baseline, preload=preload,
                                    reject_by_annotation=True)
                epoch_data = epochs.get_data()
                event_ids  = epochs.events[:, 2]
            else:
                raise RuntimeError("No matching events — falling back to fixed-length")
        except Exception:
            # No usable annotations: chop the recording into fixed-length windows
            duration = config.tmax - config.tmin
            epochs   = mne.make_fixed_length_epochs(raw, duration=duration,
                                                    preload=preload, verbose=False)
            epoch_data = epochs.get_data()
            # use a placeholder event id (same as the first wanted epoch)
            event_ids = np.full(epoch_data.shape[0], config.wanted_epochs[0], dtype='int64')

    n_epochs           = epoch_data.shape[0]
    perm_array         = np.full(n_epochs, perm_label,      dtype='int8')
    block_array        = np.arange(n_epochs,                dtype='int64')
    participant_id_arr = np.full(n_epochs, participant_id,  dtype='int32')

    zarr_group['data'].append(epoch_data)
    zarr_group['labels'].append(event_ids)
    zarr_group['perm'].append(perm_array)
    zarr_group['block'].append(block_array)
    zarr_group['participant'].append(participant_id_arr)

    print(f"  processed {n_epochs} epochs (participant {participant_id}, perm {perm_label})")


def make_epoch_data(file_list, config, zarr_group, preload, perm_label, participant_id_list=None):
    """
    Iterate over file_list, read each file, epoch it, and append to zarr_group.

    participant_id_list: optional parallel list of ids (used for .fif mode where
                         ids are parsed from filenames). Falls back to sequential
                         counter when None (original .set behaviour).
    """
    for i, filename in enumerate(file_list):
        p_id = participant_id_list[i] if participant_id_list is not None else (i + 1)
        try:
            raw = read_raw_file(filename, preload=preload)
            make_epoch_util(raw=raw, config=config, zarr_group=zarr_group,
                            preload=preload, perm_label=perm_label,
                            participant_id=p_id)
        except Exception as e:
            print(f"Skipping {filename}: {e}")
            continue


def build_path_lists(config):
    """
    Return (ci_paths, hearing_paths, ci_pids, hearing_pids).

    For 'set' mode  : scans year directories, splits by permutation via log files.
                      participant id lists are None (sequential counter used).
    For 'fif' mode  : scans flat .fif directories, parses perm + id from filenames.
    """
    if DATA_MODE == 'set':
        ci_paths      = [[], [], []]
        hearing_paths = [[], [], []]

        for year in config.years:
            raw_directory       = f'/mnt/data/PilapilData/CMPy{year}/MarkerFixed/'
            raw_data_file_paths = list_file_paths(raw_directory)
            log_paths           = f'/mnt/data/PilapilData/CMPy{year}/Logs/'
            log_files           = list_file_paths(log_paths)

            hearing_data_paths = [p for p in raw_data_file_paths if '/08' in p and '.set' in p]
            ci_data_paths      = [p for p in raw_data_file_paths if '/09' in p and '.set' in p]

            permed_hearing = permutation_divider(set_paths=hearing_data_paths, log_paths=log_files)
            permed_ci      = permutation_divider(set_paths=ci_data_paths,      log_paths=log_files)

            for i in range(3):
                ci_paths[i].extend(permed_ci[i])
                hearing_paths[i].extend(permed_hearing[i])

        return ci_paths, hearing_paths, None, None   # None → sequential ids

    elif DATA_MODE == 'fif':
        noise_dir   = '/mnt/data/PilapilData/processed_data/noise/'
        hearing_dir = '/mnt/data/PilapilData/processed_data/hearing/'

        ci_paths,      ci_pids      = build_fif_path_lists(noise_dir,   prefix='noise')
        hearing_paths, hearing_pids = build_fif_path_lists(hearing_dir, prefix='hearing')

        return ci_paths, hearing_paths, ci_pids, hearing_pids

    else:
        raise ValueError(f"Unknown DATA_MODE: '{DATA_MODE}'")


def main():
    epoch_data_storage = storage_directory

    # ── config ────────────────────────────────────────────────────────────────
    preload          = False
    CI_chs           = ['P7', 'T7', 'M2', 'M1', 'P8']
    n_components     = 0.99999
    l_freq           = 2
    years            = [2, 3, 4]
    wanted_epochs    = [10]
    tmin             = 0.0
    tmax             = 60.0
    baseline         = (0, 0)
    alpha            = 1
    channel_scaling  = {}
    gaussian_noise   = False

    config = PreprocessConfig(channels=CI_chs,
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
    # ──────────────────────────────────────────────────────────────────────────

    ci_paths, hearing_paths, ci_pids, hearing_pids = build_path_lists(config)

    # ── zarr storage ──────────────────────────────────────────────────────────
    root         = zarr.open_group(epoch_data_storage, mode='w')
    ci_group     = root.create_group('ci_trial_data')
    hearing_group = root.create_group('hearing_trial_data')

    sample_raw = read_raw_file(ci_paths[0][0], preload=preload)
    n_channels, n_times = get_dims(raw=sample_raw, config=config)

    for group in (ci_group, hearing_group):
        group.create_dataset('data',        shape=(0, n_channels, n_times), chunks=(10, n_channels, n_times), dtype='float64')
        group.create_dataset('labels',      shape=(0,),                      chunks=(10,),                    dtype='int64')
        group.create_dataset('perm',        shape=(0,),                      chunks=(10,),                    dtype='int8')
        group.create_dataset('block',       shape=(0,),                      chunks=(10,),                    dtype='int64')
        group.create_dataset('participant', shape=(0,),                      chunks=(10,),                    dtype='int32')
    # ──────────────────────────────────────────────────────────────────────────

    for perm_idx in range(3):
        perm_label = perm_idx + 1

        make_epoch_data(file_list=ci_paths[perm_idx],
                        config=config,
                        zarr_group=ci_group,
                        preload=preload,
                        perm_label=perm_label,
                        participant_id_list=ci_pids[perm_idx] if ci_pids else None)
        gc.collect()

        make_epoch_data(file_list=hearing_paths[perm_idx],
                        config=config,
                        zarr_group=hearing_group,
                        preload=preload,
                        perm_label=perm_label,
                        participant_id_list=hearing_pids[perm_idx] if hearing_pids else None)
        gc.collect()


if __name__ == "__main__":
    main()