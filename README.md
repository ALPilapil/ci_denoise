# ci_denoise

Using machine learning to remove the Cochlear Implant noise from EEG data.

# data processing
isolate_noise.py uses ICA and a low frequency filter of 2 to isolate just the noise from the CI data. Applies the same filter to hearing participants as well. 

make_raw_epochs.py makes raw epochs without any preprocessing applied and saves the permutation it came from, the block for each epoch (ie order of appearance), and which participant it came from. Everything is stored in a zarr format. 

# data
'/quobyte/millerlmgrp/processed_data/raw_epoched_data.zarr'
- Contains raw, completely unprocessed epochs
- Along with meta data for each epoch's block, label, perm, and participant

'/quobyte/millerlmgrp/processed_data/processed_raw_epoched_data.zarr'
- Contains processed epochs in which the noise represents ICA isolated noise
- Both hearing and noise have gone through a 2 low freq pass filter
- Along with meta data for each epoch's block, label, perm, and participant

/quobyte/millerlmgrp/processed_data/hearing/
- .fif files where each has had a 2 low freq pass filter applied
- file names indicate a participant and which permutation it belongs to

/quobyte/millerlmgrp/processed_data/noise/
- .fif files where each has has a 2 low freq pass filter applied 
- file names indicate a participant and which permutation it belongs to