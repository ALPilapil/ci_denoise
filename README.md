# ci_denoise

Using machine learning to remove the Cochlear Implant noise from EEG data.

# data processing
isolate_noise.py uses ICA and a low frequency filter of 2 to isolate just the noise from the CI data. Applies the same filter to hearing participants as well. 

make_raw_epochs.py makes raw epochs without any preprocessing applied and saves the permutation it came from, the block for each epoch (ie order of appearance), and which participant it came from. Everything is stored in a zarr format. 