import zarr

root = zarr.open_group('/quobyte/millerlmgrp/processed_data/raw_epoched_data.zarr', mode='r')

ci = root['ci_trial_data']
hearing = root['hearing_trial_data']

print("CI data shape:", ci['data'].shape)
print("CI labels shape:", ci['labels'].shape)
print("CI perm shape:", ci['perm'].shape)

print("\nFirst 1 CI perm labels:", ci['perm'][:1])
print("First 1 CI event labels:", ci['labels'][:1])
