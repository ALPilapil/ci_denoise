import pandas as pd

df = pd.read_csv("/quobyte/millerlmgrp/annotated_data/CMPy2/trial_onsets_0901y2.csv")
print("Blocks:", df['block'].unique())
print("Total trials:", len(df))
print("\nCondition breakdown by block:")
print(pd.crosstab(df['block'], df['condition']))

# Check if same stim_id appears in multiple blocks
stim_blocks = df.groupby(['condition', 'stim_id'])['block'].nunique()
print("\nStimuli presented across >1 block:")
print(stim_blocks[stim_blocks > 1].head(20))
print(f"Total multi-block stimuli: {(stim_blocks > 1).sum()} / {len(stim_blocks)}")
