import numpy as np
import Cell_BLAST as cb

def clean_dataset(dataset, obs_col):
    mask = na_mask(dataset.obs[obs_col])
    cb.message.info("Cleaning removed %d cells." % mask.sum())
    return dataset[~mask, :]

def na_mask(arr):
    return np.in1d(arr.astype(str), ("", "na", "NA", "nan", "NaN"))