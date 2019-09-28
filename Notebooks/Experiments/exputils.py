import sys
import numpy as np
import numba
import scipy.stats

sys.path.insert(0, "../../Evaluation")
import utils
clean_dataset = utils.clean_dataset
na_mask = utils.na_mask
pick_gpu_lowest_memory = utils.pick_gpu_lowest_memory


def subsample_roc(fpr, tpr, subsample_size=1000):
    dlength = np.concatenate([
        np.zeros(1),
        np.sqrt(np.square(fpr[1:] - fpr[:-1]) + np.square(tpr[1:] - tpr[:-1]))
    ], axis=0)
    length = dlength.sum()
    step = length / subsample_size
    cumlength = dlength.cumsum()
    nstep = np.floor(cumlength / step)
    landmark = np.concatenate([np.zeros(1).astype(np.bool_), nstep[1:] == nstep[:-1]], axis=0)
    return fpr[~landmark], tpr[~landmark]