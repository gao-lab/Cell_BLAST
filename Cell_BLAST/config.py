r"""
Global configuration
"""

import torch

from .utils import autodevice

RANDOM_SEED = 0
N_JOBS = 1
DEVICE = autodevice()


_USE_GLOBAL = "__UsE_gLoBaL__"
_NAN_REPLACEMENT = "__nAn_RePlAcEmEnT__"
_WEIGHT_PREFIX_ = "_WeIgHt_PrEfIx_"

H5_COMPRESS_OPTS = {
    "compression": "gzip",
    "compression_opts": 7,
}
H5_TRACK_OPTS = {"track_times": False}

SUPERVISION = None
RESOLUTION = 10.0
THRESHOLD = 0.5
PCA_N_COMPONENTS = 50
NO_CLUSTER = False
METRIC = "cosine"
METRIC_KWARGS = {}
MNN = True
MNN_K = 5
