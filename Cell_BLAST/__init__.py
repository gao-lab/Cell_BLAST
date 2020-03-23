r"""
The Cell_BLAST package
"""

from .utils import in_ipynb

if not in_ipynb():
    import matplotlib
    matplotlib.use("agg")

from . import (blast, config, data, directi, latent, metrics, prob, rmbatch,
               utils)


name = "Cell_BLAST"

__copyright__ = "2020, Gao Lab"

__author__ = "Zhijie Cao"

__all__ = [
    "blast",
    "data",
    "directi",
    "latent",
    "metrics",
    "prob",
    "rmbatch",
    "utils",
    "config"
]

__version__ = "0.3.6"
