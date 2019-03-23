"""
The Cell_BLAST package
"""

from . import utils
if not utils.in_ipynb():
    import matplotlib
    matplotlib.use("agg")

from . import blast
from . import data
from . import directi
from . import latent
from . import metrics
from . import prob
from . import rmbatch
from . import config

name = "Cell_BLAST"

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

__version__ = "0.1.0"
