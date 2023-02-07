r"""
The Cell_BLAST package
"""

try:
    from importlib.metadata import version
except ModuleNotFoundError:
    from pkg_resources import get_distribution

    version = lambda name: get_distribution(name).version

from .utils import in_ipynb

if not in_ipynb():
    import matplotlib

    matplotlib.use("agg")

from . import (
    blast,
    config,
    data,
    directi,
    latent,
    metrics,
    prob,
    rebuild,
    rmbatch,
    utils,
    weighting,
)

name = "Cell_BLAST"
__copyright__ = "2022, Gao Lab"
__author__ = "Zhi-Jie Cao, Runwei Lu"
__version__ = version(name)

__all__ = [
    "blast",
    "config",
    "data",
    "directi",
    "latent",
    "metrics",
    "prob",
    "rebuild",
    "rmbatch",
    "utils",
    "weighting",
]
