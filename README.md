# Cell BLAST

[![license-badge](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![pypi-badge](https://img.shields.io/pypi/v/Cell-BLAST)](https://pypi.org/project/Cell-BLAST)
[![docs-badge](https://readthedocs.org/projects/cblast/badge/?version=latest)](https://cblast.readthedocs.io/en/latest/?badge=latest)
[![build-badge](https://github.com/gao-lab/Cell_BLAST/actions/workflows/build.yml/badge.svg)](https://github.com/gao-lab/Cell_BLAST/actions/workflows/build.yml)

Cell BLAST is a cell querying tool for single-cell transcriptomics data.

For each query cell, it searches for most similar cells in the reference database.
Annotations in reference cells, e.g. cell type, can then be transfered to query
cells based on cell-to-cell similarities. See our
[publication](https://www.nature.com/articles/s41467-020-17281-7) for details.

## Installing the Python package

We only support installation via pip right now.

Installation within virtual environments are recommended, see
[virtualenv](https://virtualenv.pypa.io/en/latest/) or
[conda](https://conda.io/docs/user-guide/tasks/manage-environments.html).

For conda, here's a one-liner to set up an empty environment
for installing Cell BLAST:

`conda create -n cb python=3.9 && conda activate cb`

Then follow the instructions below to install Cell BLAST:

1. Install Cell BLAST from PyPI by running:

   `pip install Cell-BLAST`

   Or, install an editable dev version by running the following command
   under the root directory of this repo:

   `flit install -s`

2. Check if the package can be imported in the Python interpreter:

   ```python
   import Cell_BLAST as cb
   print(cb.__version__)
   ```

## Documentation

Online documentation can be found [here](https://cblast.readthedocs.org/).

## Web-based interface

We also provide a [Web-based service](http://cblast.gao-lab.org/) for
off-the-shelf querying of our ACA reference panels.

## Reproduce results in the paper

To reproduce results, please check out the `rep` branch.

## Contact

Feel free to submit an issue or contact us at
[cblast@mail.cbi.pku.edu.cn](mailto:cblast@mail.cbi.pku.edu.cn)
for problems about the Python package, website or database.
