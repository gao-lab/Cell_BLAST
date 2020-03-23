# Cell BLAST

Cell BLAST is a cell querying tool for single-cell transcriptomics data.

For each query cell, it searches for most similar cells in the reference database.
Annotations in reference cells, e.g. cell type, can then be transfered to query
cells based on cell-to-cell similarities. See our
[preprint](https://www.biorxiv.org/content/10.1101/587360v2) for details.

![flowchart](docs/_static/flowchart.svg)

## Installing the python package

We only support installation via pip right now.

Installation within virtual environments are recommended, see
[virtualenv](https://virtualenv.pypa.io/en/latest/) or
[conda](https://conda.io/docs/user-guide/tasks/manage-environments.html).

For conda, here's a one-liner to set up an empty environment
for installing Cell BLAST:

`conda create -n cb python=3.6 && source activate cb`

Now follow the instructions below to install Cell BLAST:

1. Make sure you have a working version of tensorflow or tensorflow-gpu
   (version >= 1.5). You can follow the
   [official instructions](https://www.tensorflow.org/install/)
   about how to install tensorflow (and dependencies like CUDA and CuDNN
   for the GPU version), or just install via anaconda, which handles
   dependencies automatically:

   For installing the GPU supported version:
   `conda install tensorflow-gpu=1.8`

   For installing the CPU only version:
   `conda install tensorflow=1.8`

2. Install Cell BLAST by running:
   `pip install Cell-BLAST`

3. Check if the package can be imported in python interpreter:
   `import Cell_BLAST as cb`

Online documentation can be found [here](http://cblast.gao-lab.org/doc/index.html).

## Web-based interface

We also provide a [Web-based service](http://cblast.gao-lab.org/) for
off-the-shelf querying of our ACA reference panels.

## Repository structure

* The `Cell_BLAST` directory contains the Cell BLAST python package.
* The `Datasets` directory contains scripts used for building the ACA database.
* The `Evaluation` directory contains scripts used for benchmarking
  and producing some figures of the manuscript.
* The `docs` directory contains files used to generate the online documentation.
* The `test` directory contains unit tests for the python package.

## Reproduce results

### Obtain required data files

<!-- TODO: Provide data download information here -->

### Environment setup

#### Python

First create a conda environment and install most packages via:

```bash
conda env create -n cb-gpu -f env.yml && conda activate cb-gpu
```

Optionally, if GPU does not work properly (possibly due to inconsistent conda
channels used), reinstalling tensorflow should solve the problem:

```bash
conda install tensorflow=1.8.0 tensorflow-base=1.8.0 tensorflow-gpu=1.8.0 --force-reinstall
```

Finally, install customized packages or packages unavailable in conda.
All dependencies have already been installed via `env.yml`,
so `--no-deps` is added to prevent overwriting conda installed packages:

```bash
pip install Cell-BLAST==0.2.14 --no-deps
pip install local/scScope-0.1.5.tar.gz --no-deps  # Add random seed setting
pip install local/DCA-0.2.2.tar.gz --no-deps  # Allow GPU memory growth, suppress integer warning
pip install local/DCA_modpp-0.2.2.tar.gz --no-deps  # Modify preprocessing
pip install local/ZIFA-0.1.tar.gz --no-deps  # Remove fixed random seeds
pip install local/Dhaka-0.1.tar.gz --no-deps
pip install local/scvi-0.2.3.tar.gz --no-deps  # Fix torch bugs
```

#### R

Start R (tested on version `3.6.0`) at project root and run:

```R
packrat::restore()
```

Then install the customized version of Seurat by:

```R
install.packages("local/seurat-2.3.3.tar.gz", repos=NULL, type="source")  # Remove fixed random seeds
```

For CCA anchor (Seurat v3), we used a separate packrat environment.

To build this dedicated environment, start R at directory "packrat/envs/seurat_v3" and run:

```R
packrat::restore()
```

Then install the customized version of Seurat v3 by:

```R
install.packages("../../../local/seurat-3.0.2.tar.gz", repos=NULL, type="source")  # Remove fixed random seeds
```

### Run all benchmarks and reproduce figures

Make sure the conda environment create above is activated.
Go to directory "Evaluation" and run the following command:

```bash
snakemake -prk
```

Some jobs will likely fail, e.g. due to timeout or memory issues, and cause
downstream steps, including result plotting, to fail as well.

These failing jobs will be blacklisted in future runs, so just run the above
command for a second time, and results for the successful jobs should be
summarized and plotted without error.

## Contact

Feel free to submit an issue or contact us at
[cblast@mail.cbi.pku.edu.cn](mailto:cblast@mail.cbi.pku.edu.cn)
for problems about the python package, website or database.
