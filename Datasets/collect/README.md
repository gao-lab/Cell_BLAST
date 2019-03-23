# README

This directory contains scripts for collecting data and storing them in a standardized format

Collected data are stored in hdf5 files. Every file contains:

* `exprs` that stores the expression matrix:
    - For dense format, `exprs` is a hdf5 dataset of matrix
    - For sparse format, `exprs` is a hdf5 group containing the following datasets, following
      CSR/CSC format for sparse matrices: `data`, `indices`, `indptr`, `shape`

    - `exprs` dataset are stored as $cell \times gene$ matrices in C format. Note that when using
      R to read it, they are transposed into $gene \times cell$ format.

* `obs` group that stores meta data of cells. Typical `obs` datasets include:
    - `cell_type1`: required, cell types mostly used to do evaluation, should readily map to CL
    - `cell_type2`: optional, finer cell type
    - `tissue`: optional, tissue type

    - `donor`: cell source
    - `disease`: disease state of the donor
    - `race`: race of the donor
    - `gender`: gender of the donor
    - `age`: age of the donor

    - `platform`: experimental platform
    - `batch`: experimental batch

* `var` group that stores meta data of genes

* `uns` group stores other meta data without structural constraint, typically gene selections


Data collection scripts should at least perform the following tasks:

* Build the expression matrix in a format that can be coerced to standard matrix or sparse matrix
* Build the cell meta table contains as many as possible the slots listed in `obs` (see above)
* Provide a cell type to cell ontology mapping
* Filter genes to remove ERCC
* Filter cells to remove those not well annotated by author
* Call `construct_dataset` to write the dataset to standard hdf5 format
