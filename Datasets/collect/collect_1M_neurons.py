#! /usr/bin/env python
# by caozj
# May 20, 2018
# 1:04:59 AM

import os
import numpy as np
import scipy.sparse
import h5py
from anndata import AnnData

import matplotlib as mpl
mpl.use("agg")
import scanpy.api as sc

import Cell_BLAST as cb


#===============================================================================
#
#  Preparation
#
#===============================================================================
input_file = "../download/1M_neurons/1M_neurons_filtered_gene_bc_matrices_h5.h5"
output_file = "../data/1M_neurons/data.h5"
output_half_file = "../data/1M_neurons_half/data.h5"
if not os.path.exists(os.path.dirname(output_file)):
    os.makedirs(os.path.dirname(output_file))
if not os.path.exists(os.path.dirname(output_half_file)):
    os.makedirs(os.path.dirname(output_half_file))


#===============================================================================
#
#  Read data
#
#===============================================================================
print("Reading data...")
with h5py.File(input_file, "r") as f:
    raw_data = scipy.sparse.csc_matrix((
        f['mm10/data'][...],
        f['mm10/indices'][...],
        f['mm10/indptr'][...]
    ), shape=f['mm10/shape'][...]).T
    barcodes = cb.utils.decode(f["mm10/barcodes"][...])
    gene_names = cb.utils.decode(f["mm10/gene_names"][...])
    genes = cb.utils.decode(f["mm10/genes"][...])


#===============================================================================
#
#  Use scanpy for normalization and variable gene selection
#
#===============================================================================
print("`scanpy` processing...")
adata = AnnData(X=raw_data)
adata.obs_names = barcodes
adata.var_names = gene_names
adata.var_names_make_unique()
adata.var["gene_id"] = genes
adata.raw = adata
sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
filter_result = sc.pp.filter_genes_dispersion(adata.X)
variable_genes = np.array(adata.var_names[filter_result.gene_subset])


#===============================================================================
#
#  Save results
#
#===============================================================================
print("Saving results...")
ds = cb.data.ExprDataSet(
    adata.raw.X, adata.obs, adata.var,
    dict(scanpy_genes=variable_genes)
)
ds.obs["dataset_name"] = "1M_neurons"
ds.write_dataset(output_file)

ds_half = ds[np.random.RandomState(0).choice(
    ds.shape[0], int(ds.shape[0] / 2), replace=False
), :]
ds_half.write_dataset(output_half_file)
# with h5py.File(output_file, "w") as f:
#     g = f.create_group("exprs")
#     g.create_dataset("data", data=adata.raw.X.data)
#     g.create_dataset("indices", data=adata.raw.X.indices)
#     g.create_dataset("indptr", data=adata.raw.X.indptr)
#     g.create_dataset("shape", data=adata.raw.X.shape)

#     f.create_dataset("obs_names", data=cb.utils.encode(adata.obs_names))
#     f.create_dataset("var_names", data=cb.utils.encode(adata.var_names))
#     g = f.create_group("obs")
#     g = f.create_group("var")
#     g.create_dataset("gene_id", data=cb.utils.encode(adata.var["gene_id"]))

#     g = f.create_group("uns")
#     g.create_dataset("variable_genes", data=cb.utils.encode(variable_genes))

print("Done!")
