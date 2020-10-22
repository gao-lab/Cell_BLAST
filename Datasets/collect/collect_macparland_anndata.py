#! /usr/bin/env python
# by weil
# Sep 24, 2020

# save as anndata

import pandas as pd
import numpy as np
import scipy
import os
import scanpy as sc
from anndata import AnnData

# expr matrix
expr_mat=pd.read_csv("../download/MacParland/GSE115469_Data.csv.gz", index_col=0)

# reshape to cell * gene
expr_mat=expr_mat.T

# cell meta
meta_df=pd.read_csv("../download/MacParland/Cell_clusterID_cycle.txt", sep="\t", index_col=0)
meta_df.columns=["donor", "barcode", "cluster", "cell_cycle"]
meta_df=meta_df.drop(columns="barcode")
meta_df["donor"]=meta_df["donor"].str.slice(0, 2)

# add donor meta
donor_meta=pd.read_csv("../download/MacParland/donor_annotation.csv")
meta_df["cell_id"]=meta_df.index
meta_df1=meta_df.merge(donor_meta, on="donor")

# add cluster annotation
cluster_annotation=pd.read_csv("../download/MacParland/cluster_annotation.csv")
meta_df2=meta_df1.merge(cluster_annotation, on="cluster")
meta_df2.index = meta_df2["cell_id"]
meta_df2=meta_df2.reindex(meta_df.index)
meta_df2=meta_df2.drop(columns="cell_id")

# datasets meta
datasets_meta=pd.read_csv("../ACA_datasets.csv", header=0, index_col=False)
# cell ontology
cell_ontology = pd.read_csv("../cell_ontology/liver_cell_ontology.csv",
                            usecols=["cell_type1", "cell_ontology_class", "cell_ontology_id"])

# gene_meta
gene_meta=pd.DataFrame(index=expr_mat.columns)

output_dir="../data/MacParland_anndata"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)  
    
# add dataset_meta
dataset_name="MacParland"
meta_df2["organism"]=datasets_meta.loc[datasets_meta["dataset_name"]==dataset_name, "organism"].item()
meta_df2["dataset_name"]=datasets_meta.loc[datasets_meta["dataset_name"]==dataset_name, "dataset_name"].item()
meta_df2["platform"]=datasets_meta.loc[datasets_meta["dataset_name"]==dataset_name, "platform"].item()
meta_df2["organ"]=datasets_meta.loc[datasets_meta["dataset_name"]==dataset_name, "organ"].item()

# add CL
meta_df2["cell_id"] = meta_df2.index
meta_df3 = meta_df2.merge(cell_ontology, left_on="cell_type1", right_on="cell_type1")
meta_df3.index = meta_df3["cell_id"]
meta_df3=meta_df3.reindex(meta_df2["cell_id"])
meta_df2=meta_df3.drop(columns="cell_id")
    
# AnnData
if isinstance(expr_mat, pd.DataFrame):
    adata=AnnData(X=expr_mat.values, obs=meta_df2, var=gene_meta)
else:
    adata=AnnData(X=expr_mat, obs=meta_df2, var=gene_meta)
adata.raw = adata
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
print("Selecting scanpy genes...")
sc.pp.highly_variable_genes(adata, min_mean=0.05, max_mean=3, min_disp=0.8, inplace=True)
print(np.sum(adata.var["highly_variable"]), "scanpy genes")
sc.pl.highly_variable_genes(adata, save=".pdf")
import shutil
shutil.move("./figures/filter_genes_dispersion.pdf", os.path.join(output_dir, "scanpy_genes.pdf"))

adata.X = adata.raw.X
adata.raw = None

print("Saving results...")
adata.write(os.path.join(output_dir, "data.h5ad"), compression="gzip", compression_opts=1)
