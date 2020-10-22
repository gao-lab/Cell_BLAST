#! /usr/bin/env python
# by weil
# Sep 23, 2020
# 8:10 PM

# save as anndata

import pandas as pd
import numpy as np
import scipy
import os
import scanpy as sc
from anndata import AnnData

#expression matrix
raw_data = scipy.io.mmread("../download/Cao_2019/GSE119945_gene_count.txt")
expr_mat = raw_data.tocsc()
expr_mat1 = expr_mat.T

#cell_df
cell_df=pd.read_csv("../download/Cao_2019/cell_annotate.csv", \
                    usecols=["sample", "embryo_id", "embryo_sex", "development_stage", "Main_Cluster", "Main_cell_type", "detected_doublet"])
cell_mask = cell_df["detected_doublet"] == False
cell_df = cell_df[cell_mask]
cell_df1 = cell_df.iloc[:, [0,1,2,3,6]]
cell_df1.columns = ["sample", "donor", "gender", "lifestage", "cell_type1"]

# datasets meta
datasets_meta=pd.read_csv("../ACA_datasets.csv", header=0, index_col=False)
# cell ontology
cell_ontology = pd.read_csv("../cell_ontology/mouse_embryo_cell_ontology.csv", usecols=["cell_type1", "cell_ontology_class", "cell_ontology_id"])

#gene_meta
gene_meta = pd.read_csv("../download/Cao_2019/gene_annotate.csv")
gene_mask0 = gene_meta.duplicated(["gene_short_name"])
gene_mask = []
for element in gene_mask0.values:
    gene_mask.append(not(element))
gene_meta.index = gene_meta["gene_short_name"]
gene_meta1 = gene_meta.iloc[np.where(gene_mask)[0], [0,1]]

expr_mat1 = expr_mat1[np.where(cell_mask.values)[0], :]
expr_mat1 = expr_mat1[:, np.where(gene_mask)[0]]

output_dir="../data/Cao_2019_anndata"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)  
    
# add dataset_meta
dataset_name="Cao_2019"
cell_df1["organism"]=datasets_meta.loc[datasets_meta["dataset_name"]==dataset_name, "organism"].item()
cell_df1["dataset_name"]=datasets_meta.loc[datasets_meta["dataset_name"]==dataset_name, "dataset_name"].item()
cell_df1["platform"]=datasets_meta.loc[datasets_meta["dataset_name"]==dataset_name, "platform"].item()
cell_df1["organ"]=datasets_meta.loc[datasets_meta["dataset_name"]==dataset_name, "organ"].item()

# add CL
cell_df1["cell_id"] = cell_df1.index
cell_df2 = cell_df1.merge(cell_ontology, left_on="cell_type1", right_on="cell_type1")
cell_df2.index = cell_df2["cell_id"]
cell_df2=cell_df2.reindex(cell_df1["cell_id"])
cell_df1=cell_df2.drop(columns="cell_id")

# AnnData
if isinstance(expr_mat, pd.DataFrame):
    adata=AnnData(X=expr_mat1.values, obs=cell_df1, var=gene_meta1)
else:
    adata=AnnData(X=expr_mat1, obs=cell_df1, var=gene_meta1)
adata.raw = adata

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
print("Selecting scanpy genes...")
sc.pp.highly_variable_genes(adata, min_mean=0.025, inplace=True)
print(np.sum(adata.var["highly_variable"]), "scanpy genes")
sc.pl.highly_variable_genes(adata, save=".pdf")
import shutil
shutil.move("./figures/filter_genes_dispersion.pdf", os.path.join(output_dir, "scanpy_genes.pdf"))

adata.X = adata.raw.X
adata.raw=None

print("Saving results...")
adata.write(os.path.join(output_dir, "data.h5ad"), compression="gzip", compression_opts=1)
