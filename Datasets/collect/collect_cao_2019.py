#! /usr/bin/env python
# by weil
# Jul 31, 2019
# 9:47 PM

import pandas as pd
import numpy as np
import sys
sys.path.append("../../")
import Cell_BLAST as cb
import scipy
import os
import scanpy as sc
from anndata import AnnData

output_file = "../data/Cao_2019/data.h5"
if not os.path.exists(os.path.dirname(output_file)):
    os.makedirs(os.path.dirname(output_file))

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
cell_df1["dataset_name"] = "Cao_2019"
cell_df1["organism"] = "Mus musculus"
cell_df1["organ"] = "Embryo"
cell_df1["platform"] = "sci-RNA-seq3"
cell_ontology = pd.read_csv("../cell_ontology/mouse_embryo_cell_ontology.csv", usecols=["cell_type1", "cell_ontology_class", "cell_ontology_id"])
cell_df2 = cell_df1.merge(cell_ontology, left_on="cell_type1", right_on="cell_type1")
cell_df2.index = cell_df2["sample"]
cell_df2=cell_df2.loc[cell_df["sample"]]
cell_df2=cell_df2.iloc[:, 1:11]

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

#Use scanpy for normalization and variable gene selection

adata = AnnData(X=expr_mat1)
adata.obs_names = cell_df2.index
adata.var_names = gene_meta1.index
adata.raw = adata
sc.pp.normalize_total(adata, target_sum=1)
sc_genes = sc.pp.highly_variable_genes(adata, inplace=False, min_mean=1e-5, max_mean = 8e-4, min_disp=1e-8)
scanpy_genes = gene_meta1[sc_genes["highly_variable"]].index

#expressed genes
expressed = np.sum(expr_mat1>1, axis=0)>5
expressed_genes = gene_meta1[np.array(expressed)[0]].index

#saving results
print("Saving results...")
cao_2019 = cb.data.ExprDataSet(
    expr_mat1, cell_df2, gene_meta1, {"scanpy_genes": np.array(scanpy_genes), "expressed_genes": np.array(expressed_genes)}
)

cao_2019.write_dataset(output_file)

print("Done!")

