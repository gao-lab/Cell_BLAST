#! /usr/bin/env python
# by weil
# Sep 13, 2020
# 7:47 PM


import pandas as pd
import numpy as np
import Cell_BLAST as cb
import scipy
import os
import scanpy as sc
from anndata import AnnData
from utils import construct_dataset

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

construct_dataset("../data/Cao_2019", expr_mat1, cell_df1, gene_meta1,
                  datasets_meta=datasets_meta, cell_ontology=cell_ontology)

