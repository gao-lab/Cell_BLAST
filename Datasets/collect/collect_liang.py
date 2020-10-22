#! /usr/bin/env python
# by weil
# Sep 17, 2020

import pandas as pd
import numpy as np
import Cell_BLAST as cb
import scipy
import os
import scanpy as sc
from anndata import AnnData
from utils import construct_dataset

# expr_mat
expr_mat1=pd.read_csv("../download/Liang/GSE133707_P1_Mac.txt.gz", sep=" ")
expr_mat2=pd.read_csv("../download/Liang/GSE133707_P1_Per.txt.gz", sep=" ")
expr_mat3=pd.read_csv("../download/Liang/GSE133707_P2_Mac.txt.gz", sep=" ")
expr_mat4=pd.read_csv("../download/Liang/GSE133707_P2_Per.txt.gz", sep=" ")
expr_mat5=pd.read_csv("../download/Liang/GSE133707_P3_Mac.txt.gz", sep=" ")
expr_mat6=pd.read_csv("../download/Liang/GSE133707_P3_Per.txt.gz", sep=" ")

# merge expr_mat
expr_mat=expr_mat1.merge(expr_mat2, left_index=True, right_index=True)
expr_mat=expr_mat.merge(expr_mat3, left_index=True, right_index=True)
expr_mat=expr_mat.merge(expr_mat4, left_index=True, right_index=True)
expr_mat=expr_mat.merge(expr_mat5, left_index=True, right_index=True)
expr_mat=expr_mat.merge(expr_mat6, left_index=True, right_index=True)

# reshape to cell * gene
expr_mat=expr_mat.T

# meta_df
meta_df=pd.read_csv("../download/Liang/meta.csv", index_col=0)
meta_df=meta_df[["subset", "macper", "cell_type"]]
meta_df.columns=["donor", "region", "cell_type1"]
meta_df["donor"]=meta_df["donor"].str.split("_").str[0]

# cell curation
cell_use=np.intersect1d(meta_df.index, expr_mat.index)
expr_mat=expr_mat.loc[cell_use]
meta_df=meta_df.loc[cell_use]

# datasets meta
datasets_meta=pd.read_csv("../ACA_datasets.csv", header=0, index_col=False)
# cell ontology
cell_ontology = pd.read_csv("../cell_ontology/retina_cell_ontology.csv",
                            usecols=["cell_type1", "cell_ontology_class", "cell_ontology_id"])

# gene_meta
gene_meta=pd.DataFrame(index=expr_mat.columns)

construct_dataset("../data/Liang", expr_mat, meta_df, gene_meta,
                  datasets_meta=datasets_meta, cell_ontology=cell_ontology, min_mean=0.02, max_mean=3, min_disp=0.7)
